"""Windows system-tray host for FaceFlow.

The web API remains resident and lightweight. Physical webcam capture starts
manually or when Windows reports that another application is requesting camera
access, then a demand-started session stops after the caller goes idle.
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Iterable, Optional

import psutil
import uvicorn
from PIL import Image, ImageDraw

import web_api

try:
    import pystray
except ImportError as exc:  # pragma: no cover - startup guidance
    raise SystemExit("Tray mode requires pystray. Run: python -m pip install pystray") from exc

if os.name == "nt":
    import winreg
    from ctypes import wintypes


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
WEBCAM_REGISTRY = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"


def _make_icon() -> Image.Image:
    size = 64
    image = Image.new("RGBA", (size, size), (8, 10, 15, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 60, 60), radius=16, fill=(24, 27, 38, 255), outline=(139, 124, 246, 255), width=3)
    draw.ellipse((16, 16, 48, 48), outline=(178, 167, 255, 255), width=5)
    draw.ellipse((26, 26, 38, 38), fill=(85, 214, 167, 255))
    return image


def _registry_children(root, path: str) -> Iterable[tuple[str, int, int]]:
    try:
        with winreg.OpenKey(root, path) as key:
            try:
                start = int(winreg.QueryValueEx(key, "LastUsedTimeStart")[0] or 0)
            except OSError:
                start = 0
            try:
                stop = int(winreg.QueryValueEx(key, "LastUsedTimeStop")[0] or 0)
            except OSError:
                stop = 0
            if start:
                yield path.rsplit("\\", 1)[-1], start, stop

            index = 0
            while True:
                try:
                    child = winreg.EnumKey(key, index)
                except OSError:
                    break
                index += 1
                yield from _registry_children(root, f"{path}\\{child}")
    except OSError:
        return


def _active_registry_camera_clients() -> list[str]:
    """Fallback for packaged applications reported by Windows privacy state."""
    if os.name != "nt":
        return []
    own_registry_name = str(Path(sys.executable).resolve()).replace("\\", "#").lower()
    clients = []
    for name, started, stopped in _registry_children(winreg.HKEY_CURRENT_USER, WEBCAM_REGISTRY):
        if name.lower() == own_registry_name:
            continue
        if started > 0 and (stopped == 0 or started > stopped):
            clients.append(name.replace("#", "\\"))
    return sorted(set(clients))


if os.name == "nt":
    class _SystemHandleEntry(ctypes.Structure):
        _fields_ = [
            ("object", ctypes.c_void_p),
            ("process_id", ctypes.c_size_t),
            ("handle_value", ctypes.c_size_t),
            ("granted_access", wintypes.ULONG),
            ("creator_backtrace_index", wintypes.USHORT),
            ("object_type_index", wintypes.USHORT),
            ("handle_attributes", wintypes.ULONG),
            ("reserved", wintypes.ULONG),
        ]


    class _UnicodeString(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.USHORT),
            ("maximum_length", wintypes.USHORT),
            ("buffer", wintypes.LPWSTR),
        ]


class ObsVirtualCameraConsumerProbe:
    """Find processes that actually opened the OBS virtual-camera queue.

    Windows' webcam privacy registry misses ordinary desktop DirectShow
    clients. The OBS filter opens the named shared-memory section
    ``OBSVirtualCamVideo`` while it is actively consuming frames, so matching
    that section's live process handles gives us a precise demand signal.
    """

    SYSTEM_EXTENDED_HANDLE_INFORMATION = 64
    OBJECT_NAME_INFORMATION = 1
    PROCESS_DUP_HANDLE = 0x0040
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    DUPLICATE_SAME_ACCESS = 0x00000002
    FILE_MAP_READ = 0x0004
    QUEUE_NAME = "OBSVirtualCamVideo"

    def __init__(self) -> None:
        self._negative_cache: dict[tuple[int, int], float] = {}
        self._cache_seconds = 12.0
        self._buffer_size = 32 * 1024 * 1024
        self._available = os.name == "nt"
        if not self._available:
            return
        self._ntdll = ctypes.WinDLL("ntdll")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ntdll.NtQuerySystemInformation.argtypes = [
            wintypes.ULONG,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
        ]
        self._ntdll.NtQuerySystemInformation.restype = ctypes.c_long
        self._ntdll.NtQueryObject.argtypes = [
            wintypes.HANDLE,
            wintypes.ULONG,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
        ]
        self._ntdll.NtQueryObject.restype = ctypes.c_long
        self._kernel32.OpenFileMappingW.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        self._kernel32.OpenFileMappingW.restype = wintypes.HANDLE
        self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.DuplicateHandle.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._kernel32.DuplicateHandle.restype = wintypes.BOOL
        self._kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    def _handle_snapshot(self) -> tuple[ctypes.Array, int]:
        size = self._buffer_size
        for _ in range(4):
            buffer = ctypes.create_string_buffer(size)
            required = wintypes.ULONG()
            status = self._ntdll.NtQuerySystemInformation(
                self.SYSTEM_EXTENDED_HANDLE_INFORMATION,
                buffer,
                size,
                ctypes.byref(required),
            )
            if status == 0:
                self._buffer_size = size
                return buffer, ctypes.c_size_t.from_buffer(buffer, 0).value
            size = max(size * 2, int(required.value) + 65536)
        raise OSError("Windows handle table could not be read")

    def active_clients(self) -> Optional[list[str]]:
        if not self._available:
            return None
        own_mapping = self._kernel32.OpenFileMappingW(
            self.FILE_MAP_READ,
            False,
            self.QUEUE_NAME,
        )
        if not own_mapping:
            return []

        process_handles: dict[int, int] = {}
        results: list[str] = []
        now = time.monotonic()
        try:
            buffer, count = self._handle_snapshot()
            entry_size = ctypes.sizeof(_SystemHandleEntry)
            base_offset = ctypes.sizeof(ctypes.c_size_t) * 2
            own_pid = os.getpid()
            own_handle_value = int(own_mapping)
            section_type = None
            candidates: list[tuple[int, int]] = []

            for index in range(count):
                entry = _SystemHandleEntry.from_buffer(buffer, base_offset + index * entry_size)
                pid = int(entry.process_id)
                handle_value = int(entry.handle_value)
                if pid == own_pid and handle_value == own_handle_value:
                    section_type = int(entry.object_type_index)
                if entry.granted_access == self.FILE_MAP_READ:
                    candidates.append((index, pid))

            if section_type is None:
                return None

            current_process = self._kernel32.GetCurrentProcess()
            live_signatures: set[tuple[int, int]] = set()
            for index, pid in candidates:
                entry = _SystemHandleEntry.from_buffer(buffer, base_offset + index * entry_size)
                if int(entry.object_type_index) != section_type or pid in (0, 4, own_pid):
                    continue
                handle_value = int(entry.handle_value)
                signature = (pid, handle_value)
                live_signatures.add(signature)
                cached_at = self._negative_cache.get(signature)
                if cached_at is not None and now - cached_at < self._cache_seconds:
                    continue

                if pid not in process_handles:
                    process_handle = int(
                        self._kernel32.OpenProcess(
                            self.PROCESS_DUP_HANDLE | self.PROCESS_QUERY_LIMITED_INFORMATION,
                            False,
                            pid,
                        )
                        or 0
                    )
                    process_handles[pid] = process_handle
                process_handle = process_handles[pid]
                if not process_handle:
                    continue

                duplicate = wintypes.HANDLE()
                copied = self._kernel32.DuplicateHandle(
                    wintypes.HANDLE(process_handle),
                    wintypes.HANDLE(handle_value),
                    current_process,
                    ctypes.byref(duplicate),
                    0,
                    False,
                    self.DUPLICATE_SAME_ACCESS,
                )
                if not copied:
                    continue
                try:
                    name_buffer = ctypes.create_string_buffer(1024)
                    required = wintypes.ULONG()
                    status = self._ntdll.NtQueryObject(
                        duplicate,
                        self.OBJECT_NAME_INFORMATION,
                        name_buffer,
                        len(name_buffer),
                        ctypes.byref(required),
                    )
                    object_name = ""
                    if status == 0:
                        value = _UnicodeString.from_buffer(name_buffer)
                        if value.buffer and value.length:
                            object_name = ctypes.wstring_at(value.buffer, value.length // 2)
                finally:
                    self._kernel32.CloseHandle(duplicate)

                if object_name.endswith(self.QUEUE_NAME):
                    try:
                        process = psutil.Process(pid)
                        label = f"{process.name()} ({pid})"
                    except (psutil.Error, OSError):
                        label = f"process {pid}"
                    results.append(label)
                else:
                    self._negative_cache[signature] = now

            self._negative_cache = {
                signature: timestamp
                for signature, timestamp in self._negative_cache.items()
                if signature in live_signatures and now - timestamp < self._cache_seconds
            }
            return sorted(set(results))
        except Exception as exc:
            logging.warning("OBS virtual-camera consumer detection failed: %s", exc)
            return None
        finally:
            for process_handle in process_handles.values():
                if process_handle:
                    self._kernel32.CloseHandle(wintypes.HANDLE(process_handle))
            self._kernel32.CloseHandle(own_mapping)


OBS_CONSUMER_PROBE = ObsVirtualCameraConsumerProbe()


def active_external_camera_clients() -> list[str]:
    """Return applications actively consuming the advertised virtual camera."""
    virtual_status = web_api.engine.virtual_camera_status()
    if virtual_status.get("backend") == "obs":
        exact_clients = OBS_CONSUMER_PROBE.active_clients()
        if exact_clients is not None:
            return exact_clients
    return _active_registry_camera_clients()


class DemandMonitor:
    def __init__(self, idle_seconds: float = 8.0) -> None:
        self.idle_seconds = max(2.0, float(idle_seconds))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera-demand-monitor", daemon=True)
        self.idle_since: float | None = None
        self.last_start_attempt = 0.0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            config = web_api.get_config()
            monitor_enabled = bool(config.get("demand_capture_enabled", True))
            virtual_enabled = bool(config.get("virtual_cam_enabled", False))
            clients = active_external_camera_clients() if monitor_enabled and virtual_enabled else []
            demanded = bool(clients)
            web_api.runtime.set_demand_state(monitor_enabled, demanded)
            status = web_api.runtime.status()

            if demanded:
                self.idle_since = None
                now = time.monotonic()
                if not status["running"] and now - self.last_start_attempt >= 5.0:
                    logging.info("Starting capture on Windows camera demand from %s", ", ".join(clients))
                    processing_mode = "none" if config.get("tray_start_mode") == "none" else "last"
                    self.last_start_attempt = now
                    web_api.runtime.start(reason="demand", processing_mode=processing_mode)
            elif status.get("running") and status.get("start_reason") == "demand":
                if self.idle_since is None:
                    self.idle_since = time.monotonic()
                elif time.monotonic() - self.idle_since >= self.idle_seconds:
                    logging.info("Stopping demand-started capture after %.1f idle seconds", self.idle_seconds)
                    web_api.runtime.stop()
                    self.idle_since = None
            else:
                self.idle_since = None
                self.last_start_attempt = 0.0

            self.stop_event.wait(1.0)


class TrayHost:
    def __init__(self, host: str, port: int, idle_seconds: float, enable_virtual_camera: bool) -> None:
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/"
        self.server = uvicorn.Server(uvicorn.Config(web_api.app, host=host, port=port, log_level="info"))
        self.server_thread = threading.Thread(target=self.server.run, name="faceflow-web-api", daemon=True)
        self.demand_monitor = DemandMonitor(idle_seconds)
        if enable_virtual_camera:
            previous = web_api.get_config()
            if not previous.get("virtual_cam_enabled"):
                web_api.save_config({"virtual_cam_enabled": True})
        config = web_api.get_config()
        web_api.engine.configure_virtual_camera(
            bool(config.get("virtual_cam_enabled")),
            int(config.get("width") or 1280),
            int(config.get("height") or 720),
            float(config.get("fps") or 30.0),
            persistent=True,
        )

        self.icon = pystray.Icon(
            "FaceFlow",
            _make_icon(),
            "FaceFlow - camera off",
            menu=pystray.Menu(
                pystray.MenuItem("Open FaceFlow controls", self.open_controls, default=True),
                pystray.MenuItem("Open API options", self.open_api),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(self.camera_label, self.toggle_camera),
                pystray.MenuItem(self.streaming_label, self.toggle_virtual, checked=self.virtual_checked),
                pystray.MenuItem("Start on app demand", self.toggle_demand, checked=self.demand_checked),
                pystray.MenuItem(
                    "Camera start mode",
                    pystray.Menu(
                        pystray.MenuItem(
                            "No processing (passthrough)",
                            self.select_passthrough,
                            checked=self.passthrough_checked,
                            radio=True,
                        ),
                        pystray.MenuItem(
                            "Last processing setup",
                            self.select_last_processing,
                            checked=self.last_processing_checked,
                            radio=True,
                        ),
                    ),
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Exit FaceFlow", self.exit),
            ),
        )

    def camera_label(self, _item) -> str:
        status = web_api.runtime.status()
        return "Stop camera" if status["running"] else "Start camera"

    def virtual_checked(self, _item) -> bool:
        return bool(web_api.get_config().get("virtual_cam_enabled"))

    def streaming_label(self, _item) -> str:
        return "Disable virtual-camera streaming" if self.virtual_checked(_item) else "Enable virtual-camera streaming"

    def demand_checked(self, _item) -> bool:
        return bool(web_api.get_config().get("demand_capture_enabled", True))

    def passthrough_checked(self, _item) -> bool:
        return web_api.get_config().get("tray_start_mode", "last") == "none"

    def last_processing_checked(self, _item) -> bool:
        return not self.passthrough_checked(_item)

    def _selected_processing_mode(self) -> str:
        return "none" if web_api.get_config().get("tray_start_mode") == "none" else "last"

    def _save_live(self, values: dict) -> None:
        previous = web_api.get_config()
        if web_api.runtime.status()["running"]:
            web_api.save_live_config(values, previous)
        else:
            web_api.save_config(values)

    def open_controls(self, _icon=None, _item=None) -> None:
        webbrowser.open(self.url)

    def open_api(self, _icon=None, _item=None) -> None:
        webbrowser.open(f"{self.url}docs")

    def toggle_camera(self, _icon=None, _item=None) -> None:
        if web_api.runtime.status()["running"]:
            web_api.runtime.stop()
        else:
            web_api.runtime.start(reason="manual", processing_mode=self._selected_processing_mode())
        self.icon.update_menu()

    def toggle_virtual(self, _icon=None, _item=None) -> None:
        enabled = not bool(web_api.get_config().get("virtual_cam_enabled"))
        self._save_live({"virtual_cam_enabled": enabled})
        self.icon.update_menu()

    def toggle_demand(self, _icon=None, _item=None) -> None:
        enabled = not bool(web_api.get_config().get("demand_capture_enabled", True))
        self._save_live({"demand_capture_enabled": enabled})
        self.icon.update_menu()

    def select_passthrough(self, _icon=None, _item=None) -> None:
        web_api.save_config({"tray_start_mode": "none"})
        self.icon.update_menu()

    def select_last_processing(self, _icon=None, _item=None) -> None:
        web_api.save_config({"tray_start_mode": "last"})
        self.icon.update_menu()

    def _update_title(self) -> None:
        while not self.server.should_exit:
            status = web_api.runtime.status()
            if status["capture_active"]:
                mode = "passthrough" if status.get("processing_mode") == "none" else "processing"
                suffix = f"camera live ({mode})"
            elif status["running"]:
                suffix = "starting camera"
            elif status.get("virtual_camera_advertised"):
                suffix = "virtual camera ready"
            elif status.get("virtual_camera_error"):
                suffix = "virtual camera unavailable"
            else:
                suffix = "streaming disabled"
            self.icon.title = f"FaceFlow - {suffix}"
            self.icon.update_menu()
            time.sleep(1.5)

    def run(self) -> None:
        self.server_thread.start()
        self.demand_monitor.start()
        threading.Thread(target=self._update_title, name="tray-status", daemon=True).start()
        self.icon.run()

    def exit(self, _icon=None, _item=None) -> None:
        self.demand_monitor.stop()
        web_api.runtime.stop()
        web_api.engine.set_virtual_camera_persistent(False)
        web_api.engine.shutdown_virtual_camera()
        self.server.should_exit = True
        self.icon.stop()
        if self.server_thread.is_alive():
            self.server_thread.join(timeout=8.0)


def port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
        return True


def main() -> None:
    parser = argparse.ArgumentParser("FaceFlow tray application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--idle-seconds", type=float, default=8.0)
    parser.add_argument("--enable-virtual-camera", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / "tray_app.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    if not port_is_available(args.host, args.port):
        raise SystemExit(f"FaceFlow cannot start: http://{args.host}:{args.port}/ is already in use")

    TrayHost(args.host, args.port, args.idle_seconds, args.enable_virtual_camera).run()


if __name__ == "__main__":
    main()
