from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Optional


DFL_REPOSITORY = "https://github.com/volnas10/DeepFaceLab-RTX5000.git"
RECOMMENDED_DISTRO = "Ubuntu-24.04"


class DfmTrainingError(RuntimeError):
    pass


def _decode_windows_output(payload: bytes) -> str:
    if not payload:
        return ""
    if b"\x00" in payload[:100]:
        return payload.decode("utf-16-le", errors="replace").replace("\ufeff", "")
    return payload.decode("utf-8", errors="replace")


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive or not re.fullmatch(r"[a-z]", drive):
        raise DfmTrainingError(f"DFM training needs a drive-letter path, not {resolved}")
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


class DfmTrainingManager:
    """Bridge FaceFlow identity datasets to the WSL DeepFaceLab SAEHD workflow."""

    def __init__(
        self,
        storage_dir: Path,
        custom_model_dir: Path,
        read_profile: Callable[[str], Dict[str, Any]],
        register_model: Callable[[str], None],
    ) -> None:
        self.storage_dir = storage_dir
        self.custom_model_dir = custom_model_dir
        self.read_profile = read_profile
        self.register_model = register_model
        self.jobs_dir = storage_dir / "jobs"
        self._lock = threading.RLock()
        self._processes: Dict[str, subprocess.Popen[bytes]] = {}
        self._environment_cache: Optional[tuple[float, Dict[str, Any]]] = None
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _run(args: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess[bytes]:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.run(args, capture_output=True, timeout=timeout, creationflags=flags, check=False)

    @classmethod
    def _wsl(cls, distro: str, command: str, timeout: float = 30.0) -> subprocess.CompletedProcess[bytes]:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.run(
            ["wsl.exe", "-d", distro, "--", "bash", "-s"],
            input=command.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            creationflags=flags,
            check=False,
        )

    @staticmethod
    def _validate_distro(value: str) -> str:
        distro = str(value or "").strip()
        if not distro or not re.fullmatch(r"[A-Za-z0-9._-]+", distro):
            raise DfmTrainingError("Choose a valid WSL distribution")
        return distro

    @staticmethod
    def _validate_linux_path(value: str, label: str) -> str:
        path = str(value or "").strip().rstrip("/")
        if not path.startswith("/") or any(character in path for character in ("\n", "\r", "\x00")):
            raise DfmTrainingError(f"{label} must be an absolute WSL path")
        return path

    def _installed_distros(self) -> list[str]:
        try:
            result = self._run(["wsl.exe", "--list", "--quiet"], timeout=10.0)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        output = _decode_windows_output(result.stdout)
        return [line.strip().replace("\x00", "") for line in output.splitlines() if line.strip()]

    def environment(self, refresh: bool = False, distro: Optional[str] = None, engine_root: Optional[str] = None) -> Dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if not refresh and distro is None and engine_root is None and self._environment_cache and now - self._environment_cache[0] < 5.0:
                return dict(self._environment_cache[1])
        distros = self._installed_distros()
        distro_versions: Dict[str, str] = {}
        for candidate in (item for item in distros if item.lower().startswith("ubuntu")):
            try:
                version_probe = self._wsl(candidate, "grep '^VERSION_ID=' /etc/os-release | cut -d= -f2 | tr -d '\"'", timeout=8.0)
                distro_versions[candidate] = _decode_windows_output(version_probe.stdout).strip().strip('"')
            except Exception:
                distro_versions[candidate] = ""
        ubuntu_24 = next((item for item, version in distro_versions.items() if version.startswith("24.04")), "")
        chosen = distro if distro in distros else (RECOMMENDED_DISTRO if RECOMMENDED_DISTRO in distros else ubuntu_24 or next((item for item in distros if item.lower().startswith("ubuntu")), ""))
        user = ""
        gpu = ""
        default_root = ""
        ready = False
        repository_present = False
        setup_present = False
        error = None
        if chosen:
            try:
                probe = self._wsl(
                    chosen,
                    "printf '%s\\n' \"$USER\"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1",
                    timeout=15.0,
                )
                lines = [line.strip() for line in _decode_windows_output(probe.stdout).splitlines() if line.strip()]
                user = lines[0] if lines else ""
                gpu = lines[1] if len(lines) > 1 else ""
                default_root = f"/home/{user}/DeepFaceLab-RTX5000" if user else ""
                root = self._validate_linux_path(engine_root, "DeepFaceLab root") if engine_root else default_root
                quoted_root = shlex.quote(root)
                check = self._wsl(
                    chosen,
                    f"if test -f {quoted_root}/_internal/DeepFaceLab/main.py; then echo repo=1; else echo repo=0; fi; "
                    f"if test -f {quoted_root}/setup.sh -o -f {quoted_root}/_internal/DeepFaceLab/setup.sh; then echo setup=1; else echo setup=0; fi; "
                    f"if test -x {quoted_root}/_internal/DeepFaceLab/venv/bin/python; then echo venv=1; else echo venv=0; fi",
                    timeout=12.0,
                )
                values = {}
                for line in _decode_windows_output(check.stdout).splitlines():
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        values[key] = value
                repository_present = values.get("repo") == "1"
                setup_present = values.get("setup") == "1"
                ready = repository_present and values.get("venv") == "1"
            except Exception as exc:
                error = str(exc) or type(exc).__name__
        result = {
            "wsl_available": bool(distros),
            "distros": distros,
            "distro_versions": distro_versions,
            "recommended_distro": RECOMMENDED_DISTRO,
            "distro": chosen,
            "user": user,
            "gpu": gpu,
            "engine_root": engine_root or default_root,
            "repository_present": repository_present,
            "setup_present": setup_present,
            "ready": ready,
            "error": error,
            "repository": DFL_REPOSITORY,
        }
        if distro is None and engine_root is None:
            with self._lock:
                self._environment_cache = (now, dict(result))
        return result

    def open_setup(self, distro: str, engine_root: str) -> Dict[str, Any]:
        distro = self._validate_distro(distro)
        distros = self._installed_distros()
        if distro not in distros:
            return {
                "launched": False,
                "needs_distro": True,
                "command": f"wsl --install -d {RECOMMENDED_DISTRO}",
                "message": f"Install {RECOMMENDED_DISTRO}, restart Windows if requested, then return here.",
            }
        version_probe = self._wsl(distro, "grep '^VERSION_ID=' /etc/os-release | cut -d= -f2 | tr -d '\"'", timeout=8.0)
        version = _decode_windows_output(version_probe.stdout).strip().strip('"')
        if not version.startswith("24.04"):
            return {
                "launched": False,
                "needs_distro": True,
                "command": f"wsl --install -d {RECOMMENDED_DISTRO}",
                "message": "The RTX 5000 trainer setup expects Ubuntu 24.04 and will not modify your older Linux environment.",
            }
        root = self._validate_linux_path(engine_root, "DeepFaceLab root")
        root_path = PurePosixPath(root)
        parent = str(root_path.parent)
        folder = root_path.name
        script = (
            "set -e; "
            f"mkdir -p {shlex.quote(parent)}; cd {shlex.quote(parent)}; "
            f"if [ ! -d {shlex.quote(folder)}/.git ]; then git clone {shlex.quote(DFL_REPOSITORY)} {shlex.quote(folder)}; fi; "
            f"cd {shlex.quote(folder)}; chmod +x setup.sh; ./setup.sh; "
            "printf '\\nSetup complete. You can close this window and refresh FaceFlow.\\n'; read -r _"
        )
        setup_script = self.storage_dir / "install-deepfacelab.sh"
        setup_script.parent.mkdir(parents=True, exist_ok=True)
        setup_script.write_text("#!/bin/bash\n" + script + "\nexec bash\n", encoding="utf-8", newline="\n")
        setup_script_wsl = _windows_to_wsl(setup_script)
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(["wsl.exe", "-d", distro, "--", "bash", setup_script_wsl], creationflags=flags)
        with self._lock:
            self._environment_cache = None
        return {"launched": True, "needs_distro": False, "message": "DeepFaceLab setup opened in a terminal. It may ask for the Linux password."}

    def _job_path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9-]+", job_id):
            raise DfmTrainingError("Unknown DFM training job")
        path = self.jobs_dir / job_id
        if not path.is_dir():
            raise DfmTrainingError("Unknown DFM training job")
        return path

    def _load_job(self, job_id: str) -> Dict[str, Any]:
        path = self._job_path(job_id) / "job.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DfmTrainingError(f"DFM job metadata is damaged: {exc}") from exc

    def _save_job(self, job: Dict[str, Any]) -> None:
        path = self.jobs_dir / str(job["id"])
        path.mkdir(parents=True, exist_ok=True)
        temp = path / "job.json.tmp"
        temp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        os.replace(temp, path / "job.json")

    def _update_job(self, job_id: str, **values: Any) -> Dict[str, Any]:
        with self._lock:
            job = self._load_job(job_id)
            job.update(values)
            job["updated_at"] = time.time()
            self._save_job(job)
            return job

    @staticmethod
    def _tail(path: Path, limit: int = 12000) -> str:
        if not path.is_file():
            return ""
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode("utf-8", errors="replace")[-limit:]

    def _summarize_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        job_id = str(job["id"])
        process = self._processes.get(job_id)
        running = bool(process and process.poll() is None)
        log_tail = self._tail(self.jobs_dir / job_id / "training.log")
        iterations = [int(value) for value in re.findall(r"\[#?0*(\d{2,})\]", log_tail)]
        current_iteration = max(iterations) if iterations else int(job.get("current_iteration") or 0)
        target = int(job.get("target_iterations") or 1_000_000)
        if current_iteration != job.get("current_iteration"):
            job = self._update_job(job_id, current_iteration=current_iteration)
        result = dict(job)
        result.update(
            {
                "running": running,
                "current_iteration": current_iteration,
                "progress": min(1.0, current_iteration / max(1, target)),
                "log_tail": log_tail[-6000:],
            }
        )
        return result

    def status(self, refresh_environment: bool = False) -> Dict[str, Any]:
        jobs = []
        for manifest in self.jobs_dir.glob("*/job.json"):
            try:
                jobs.append(self._summarize_job(json.loads(manifest.read_text(encoding="utf-8"))))
            except Exception:
                continue
        jobs.sort(key=lambda item: float(item.get("created_at") or 0.0), reverse=True)
        return {"environment": self.environment(refresh=refresh_environment), "jobs": jobs}

    def job(self, job_id: str) -> Dict[str, Any]:
        return self._summarize_job(self._load_job(job_id))

    def prepare(
        self,
        profile_id: str,
        distro: str,
        engine_root: str,
        base_workspace: str,
        target_iterations: int,
    ) -> Dict[str, Any]:
        profile = self.read_profile(profile_id)
        distro = self._validate_distro(distro)
        root = self._validate_linux_path(engine_root, "DeepFaceLab root")
        base = self._validate_linux_path(base_workspace, "Base workspace")
        environment = self.environment(refresh=True, distro=distro, engine_root=root)
        if not environment.get("ready"):
            raise DfmTrainingError("DeepFaceLab is not set up in the selected WSL distribution")
        check = self._wsl(distro, f"test -d {shlex.quote(base)}/data_dst/aligned -a -d {shlex.quote(base)}/model", timeout=12.0)
        if check.returncode != 0:
            raise DfmTrainingError("The base workspace must contain data_dst/aligned and a pretrained model folder")
        profile_path = Path(str(profile.get("frames", [{}])[0].get("path") or "")).parent
        if not profile_path.is_dir():
            raise DfmTrainingError("The identity profile has no recorded dataset")
        job_id = f"{re.sub(r'[^a-z0-9]+', '-', str(profile.get('name') or profile_id).lower()).strip('-')[:32]}-{int(time.time())}-{uuid.uuid4().hex[:5]}"
        workspace = f"{root}/workspaces/{job_id}"
        job = {
            "id": job_id,
            "profile_id": profile_id,
            "name": str(profile.get("name") or profile_id),
            "distro": distro,
            "engine_root": root,
            "base_workspace": base,
            "workspace": workspace,
            "source_frames": int(profile.get("frame_count") or len(profile.get("frames") or [])),
            "aligned_frames": 0,
            "target_iterations": max(25_000, min(5_000_000, int(target_iterations))),
            "current_iteration": 0,
            "phase": "preparing",
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            "model_id": None,
        }
        self._save_job(job)
        thread = threading.Thread(target=self._prepare_worker, args=(job_id, profile_path), name=f"dfm-prepare-{job_id}", daemon=True)
        thread.start()
        return self._summarize_job(job)

    def _prepare_worker(self, job_id: str, profile_path: Path) -> None:
        try:
            job = self._load_job(job_id)
            source = _windows_to_wsl(profile_path)
            workspace = str(job["workspace"])
            base = str(job["base_workspace"])
            command = (
                "set -e; "
                f"mkdir -p {shlex.quote(workspace)}/data_src {shlex.quote(workspace)}/data_dst/aligned {shlex.quote(workspace)}/model; "
                f"cp -a {shlex.quote(source)}/. {shlex.quote(workspace)}/data_src/; "
                f"cp -a {shlex.quote(base)}/data_dst/aligned/. {shlex.quote(workspace)}/data_dst/aligned/; "
                f"cp -a {shlex.quote(base)}/model/. {shlex.quote(workspace)}/model/"
            )
            result = self._wsl(str(job["distro"]), command, timeout=1800.0)
            if result.returncode != 0:
                raise DfmTrainingError(_decode_windows_output(result.stderr).strip() or "Could not copy the DFM workspace")
            self._update_job(job_id, phase="prepared", error=None)
        except Exception as exc:
            self._update_job(job_id, phase="error", error=str(exc) or type(exc).__name__)

    def _launch_job_process(self, job_id: str, phase: str, command: str, log_name: str) -> Dict[str, Any]:
        with self._lock:
            current = self._processes.get(job_id)
            if current and current.poll() is None:
                raise DfmTrainingError("This DFM job is already running")
            job = self._update_job(job_id, phase=phase, error=None)
            log_path = self.jobs_dir / job_id / log_name
            log_stream = log_path.open("ab", buffering=0)
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                ["wsl.exe", "-d", str(job["distro"]), "--", "bash", "-s"],
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                creationflags=flags,
            )
            if process.stdin is not None:
                process.stdin.write(command.encode("utf-8"))
                process.stdin.close()
            self._processes[job_id] = process
            monitor = threading.Thread(target=self._monitor_process, args=(job_id, phase, process, log_stream), name=f"dfm-{phase}-{job_id}", daemon=True)
            monitor.start()
        return self._summarize_job(job)

    def _monitor_process(self, job_id: str, phase: str, process: subprocess.Popen[bytes], log_stream: Any) -> None:
        return_code = process.wait()
        log_stream.close()
        with self._lock:
            self._processes.pop(job_id, None)
        job = self._load_job(job_id)
        if job.get("phase") == "stopping":
            self._update_job(job_id, phase="stopped", error=None)
            return
        if return_code != 0:
            self._update_job(job_id, phase="error", error=f"DeepFaceLab {phase} exited with code {return_code}")
            return
        if phase == "extracting":
            count_result = self._wsl(
                str(job["distro"]),
                f"find {shlex.quote(str(job['workspace']))}/data_src/aligned -maxdepth 1 -type f -iname '*.jpg' | wc -l",
                timeout=20.0,
            )
            try:
                count = int(_decode_windows_output(count_result.stdout).strip() or "0")
            except ValueError:
                count = 0
            self._update_job(job_id, phase="ready", aligned_frames=count, error=None)
        elif phase == "training":
            self._update_job(job_id, phase="stopped", error=None)
        elif phase == "exporting":
            try:
                self._import_export(job_id)
            except Exception as exc:
                self._update_job(job_id, phase="error", error=str(exc) or type(exc).__name__)

    def extract(self, job_id: str) -> Dict[str, Any]:
        job = self._load_job(job_id)
        if job.get("phase") not in {"prepared", "ready", "error", "stopped"}:
            raise DfmTrainingError("Finish workspace preparation before extracting faces")
        root = str(job["engine_root"])
        workspace = str(job["workspace"])
        python = f"{root}/_internal/DeepFaceLab/venv/bin/python"
        main = f"{root}/_internal/DeepFaceLab/main.py"
        command = (
            f"rm -rf {shlex.quote(workspace)}/data_src/aligned; mkdir -p {shlex.quote(workspace)}/data_src/aligned; "
            f"exec -a faceflow-dfm-{shlex.quote(job_id)} {shlex.quote(python)} {shlex.quote(main)} extract "
            f"--input-dir {shlex.quote(workspace + '/data_src')} --output-dir {shlex.quote(workspace + '/data_src/aligned')} "
            "--detector s3fd --face-type whole_face --max-faces-from-image 1 --image-size 512 --jpeg-quality 95 "
            "--no-output-debug --force-gpu-idxs 0 --num-gpu-sessions 1"
        )
        return self._launch_job_process(job_id, "extracting", command, "training.log")

    def train(self, job_id: str) -> Dict[str, Any]:
        job = self._load_job(job_id)
        if int(job.get("aligned_frames") or 0) < 50:
            raise DfmTrainingError("Extract at least 50 aligned source faces before starting training")
        root = str(job["engine_root"])
        workspace = str(job["workspace"])
        python = f"{root}/_internal/DeepFaceLab/venv/bin/python"
        main = f"{root}/_internal/DeepFaceLab/main.py"
        command = (
            f"exec -a faceflow-dfm-{shlex.quote(job_id)} {shlex.quote(python)} {shlex.quote(main)} train "
            f"--training-data-src-dir {shlex.quote(workspace + '/data_src/aligned')} "
            f"--training-data-dst-dir {shlex.quote(workspace + '/data_dst/aligned')} "
            f"--model-dir {shlex.quote(workspace + '/model')} --model SAEHD --silent-start --no-preview --force-gpu-idxs 0"
        )
        return self._launch_job_process(job_id, "training", command, "training.log")

    def stop(self, job_id: str) -> Dict[str, Any]:
        job = self._load_job(job_id)
        process = self._processes.get(job_id)
        self._update_job(job_id, phase="stopping")
        self._wsl(str(job["distro"]), f"pkill -INT -f {shlex.quote('faceflow-dfm-' + job_id)} || true", timeout=12.0)
        if process and process.poll() is None:
            try:
                process.wait(timeout=12.0)
            except subprocess.TimeoutExpired:
                process.terminate()
        return self._summarize_job(self._load_job(job_id))

    def export(self, job_id: str) -> Dict[str, Any]:
        job = self._load_job(job_id)
        if self._processes.get(job_id) and self._processes[job_id].poll() is None:
            raise DfmTrainingError("Stop training and wait for the model to save before exporting")
        root = str(job["engine_root"])
        workspace = str(job["workspace"])
        python = f"{root}/_internal/DeepFaceLab/venv/bin/python"
        main = f"{root}/_internal/DeepFaceLab/main.py"
        command = (
            f"cd {shlex.quote(workspace + '/model')}; "
            f"exec -a faceflow-dfm-{shlex.quote(job_id)} {shlex.quote(python)} {shlex.quote(main)} exportdfm "
            f"--model-dir {shlex.quote(workspace + '/model')} --model SAEHD"
        )
        return self._launch_job_process(job_id, "exporting", command, "training.log")

    def _import_export(self, job_id: str) -> None:
        job = self._load_job(job_id)
        workspace = str(job["workspace"])
        find_result = self._wsl(
            str(job["distro"]),
            f"find {shlex.quote(workspace + '/model')} -maxdepth 1 -type f -iname '*.dfm' -printf '%T@ %p\\n' | sort -nr | head -n 1 | cut -d' ' -f2-",
            timeout=20.0,
        )
        exported = _decode_windows_output(find_result.stdout).strip()
        if not exported:
            raise DfmTrainingError("DeepFaceLab finished without producing a .dfm file")
        model_stem = re.sub(r"[^a-z0-9]+", "_", str(job.get("name") or job_id).lower()).strip("_")[:48]
        model_name = f"trained_{model_stem}_{int(time.time())}"
        destination = self.custom_model_dir / f"{model_name}.dfm"
        self.custom_model_dir.mkdir(parents=True, exist_ok=True)
        destination_wsl = _windows_to_wsl(destination)
        copy_result = self._wsl(str(job["distro"]), f"cp {shlex.quote(exported)} {shlex.quote(destination_wsl)}", timeout=600.0)
        if copy_result.returncode != 0 or not destination.is_file():
            raise DfmTrainingError(_decode_windows_output(copy_result.stderr).strip() or "Could not import the exported DFM")
        model_id = f"custom/{model_name}"
        self.register_model(model_id)
        self._update_job(job_id, phase="complete", model_id=model_id, exported_path=str(destination), error=None)
