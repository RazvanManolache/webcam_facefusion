import argparse
import sys
import time

print("[INFO] Initializing Python subsystems...", flush=True)

from typing import Optional, Tuple, Dict, List
import subprocess
import os
import threading
import importlib
import re
from collections import defaultdict, deque
from functools import wraps

print("[INFO] Setting up paths...", flush=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FF_ROOT = os.path.join(BASE_DIR, "facefusion_mrg")
if FF_ROOT not in sys.path:
    sys.path.insert(0, FF_ROOT)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

print("[INFO] Importing core libraries...", flush=True)

print("  - importing cv2...", flush=True)
import cv2
print("  - importing numpy...", flush=True)
import numpy as np
# Lazy import Gradio only if requested
gr = None
if "--gui" in sys.argv:
    print("  - importing gradio (GUI requested)...", flush=True)
    try:
        import gradio as gr
    except Exception as e:
        print(f"[ERROR] Failed to import Gradio: {e}", flush=True)

if gr is None:
    print("  - skipping gradio (headless or failed)...", flush=True)
    # Minimal mock for headless mode runtime compatibility
    class _MockGR:
        class Error(Exception): pass
        def update(self, *args, **kwargs): return None
    gr = _MockGR()

print("  - importing logging...", flush=True)
import logging
print("  - preloading TensorRT...", flush=True)
try:
    # Importing TensorRT first loads its Windows DLLs before ONNX Runtime tries
    # to create a TensorRT execution provider.
    import tensorrt as trt
    HAS_TENSORRT = str(getattr(trt, "__version__", "")).startswith("10.")
except Exception as tensor_rt_error:
    HAS_TENSORRT = False
    print(f"  - TensorRT unavailable ({tensor_rt_error}); CUDA remains available", flush=True)
print("  - importing onnxruntime...", flush=True)
import onnxruntime as ort
print("  - importing gc...", flush=True)
import gc
from concurrent.futures import ThreadPoolExecutor, Future
import json
import shutil
print("  - importing pyvirtualcam...", flush=True)
try:
    import pyvirtualcam
    HAS_PYVIRTUALCAM = True
except Exception:
    HAS_PYVIRTUALCAM = False

print("[INFO] Importing FaceFusion modules (this may take a few seconds)...", flush=True)

from facefusion import state_manager
from facefusion.types import Face, VisionFrame
from facefusion.face_selector import select_faces, sort_and_filter_faces
from facefusion.face_analyser import scale_face
from facefusion.face_helper import apply_nms, convert_to_face_landmark_5, estimate_face_angle, get_nms_threshold
from facefusion.processors.modules.deep_swapper import core as deep_swapper
from facefusion.processors.modules.face_swapper import core as ff_face_swapper
from facefusion.processors.modules.face_enhancer import core as ff_face_enhancer
from facefusion.processors.modules.frame_enhancer import core as ff_frame_enhancer
from facefusion.processors.modules.frame_colorizer import core as ff_frame_colorizer
from facefusion.processors.modules.expression_restorer import core as ff_expr_restorer
from facefusion.processors.modules.age_modifier import core as ff_age_modifier
from facefusion.processors.modules.face_editor import core as ff_face_editor
from facefusion.processors.modules.face_debugger import core as ff_face_debugger
from facefusion.processors.modules.lip_syncer import core as ff_lip_syncer
from facefusion import choices as ff_choices
from facefusion.processors import choices as proc_choices
from facefusion.processors import choices as proc_choices
from facefusion.processors.modules.frame_colorizer import choices as frame_colorizer_choices
from facefusion.processors.modules.expression_restorer import choices as expression_restorer_choices
from facefusion.processors.modules.age_modifier import choices as age_modifier_choices
from facefusion.processors.modules.face_editor import choices as face_editor_choices
from facefusion.processors.modules.face_debugger import choices as face_debugger_choices
from facefusion.processors.modules.lip_syncer import choices as lip_syncer_choices
from facefusion import face_detector as ff_detector
from facefusion import face_landmarker as ff_landmarker
from facefusion import face_masker as ff_masker
from facefusion import face_recognizer as ff_recognizer
from facefusion import content_analyser as ff_content
from facefusion import face_classifier as ff_classifier
from facefusion import core as ff_core
from facefusion import inference_manager as ff_inference_manager
from facefusion.filesystem import resolve_file_paths, get_file_name
from facefusion.download import conditional_download_hashes, conditional_download_sources

print("[INFO] Imports complete. Starting application...", flush=True)


def _raise_embedded_inference_error(error_code: int) -> None:
    """Keep a model-load failure from terminating the entire webcam server."""
    raise RuntimeError(f"FaceFusion inference session failed to load (code {error_code})")


# FaceFusion's CLI-oriented inference manager normally calls os._exit() when
# ONNX Runtime cannot create a session. In an embedded web server that would
# also kill the camera controls and API, so surface the failure to the runtime
# controller instead; it can then release capture and display the error.
ff_inference_manager.fatal_exit = _raise_embedded_inference_error

LOGGER = logging.getLogger("webcam_deep_swap")

try:
    import gpu_face_pipeline
    HAS_GPU_FACE_PIPELINE = gpu_face_pipeline.is_available()
except Exception as gpu_pipeline_import_error:
    gpu_face_pipeline = None
    HAS_GPU_FACE_PIPELINE = False
    LOGGER.warning("CUDA crop pipeline unavailable: %s", gpu_pipeline_import_error)

# Global cache for live face swapper sources
_source_imgs = []
_virt_cam = None
_benchmark_generation = 0
_benchmark_completed_generation = 0
_benchmark_lock = threading.Lock()
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_original_deep_swapper_occlusion_mask = deep_swapper.create_occlusion_mask


class TemporalOcclusionMaskCache:
    """Reuse a stable aligned-face occlusion mask for a small number of frames.

    Deep swapper crops are aligned before the mask is requested, so adjacent
    crops are normally almost identical even when the head moves slightly in
    the source frame. A small appearance signature prevents reuse when a hand,
    microphone or fast pose change enters the crop.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._mask: Optional[np.ndarray] = None
        self._signature: Optional[np.ndarray] = None
        self._cache_key: Optional[Tuple[object, ...]] = None
        self._age = 0
        self._computed = 0
        self._reused = 0
        self._motion_refreshes = 0
        self._last_motion = 0.0

    @staticmethod
    def _appearance_signature(crop: np.ndarray) -> np.ndarray:
        small = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)

    def reset(self) -> None:
        with self._lock:
            self._mask = None
            self._signature = None
            self._cache_key = None
            self._age = 0
            self._computed = 0
            self._reused = 0
            self._motion_refreshes = 0
            self._last_motion = 0.0

    def create(self, crop: np.ndarray) -> np.ndarray:
        enabled_value = state_manager.get_item("temporal_occlusion_reuse")
        enabled = bool(enabled_value) if enabled_value is not None else True
        interval = max(1, min(8, int(state_manager.get_item("temporal_occlusion_interval") or 3)))
        selector_mode = str(state_manager.get_item("face_selector_mode") or "one")
        if not enabled or interval <= 1 or selector_mode == "many":
            return _original_deep_swapper_occlusion_mask(crop)

        signature = self._appearance_signature(crop)
        cache_key = (
            tuple(crop.shape),
            str(state_manager.get_item("face_occluder_model") or "xseg_1"),
        )
        motion_threshold = 0.035

        with self._lock:
            can_reuse = (
                self._mask is not None
                and self._signature is not None
                and self._cache_key == cache_key
                and self._age < interval - 1
            )
            if can_reuse:
                motion = float(np.mean(np.abs(signature - self._signature)) / 255.0)
                self._last_motion = motion
                if motion <= motion_threshold:
                    self._age += 1
                    self._reused += 1
                    return self._mask
                self._motion_refreshes += 1

            mask = _original_deep_swapper_occlusion_mask(crop)
            self._mask = np.ascontiguousarray(mask, dtype=np.float32)
            self._signature = signature
            self._cache_key = cache_key
            self._age = 0
            self._computed += 1
            return self._mask

    def status(self) -> Dict[str, object]:
        with self._lock:
            total = self._computed + self._reused
            enabled_value = state_manager.get_item("temporal_occlusion_reuse")
            return {
                "enabled": bool(enabled_value) if enabled_value is not None else True,
                "interval": max(1, min(8, int(state_manager.get_item("temporal_occlusion_interval") or 3))),
                "computed": self._computed,
                "reused": self._reused,
                "reuse_percent": round(100.0 * self._reused / total, 1) if total else 0.0,
                "motion_refreshes": self._motion_refreshes,
                "last_motion": round(self._last_motion, 4),
            }


_temporal_occlusion_cache = TemporalOcclusionMaskCache()


class TemporalFaceTracker:
    """Reuse full-analysis landmarks briefly, updating them with optical flow."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._previous_gray: Optional[np.ndarray] = None
        self._face: Optional[Face] = None
        self._cache_key: Optional[Tuple[object, ...]] = None
        self._age = 0
        self._analysed = 0
        self._tracked = 0
        self._fallbacks = 0
        self._last_error = 0.0

    def reset(self) -> None:
        with self._lock:
            self._previous_gray = None
            self._face = None
            self._cache_key = None
            self._age = 0
            self._analysed = 0
            self._tracked = 0
            self._fallbacks = 0
            self._last_error = 0.0

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    @staticmethod
    def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        transformed = cv2.transform(points.astype(np.float32).reshape(1, -1, 2), matrix)
        return transformed.reshape(-1, 2)

    def _track(self, previous_gray: np.ndarray, current_gray: np.ndarray, face: Face) -> Optional[Face]:
        landmarks = face.landmark_set or {}
        anchors = landmarks.get("68")
        if not isinstance(anchors, np.ndarray) or len(anchors) < 5:
            anchors = landmarks.get("5/68")
        if not isinstance(anchors, np.ndarray) or len(anchors) < 4:
            return None

        source = anchors.astype(np.float32).reshape(-1, 1, 2)
        tracked, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            source,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
        )
        if tracked is None or status is None:
            return None
        backtracked, back_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            tracked,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
        )
        if backtracked is None or back_status is None:
            return None

        good = status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool)
        forward_backward = np.linalg.norm(source.reshape(-1, 2) - backtracked.reshape(-1, 2), axis=1)
        good &= forward_backward <= 2.5
        if int(good.sum()) < max(4, int(len(source) * 0.6)):
            return None

        source_good = source.reshape(-1, 2)[good]
        tracked_good = tracked.reshape(-1, 2)[good]
        matrix, inliers = cv2.estimateAffinePartial2D(
            source_good,
            tracked_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=100,
            confidence=0.99,
            refineIters=10,
        )
        if matrix is None:
            return None
        inlier_ratio = float(np.mean(inliers)) if inliers is not None and len(inliers) else 0.0
        scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
        mean_error = float(np.mean(error.reshape(-1)[good])) if error is not None else 0.0
        self._last_error = mean_error
        if inlier_ratio < 0.65 or not 0.85 <= scale <= 1.18 or mean_error > 18.0:
            return None

        transformed_landmarks = {
            key: self._transform_points(value, matrix) if isinstance(value, np.ndarray) else value
            for key, value in landmarks.items()
        }
        x1, y1, x2, y2 = np.asarray(face.bounding_box, dtype=np.float32)
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        transformed_corners = self._transform_points(corners, matrix)
        height, width = current_gray.shape[:2]
        new_box = np.array(
            [
                np.clip(transformed_corners[:, 0].min(), 0, width - 1),
                np.clip(transformed_corners[:, 1].min(), 0, height - 1),
                np.clip(transformed_corners[:, 0].max(), 0, width - 1),
                np.clip(transformed_corners[:, 1].max(), 0, height - 1),
            ],
            dtype=np.float32,
        )
        angle_landmarks = transformed_landmarks.get("68")
        new_angle = estimate_face_angle(angle_landmarks) if isinstance(angle_landmarks, np.ndarray) else face.angle
        return face._replace(bounding_box=new_box, landmark_set=transformed_landmarks, angle=new_angle)

    def select(self, frame: np.ndarray) -> List[Face]:
        enabled_value = state_manager.get_item("temporal_face_tracking")
        enabled = bool(enabled_value) if enabled_value is not None else False
        interval = max(1, min(8, int(state_manager.get_item("temporal_face_interval") or 3)))
        selector_mode = str(state_manager.get_item("face_selector_mode") or "one")
        if not enabled or interval <= 1 or selector_mode != "one":
            result = select_faces(reference_vision_frame=frame, target_vision_frame=frame)
            return result[0] if isinstance(result, tuple) else result

        current_gray = self._gray(frame)
        cache_key = (
            tuple(frame.shape),
            str(state_manager.get_item("face_detector_model")),
            str(state_manager.get_item("face_detector_size")),
            str(state_manager.get_item("face_landmarker_model")),
        )
        with self._lock:
            can_track = (
                self._face is not None
                and self._previous_gray is not None
                and self._cache_key == cache_key
                and self._age < interval - 1
            )
            if can_track:
                tracked_face = self._track(self._previous_gray, current_gray, self._face)
                if tracked_face is not None:
                    self._face = tracked_face
                    self._previous_gray = current_gray
                    self._age += 1
                    self._tracked += 1
                    return [tracked_face]
                self._fallbacks += 1

            result = select_faces(reference_vision_frame=frame, target_vision_frame=frame)
            faces = result[0] if isinstance(result, tuple) else result
            face = faces[0] if faces else None
            self._face = face
            self._previous_gray = current_gray
            self._cache_key = cache_key
            self._age = 0
            self._analysed += 1
            return [face] if face is not None else []

    def status(self) -> Dict[str, object]:
        with self._lock:
            total = self._analysed + self._tracked
            enabled_value = state_manager.get_item("temporal_face_tracking")
            return {
                "enabled": bool(enabled_value) if enabled_value is not None else False,
                "interval": max(1, min(8, int(state_manager.get_item("temporal_face_interval") or 3))),
                "analysed": self._analysed,
                "tracked": self._tracked,
                "tracking_percent": round(100.0 * self._tracked / total, 1) if total else 0.0,
                "fallbacks": self._fallbacks,
                "last_error": round(self._last_error, 3),
            }


_temporal_face_tracker = TemporalFaceTracker()


def _create_temporal_deep_swapper_occlusion_mask(crop: np.ndarray) -> np.ndarray:
    return _temporal_occlusion_cache.create(crop)


def reset_temporal_occlusion_cache() -> None:
    _temporal_occlusion_cache.reset()


def temporal_occlusion_status() -> Dict[str, object]:
    return _temporal_occlusion_cache.status()


def reset_temporal_face_tracker() -> None:
    _temporal_face_tracker.reset()


def temporal_face_tracking_status() -> Dict[str, object]:
    return _temporal_face_tracker.status()


# Deep swapper imports the masker function directly, so patch only its local
# reference. Other processors retain their normal per-frame mask behavior.
deep_swapper.create_occlusion_mask = _create_temporal_deep_swapper_occlusion_mask


class VirtualCameraPublisher:
    """Own the virtual-camera sender and keep it available between captures.

    pyvirtualcam exposes a producer, not a consumer-count API. Tray mode keeps
    this producer open with a lightweight standby frame so applications can
    discover and open the virtual device before the physical camera starts.
    The processing loop only replaces the latest frame; it never has to reopen
    the virtual camera during a demand-triggered handoff.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._enabled = False
        self._persistent = False
        self._source_active = False
        self._width = 1280
        self._height = 720
        self._fps = 30.0
        self._latest_rgb: Optional[np.ndarray] = None
        self._camera = None
        self._error: Optional[str] = None

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="virtual-camera-publisher",
                daemon=True,
            )
            self._thread.start()

    def configure(
        self,
        enabled: bool,
        width: int,
        height: int,
        fps: float,
        persistent: Optional[bool] = None,
    ) -> None:
        with self._lock:
            self._enabled = bool(enabled and HAS_PYVIRTUALCAM)
            self._width = max(1, int(width or self._width))
            self._height = max(1, int(height or self._height))
            self._fps = max(1.0, float(fps or self._fps))
            if persistent is not None:
                self._persistent = bool(persistent)
            if self._enabled and (self._persistent or self._source_active):
                self._ensure_thread()
            self._wake.set()

    def set_persistent(self, persistent: bool) -> None:
        with self._lock:
            self._persistent = bool(persistent)
            if self._enabled and (self._persistent or self._source_active):
                self._ensure_thread()
            self._wake.set()

    def publish(self, frame: np.ndarray, input_is_rgb: bool) -> bool:
        if not HAS_PYVIRTUALCAM or not isinstance(frame, np.ndarray) or not frame.size:
            return False
        if input_is_rgb:
            rgb = frame
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._lock:
            if not self._enabled:
                return False
            if rgb.shape[1] != self._width or rgb.shape[0] != self._height:
                rgb = cv2.resize(rgb, (self._width, self._height), interpolation=cv2.INTER_AREA)
            self._latest_rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            self._source_active = True
            self._ensure_thread()
            self._wake.set()
        return True

    def source_stopped(self) -> None:
        with self._lock:
            self._source_active = False
            self._latest_rgb = None
            self._wake.set()

    def disable(self) -> bool:
        with self._lock:
            was_open = self._camera is not None or self._enabled
            self._enabled = False
            self._source_active = False
            self._latest_rgb = None
            self._wake.set()
            return was_open

    def shutdown(self) -> None:
        with self._lock:
            self._enabled = False
            self._persistent = False
            self._source_active = False
            self._latest_rgb = None
            self._stop.set()
            self._wake.set()
            thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)

    def status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "persistent": self._persistent,
                "advertised": self._camera is not None,
                "source_active": self._source_active,
                "width": self._width,
                "height": self._height,
                "fps": self._fps,
                "backend": getattr(self._camera, "backend", None) if self._camera else None,
                "error": self._error,
            }

    def _standby_frame(self, width: int, height: int) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (10, 12, 18)
        scale = max(0.7, min(width / 900.0, height / 500.0))
        title = "FaceFlow virtual camera"
        subtitle = "Standby - waiting for camera demand"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.58, 1)[0]
        center_y = height // 2
        cv2.putText(
            frame,
            title,
            ((width - title_size[0]) // 2, center_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (190, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            subtitle,
            ((width - sub_size[0]) // 2, center_y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.58,
            (155, 160, 175),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _close_sender(self) -> None:
        global _virt_cam
        camera = self._camera
        self._camera = None
        _virt_cam = None
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass

    def _run(self) -> None:
        global _virt_cam
        standby: Optional[np.ndarray] = None
        standby_shape = (0, 0)
        while not self._stop.is_set():
            with self._lock:
                should_open = self._enabled and (self._persistent or self._source_active)
                width, height, fps = self._width, self._height, self._fps
                latest = self._latest_rgb
                camera = self._camera

            if not should_open:
                with self._lock:
                    self._close_sender()
                self._wake.wait(0.25)
                self._wake.clear()
                continue

            if camera is None or camera.width != width or camera.height != height or abs(camera.fps - fps) > 0.01:
                with self._lock:
                    self._close_sender()
                try:
                    camera = pyvirtualcam.Camera(
                        width=width,
                        height=height,
                        fps=fps,
                        fmt=pyvirtualcam.PixelFormat.RGB,
                    )
                    with self._lock:
                        self._camera = camera
                        _virt_cam = camera
                        self._error = None
                    LOGGER.info(
                        "[virtual-cam] Advertised %dx%d at %.2f FPS via %s",
                        width,
                        height,
                        fps,
                        getattr(camera, "backend", "unknown"),
                    )
                except Exception as exc:
                    with self._lock:
                        self._error = str(exc) or type(exc).__name__
                    LOGGER.warning("[virtual-cam] Could not advertise device: %s", self._error)
                    self._wake.wait(3.0)
                    self._wake.clear()
                    continue

            if standby is None or standby_shape != (width, height):
                standby = self._standby_frame(width, height)
                standby_shape = (width, height)
            frame = latest if latest is not None else standby
            try:
                camera.send(frame)
                camera.sleep_until_next_frame()
            except Exception as exc:
                with self._lock:
                    self._error = str(exc) or type(exc).__name__
                    self._close_sender()
                self._wake.wait(1.0)
                self._wake.clear()

        with self._lock:
            self._close_sender()


_virtual_camera_publisher = VirtualCameraPublisher()


_models_downloaded = False
_camera_capability_cache: Dict[str, Dict[str, float]] = {}
_fallback_camera_capabilities: Dict[str, float] = {
    "640x480": 30.0,
    "1280x720": 30.0,
    "1920x1080": 30.0,
}

# Simple preferences store
PREFS_PATH = os.path.join(BASE_DIR, 'user_prefs.json')
_prefs_cache = None

def _load_prefs() -> dict:
    global _prefs_cache
    try:
        with open(PREFS_PATH, 'r', encoding='utf-8') as f:
            _prefs_cache = json.load(f) or {}
    except Exception:
        _prefs_cache = {}
    return _prefs_cache

def _save_prefs(prefs: dict) -> None:
    global _prefs_cache
    try:
        with open(PREFS_PATH, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=2)
        _prefs_cache = dict(prefs)
    except Exception:
        pass

def _get_pref(key: str, default):
    return _load_prefs().get(key, default)

def _persist_state() -> None:
    try:
        prefs = _load_prefs().copy()
        # Collect a curated set of keys to persist
        keys = [
            'backend','camera_choice','resolution_preset','width','height','fps',
            'dshow_name_device','dshow_name_text','convert_rgb','force_fourcc',
            'retry_black','gentle_mode','auto_repair','color_mode',
            'lock_exposure','exposure_value','lock_wb','wb_temperature','virtual_cam_enabled',
            'swap_mode','deep_swapper_model','morph',
            'face_swapper_model','face_swapper_pixel_boost','face_swapper_weight','source_paths',
            'colorizer_enabled','expression_restorer_enabled','age_modifier_enabled','face_editor_enabled','face_debugger_enabled','lip_syncer_enabled',
            'frame_colorizer_model','frame_colorizer_size','frame_colorizer_blend',
            'expression_restorer_model','expression_restorer_factor','expression_restorer_areas',
            'age_modifier_model','age_modifier_direction',
            'face_editor_model','face_editor_eyebrow_direction','face_editor_eye_gaze_horizontal','face_editor_eye_gaze_vertical','face_editor_eye_open_ratio','face_editor_lip_open_ratio','face_editor_mouth_smile','face_editor_mouth_grim','face_editor_mouth_pout','face_editor_mouth_purse','face_editor_mouth_position_horizontal','face_editor_mouth_position_vertical','face_editor_head_pitch','face_editor_head_yaw','face_editor_head_roll',
            'lip_syncer_model','lip_syncer_weight',
            'execution_providers','execution_device_ids','video_memory_strategy',
            'detector_model','detector_size','detector_score','landmarker_model','landmarker_score','selector_mode','auto_fallback',
            'occluder_model','parser_model','use_occlusion','realtime_fast_analysis',
            'gpu_pipeline_enabled',
            'provider_detector','provider_landmarker','provider_recognizer','provider_classifier',
            'provider_masker','provider_deep_swapper','provider_face_swapper',
            'provider_face_enhancer','provider_frame_enhancer','provider_colorizer',
            'provider_expression_restorer','provider_age_modifier','provider_face_editor',
            'provider_lip_syncer','provider_content_analyser',
            'show_overlay','debug_logs','show_boxes','show_native','fast_startup'
        ]
        for k in keys:
            v = state_manager.get_item(k)
            if v is not None:
                prefs[k] = v
        _save_prefs(prefs)
    except Exception:
        pass

# Default fast startup to True to avoid redundant pre_check when models are present
try:
    if state_manager.get_item('fast_startup') is None:
        state_manager.set_item('fast_startup', True)
    if state_manager.get_item('realtime_fast_analysis') is None:
        state_manager.set_item('realtime_fast_analysis', bool(_get_pref('realtime_fast_analysis', True)))
    if state_manager.get_item('gpu_pipeline_enabled') is None:
        state_manager.set_item('gpu_pipeline_enabled', bool(_get_pref('gpu_pipeline_enabled', False)))
except Exception:
    pass


def ensure_models_downloaded() -> None:
    global _models_downloaded
    if _models_downloaded:
        return
    try:
        if state_manager.get_item('fast_startup'):
            _models_downloaded = True
            return
        else:
            print("[INFO] Preparing FaceFusion models (first run may download files)...")
            LOGGER.info("Preparing FaceFusion models (first run may download files)...")
        # Seed essential state so pre_check() knows what to fetch
        try:
            if state_manager.get_item("download_scope") is None:
                state_manager.set_item("download_scope", "full")
            # Ensure margin exists as a 4-int list in 0..100
            if state_manager.get_item("face_detector_margin") is None:
                state_manager.set_item("face_detector_margin", [0, 0, 0, 0])
            if not state_manager.get_item("face_landmarker_model"):
                state_manager.set_item("face_landmarker_model", "many")
            if not state_manager.get_item("deep_swapper_model"):
                state_manager.set_item("deep_swapper_model", "iperov/james_carrey_224")
            # Reasonable defaults for optional processors to avoid None
            state_manager.set_item("background_remover_model", state_manager.get_item("background_remover_model") or "rmbg_2.0")
            state_manager.set_item("face_enhancer_model", state_manager.get_item("face_enhancer_model") or "gfpgan_1.4")
            state_manager.set_item("frame_enhancer_model", state_manager.get_item("frame_enhancer_model") or "span_kendata_x4")
            state_manager.set_item("face_swapper_model", state_manager.get_item("face_swapper_model") or "inswapper_128")
        except Exception:
            pass
        # 1) Ensure common modules are ready
        try:
            ff_core.common_pre_check()
        except Exception:
            pass
        # 2) Detect available processor module directories and set processors list
        try:
            modules_dir = os.path.join(FF_ROOT, 'facefusion', 'processors', 'modules')
            names = []
            if os.path.isdir(modules_dir):
                for entry in os.listdir(modules_dir):
                    entry_path = os.path.join(modules_dir, entry)
                    if os.path.isdir(entry_path) and not entry.startswith('__'):
                        names.append(entry)
           
            #remove deep_swapper from names
            if 'deep_swapper' in names:
                names.remove('deep_swapper')
            if not names:
                # Fallback to a curated list if discovery fails
                names = [
                    #'deep_swapper',
                    'face_swapper',
                    'frame_enhancer',
                    'face_enhancer',
                    'background_remover',
                    'expression_restorer',
                    'age_modifier',
                    'frame_colorizer',
                    'lip_syncer',
                    'face_debugger',
                ]
            state_manager.set_item('processors', names)
        except Exception:
            pass
        # 3) Download ALL model files for common + each processor by iterating create_static_model_set('full')
        def _download_model_set(model_set: dict) -> int:
            cnt = 0
            for cfg in (model_set or {}).values():
                hashes = cfg.get('hashes')
                sources = cfg.get('sources')
                if hashes:
                    conditional_download_hashes(hashes)
                if sources:
                    conditional_download_sources(sources)
                cnt += 1
            return cnt

        total = 0
        # Common modules we care about
        for mod in [ff_detector, ff_landmarker, ff_masker, ff_recognizer, ff_content, ff_classifier, ff_face_enhancer, ff_frame_enhancer, ff_face_swapper]:
            try:
                if hasattr(mod, 'create_static_model_set'):
                    total += _download_model_set(mod.create_static_model_set('full'))
            except Exception:
                continue
        # Processor modules
        prepared = 0
        for proc in (state_manager.get_item('processors') or []):
            try:
                mod = importlib.import_module(f'facefusion.processors.modules.{proc}.core')
                if hasattr(mod, 'create_static_model_set'):
                    prepared += _download_model_set(mod.create_static_model_set('full'))
            except Exception as e:
                LOGGER.warning(f"download_all({proc}) failed: {e}")
        LOGGER.info(f"Model sets enumerated: common+processors={total+prepared}")
        # 4) Explicitly ensure critical models are present
        try:
            ff_detector.pre_check()
        except Exception:
            pass
        # try:
        #     deep_swapper.pre_check()
        # except Exception:
        #     pass
    except Exception as e:
        LOGGER.warning(f"model preparation failed or partial: {e}")
    _models_downloaded = True


def list_cameras(max_index: int = 10) -> None:
    print("Detecting available cameras...")
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            if ret:
                print(f"[{idx}] - available")
            else:
                print(f"[{idx}] - openable, no frames")
            cap.release()
        else:
            print(f"[{idx}] - not available")


def _ffmpeg_camera_names() -> List[str]:
    try:
        # Probe DirectShow devices on Windows using ffmpeg if present
        proc = subprocess.run([
            "ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5)
        out = proc.stdout.splitlines()
        names = []
        capture = False
        for line in out:
            if "DirectShow video devices" in line:
                capture = True
                continue
            if "DirectShow audio devices" in line:
                capture = False
                continue
            # Recent FFmpeg builds omit the section headings and append a
            # media-type marker directly to each device line.
            is_video_device = line.rstrip().endswith("(video)")
            if (capture or is_video_device) and "Alternative name" not in line and '"' in line:
                name = line.split('"')
                if len(name) >= 2 and name[1] not in names:
                    names.append(name[1])
        return names
    except Exception:
        return []


def _camera_name_from_choice(camera_choice: Optional[str]) -> str:
    value = str(camera_choice or "").strip()
    if value.startswith("[") and "]" in value:
        return value.split("]", 1)[1].strip()
    return value


def _normalize_camera_fps(value: float) -> float:
    rounded = round(value)
    if abs(value - rounded) < 0.05:
        return float(rounded)
    return round(value, 2)


def _parse_ffmpeg_camera_capabilities(output: str) -> Dict[str, float]:
    modes: Dict[str, float] = {}
    range_pattern = re.compile(
        r"min s=(\d+)x(\d+) fps=([\d.]+) max s=(\d+)x(\d+) fps=([\d.]+)",
        re.IGNORECASE,
    )
    simple_pattern = re.compile(r"(?:^|\s)s=(\d+)x(\d+).*?fps=([\d.]+)", re.IGNORECASE)

    def _add_mode(width: int, height: int, fps_value: float) -> None:
        if width <= 0 or height <= 0 or fps_value <= 0:
            return
        resolution = f"{width}x{height}"
        modes[resolution] = max(modes.get(resolution, 0.0), _normalize_camera_fps(fps_value))

    for line in output.splitlines():
        range_match = range_pattern.search(line)
        if range_match:
            min_width, min_height, min_fps, max_width, max_height, max_fps = range_match.groups()
            _add_mode(int(min_width), int(min_height), float(min_fps))
            _add_mode(int(max_width), int(max_height), float(max_fps))
            continue
        simple_match = simple_pattern.search(line)
        if simple_match:
            width, height, fps_value = simple_match.groups()
            _add_mode(int(width), int(height), float(fps_value))
    return modes


def detect_camera_capabilities(camera_name: str, force_refresh: bool = False) -> Dict[str, float]:
    if not camera_name:
        return {}
    if not force_refresh and camera_name in _camera_capability_cache:
        return dict(_camera_capability_cache[camera_name])
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-f", "dshow", "-list_options", "true",
                "-i", f"video={camera_name}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        modes = _parse_ffmpeg_camera_capabilities(proc.stdout)
    except Exception as exc:
        LOGGER.warning(f"[cam] Could not detect modes for '{camera_name}': {exc}")
        modes = {}
    _camera_capability_cache[camera_name] = dict(modes)
    return modes


def _sorted_camera_resolutions(modes: Dict[str, float]) -> List[str]:
    def _area(resolution: str) -> int:
        try:
            width, height = (int(part) for part in resolution.split("x", 1))
            return width * height
        except Exception:
            return 0
    return sorted(modes, key=_area)


def _camera_mode_status(camera_name: str, modes: Dict[str, float], resolution: str) -> str:
    if not modes:
        return f"No driver modes were reported for **{camera_name}**. Using safe fallback choices."
    maximum_resolution = max(modes, key=lambda item: int(item.split("x")[0]) * int(item.split("x")[1]))
    selected_fps = modes.get(resolution, 0.0)
    return (
        f"Detected **{len(modes)} resolutions** for **{camera_name}**. "
        f"Selected **{resolution} up to {selected_fps:g} FPS**. "
        f"Highest resolution: **{maximum_resolution}**."
    )


def _powershell_pnp_camera_names() -> List[str]:
    # Fallback listing via PowerShell for friendly names
    try:
        ps = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -in @('Camera','Image') } | Select-Object -ExpandProperty Name"
        ]
        proc = subprocess.run(ps, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
        names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        # Deduplicate
        seen = set()
        uniq = []
        for n in names:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return uniq
    except Exception:
        return []


BACKEND_MAP: Dict[str, Optional[int]] = {
    "Auto": None,
    "DirectShow": cv2.CAP_DSHOW,
    "Media Foundation": cv2.CAP_MSMF,
}


def _open_capture(index: int, backend_name: str = "Auto", dshow_name: Optional[str] = None) -> Optional[cv2.VideoCapture]:
    # If user provided a DirectShow device name, try that first with DSHOW backend
    if dshow_name:
        cap = cv2.VideoCapture(f"video={dshow_name}", cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    # Backend selection
    preferred = BACKEND_MAP.get(backend_name, None)
    backends = []
    if preferred is None:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [preferred]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def _apply_fourcc(cap: cv2.VideoCapture, fourcc_name: str) -> bool:
    try:
        if not fourcc_name or fourcc_name == "Auto":
            return True
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        return bool(cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    except Exception:
        return False


def _set_convert_rgb(cap: cv2.VideoCapture, convert: bool) -> None:
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1 if convert else 0)
    except Exception:
        pass
    # Ensure execution settings
    try:
        if state_manager.get_item("execution_device_ids") is None:
            state_manager.set_item("execution_device_ids", ["0"])  # default single device
        if state_manager.get_item("execution_providers") is None:
            state_manager.set_item("execution_providers", ["cpu"])  # safe default
    except Exception:
        pass


def _apply_camera_controls(
    cap: cv2.VideoCapture,
    lock_exposure: bool,
    exposure_value: float,
    lock_wb: bool,
    wb_temperature: int,
) -> None:
    # Exposure control
    try:
        try:
            backend_name = str(cap.getBackendName()).upper()
        except Exception:
            backend_name = ""
        # DirectShow uses 0.25/0.75 for manual/automatic exposure.  Sending a
        # second generic 0/1 value immediately afterwards put some UVC camera
        # drivers into a broken ISP mode with strong magenta/green edge halos.
        # Pick the convention for the active backend and write it only once.
        manual_exposure_mode = 0.25 if "DSHOW" in backend_name else 0.0
        automatic_exposure_mode = 0.75 if "DSHOW" in backend_name else 1.0
        if lock_exposure:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual_exposure_mode)
            # Apply exposure value (note: many drivers expect negative log-scale)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_value))
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, automatic_exposure_mode)
    except Exception:
        pass
    # White balance control
    try:
        if lock_wb:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            # Temperature in Kelvin if supported
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, int(wb_temperature))
        else:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    except Exception:
        pass


def get_camera_choices(max_index: int = 5, backend_name: str = "Auto", dshow_name_hint: Optional[str] = None) -> List[str]:
    # SAFE enumeration: do not open devices here to avoid LED blinking and driver resets
    names = _ffmpeg_camera_names()
    choices: List[str] = []
    if names:
        for idx, label in enumerate(names[:max_index]):
            choices.append(f"[{idx}] {label}")
    else:
        for idx in range(max_index):
            choices.append(f"[{idx}] Camera {idx}")
    return choices


def init_state(
    model_id: str,
    use_occlusion: bool = True,
    morph: int = 100,
    selector_mode: str = "one",
) -> None:
    # Ensure download settings exist so model prechecks can resolve URLs
    try:
        if state_manager.get_item("download_providers") is None:
            state_manager.set_item("download_providers", list(ff_choices.download_providers))
        if state_manager.get_item("download_scope") is None:
            state_manager.set_item("download_scope", "full")
        if state_manager.get_item("log_level") is None:
            state_manager.set_item("log_level", "info")
    except Exception:
        pass
    # Processor-specific
    state_manager.set_item("deep_swapper_model", model_id)
    state_manager.set_item("deep_swapper_morph", int(max(0, min(100, morph))))
    # Ensure face mask defaults (avoid None in deep_swapper)
    try:
        if state_manager.get_item("face_mask_blur") is None:
            state_manager.set_item("face_mask_blur", 0.25)
        pad = state_manager.get_item("face_mask_padding")
        if not (isinstance(pad, (list, tuple)) and len(pad) == 4):
            state_manager.set_item("face_mask_padding", (0, 0, 0, 0))
    except Exception:
        pass

    # Face selector
    state_manager.set_item("face_selector_mode", selector_mode)  # 'one' | 'many' | 'reference'
    state_manager.set_item("reference_face_position", 0)
    state_manager.set_item("reference_face_distance", 0.5)
    state_manager.set_item("face_selector_order", "large-small")
    state_manager.set_item("face_selector_gender", None)
    state_manager.set_item("face_selector_race", None)
    state_manager.set_item("face_selector_age_start", 0)
    state_manager.set_item("face_selector_age_end", 100)

    # Face detector/landmarker/recognizer sensible defaults
    state_manager.set_item("face_detector_angles", [0])
    state_manager.set_item("face_detector_score", 0.5)
    state_manager.set_item("face_landmarker_score", 0.5)

    # Masking options
    mask_types = ["box"]
    if use_occlusion:
        mask_types.append("occlusion")
    state_manager.set_item("face_mask_types", mask_types)
    # Ensure face mask defaults every start
    try:
        if state_manager.get_item("face_mask_blur") is None:
            state_manager.set_item("face_mask_blur", 0.25)
        pad = state_manager.get_item("face_mask_padding")
        if not (isinstance(pad, (list, tuple)) and len(pad) == 4):
            state_manager.set_item("face_mask_padding", (0, 0, 0, 0))
    except Exception:
        pass
    state_manager.set_item("face_mask_blur", 0.7)
    state_manager.set_item("face_mask_padding", (0, 0, 0, 0))
    state_manager.set_item("face_mask_areas", [])
    state_manager.set_item("face_mask_regions", [])

    # Occluder/parser models
    state_manager.set_item("face_occluder_model", "xseg_2")
    state_manager.set_item("face_parser_model", "bisenet_resnet_18")

    # Memory strategy to allow clearing pools if needed
    state_manager.set_item("video_memory_strategy", "moderate")  # 'strict'|'moderate'|'permissive'

    # Needed by some pre/post hooks even if not used for files
    state_manager.set_item("target_path", "")
    state_manager.set_item("output_path", "")

    # Force download all common and processor models once at startup
    try:
        LOGGER.info("Force downloading FaceFusion models (this may take a while on first run)...")
        ff_core.force_download()
    except Exception as e:
        LOGGER.warning(f"force_download failed or partial: {e}")

    # Ensure models are present
    deep_swapper.pre_check()
    try:
        ff_detector.pre_check()
    except Exception:
        pass


def apply_state_from_ui(
    detector_model: str,
    detector_size: str,
    detector_score: float,
    landmarker_model: str,
    landmarker_score: float,
    occluder_model: str,
    parser_model: str,
    deep_model: str,
    morph: int,
    use_occlusion: bool,
    selector_mode: str = "one",
) -> None:
    state_manager.set_item("face_detector_model", detector_model)
    # validate detector size against choices
    try:
        valid_sizes = ff_choices.face_detector_set.get(detector_model, [])
        size_to_use = detector_size if detector_size in valid_sizes else (valid_sizes[0] if valid_sizes else detector_size)
    except Exception:
        size_to_use = detector_size
    state_manager.set_item("face_detector_size", size_to_use)
    state_manager.set_item("face_detector_score", detector_score)
    # A webcam frame is already upright. Running the detector again at 90,
    # 180 and 270 degrees quadruples the most expensive analysis stage and
    # provides no benefit for the normal live-camera case.
    state_manager.set_item("face_detector_angles", [0])
    state_manager.set_item("face_landmarker_model", landmarker_model)
    state_manager.set_item("face_landmarker_score", landmarker_score)
    state_manager.set_item("face_occluder_model", occluder_model)
    state_manager.set_item("face_parser_model", parser_model)
    state_manager.set_item("deep_swapper_model", deep_model)
    state_manager.set_item("deep_swapper_morph", morph)
    mask_types = ["box"]
    if use_occlusion:
        mask_types.append("occlusion")
    state_manager.set_item("face_mask_types", mask_types)
    # selector
    state_manager.set_item("face_selector_mode", selector_mode)


def draw_info(frame: VisionFrame, info: str, org: Tuple[int, int] = (10, 24)) -> None:
    cv2.putText(frame, info, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, info, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


_FAST_FACE_EMBEDDING = np.zeros(512, dtype=np.float32)


def _live_selector_requires_demographics() -> bool:
    """Return whether selector mode ``one`` genuinely needs classification."""
    if state_manager.get_item("face_selector_gender") is not None:
        return True
    if state_manager.get_item("face_selector_race") is not None:
        return True
    age_start = int(state_manager.get_item("face_selector_age_start") or 0)
    age_end = int(state_manager.get_item("face_selector_age_end") or 100)
    return age_start > 0 or age_end < 100


def _select_live_faces_refined(vision_frame: VisionFrame) -> List[Face]:
    """Keep refined landmarks while skipping unused identity analysis.

    The normal FaceFusion analyser always calculates an ArcFace embedding and
    age/gender/race classification for every detected face. Live selector mode
    ``one`` consumes neither result unless a demographic filter is active.
    This path otherwise mirrors ``face_analyser.create_faces`` so crop
    alignment and visual quality remain unchanged.
    """
    if _live_selector_requires_demographics():
        result = select_faces(reference_vision_frame=vision_frame, target_vision_frame=vision_frame)
        return result[0] if isinstance(result, tuple) else result

    bounding_boxes, face_scores, face_landmarks_5 = ff_detector.detect_faces(vision_frame)
    detector_score = float(state_manager.get_item("face_detector_score") or 0)
    if not bounding_boxes or detector_score <= 0:
        return []

    keep_indices = apply_nms(
        bounding_boxes,
        face_scores,
        detector_score,
        get_nms_threshold(
            state_manager.get_item("face_detector_model"),
            state_manager.get_item("face_detector_angles") or [0],
        ),
    )
    landmarker_threshold = float(state_manager.get_item("face_landmarker_score") or 0)
    faces: List[Face] = []
    for index in keep_indices:
        bounding_box = bounding_boxes[index]
        landmark_5 = face_landmarks_5[index]
        landmark_5_refined = landmark_5
        landmark_68_from_5 = ff_landmarker.estimate_face_landmark_68_5(landmark_5)
        landmark_68 = landmark_68_from_5
        landmark_score = 0.0
        face_angle = estimate_face_angle(landmark_68_from_5)

        if landmarker_threshold > 0:
            landmark_68, landmark_score = ff_landmarker.detect_face_landmark(
                vision_frame,
                bounding_box,
                face_angle,
            )
        if landmark_score > landmarker_threshold:
            landmark_5_refined = convert_to_face_landmark_5(landmark_68)

        faces.append(Face(
            bounding_box=bounding_box,
            score_set={
                "detector": face_scores[index],
                "landmarker": landmark_score,
            },
            landmark_set={
                "5": landmark_5,
                "5/68": landmark_5_refined,
                "68": landmark_68,
                "68/5": landmark_68_from_5,
            },
            angle=face_angle,
            embedding=_FAST_FACE_EMBEDDING,
            embedding_norm=_FAST_FACE_EMBEDDING,
            gender=None,
            age=range(0, 100),
            race=None,
        ))

    return sort_and_filter_faces(faces)[:1]


def _select_live_faces_fast(vision_frame: VisionFrame) -> List[Face]:
    """Select the largest live face without identity/demographic analysis.

    DeepFaceLive only consumes the target landmarks. Face recognition,
    classification and the separate 68-point refinement add latency but do
    not affect selector mode ``one`` or the deep-swap model itself.
    """
    bounding_boxes, face_scores, face_landmarks_5 = ff_detector.detect_faces(vision_frame)
    if not bounding_boxes or float(state_manager.get_item('face_detector_score') or 0) <= 0:
        return []

    nms_threshold = get_nms_threshold(
        state_manager.get_item('face_detector_model'),
        state_manager.get_item('face_detector_angles') or [0],
    )
    keep_indices = apply_nms(
        bounding_boxes,
        face_scores,
        float(state_manager.get_item('face_detector_score') or 0.5),
        nms_threshold,
    )
    faces: List[Face] = []
    for index in keep_indices:
        landmark_5 = face_landmarks_5[index]
        landmark_68_5 = ff_landmarker.estimate_face_landmark_68_5(landmark_5)
        faces.append(Face(
            bounding_box=bounding_boxes[index],
            score_set={
                'detector': face_scores[index],
                'landmarker': 0.0,
            },
            landmark_set={
                '5': landmark_5,
                '5/68': landmark_5,
                '68': landmark_68_5,
                '68/5': landmark_68_5,
            },
            angle=estimate_face_angle(landmark_68_5),
            embedding=_FAST_FACE_EMBEDDING,
            embedding_norm=_FAST_FACE_EMBEDDING,
            gender=None,
            age=range(0, 100),
            race=None,
        ))

    sorted_faces = sort_and_filter_faces(faces)
    return sorted_faces[:1]


def process_frame(frame: VisionFrame, do_swap: bool, show_debug: bool = False, debug_log: bool = False, show_boxes: bool = False) -> VisionFrame:
    if frame is None:
        return None
    # Determine swap on/off from swap_mode for real-time behavior
    try:
        sm = (state_manager.get_item('swap_mode') or 'deep')
        do_swap = (sm != 'none')
    except Exception:
        pass
    try:
        st_boxes = state_manager.get_item('show_boxes')
        if isinstance(st_boxes, bool):
            show_boxes = st_boxes
    except Exception:
        pass
    # Runtime guard to avoid NoneType margin in face_detector.prepare_margin
    try:
        margin = state_manager.get_item("face_detector_margin")
        if margin is None or (isinstance(margin, (list, tuple)) and len(margin) != 4):
            state_manager.set_item("face_detector_margin", [0, 0, 0, 0])
    except Exception:
        pass
    # Ensure detector score is a float, not a string
    try:
        det_score = state_manager.get_item("face_detector_score")
        if isinstance(det_score, str):
            state_manager.set_item("face_detector_score", float(det_score))
        elif det_score is None:
            state_manager.set_item("face_detector_score", 0.3)
    except Exception:
        try:
            state_manager.set_item("face_detector_score", 0.3)
        except Exception:
            pass
    # Prepare temp/out frame first so processors can run even when no swap
    temp = frame.copy() if hasattr(frame, "copy") else frame
    out = temp

    if not do_swap:
        if show_debug:
            try:
                faces_dbg_res = select_faces(reference_vision_frame=frame, target_vision_frame=frame)
                faces_dbg = faces_dbg_res[0] if isinstance(faces_dbg_res, tuple) else faces_dbg_res
                draw_info(frame, f"faces: {len(faces_dbg)}")
                if show_boxes:
                    try:
                        for f in faces_dbg:
                            x1, y1, x2, y2 = map(int, f.bounding_box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except Exception:
                        pass
                if debug_log:
                    LOGGER.info(f"[process_frame] no-swap: faces={len(faces_dbg)}")
            except Exception:
                pass
    else:
        # Build inputs for swapping path
        inputs = {
            "reference_vision_frame": frame,  # unused when selector_mode='one'
            "target_vision_frame": frame,
            "temp_vision_frame": temp,
        }

    try:
        # Optimization: Downscale for detection if frame is large
        det_frame = frame
        det_scale = 1.0
        h, w = frame.shape[:2]
        MAX_DET_W = 640
        if w > MAX_DET_W:
            det_scale = w / MAX_DET_W
            new_h = int(h / det_scale)
            det_frame = cv2.resize(frame, (MAX_DET_W, new_h))
            # if debug_log: LOGGER.info(f"[process_frame] Downscaled detection: {w}x{h} -> {MAX_DET_W}x{new_h} (scale={det_scale:.2f})")

        # Ensure faces are selected; if none, we will skip swapping but still allow processors
        try:
            use_fast_analysis = (
                bool(state_manager.get_item('realtime_fast_analysis'))
                and state_manager.get_item('face_selector_mode') == 'one'
            )
            selector_mode = state_manager.get_item("face_selector_mode")
            if use_fast_analysis:
                faces_res = _select_live_faces_fast(det_frame)
            elif (
                bool(state_manager.get_item("temporal_face_tracking"))
                and selector_mode == "one"
            ):
                faces_res = _temporal_face_tracker.select(det_frame)
            elif selector_mode == "one":
                faces_res = _select_live_faces_refined(det_frame)
            else:
                faces_res = select_faces(reference_vision_frame=det_frame, target_vision_frame=det_frame)
            faces_det = faces_res[0] if isinstance(faces_res, tuple) else faces_res

            # Scale faces back to original resolution if we downscaled
            faces = []
            if faces_det:
                if det_scale != 1.0:
                    for f in faces_det:
                        # Scale bounding box
                        new_bbox = f.bounding_box * det_scale
                        # Scale landmarks
                        new_lm_set = {}
                        if f.landmark_set:
                            for k, v in f.landmark_set.items():
                                if isinstance(v, np.ndarray):
                                    new_lm_set[k] = v * det_scale
                                else:
                                    new_lm_set[k] = v
                        # Create new face with scaled coords
                        faces.append(f._replace(bounding_box=new_bbox, landmark_set=new_lm_set))
                else:
                    faces = faces_det

            # track for fallback logic
            global _last_faces_count
            _last_faces_count = len(faces) if faces is not None else 0
            if show_debug:
                try:
                    draw_info(frame, f"faces: {len(faces)}")
                except Exception:
                    pass
            if show_boxes and faces:
                try:
                    for f in faces:
                        x1, y1, x2, y2 = map(int, f.bounding_box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except Exception:
                    pass
            if not faces and debug_log:
                LOGGER.info("[process_frame] no faces: swapping is skipped but processors may run")
        except Exception as e:
            # Selection failure: skip swapping but continue to processors
            faces = []
            if debug_log:
                LOGGER.exception(f"[process_frame] select_faces error; continuing without swap: {e}")
        # Ensure face mask parameters are present before swapping (runtime guard)
        try:
            blur_val = state_manager.get_item("face_mask_blur")
            if blur_val is None:
                state_manager.set_item("face_mask_blur", 0.5)
            pad_val = state_manager.get_item("face_mask_padding")
            if not (isinstance(pad_val, (list, tuple)) and len(pad_val) == 4):
                state_manager.set_item("face_mask_padding", (0, 0, 0, 0))
        except Exception:
            pass
        # Choose swapper guarded by do_swap; otherwise keep 'out' as temp and let processors run
        swap_mode = (state_manager.get_item("swap_mode") or "deep")
        out = temp
        if do_swap and faces and len(faces) > 0:
            if swap_mode == "face":
                # Prepare source frames from uploaded images if any
                try:
                    cached_face = _get_cached_source_face()
                    if cached_face is not None:
                        for target_face in faces:
                            try:
                                t_scaled = scale_face(target_face, frame, out)
                                out = ff_face_swapper.swap_face(cached_face, t_scaled, out)
                            except Exception:
                                continue
                    else:
                        if _source_imgs:
                            fs_inputs = {
                                "reference_vision_frame": frame,
                                "source_vision_frames": _source_imgs,
                                "target_vision_frame": frame,
                                "temp_vision_frame": out,
                            }
                        else:
                            fs_inputs = {
                                "reference_vision_frame": frame,
                                "source_vision_frames": None,
                                "target_vision_frame": frame,
                                "temp_vision_frame": out,
                            }
                        try:
                            ff_face_swapper.pre_check()
                        except Exception:
                            pass
                        out = ff_face_swapper.process_frame(fs_inputs)
                        # face_swapper may return (frame, mask); extract frame
                        try:
                            if isinstance(out, tuple) and len(out) > 0:
                                out = out[0]
                        except Exception:
                            pass
                except Exception as e:
                    if debug_log:
                        LOGGER.exception(f"[process_frame] face_swapper failed: {e}")
                    out = temp
            else:
                # Deep swap path
                try:
                    _ = deep_swapper.get_model_size()
                except Exception as _e:
                    try:
                        if debug_log:
                            LOGGER.info("[process_frame] deep_swapper not ready; running pre_check and clearing pool")
                        if deep_swapper.pre_check():
                            deep_swapper.clear_inference_pool()
                    except Exception:
                        pass
                try:
                    # The faces above were already detected and scaled back to
                    # output coordinates. Calling deep_swapper.process_frame()
                    # here would run the entire detector, landmarker,
                    # recognizer and classifier a second time on the full-size
                    # frame. Reuse the existing Face objects and invoke only
                    # the actual crop/model/mask/paste operation.
                    for target_face in faces:
                        if (
                            HAS_GPU_FACE_PIPELINE
                            and gpu_face_pipeline is not None
                            and bool(state_manager.get_item('gpu_pipeline_enabled'))
                        ):
                            try:
                                out = gpu_face_pipeline.swap_face(target_face, out)
                            except Exception as gpu_pipeline_error:
                                gpu_face_pipeline.record_failure(gpu_pipeline_error)
                                if gpu_face_pipeline.status().get('failures', 0) <= 2:
                                    LOGGER.warning(
                                        "CUDA crop pipeline failed; using the compatible CPU image path: %s",
                                        gpu_pipeline_error,
                                    )
                                out = deep_swapper.swap_face(target_face, out)
                        else:
                            out = deep_swapper.swap_face(target_face, out)
                except Exception as e:
                    if debug_log:
                        LOGGER.exception(f"[process_frame] deep swap failed: {e}")
                    out = temp
        # Accept only non-empty numpy image outputs; otherwise, fall back
        if isinstance(out, np.ndarray) and getattr(out, 'size', 0) > 0:
            # Optional post processors that run in real-time
            try:
                if state_manager.get_item('frame_colorizer_enabled'):
                    # Run pre_check only once per session
                    try:
                        if not state_manager.get_item('fast_startup') and not state_manager.get_item('frame_colorizer_ready'):
                            if ff_frame_colorizer.pre_check():
                                state_manager.set_item('frame_colorizer_ready', True)
                    except Exception:
                        pass
                    out = ff_frame_colorizer.colorize_frame(out)
                if state_manager.get_item('expression_restorer_enabled'):
                    try:
                        if not state_manager.get_item('fast_startup') and not state_manager.get_item('expression_restorer_ready'):
                            if ff_expr_restorer.pre_check():
                                state_manager.set_item('expression_restorer_ready', True)
                    except Exception:
                        pass
                    try:
                        res = ff_expr_restorer.process_frame({
                            'reference_vision_frame': frame,
                            'target_vision_frame': frame,
                            'temp_vision_frame': out,
                            'temp_vision_mask': None,
                        })
                        if isinstance(res, tuple) and len(res) > 0:
                            out = res[0]
                        elif isinstance(res, np.ndarray):
                            out = res
                    except Exception:
                        pass
                if state_manager.get_item('age_modifier_enabled'):
                    try:
                        if not state_manager.get_item('fast_startup') and not state_manager.get_item('age_modifier_ready'):
                            if ff_age_modifier.pre_check():
                                state_manager.set_item('age_modifier_ready', True)
                    except Exception:
                        pass
                    try:
                        res = ff_age_modifier.process_frame({
                            'reference_vision_frame': frame,
                            'target_vision_frame': frame,
                            'temp_vision_frame': out,
                            'temp_vision_mask': None,
                        })
                        if isinstance(res, tuple) and len(res) > 0:
                            out = res[0]
                        elif isinstance(res, np.ndarray):
                            out = res
                    except Exception:
                        pass
                if state_manager.get_item('face_editor_enabled'):
                    try:
                        if not state_manager.get_item('fast_startup') and not state_manager.get_item('face_editor_ready'):
                            if ff_face_editor.pre_check():
                                state_manager.set_item('face_editor_ready', True)
                    except Exception:
                        LOGGER.exception("[process_frame] face_editor pre_check error")
                    # Ensure all editor sliders are initialized to 0.0 to avoid NoneType errors
                    try:
                        _editor_keys = [
                            'face_editor_eyebrow_direction',
                            'face_editor_eye_gaze_horizontal',
                            'face_editor_eye_gaze_vertical',
                            'face_editor_eye_open_ratio',
                            'face_editor_lip_open_ratio',
                            'face_editor_mouth_smile',
                            'face_editor_mouth_grim',
                            'face_editor_mouth_pout',
                            'face_editor_mouth_purse',
                            'face_editor_mouth_position_horizontal',
                            'face_editor_mouth_position_vertical',
                            'face_editor_head_pitch',
                            'face_editor_head_yaw',
                            'face_editor_head_roll',
                        ]
                        for _k in _editor_keys:
                            _v = state_manager.get_item(_k)
                            if _v is None:
                                state_manager.set_item(_k, 0.0)
                    except Exception:
                        pass
                    try:
                        res = ff_face_editor.process_frame({
                            'reference_vision_frame': frame,
                            'target_vision_frame': frame,
                            'temp_vision_frame': out,
                            'temp_vision_mask': None,
                        })
                        #LOGGER.info(f"[face_editor] res type={type(res)} is_tuple={isinstance(res, tuple)} is_ndarray={isinstance(res, np.ndarray)}")
                        if isinstance(res, tuple) and len(res) > 0:
                            out = res[0]
                            #LOGGER.info("[process_frame] face_editor success tuple")
                        elif isinstance(res, np.ndarray):
                            out = res
                            #LOGGER.info("[process_frame] face_editor success ndarray")
                        #else:
                            #LOGGER.warning(f"[process_frame] face_editor returned unexpected type: {type(res)}")
                    except Exception as e:
                        LOGGER.exception(f"[process_frame] face_editor error: {e}")
                        pass
                if state_manager.get_item('lip_syncer_enabled'):
                    try:
                        if not state_manager.get_item('fast_startup') and not state_manager.get_item('lip_syncer_ready'):
                            if ff_lip_syncer.pre_check():
                                state_manager.set_item('lip_syncer_ready', True)
                    except Exception:
                        pass
                    try:
                        res = ff_lip_syncer.process_frame({
                            'reference_vision_frame': frame,
                            'source_voice_frame': None,
                            'target_vision_frame': frame,
                            'temp_vision_frame': out,
                            'temp_vision_mask': None,
                        })
                        if isinstance(res, tuple) and len(res) > 0:
                            out = res[0]
                        elif isinstance(res, np.ndarray):
                            out = res
                    except Exception:
                        pass
                if state_manager.get_item('face_debugger_enabled'):
                    try:
                        res = ff_face_debugger.process_frame({
                            'reference_vision_frame': frame,
                            'target_vision_frame': frame,
                            'temp_vision_frame': out,
                            'temp_vision_mask': None,
                        })
                        if isinstance(res, tuple) and len(res) > 0:
                            out = res[0]
                        elif isinstance(res, np.ndarray):
                            out = res
                    except Exception:
                        pass
            except Exception:
                pass
            if show_debug:
                try:
                    draw_info(out, f"faces: {len(faces)}")
                except Exception:
                    pass
            if show_boxes and faces:
                try:
                    for f in faces:
                        x1, y1, x2, y2 = map(int, f.bounding_box)
                        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except Exception:
                    pass
            # Optional enhancers after swap (skip here if async enhance is enabled)
            try:
                if not state_manager.get_item("enhance_async"):
                    if state_manager.get_item("face_enhancer_enabled"):
                        try:
                            ff_face_enhancer.pre_check()
                        except Exception:
                            pass
                        out = ff_face_enhancer.process_frame({
                            "reference_vision_frame": frame,
                            "target_vision_frame": frame,
                            "temp_vision_frame": out,
                        })
                    if state_manager.get_item("frame_enhancer_enabled"):
                        try:
                            ff_frame_enhancer.pre_check()
                        except Exception:
                            pass
                        out = ff_frame_enhancer.process_frame({
                            "temp_vision_frame": out,
                        })
            except Exception as e:
                if debug_log:
                    LOGGER.exception(f"[process_frame] enhancers failed: {e}")
            if debug_log:
                LOGGER.info(f"[process_frame] swapped({swap_mode}) faces={len(faces)} size={out.shape}")
            return out
        if debug_log:
            LOGGER.info("[process_frame] swap returned empty; falling back to original frame")
        return frame
    except Exception as e:
        LOGGER.exception(f"Deep swap failed this frame: {e}")
        return frame


def get_model_choices() -> Dict[str, List[str]]:
    return {
        "detector_models": ff_choices.face_detector_models,
        "detector_sizes_map": {k: v for k, v in ff_choices.face_detector_set.items()},
        "landmarker_models": ff_choices.face_landmarker_models,
        "occluder_models": ff_choices.face_occluder_models,
        "parser_models": ff_choices.face_parser_models,
        "deep_models": proc_choices.deep_swapper_models,
        "face_swapper_models": proc_choices.face_swapper_models,
        "frame_enhancer_models": proc_choices.frame_enhancer_models,
        "face_enhancer_models": proc_choices.face_enhancer_models,
    }


def _available_execution_provider_keys() -> List[str]:
    # Map ORT provider names to FaceFusion provider keys
    try:
        ort_avail = set(ort.get_available_providers())
    except Exception:
        ort_avail = set()
    key_to_ort = ff_choices.execution_provider_set  # key -> ORT name
    avail_keys: List[str] = []
    for k, v in key_to_ort.items():
        if k == 'tensorrt' and not HAS_TENSORRT:
            continue
        if v in ort_avail:
            avail_keys.append(k)
    # Always include cpu as fallback
    if 'cpu' not in avail_keys:
        avail_keys.append('cpu')
    return avail_keys


_MODULE_PROVIDER_ROUTES = [
    ('provider_detector', 'Detector', ff_detector),
    ('provider_landmarker', 'Landmarker', ff_landmarker),
    ('provider_recognizer', 'Recognizer', ff_recognizer),
    ('provider_classifier', 'Classifier', ff_classifier),
    ('provider_masker', 'Masker', ff_masker),
    ('provider_deep_swapper', 'Deep swapper', deep_swapper),
    ('provider_face_swapper', 'Face swapper', ff_face_swapper),
    ('provider_face_enhancer', 'Face enhancer', ff_face_enhancer),
    ('provider_frame_enhancer', 'Frame enhancer', ff_frame_enhancer),
    ('provider_colorizer', 'Colorizer', ff_frame_colorizer),
    ('provider_expression_restorer', 'Expression restorer', ff_expr_restorer),
    ('provider_age_modifier', 'Age modifier', ff_age_modifier),
    ('provider_face_editor', 'Face editor', ff_face_editor),
    ('provider_lip_syncer', 'Lip syncer', ff_lip_syncer),
    ('provider_content_analyser', 'Content analyser', ff_content),
]

_INFERENCE_MODULE_ROUTE_PATTERNS = [
    ('processors.modules.deep_swapper', 'provider_deep_swapper'),
    ('processors.modules.face_swapper', 'provider_face_swapper'),
    ('processors.modules.face_enhancer', 'provider_face_enhancer'),
    ('processors.modules.frame_enhancer', 'provider_frame_enhancer'),
    ('processors.modules.frame_colorizer', 'provider_colorizer'),
    ('processors.modules.expression_restorer', 'provider_expression_restorer'),
    ('processors.modules.age_modifier', 'provider_age_modifier'),
    ('processors.modules.face_editor', 'provider_face_editor'),
    ('processors.modules.lip_syncer', 'provider_lip_syncer'),
    ('face_detector', 'provider_detector'),
    ('face_landmarker', 'provider_landmarker'),
    ('face_recognizer', 'provider_recognizer'),
    ('face_classifier', 'provider_classifier'),
    ('face_masker', 'provider_masker'),
    ('content_analyser', 'provider_content_analyser'),
]
_inference_timing_lock = threading.RLock()
_inference_call_timings = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1200)))
_inference_frame_timings = defaultdict(lambda: defaultdict(lambda: deque(maxlen=600)))
_inference_profile_local = threading.local()


def _profile_route_for_module(module_name: str) -> Optional[str]:
    for pattern, route_key in _INFERENCE_MODULE_ROUTE_PATTERNS:
        if pattern in module_name:
            return route_key
    return None


def _profile_provider_key(provider_name: str) -> str:
    provider_lower = str(provider_name).lower()
    if 'cuda' in provider_lower:
        return 'cuda'
    if 'tensorrt' in provider_lower:
        return 'tensorrt'
    if 'directml' in provider_lower or 'dml' in provider_lower:
        return 'directml'
    if 'coreml' in provider_lower:
        return 'coreml'
    return 'cpu' if 'cpu' in provider_lower else provider_lower.replace('executionprovider', '')


def _record_inference_timing(module_name: str, model_name: str, provider_name: str, elapsed_ms: float) -> None:
    route_key = _profile_route_for_module(module_name)
    if route_key is None:
        return
    provider_key = _profile_provider_key(provider_name)
    timestamp = time.time()
    with _inference_timing_lock:
        _inference_call_timings[route_key][provider_key].append((timestamp, float(elapsed_ms), str(model_name)))
    if bool(getattr(_inference_profile_local, 'active', False)):
        totals = getattr(_inference_profile_local, 'totals', None)
        if totals is not None:
            totals[(route_key, provider_key)] += float(elapsed_ms)


def _profile_inference_frame(function):
    @wraps(function)
    def _wrapped(*args, **kwargs):
        if bool(getattr(_inference_profile_local, 'active', False)):
            return function(*args, **kwargs)
        _inference_profile_local.active = True
        _inference_profile_local.totals = defaultdict(float)
        try:
            return function(*args, **kwargs)
        finally:
            timestamp = time.time()
            totals = dict(getattr(_inference_profile_local, 'totals', {}))
            with _inference_timing_lock:
                for (route_key, provider_key), elapsed_ms in totals.items():
                    _inference_frame_timings[route_key][provider_key].append((timestamp, float(elapsed_ms)))
            _inference_profile_local.active = False
            _inference_profile_local.totals = defaultdict(float)
    return _wrapped


def reset_inference_timings() -> None:
    with _inference_timing_lock:
        _inference_call_timings.clear()
        _inference_frame_timings.clear()


def _timing_stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    sorted_values = sorted(values)
    p95_index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * 0.95) - 1))
    return {
        'mean_ms': round(sum(values) / len(values), 3),
        'median_ms': round(sorted_values[len(sorted_values) // 2], 3),
        'p95_ms': round(sorted_values[p95_index], 3),
        'samples': len(values),
    }


def get_inference_timing_summary() -> Dict[str, object]:
    label_by_key = {state_key: label for state_key, label, _module in _MODULE_PROVIDER_ROUTES}
    now = time.time()
    result: Dict[str, object] = {'generated_at': now, 'modules': {}}
    with _inference_timing_lock:
        for route_key, label in label_by_key.items():
            providers: Dict[str, object] = {}
            provider_keys = set(_inference_call_timings.get(route_key, {}).keys()) | set(_inference_frame_timings.get(route_key, {}).keys())
            for provider_key in sorted(provider_keys):
                calls = list(_inference_call_timings.get(route_key, {}).get(provider_key, []))
                frames = list(_inference_frame_timings.get(route_key, {}).get(provider_key, []))
                model_values = defaultdict(list)
                for _timestamp, elapsed_ms, model_name in calls:
                    model_values[model_name].append(elapsed_ms)
                last_timestamp = max(
                    [entry[0] for entry in calls] + [entry[0] for entry in frames] + [0.0]
                )
                providers[provider_key] = {
                    'per_call': _timing_stats([entry[1] for entry in calls]),
                    'per_frame': _timing_stats([entry[1] for entry in frames]),
                    'models': {model_name: _timing_stats(values) for model_name, values in sorted(model_values.items())},
                    'age_seconds': round(max(0.0, now - last_timestamp), 1) if last_timestamp else None,
                }
            result['modules'][route_key] = {
                'label': label,
                'providers': providers,
            }
    return result


ff_inference_manager.set_inference_timing_callback(_record_inference_timing)
process_frame = _profile_inference_frame(process_frame)


def _normalize_module_provider(provider: Optional[str]) -> str:
    available = _available_execution_provider_keys()
    if provider in available:
        return str(provider)
    main_provider = (state_manager.get_item('execution_providers') or ['cuda'])[0]
    if main_provider in available:
        return str(main_provider)
    return 'cuda' if 'cuda' in available else 'cpu'


def _module_provider_resolver(state_key: str):
    def _resolve() -> List[str]:
        return [_normalize_module_provider(state_manager.get_item(state_key))]
    return _resolve


def _install_module_provider_routing(prefer_current_state: bool = False) -> None:
    available_providers = _available_execution_provider_keys()
    default_provider = 'cuda' if 'cuda' in available_providers else 'cpu'
    # XSeg TensorRT FP16 is fast but can create visible mask-edge corruption.
    # Keep the stable CUDA mask path as the default; DFM TensorRT FP32 retains
    # its measured speedup without changing the rendered face materially.
    tensorrt_routes = {'provider_deep_swapper'}
    for state_key, _label, module in _MODULE_PROVIDER_ROUTES:
        measured_default = 'tensorrt' if state_key in tensorrt_routes and 'tensorrt' in available_providers else default_provider
        configured_provider = state_manager.get_item(state_key) if prefer_current_state else _get_pref(state_key, measured_default)
        selected = _normalize_module_provider(configured_provider)
        state_manager.set_item(state_key, selected)
        setattr(module, 'resolve_execution_providers', _module_provider_resolver(state_key))


def _module_routing_summary() -> str:
    routes = [
        f"{label}: {_normalize_module_provider(state_manager.get_item(state_key)).upper()}"
        for state_key, label, _module in _MODULE_PROVIDER_ROUTES
    ]
    return "**Configured routing:** " + " · ".join(routes)


_install_module_provider_routing()


_stop_stream = False
_stream_generation = 0
_stream_lock = threading.Lock()
_capture_lock = threading.RLock()
_live_processing_lock = threading.RLock()
_cap: Optional[cv2.VideoCapture] = None
_cap_settings: Dict[str, object] = {}
_last_faces_count: int = 0
_zero_face_streak: int = 0
_source_imgs: List[np.ndarray] = []
_cached_source_face = None
_cached_source_key: Optional[tuple] = None
_latest_raw_frame: Optional[np.ndarray] = None
_latest_raw_frame_sequence: int = 0
_latest_raw_frame_captured_at: float = 0.0
_latest_raw_frame_condition = threading.Condition()


def _publish_latest_raw_frame(frame: np.ndarray, captured_at: float = 0.0) -> None:
    """Expose the newest unprocessed camera frame to guided capture tools.

    The camera reader already owns each returned ndarray until it replaces it
    with the next frame, so retaining the reference costs no extra full-HD copy
    in the realtime path. Consumers receive their own copy below.
    """
    global _latest_raw_frame, _latest_raw_frame_sequence, _latest_raw_frame_captured_at
    if not isinstance(frame, np.ndarray) or not frame.size:
        return
    with _latest_raw_frame_condition:
        _latest_raw_frame = frame
        _latest_raw_frame_sequence += 1
        _latest_raw_frame_captured_at = float(captured_at or time.perf_counter())
        _latest_raw_frame_condition.notify_all()


def get_latest_raw_frame(
    after_sequence: int = -1,
    timeout: float = 2.0,
) -> Tuple[Optional[np.ndarray], int, float]:
    """Return a copy of the newest raw frame, optionally waiting for a newer one."""
    deadline = time.perf_counter() + max(0.0, float(timeout))
    with _latest_raw_frame_condition:
        while _latest_raw_frame is None or _latest_raw_frame_sequence <= after_sequence:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            _latest_raw_frame_condition.wait(remaining)
        if _latest_raw_frame is None or _latest_raw_frame_sequence <= after_sequence:
            return None, _latest_raw_frame_sequence, _latest_raw_frame_captured_at
        return _latest_raw_frame.copy(), _latest_raw_frame_sequence, _latest_raw_frame_captured_at


def _live_bool(key: str, fallback: bool = False) -> bool:
    value = state_manager.get_item(key)
    return fallback if value is None else bool(value)


def _live_text(key: str, fallback: str = "") -> str:
    value = state_manager.get_item(key)
    return fallback if value is None else str(value)


def _invalidate_source_cache():
    global _cached_source_face, _cached_source_key
    _cached_source_face = None
    _cached_source_key = None


def set_source_paths(paths) -> None:
    """Replace live face-swap source images without restarting capture."""
    global _source_imgs
    clean_paths = [str(path) for path in (paths or []) if path]
    source_images: List[np.ndarray] = []
    for path in clean_paths:
        try:
            image = cv2.imread(path)
            if isinstance(image, np.ndarray) and image.size:
                source_images.append(image)
        except Exception:
            pass
    state_manager.set_item('source_paths', clean_paths)
    _source_imgs = source_images
    _invalidate_source_cache()


def close_virtual_camera() -> bool:
    """Close only the virtual output while leaving physical capture running."""
    return _virtual_camera_publisher.disable()


def configure_virtual_camera(
    enabled: bool,
    width: int,
    height: int,
    fps: float,
    persistent: Optional[bool] = None,
) -> None:
    _virtual_camera_publisher.configure(enabled, width, height, fps, persistent)


def set_virtual_camera_persistent(persistent: bool) -> None:
    _virtual_camera_publisher.set_persistent(persistent)


def publish_virtual_camera_frame(frame: np.ndarray, input_is_rgb: bool = False) -> bool:
    return _virtual_camera_publisher.publish(frame, input_is_rgb)


def release_virtual_camera_source() -> None:
    _virtual_camera_publisher.source_stopped()


def shutdown_virtual_camera() -> None:
    _virtual_camera_publisher.shutdown()


def virtual_camera_status() -> Dict[str, object]:
    return _virtual_camera_publisher.status()


def _current_source_key() -> tuple:
    try:
        paths = tuple(state_manager.get_item('source_paths') or [])
    except Exception:
        paths = tuple()
    try:
        model = state_manager.get_item('face_swapper_model') or ''
    except Exception:
        model = ''
    try:
        pixel = state_manager.get_item('face_swapper_pixel_boost') or ''
    except Exception:
        pixel = ''
    return (model, pixel, paths)


def _get_cached_source_face():
    global _cached_source_face, _cached_source_key
    key = _current_source_key()
    if _cached_source_face is not None and _cached_source_key == key:
        return _cached_source_face
    # Recompute if we have source images
    try:
        if _source_imgs:
            try:
                ff_face_swapper.pre_check()
            except Exception:
                pass
            face = ff_face_swapper.extract_source_face(_source_imgs)
            _cached_source_face = face
            _cached_source_key = key
            return face
    except Exception:
        pass
    return None


def _parse_cam_index(camera_choice: str) -> int:
    try:
        return int(camera_choice.split(']')[0][1:])
    except Exception:
        return 0


def _cleanup_inference() -> None:
    # Clear all known inference pools so ORT sessions are released and VRAM is freed
    reset_temporal_occlusion_cache()
    reset_temporal_face_tracker()
    if gpu_face_pipeline is not None:
        gpu_face_pipeline.reset_status()
    try:
        # A routing change produces a different inference-context key. Clear
        # every context, including sessions created by the previous provider.
        ff_inference_manager.INFERENCE_POOL_SET['cli'].clear()
        ff_inference_manager.INFERENCE_POOL_SET['ui'].clear()
    except Exception:
        pass
    try:
        deep_swapper.clear_inference_pool()
    except Exception:
        pass
    try:
        ff_face_swapper.clear_inference_pool()
    except Exception:
        pass
    try:
        ff_detector.clear_inference_pool()
    except Exception:
        pass
    try:
        ff_landmarker.clear_inference_pool()
    except Exception:
        pass
    try:
        ff_masker.clear_inference_pool()
    except Exception:
        pass
    try:
        ff_recognizer.clear_inference_pool()
    except Exception:
        pass
    try:
        ff_content.clear_inference_pool()
    except Exception:
        pass


def gpu_pipeline_status() -> Dict[str, object]:
    if gpu_face_pipeline is None:
        return {
            'available': False,
            'enabled': bool(state_manager.get_item('gpu_pipeline_enabled')),
            'active': False,
            'mean_ms': None,
            'samples': 0,
            'failures': 0,
            'last_error': 'CUDA PyTorch pipeline is unavailable',
        }
    return gpu_face_pipeline.status()
    try:
        ff_classifier.clear_inference_pool()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def _settings_from_inputs(
    camera_choice: str,
    backend_name: str,
    dshow_name: Optional[str],
    width: Optional[int],
    height: Optional[int],
    target_fps: Optional[float],
    convert_rgb: bool,
    force_fourcc: str,
) -> Dict[str, object]:
    cam_index = _parse_cam_index(camera_choice)
    resolved_backend = backend_name
    if dshow_name:
        try:
            # The selected dropdown already carries the DirectShow index and
            # label. Avoid re-enumerating devices during every start: on some
            # Windows drivers that briefly toggles the camera before capture.
            selected_name = _camera_name_from_choice(camera_choice)
            if selected_name == dshow_name:
                resolved_backend = "DirectShow"
            else:
                dshow_names = _ffmpeg_camera_names()
                if dshow_name not in dshow_names:
                    raise ValueError(f"DirectShow device '{dshow_name}' was not found")
                cam_index = dshow_names.index(dshow_name)
                resolved_backend = "DirectShow"
            LOGGER.info(f"[cam] Resolved DirectShow device '{dshow_name}' to camera index {cam_index}")
        except Exception as exc:
            LOGGER.warning(f"[cam] Could not resolve DirectShow device name: {exc}")
    return {
        "cam_index": cam_index,
        "backend": resolved_backend,
        "dshow_name": dshow_name or None,
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "fps": float(target_fps) if target_fps else None,
        "convert_rgb": bool(convert_rgb),
        "fourcc": force_fourcc or "Auto",
    }


def _settings_equal(a: Dict[str, object], b: Dict[str, object]) -> bool:
    return all(a.get(k) == b.get(k) for k in ("cam_index","backend","dshow_name","width","height","fps","convert_rgb","fourcc"))


def _fourcc_string(cap: cv2.VideoCapture) -> str:
    try:
        fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
        return ''.join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    except Exception:
        return "????"


class LatestFrameReader:
    """Continuously capture and retain only the newest webcam frame."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._captured_at = 0.0
        self._sequence = -1
        self._thread = threading.Thread(target=self._capture_loop, name="latest-webcam-frame", daemon=True)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if ok and isinstance(frame, np.ndarray) and frame.size:
                with self._condition:
                    self._frame = frame
                    self._captured_at = time.perf_counter()
                    self._sequence += 1
                    self._condition.notify_all()
            else:
                self._stop_event.wait(0.01)

    def read_after(
        self,
        previous_sequence: int,
        timeout: float = 2.0,
    ) -> Tuple[bool, Optional[np.ndarray], int, int, float]:
        deadline = time.perf_counter() + timeout
        with self._condition:
            while self._sequence <= previous_sequence and not self._stop_event.is_set():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._frame is None or self._sequence <= previous_sequence:
                return False, None, previous_sequence, 0, 0.0
            sequence = self._sequence
            dropped = max(0, sequence - previous_sequence - 1) if previous_sequence >= 0 else 0
            return True, self._frame, sequence, dropped, self._captured_at

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def _run_decoupled_fast_stream(
    cap: cv2.VideoCapture,
    stream_generation: int,
    settings: Dict[str, object],
    target_fps: float,
    color_mode: str,
    show_overlay: bool,
    debug_logs: bool,
    show_boxes: bool,
    show_native_window: bool,
    virtual_cam_enabled: bool,
    auto_fallback: bool,
):
    """Process continuously while Gradio displays only the newest frame.

    Encoding and delivering a 1080p Gradio image can take much longer than a
    swap. A generator normally pauses at every yield, which used to throttle
    the swap and virtual-camera loop to the browser preview rate.
    """
    global _benchmark_completed_generation

    frame_reader = LatestFrameReader(cap)
    worker_stop = threading.Event()
    packet_condition = threading.Condition()
    shared: Dict[str, object] = {
        'sequence': -1,
        'preview': None,
        'performance': '',
        'done': False,
        'error': None,
    }
    native_window_name = "Webcam (GUI native)"

    def _processing_worker() -> None:
        global _stop_stream, _virt_cam, _benchmark_completed_generation, _zero_face_streak
        last_sequence = -1
        completed_sequence = 0
        dropped_frame_count = 0
        frame_count = 0
        completed_at_previous = 0.0
        last_preview_published_at = 0.0
        fps_history: List[float] = []
        latency_history: List[float] = []
        active_benchmark_generation: Optional[int] = None
        benchmark_started_at = 0.0
        benchmark_processed_start = 0
        benchmark_dropped_start = 0
        benchmark_latencies: List[float] = []
        benchmark_face_frames = 0
        benchmark_virtual_sent_start = 0
        virtual_camera_frames = 0
        virtual_camera_started_at = 0.0
        virtual_camera_error: Optional[str] = None
        native_window_open = False

        try:
            while (
                not worker_stop.is_set()
                and not _stop_stream
                and stream_generation == _stream_generation
            ):
                # DirectShow may need a few seconds to deliver the very first
                # frame even though atomic mode negotiation has already
                # opened the device. Reopening after a short first-frame
                # timeout causes the exact start/stop/start flicker we avoid.
                read_timeout = 6.0 if last_sequence < 0 else 2.0
                ok, frame, sequence, dropped, captured_at = frame_reader.read_after(
                    last_sequence,
                    timeout=read_timeout,
                )
                if not ok or frame is None or not getattr(frame, 'size', 0):
                    if worker_stop.is_set() or _stop_stream or stream_generation != _stream_generation:
                        break
                    raise RuntimeError("Camera stopped delivering frames.")
                last_sequence = sequence
                dropped_frame_count += dropped
                frame_count += 1
                _publish_latest_raw_frame(frame, captured_at)

                requested_width = int(settings.get('width') or 0)
                requested_height = int(settings.get('height') or 0)
                frame_height, frame_width = frame.shape[:2]
                if (
                    (requested_width and frame_width != requested_width)
                    or (requested_height and frame_height != requested_height)
                ):
                    raise RuntimeError(
                        f"Camera changed to {frame_width}x{frame_height}; "
                        f"expected {requested_width}x{requested_height}."
                    )

                # Camera-health sampling is intentionally sparse and tiny.
                # The previous full 1080p row-mean check ran every frame.
                if frame_count % 60 == 0:
                    sample = frame[::16, ::16]
                    if sample.size and float(sample.std()) < 1.0:
                        LOGGER.warning("[cam] Live frame appears nearly uniform or black")

                loop_started_at = time.perf_counter()
                live_show_overlay = _live_bool('show_overlay', show_overlay)
                live_debug_logs = _live_bool('debug_logs', debug_logs)
                live_show_boxes = _live_bool('show_boxes', show_boxes)
                live_show_native = _live_bool('show_native', show_native_window)
                live_virtual_cam = _live_bool('virtual_cam_enabled', virtual_cam_enabled)
                live_auto_fallback = _live_bool('auto_fallback', auto_fallback)
                live_color_mode = _live_text('color_mode', color_mode)
                live_passthrough = _live_bool('realtime_passthrough', False)
                if live_passthrough:
                    frame_out = frame
                else:
                    with _live_processing_lock:
                        processed = process_frame(frame, True, live_show_overlay, live_debug_logs, live_show_boxes)
                    frame_out = processed if isinstance(processed, np.ndarray) and processed.size else frame
                completed_at = time.perf_counter()
                processing_latency_ms = (completed_at - loop_started_at) * 1000.0
                end_to_end_latency_ms = (
                    (completed_at - captured_at) * 1000.0 if captured_at else processing_latency_ms
                )

                if completed_at_previous:
                    fps_history.append(1.0 / max(completed_at - completed_at_previous, 1e-6))
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                completed_at_previous = completed_at
                latency_history.append(processing_latency_ms)
                if len(latency_history) > 30:
                    latency_history.pop(0)
                processed_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
                mean_latency_ms = sum(latency_history) / len(latency_history)
                total_camera_frames = frame_count + dropped_frame_count
                drop_percent = (
                    100.0 * dropped_frame_count / total_camera_frames if total_camera_frames else 0.0
                )
                camera_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(target_fps or 0)

                with _benchmark_lock:
                    requested_benchmark_generation = _benchmark_generation
                if (
                    active_benchmark_generation is None
                    and requested_benchmark_generation > _benchmark_completed_generation
                ):
                    active_benchmark_generation = requested_benchmark_generation
                    benchmark_started_at = completed_at
                    benchmark_processed_start = frame_count
                    benchmark_dropped_start = dropped_frame_count
                    benchmark_latencies = []
                    benchmark_face_frames = 0
                    benchmark_virtual_sent_start = virtual_camera_frames

                benchmark_message = ''
                if active_benchmark_generation is not None:
                    benchmark_latencies.append(processing_latency_ms)
                    if _last_faces_count > 0:
                        benchmark_face_frames += 1
                    benchmark_elapsed = completed_at - benchmark_started_at
                    if benchmark_elapsed < 10.0:
                        benchmark_message = f" | benchmark {10.0 - benchmark_elapsed:.1f}s remaining"
                    else:
                        benchmark_processed = frame_count - benchmark_processed_start
                        benchmark_dropped = dropped_frame_count - benchmark_dropped_start
                        benchmark_fps = benchmark_processed / benchmark_elapsed if benchmark_elapsed else 0.0
                        sorted_latencies = sorted(benchmark_latencies)
                        p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)
                        benchmark_result = {
                            'camera': str(
                                settings.get('dshow_name')
                                or f"Camera {settings.get('cam_index')}"
                            ),
                            'resolution': f"{frame_out.shape[1]}x{frame_out.shape[0]}",
                            'camera_fps': round(camera_fps, 3),
                            'duration_seconds': round(benchmark_elapsed, 3),
                            'processed_frames': benchmark_processed,
                            'dropped_camera_frames': benchmark_dropped,
                            'processed_fps': round(benchmark_fps, 3),
                            'mean_processing_latency_ms': round(
                                sum(benchmark_latencies) / len(benchmark_latencies), 2
                            ),
                            'p95_processing_latency_ms': round(sorted_latencies[p95_index], 2),
                            'frames_with_face': benchmark_face_frames,
                            'valid_face_benchmark': benchmark_face_frames > 0,
                            'virtual_camera_enabled': bool(virtual_cam_enabled),
                            'virtual_camera_frames_sent': virtual_camera_frames - benchmark_virtual_sent_start,
                            'virtual_camera_backend': getattr(_virt_cam, 'backend', None) if _virt_cam else None,
                            'virtual_camera_error': virtual_camera_error,
                        }
                        try:
                            benchmark_dir = os.path.join(BASE_DIR, 'benchmarks')
                            os.makedirs(benchmark_dir, exist_ok=True)
                            with open(
                                os.path.join(benchmark_dir, 'realtime_benchmark_latest.json'),
                                'w',
                                encoding='utf-8',
                            ) as benchmark_file:
                                json.dump(benchmark_result, benchmark_file, indent=2)
                        except Exception as exc:
                            LOGGER.warning(f"[benchmark] Could not save result: {exc}")
                        benchmark_message = (
                            f" | benchmark complete: {benchmark_fps:.1f} FPS"
                            if benchmark_face_frames > 0
                            else " | benchmark invalid: no face detected"
                        )
                        with _benchmark_lock:
                            _benchmark_completed_generation = active_benchmark_generation
                        active_benchmark_generation = None

                if live_show_overlay:
                    try:
                        draw_info(
                            frame_out,
                            f"Process: {processed_fps:.1f} FPS | {mean_latency_ms:.0f}ms | Drop: {drop_percent:.0f}%",
                            (10, frame_out.shape[0] - 20),
                        )
                    except Exception:
                        pass

                try:
                    if live_show_native:
                        cv2.imshow(native_window_name, frame_out)
                        native_window_open = True
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            _stop_stream = True
                    elif native_window_open:
                        cv2.destroyWindow(native_window_name)
                        native_window_open = False
                except Exception:
                    pass

                try:
                    if live_virtual_cam and HAS_PYVIRTUALCAM:
                        height, width = frame_out.shape[:2]
                        configure_virtual_camera(True, width, height, target_fps)
                        if publish_virtual_camera_frame(
                            frame_out,
                            input_is_rgb=live_color_mode.startswith('Assume RGB'),
                        ):
                            virtual_camera_frames += 1
                        if virtual_camera_started_at == 0.0:
                            virtual_camera_started_at = time.perf_counter()
                        virtual_camera_error = None
                    elif live_virtual_cam and not HAS_PYVIRTUALCAM:
                        virtual_camera_error = 'pyvirtualcam is not installed'
                    elif virtual_camera_started_at or _virt_cam is not None:
                        close_virtual_camera()
                        virtual_camera_started_at = 0.0
                        virtual_camera_frames = 0
                        virtual_camera_error = None
                except Exception as exc:
                    error_text = str(exc) or type(exc).__name__
                    if virtual_camera_error != error_text:
                        LOGGER.warning(f"[virtual-cam] {error_text}")
                    virtual_camera_error = error_text

                # Preparing/encoding every full-resolution result for Gradio
                # used to throttle the swap loop to the browser's ~4 FPS. The
                # web preview is intentionally smaller and sampled at 5 FPS;
                # native and virtual-camera outputs above remain full size.
                if completed_at - last_preview_published_at >= 0.2:
                    preview_source = frame_out
                    preview_height, preview_width = preview_source.shape[:2]
                    if preview_width > 640:
                        scaled_height = max(1, int(preview_height * 640 / preview_width))
                        preview_source = cv2.resize(
                            preview_source,
                            (640, scaled_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    preview_frame = (
                        preview_source.copy()
                        if live_color_mode.startswith('Assume RGB')
                        else cv2.cvtColor(preview_source, cv2.COLOR_BGR2RGB)
                    )
                    performance_text = (
                        f"**Live performance:** {processed_fps:.1f} processed FPS | "
                        f"{mean_latency_ms:.0f} ms processing | {end_to_end_latency_ms:.0f} ms end-to-end | "
                        f"{drop_percent:.0f}% camera frames skipped{benchmark_message}"
                    )
                    if live_virtual_cam:
                        if virtual_camera_error:
                            performance_text += f" | virtual cam unavailable: {virtual_camera_error[:100]}"
                        elif virtual_camera_frames > 0 and virtual_camera_started_at:
                            virtual_elapsed = max(time.perf_counter() - virtual_camera_started_at, 1e-6)
                            virtual_fps = virtual_camera_frames / virtual_elapsed
                            virtual_backend = getattr(_virt_cam, 'backend', 'unknown') if _virt_cam else 'unknown'
                            performance_text += (
                                f" | virtual cam {frame_out.shape[1]}x{frame_out.shape[0]} "
                                f"at {virtual_fps:.1f} FPS via {virtual_backend}"
                            )
                    completed_sequence += 1
                    last_preview_published_at = completed_at
                    with packet_condition:
                        shared['sequence'] = completed_sequence
                        shared['preview'] = preview_frame
                        shared['performance'] = performance_text
                        packet_condition.notify_all()

                if live_auto_fallback:
                    if _last_faces_count == 0:
                        _zero_face_streak += 1
                    else:
                        _zero_face_streak = 0
        except Exception as exc:
            LOGGER.exception("Decoupled live processing stopped")
            with packet_condition:
                shared['error'] = exc
                packet_condition.notify_all()
        finally:
            frame_reader.stop()
            if native_window_open:
                try:
                    cv2.destroyWindow(native_window_name)
                except Exception:
                    pass
            with packet_condition:
                shared['done'] = True
                packet_condition.notify_all()

    worker = threading.Thread(
        target=_processing_worker,
        name='facefusion-live-processor',
        daemon=True,
    )
    worker.start()
    delivered_sequence = -1
    preview_fps_history: List[float] = []
    previous_delivery_at = 0.0
    try:
        while worker.is_alive() or int(shared['sequence']) > delivered_sequence:
            with packet_condition:
                packet_condition.wait_for(
                    lambda: (
                        int(shared['sequence']) > delivered_sequence
                        or bool(shared['done'])
                        or shared['error'] is not None
                    ),
                    timeout=2.0,
                )
                sequence = int(shared['sequence'])
                preview_frame = shared['preview']
                performance_text = str(shared['performance'])
                worker_error = shared['error']
            if worker_error is not None:
                raise gr.Error(f"Live processing stopped: {worker_error}")
            if sequence <= delivered_sequence or not isinstance(preview_frame, np.ndarray):
                if bool(shared['done']):
                    break
                continue
            delivered_sequence = sequence
            delivered_at = time.perf_counter()
            if previous_delivery_at:
                preview_fps_history.append(1.0 / max(delivered_at - previous_delivery_at, 1e-6))
                if len(preview_fps_history) > 10:
                    preview_fps_history.pop(0)
            previous_delivery_at = delivered_at
            preview_fps = (
                sum(preview_fps_history) / len(preview_fps_history) if preview_fps_history else 0.0
            )
            yield preview_frame, f"{performance_text} | UI preview {preview_fps:.1f} FPS"
    finally:
        worker_stop.set()
        with packet_condition:
            packet_condition.notify_all()
        worker.join(timeout=3.0)


def request_realtime_benchmark() -> str:
    global _benchmark_generation
    with _benchmark_lock:
        _benchmark_generation += 1
    return "**10-second benchmark armed.** Keep your face visible and look toward the camera."


def _capture_backend_candidates(settings: Dict[str, object]) -> List[Tuple[int, str]]:
    backend_name = str(settings.get("backend") or "Auto")
    if backend_name == "DirectShow":
        return [(cv2.CAP_DSHOW, "DirectShow")]
    if backend_name == "Media Foundation":
        return [(cv2.CAP_MSMF, "Media Foundation")]
    if os.name == "nt":
        # Camera labels are enumerated through DirectShow, so DirectShow indices
        # are the only ones guaranteed to refer to the displayed device.
        return [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "OpenCV Auto")]
    return [(cv2.CAP_ANY, "OpenCV Auto")]


def _open_configured_capture(
    settings: Dict[str, object],
    gentle_mode: bool = True,
) -> Tuple[Optional[cv2.VideoCapture], Dict[str, object], List[str]]:
    camera_index = int(settings["cam_index"])
    width_requested = int(settings.get("width") or 0)
    height_requested = int(settings.get("height") or 0)
    fps_requested = float(settings.get("fps") or 0)
    requested_fourcc = str(settings.get("fourcc") or "Auto")
    convert_rgb = bool(settings.get("convert_rgb", True))

    if requested_fourcc != "Auto":
        fourcc_candidates = [requested_fourcc]
    elif width_requested >= 1280 or height_requested >= 720:
        fourcc_candidates = ["MJPG", "YUY2", "H264", "NV12", "Auto"]
    else:
        fourcc_candidates = ["Auto", "MJPG", "YUY2"]

    fps_candidates: List[float] = [fps_requested]
    for fallback_fps in (30.0, 15.0):
        if fps_requested > fallback_fps and fallback_fps not in fps_candidates:
            fps_candidates.append(fallback_fps)

    attempts: List[str] = []
    for backend_id, backend_label in _capture_backend_candidates(settings):
        for fourcc_name in fourcc_candidates:
            for fps_value in fps_candidates:
                LOGGER.info(
                    f"[cam] Trying camera {camera_index} with {backend_label}, "
                    f"FOURCC={fourcc_name}, {width_requested}x{height_requested}@{fps_value:g}"
                )
                # DirectShow briefly starts its default mode and renegotiates
                # every property assigned after open(). On some cameras that
                # looks like an open/close/open cycle and costs several
                # seconds. Passing the complete mode into the constructor lets
                # the driver negotiate once before it begins streaming.
                cap = None
                if gentle_mode:
                    open_params: List[int] = []
                    if fourcc_name != "Auto":
                        open_params.extend(
                            [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_name)]
                        )
                    if width_requested:
                        open_params.extend([cv2.CAP_PROP_FRAME_WIDTH, width_requested])
                    if height_requested:
                        open_params.extend([cv2.CAP_PROP_FRAME_HEIGHT, height_requested])
                    if fps_value:
                        open_params.extend([cv2.CAP_PROP_FPS, int(round(fps_value))])
                    open_params.extend([cv2.CAP_PROP_CONVERT_RGB, int(convert_rgb)])
                    try:
                        cap = cv2.VideoCapture(camera_index, backend_id, open_params)
                    except (TypeError, cv2.error):
                        cap = None

                    if cap is not None and cap.isOpened():
                        property_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        property_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(fps_value)
                        actual_fourcc = _fourcc_string(cap)
                        property_resolution_matches = (
                            (not width_requested or property_width == width_requested)
                            and (not height_requested or property_height == height_requested)
                        )
                        if property_resolution_matches:
                            result = (
                                f"{backend_label}/{fourcc_name}: "
                                f"{property_width}x{property_height}@{actual_fps:.2f} "
                                f"FOURCC={actual_fourcc}"
                            )
                            attempts.append(result)
                            effective = {
                                "backend": backend_label,
                                "fourcc": actual_fourcc,
                                "width": property_width,
                                "height": property_height,
                                "fps": actual_fps,
                            }
                            LOGGER.info(
                                f"[cam] Selected {result} "
                                "(atomic mode negotiation; first frame warming asynchronously)"
                            )
                            return cap, effective, attempts

                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(camera_index, backend_id)
                if not cap.isOpened():
                    attempts.append(f"{backend_label}/{fourcc_name}: open failed")
                    cap.release()
                    break

                # Windows webcam drivers usually negotiate correctly only when
                # format is selected before dimensions and frame rate.
                _set_convert_rgb(cap, convert_rgb)
                if fourcc_name != "Auto":
                    _apply_fourcc(cap, fourcc_name)
                if width_requested:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_requested)
                if height_requested:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_requested)
                if fps_value:
                    cap.set(cv2.CAP_PROP_FPS, fps_value)

                # In gentle mode, accept an exact driver negotiation without a
                # blocking warm-up read. DirectShow can spend many seconds in
                # the first read while toggling its indicator; the persistent
                # LatestFrameReader below is the right place to wait for that
                # first frame. Any real frame-size mismatch is still detected
                # and repaired by the stream loop.
                property_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                property_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(fps_value)
                actual_fourcc = _fourcc_string(cap)
                property_resolution_matches = (
                    (not width_requested or property_width == width_requested)
                    and (not height_requested or property_height == height_requested)
                )
                if gentle_mode and property_resolution_matches:
                    result = (
                        f"{backend_label}/{fourcc_name}: "
                        f"{property_width}x{property_height}@{actual_fps:.2f} FOURCC={actual_fourcc}"
                    )
                    attempts.append(result)
                    effective = {
                        "backend": backend_label,
                        "fourcc": actual_fourcc,
                        "width": property_width,
                        "height": property_height,
                        "fps": actual_fps,
                    }
                    LOGGER.info(f"[cam] Selected {result} (first frame warming asynchronously)")
                    return cap, effective, attempts

                frame = None
                ok = False
                # One successful frame is enough to validate the negotiated
                # mode. Extra synchronous warm-up reads made camera startup
                # look like an open/close/open cycle and added several seconds
                # on slow DirectShow devices; the background reader handles
                # subsequent warm-up frames without releasing the device.
                read_count = 1 if gentle_mode else 3
                for _ in range(read_count):
                    try:
                        ok, frame = cap.read()
                    except Exception:
                        ok, frame = False, None
                    if ok and isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size:
                        break

                property_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                property_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
                actual_width = int(frame.shape[1]) if ok and isinstance(frame, np.ndarray) and frame.ndim >= 2 else property_width
                actual_height = int(frame.shape[0]) if ok and isinstance(frame, np.ndarray) and frame.ndim >= 2 else property_height
                actual_fourcc = _fourcc_string(cap)
                result = (
                    f"{backend_label}/{fourcc_name}: "
                    f"{actual_width}x{actual_height}@{actual_fps:.2f} FOURCC={actual_fourcc}"
                )
                attempts.append(result)

                valid_frame = bool(ok and isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size)
                exact_resolution = (
                    (not width_requested or actual_width == width_requested)
                    and (not height_requested or actual_height == height_requested)
                )
                if valid_frame and exact_resolution:
                    effective = {
                        "backend": backend_label,
                        "fourcc": actual_fourcc,
                        "width": actual_width,
                        "height": actual_height,
                        "fps": actual_fps,
                    }
                    LOGGER.info(f"[cam] Selected {result}")
                    return cap, effective, attempts

                if valid_frame:
                    LOGGER.warning(
                        f"[cam] Rejected {actual_width}x{actual_height}; "
                        f"the requested resolution is {width_requested}x{height_requested}"
                    )
                else:
                    LOGGER.warning(f"[cam] Rejected invalid frame from {backend_label}/{fourcc_name}")
                cap.release()

    return None, {}, attempts


def is_camera_capture_active() -> bool:
    """Return whether this process currently owns an open webcam handle."""
    with _capture_lock:
        try:
            return bool(_cap is not None and _cap.isOpened())
        except Exception:
            return False


def release_camera_capture() -> bool:
    """Release the webcam handle and forget its negotiated settings."""
    global _cap, _cap_settings
    with _capture_lock:
        cap = _cap
        _cap = None
        _cap_settings = {}
        if cap is None:
            return False
        try:
            cap.release()
        except Exception:
            pass
        return True


def _ensure_capture(settings: Dict[str, object], gentle_mode: bool = True, force_reopen: bool = False,
                    lock_exposure: bool = False, exposure_value: float = -6.0,
                    lock_wb: bool = False, wb_temperature: int = 4500) -> cv2.VideoCapture:
    global _cap, _cap_settings
    with _capture_lock:
        # Reuse existing if matches and is opened
        if not force_reopen and _cap is not None and _cap.isOpened() and _settings_equal(settings, _cap_settings):
            return _cap
        # Close previous
        if _cap is not None:
            try:
                _cap.release()
            except Exception:
                pass
            _cap = None
        cap, effective, attempts = _open_configured_capture(settings, gentle_mode=gentle_mode)
        if cap is None:
            requested = f"{settings.get('width')}x{settings.get('height')}@{settings.get('fps')}"
            detail = "; ".join(attempts[-8:]) if attempts else "no capture backend opened"
            raise gr.Error(
                f"Camera {settings['cam_index']} could not provide {requested}. "
                f"Tried: {detail}"
            )

        _apply_camera_controls(cap, lock_exposure, exposure_value, lock_wb, wb_temperature)

        LOGGER.info(
            f"[cam] Active capture: {effective.get('width')}x{effective.get('height')}"
            f"@{float(effective.get('fps') or 0):.2f} via {effective.get('backend')}"
        )
        _cap = cap
        _cap_settings = dict(settings)
        return cap


def gr_stream(
    camera_choice: str,
    width: int,
    height: int,
    use_occlusion: bool,
    target_fps: float,
    backend_name: str,
    dshow_name: str,
    convert_rgb: bool,
    force_fourcc: str,
    retry_black: int,
    gentle_mode: bool,
    auto_repair: bool,
    color_mode: str,
    lock_exposure: bool,
    exposure_value: float,
    lock_wb: bool,
    wb_temperature: int,
    show_overlay: bool,
    debug_logs: bool,
    show_boxes: bool,
    show_native_window: bool,
    virtual_cam_enabled: bool,
    colorizer_enabled: bool,
    expr_enabled: bool,
    age_enabled: bool,
    editor_enabled: bool,
    debugger_enabled: bool,
    lip_enabled: bool,
    selector_mode: str,
    auto_fallback: bool,
    exec_provider_key: str,
    exec_device_id: str,
    video_memory_strategy: str,
    detector_model: str,
    detector_size: str,
    detector_score: float,
    landmarker_model: str,
    landmarker_score: float,
    occluder_model: str,
    parser_model: str,
    swap_mode: str,
    source_files,
    deep_model: str,
    morph: int,
    face_swapper_model: str,
    face_swapper_pixel_boost: str,
    face_swapper_weight: float,
    face_enhancer_enabled: bool,
    face_enhancer_model: str,
    face_enhancer_blend: int,
    face_enhancer_weight: float,
    frame_enhancer_enabled: bool,
    frame_enhancer_model: str,
    frame_enhancer_blend: int,
    enhance_async: bool,
):
    try:
        global _stop_stream, _stream_generation, _benchmark_completed_generation
        with _stream_lock:
            _stream_generation += 1
            stream_generation = _stream_generation
            _stop_stream = False
        reset_temporal_occlusion_cache()
        reset_temporal_face_tracker()
        if not detector_size:
            raise gr.Error("Detector size is empty; pick a valid size for the selected detector.")
        # Parse camera index from choice like "[0] Name"
        if not camera_choice.startswith("[") or "]" not in camera_choice:
            raise gr.Error(f"Invalid camera selection: {camera_choice}")
        cam_index = _parse_cam_index(camera_choice)
        if not camera_choice.startswith("["):
            raise gr.Error(f"Invalid camera selection: {camera_choice}")

        # Ensure download & execution settings exist (GUI path) before any model prechecks/downloads
        try:
            if state_manager.get_item("download_providers") is None:
                state_manager.set_item("download_providers", list(ff_choices.download_providers))
            if state_manager.get_item("download_scope") is None:
                state_manager.set_item("download_scope", "full")
            if state_manager.get_item("log_level") is None:
                state_manager.set_item("log_level", "info")
            # Apply chosen execution provider/device
            ep_key = (exec_provider_key or 'cpu')
            if ep_key not in ff_choices.execution_providers:
                ep_key = 'cpu'
            dev_ids = [str(exec_device_id or '0')]
            state_manager.set_item("execution_device_ids", dev_ids)
            state_manager.set_item("execution_providers", [ep_key])
            # Apply video memory strategy
            if video_memory_strategy in ("strict","moderate","relaxed"):
                state_manager.set_item("video_memory_strategy", video_memory_strategy)
            # Swap enablement is derived from swap_mode; no explicit deep_enabled state
            # Initialize enabled processors on stream start
            if state_manager.get_item('frame_colorizer_enabled') and not state_manager.get_item('fast_startup'):
                try: ff_frame_colorizer.pre_check()
                except Exception: pass
            if state_manager.get_item('expression_restorer_enabled') and not state_manager.get_item('fast_startup'):
                try: ff_expr_restorer.pre_check()
                except Exception: pass
            if state_manager.get_item('age_modifier_enabled') and not state_manager.get_item('fast_startup'):
                try: ff_age_modifier.pre_check()
                except Exception: pass
            if state_manager.get_item('face_editor_enabled') and not state_manager.get_item('fast_startup'):
                try: ff_face_editor.pre_check()
                except Exception: pass
        except Exception:
            pass

        apply_state_from_ui(
            detector_model,
            detector_size,
            detector_score,
            landmarker_model,
            landmarker_score,
            occluder_model,
            parser_model,
            deep_model,
            morph,
            use_occlusion,
            selector_mode,
        )
        try:
            _persist_state()
        except Exception:
            pass
        # Swap mode & face swapper setup
        try:
            state_manager.set_item("swap_mode", swap_mode or 'deep')
            state_manager.set_item('face_swapper_model', face_swapper_model)
            state_manager.set_item('face_swapper_pixel_boost', face_swapper_pixel_boost)
            state_manager.set_item('face_swapper_weight', float(face_swapper_weight))
        except Exception:
            pass
        # Enhancers setup
        try:
            state_manager.set_item('face_enhancer_enabled', bool(face_enhancer_enabled))
            state_manager.set_item('face_enhancer_model', face_enhancer_model)
            state_manager.set_item('face_enhancer_blend', int(face_enhancer_blend))
            state_manager.set_item('face_enhancer_weight', float(face_enhancer_weight))
            state_manager.set_item('frame_enhancer_enabled', bool(frame_enhancer_enabled))
            state_manager.set_item('frame_enhancer_model', frame_enhancer_model)
            state_manager.set_item('frame_enhancer_blend', int(frame_enhancer_blend))
            state_manager.set_item('enhance_async', bool(enhance_async))
            # Other processors (toggles)
            state_manager.set_item('frame_colorizer_enabled', bool(colorizer_enabled))
            state_manager.set_item('expression_restorer_enabled', bool(expr_enabled))
            state_manager.set_item('age_modifier_enabled', bool(age_enabled))
            state_manager.set_item('face_editor_enabled', bool(editor_enabled))
            state_manager.set_item('face_debugger_enabled', bool(debugger_enabled))
            state_manager.set_item('lip_syncer_enabled', bool(lip_enabled))
        except Exception:
            pass

        # Source files ingestion
        try:
            if source_files:
                paths = [f.name if hasattr(f, 'name') else f for f in source_files]
                state_manager.set_item('source_paths', paths)
                global _source_imgs
                _source_imgs = []
                for p in paths:
                    try:
                        img = cv2.imread(p)
                        if isinstance(img, np.ndarray) and getattr(img, 'size', 0) > 0:
                            _source_imgs.append(img)
                    except Exception:
                        pass
        except Exception:
            pass

        # Verify model files, but preserve already-created ONNX sessions. Model
        # and provider changes invalidate the bounded cache in web_api; clearing
        # here on every camera start made demand-start pay the full load cost.
        try:
            active_swap_mode = swap_mode or 'deep'
            if active_swap_mode == 'deep':
                if debug_logs:
                    LOGGER.info(f"[model] preparing deep swapper '{deep_model}' (scope={state_manager.get_item('download_scope')})")
                deep_swapper.pre_check()
            elif active_swap_mode == 'face':
                try:
                    LOGGER.info(f"[model] preparing face swapper '{state_manager.get_item('face_swapper_model')}'")
                    ff_face_swapper.pre_check()
                except Exception:
                    pass
            if state_manager.get_item('face_enhancer_enabled'):
                try:
                    LOGGER.info(f"[model] preparing face enhancer '{state_manager.get_item('face_enhancer_model')}'")
                    ff_face_enhancer.pre_check()
                except Exception:
                    pass
            if state_manager.get_item('frame_enhancer_enabled'):
                try:
                    LOGGER.info(f"[model] preparing frame enhancer '{state_manager.get_item('frame_enhancer_model')}'")
                    ff_frame_enhancer.pre_check()
                except Exception:
                    pass
        except Exception:
            pass

        settings = _settings_from_inputs(camera_choice, backend_name, dshow_name, width, height, target_fps, convert_rgb, force_fourcc)
        # A stop request can arrive while models are loading. Do not open the
        # camera after the user has already switched capture off.
        if _stop_stream or stream_generation != _stream_generation:
            return
        cap = _ensure_capture(settings, gentle_mode=gentle_mode, force_reopen=False,
                              lock_exposure=lock_exposure, exposure_value=exposure_value,
                              lock_wb=lock_wb, wb_temperature=wb_temperature)
        if (
            bool(state_manager.get_item('realtime_fast_analysis'))
            and not state_manager.get_item('face_enhancer_enabled')
            and not state_manager.get_item('frame_enhancer_enabled')
        ):
            yield from _run_decoupled_fast_stream(
                cap=cap,
                stream_generation=stream_generation,
                settings=settings,
                target_fps=target_fps,
                color_mode=color_mode,
                show_overlay=show_overlay,
                debug_logs=debug_logs,
                show_boxes=show_boxes,
                show_native_window=show_native_window,
                virtual_cam_enabled=virtual_cam_enabled,
                auto_fallback=auto_fallback,
            )
            return
        frame_reader = LatestFrameReader(cap)
        last_sequence = -1
        dropped_frame_count = 0
        switched_backend_once = False
        if debug_logs:
            LOGGER.info(f"[start] backend={backend_name} cam={settings['cam_index']} fourcc={settings['fourcc']} convert_rgb={settings['convert_rgb']} res=({settings['width']},{settings['height']}) fps={settings['fps']} color_mode={color_mode} gentle={gentle_mode} auto_repair={auto_repair} lock_exp={lock_exposure} lock_wb={lock_wb}")
            try:
                LOGGER.info(f"[exec] providers={state_manager.get_item('execution_providers')} device_ids={state_manager.get_item('execution_device_ids')} detector={detector_model} size={state_manager.get_item('face_detector_size')} score={detector_score}")
            except Exception:
                pass
        frame_count = 0
        # Async enhancement pipeline state
        executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        pending: Optional[Future] = None
        latest_enhanced: Optional[np.ndarray] = None
        native_window_name = "Webcam (GUI native)"
        native_window_open = False

        # Performance Monitoring
        fps_history: List[float] = []
        latency_history: List[float] = []
        fps_len = 30
        last_yield_time = time.perf_counter()
        stream_started_at = last_yield_time
        active_benchmark_generation: Optional[int] = None
        benchmark_started_at = 0.0
        benchmark_processed_start = 0
        benchmark_dropped_start = 0
        benchmark_latencies: List[float] = []
        benchmark_face_frames = 0

        def _run_enhancers(img: np.ndarray) -> np.ndarray:
            try:
                out = img
                if state_manager.get_item('face_enhancer_enabled'):
                    try:
                        ff_face_enhancer.pre_check()
                    except Exception:
                        pass
                    out = ff_face_enhancer.process_frame({
                        'reference_vision_frame': out,
                        'target_vision_frame': out,
                        'temp_vision_frame': out,
                    })
                if state_manager.get_item('frame_enhancer_enabled'):
                    try:
                        ff_frame_enhancer.pre_check()
                    except Exception:
                        pass
                    out = ff_frame_enhancer.process_frame({
                        'temp_vision_frame': out,
                    })
                return out
            except Exception:
                return img
        while not _stop_stream and stream_generation == _stream_generation:
            read_timeout = 6.0 if last_sequence < 0 else 2.0
            ok, frame, sequence, dropped, captured_at = frame_reader.read_after(
                last_sequence,
                timeout=read_timeout,
            )
            if not ok or frame is None or getattr(frame, 'size', 0) == 0:
                frame_reader.stop()
                if _stop_stream or stream_generation != _stream_generation:
                    break
                cap = _ensure_capture(settings, gentle_mode=gentle_mode, force_reopen=True)
                frame_reader = LatestFrameReader(cap)
                last_sequence = -1
                ok, frame, sequence, dropped, captured_at = frame_reader.read_after(last_sequence, timeout=3.0)
                if not ok or frame is None or getattr(frame, 'size', 0) == 0:
                    raise gr.Error("Camera stopped delivering frames and could not be reconnected.")
            last_sequence = sequence
            dropped_frame_count += dropped
            _publish_latest_raw_frame(frame, captured_at)
            loop_start = time.perf_counter()
            frame_count += 1
            if debug_logs and (frame_count % 15 == 0):
                try:
                    LOGGER.info(f"[captured] shape={frame.shape} mean={frame.mean():.2f}")
                except Exception:
                    pass
            # Ignore transient black frames without mutating a live capture
            # handle from two threads. The reader always advances to the newest
            # available frame.
            if retry_black > 0 and not gentle_mode and frame.mean() < 1.0:
                for _ in range(int(retry_black)):
                    if _stop_stream or stream_generation != _stream_generation:
                        break
                    ok, candidate, sequence, dropped, candidate_at = frame_reader.read_after(last_sequence, timeout=1.0)
                    if not ok or candidate is None:
                        continue
                    last_sequence = sequence
                    dropped_frame_count += dropped
                    if candidate.mean() >= 1.0:
                        frame = candidate
                        captured_at = candidate_at
                        break
                if frame is None or frame.mean() < 1.0:
                    raise gr.Error("Camera continues to return black frames after retrying.")

            # Auto-repair for corrupted frames (horizontal bands, wrong stride)
            try:
                if auto_repair and frame is not None and getattr(frame, 'size', 0) > 0:
                    h, w = frame.shape[:2]
                    # Heuristics: very low vertical variance or too few unique rows indicates corruption
                    row_means = frame.mean(axis=1)
                    vstd = float(row_means.std()) if row_means.size > 0 else 0.0
                    if (vstd < 1.0 or h < 40) and not switched_backend_once:
                        frame_reader.stop()
                        if _stop_stream or stream_generation != _stream_generation:
                            break
                        cap = _ensure_capture(settings, gentle_mode=gentle_mode, force_reopen=True)
                        frame_reader = LatestFrameReader(cap)
                        last_sequence = -1
                        switched_backend_once = True
                        ok, frame, sequence, dropped, captured_at = frame_reader.read_after(last_sequence, timeout=3.0)
                        if ok and frame is not None:
                            last_sequence = sequence
                            dropped_frame_count += dropped
            except Exception:
                pass

            # Codec changes made by a driver or repair attempt can silently
            # renegotiate to VGA. Never feed a lower-resolution frame into the
            # processing or virtual-camera pipeline when HD was requested.
            requested_width = int(settings.get("width") or 0)
            requested_height = int(settings.get("height") or 0)
            frame_height, frame_width = frame.shape[:2]
            if (
                (requested_width and frame_width != requested_width)
                or (requested_height and frame_height != requested_height)
            ):
                LOGGER.warning(
                    f"[cam] Stream changed to {frame_width}x{frame_height}; "
                    f"reopening at {requested_width}x{requested_height}"
                )
                frame_reader.stop()
                if _stop_stream or stream_generation != _stream_generation:
                    break
                cap = _ensure_capture(settings, gentle_mode=gentle_mode, force_reopen=True)
                frame_reader = LatestFrameReader(cap)
                last_sequence = -1
                ok, frame, sequence, dropped, captured_at = frame_reader.read_after(last_sequence, timeout=3.0)
                if ok:
                    last_sequence = sequence
                    dropped_frame_count += dropped
                restored_height, restored_width = frame.shape[:2] if ok and frame is not None else (0, 0)
                resolution_restored = (
                    (not requested_width or restored_width == requested_width)
                    and (not requested_height or restored_height == requested_height)
                )
                if not ok or frame is None or not resolution_restored:
                    raise gr.Error(
                        f"Camera resolution changed unexpectedly and could not be restored to "
                        f"{requested_width}x{requested_height}."
                    )

            # PROCESS
            live_show_overlay = _live_bool('show_overlay', show_overlay)
            live_debug_logs = _live_bool('debug_logs', debug_logs)
            live_show_boxes = _live_bool('show_boxes', show_boxes)
            live_show_native = _live_bool('show_native', show_native_window)
            live_virtual_cam = _live_bool('virtual_cam_enabled', virtual_cam_enabled)
            live_auto_fallback = _live_bool('auto_fallback', auto_fallback)
            live_enhance_async = _live_bool('enhance_async', enhance_async)
            live_color_mode = _live_text('color_mode', color_mode)
            live_passthrough = _live_bool('realtime_passthrough', False)
            if live_passthrough:
                processed = frame
            else:
                with _live_processing_lock:
                    processed = process_frame(frame, True, live_show_overlay, live_debug_logs, live_show_boxes)

            # Async Enhancement Logic
            frame_out = processed if (isinstance(processed, np.ndarray) and getattr(processed, 'size', 0) > 0) else frame
            if live_enhance_async and (state_manager.get_item('face_enhancer_enabled') or state_manager.get_item('frame_enhancer_enabled')):
                if latest_enhanced is None:
                     latest_enhanced = frame_out
                # Check previous job
                if pending is not None and pending.done():
                    try:
                        res = pending.result()
                        if isinstance(res, np.ndarray) and res.shape == frame_out.shape:
                             latest_enhanced = res
                    except Exception:
                        pass
                    pending = None
                # Submit new job if idle (simple 1-depth queue)
                if pending is None:
                     # Copy input to avoid threading mutation issues
                     inp = frame_out.copy()
                     pending = executor.submit(_run_enhancers, inp)
                # Always show latest stable enhanced frame
                frame_out = latest_enhanced
            elif (state_manager.get_item('face_enhancer_enabled') or state_manager.get_item('frame_enhancer_enabled')):
                # Sync enhancement
                frame_out = _run_enhancers(frame_out)

            # Performance Calculations
            now = time.perf_counter()
            dt = now - last_yield_time
            last_yield_time = now
            if dt > 0:
                instantaneous_fps = 1.0 / dt
                fps_history.append(instantaneous_fps)
                if len(fps_history) > fps_len:
                    fps_history.pop(0)

            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
            processing_latency_ms = (now - loop_start) * 1000.0
            end_to_end_latency_ms = (now - captured_at) * 1000.0 if captured_at else processing_latency_ms
            latency_history.append(processing_latency_ms)
            if len(latency_history) > fps_len:
                latency_history.pop(0)
            avg_latency_ms = sum(latency_history) / len(latency_history) if latency_history else 0.0
            total_camera_frames = frame_count + dropped_frame_count
            drop_percent = (100.0 * dropped_frame_count / total_camera_frames) if total_camera_frames else 0.0
            camera_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(target_fps or 0)

            with _benchmark_lock:
                requested_benchmark_generation = _benchmark_generation
            if (
                active_benchmark_generation is None
                and requested_benchmark_generation > _benchmark_completed_generation
            ):
                active_benchmark_generation = requested_benchmark_generation
                benchmark_started_at = now
                benchmark_processed_start = frame_count
                benchmark_dropped_start = dropped_frame_count
                benchmark_latencies = []
                benchmark_face_frames = 0

            benchmark_message = ""
            if active_benchmark_generation is not None:
                benchmark_latencies.append(processing_latency_ms)
                if _last_faces_count > 0:
                    benchmark_face_frames += 1
                benchmark_elapsed = now - benchmark_started_at
                if benchmark_elapsed < 10.0:
                    benchmark_message = f" · benchmark {10.0 - benchmark_elapsed:.1f}s remaining"
                else:
                    benchmark_processed = frame_count - benchmark_processed_start
                    benchmark_dropped = dropped_frame_count - benchmark_dropped_start
                    benchmark_fps = benchmark_processed / benchmark_elapsed if benchmark_elapsed else 0.0
                    sorted_latencies = sorted(benchmark_latencies)
                    p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)
                    benchmark_result = {
                        "camera": _camera_name_from_choice(camera_choice),
                        "resolution": f"{frame_out.shape[1]}x{frame_out.shape[0]}",
                        "camera_fps": round(camera_fps, 3),
                        "duration_seconds": round(benchmark_elapsed, 3),
                        "processed_frames": benchmark_processed,
                        "dropped_camera_frames": benchmark_dropped,
                        "processed_fps": round(benchmark_fps, 3),
                        "mean_processing_latency_ms": round(sum(benchmark_latencies) / len(benchmark_latencies), 2),
                        "p95_processing_latency_ms": round(sorted_latencies[p95_index], 2),
                        "frames_with_face": benchmark_face_frames,
                        "valid_face_benchmark": benchmark_face_frames > 0,
                    }
                    try:
                        benchmark_dir = os.path.join(BASE_DIR, "benchmarks")
                        os.makedirs(benchmark_dir, exist_ok=True)
                        benchmark_path = os.path.join(benchmark_dir, "realtime_benchmark_latest.json")
                        with open(benchmark_path, "w", encoding="utf-8") as benchmark_file:
                            json.dump(benchmark_result, benchmark_file, indent=2)
                    except Exception as exc:
                        LOGGER.warning(f"[benchmark] Could not save result: {exc}")
                    if benchmark_face_frames > 0:
                        benchmark_message = f" · benchmark complete: {benchmark_fps:.1f} FPS"
                    else:
                        benchmark_message = " · benchmark invalid: no face detected"
                    with _benchmark_lock:
                        _benchmark_completed_generation = active_benchmark_generation
                    active_benchmark_generation = None

            performance_text = (
                f"**Live performance:** {avg_fps:.1f} processed/displayed FPS · "
                f"{avg_latency_ms:.0f} ms processing · {end_to_end_latency_ms:.0f} ms end-to-end · "
                f"{drop_percent:.0f}% camera frames skipped{benchmark_message}"
            )

            if live_show_overlay:
                info_str = f"FPS: {avg_fps:.1f} | Process: {avg_latency_ms:.0f}ms | Drop: {drop_percent:.0f}%"
                try:
                    draw_info(frame_out, info_str, (10, frame_out.shape[0] - 20))
                except Exception:
                    pass

            # Mirror to native window if requested
            try:
                if live_show_native:
                    cv2.imshow(native_window_name, frame_out)
                    native_window_open = True
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                         _stop_stream = True
                elif native_window_open:
                    cv2.destroyWindow(native_window_name)
                    native_window_open = False
            except Exception:
                pass
            # Virtual camera output
            try:
                if live_virtual_cam and HAS_PYVIRTUALCAM:
                    h, w = frame_out.shape[:2]
                    configure_virtual_camera(True, w, h, target_fps)
                    publish_virtual_camera_frame(
                        frame_out,
                        input_is_rgb=live_color_mode.startswith("Assume RGB"),
                    )
                elif _virt_cam is not None:
                    close_virtual_camera()
            except Exception:
                pass
            # Stream to Gradio output
            if live_color_mode.startswith("Assume RGB"):
                preview_frame = frame_out.copy()
            else:
                preview_frame = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            yield preview_frame, performance_text
            # Optional auto-fallback to other detectors if no faces for a while
            if live_auto_fallback:
                try:
                    global _zero_face_streak
                    if _last_faces_count == 0:
                        _zero_face_streak += 1
                    else:
                        _zero_face_streak = 0
                    if _zero_face_streak >= 45:
                        # rotate detector model: retinaface -> yunet -> scrfd -> retinaface
                        det_order = ["retinaface", "yunet", "scrfd"]
                        try:
                            cur = state_manager.get_item("face_detector_model") or detector_model
                            nxt = det_order[(det_order.index(cur) + 1) % len(det_order)] if cur in det_order else det_order[0]
                        except Exception:
                            nxt = "yunet"
                        sizes = ff_choices.face_detector_set.get(nxt, ["640x640"]) or ["640x640"]
                        state_manager.set_item("face_detector_model", nxt)
                        state_manager.set_item("face_detector_size", sizes[0])
                        if live_debug_logs:
                            LOGGER.info(f"[fallback] switching detector to {nxt} size={sizes[0]}")
                        _zero_face_streak = 0
                except Exception:
                    pass

    except gr.Error as e:
        # Re-raise to show a visible banner in Gradio
        raise e
    except Exception as e:
        raise gr.Error(f"Start failed: {e}")
    finally:
        # A stopped stream must never leave the physical webcam owned by the
        # process. This also unblocks a capture read that is still waiting in a
        # background reader thread.
        release_camera_capture()
        try:
            if 'frame_reader' in locals() and frame_reader is not None:
                frame_reader.stop()
        except Exception:
            pass
        try:
            if 'pending' in locals() and pending is not None:
                try:
                    pending.cancel()
                except Exception:
                    pass
            if 'executor' in locals() and executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        if 'native_window_open' in locals() and native_window_open:
            try:
                cv2.destroyWindow(native_window_name)
            except Exception:
                pass


def gr_stop_stream():
    global _stop_stream, _stream_generation
    with _stream_lock:
        _stop_stream = True
        _stream_generation += 1
    release_camera_capture()
    reset_temporal_occlusion_cache()
    reset_temporal_face_tracker()
    # In tray mode the sender remains open and falls back to its standby
    # frame. Normal mode closes it because persistence is disabled.
    release_virtual_camera_source()
    return None


def gr_shutdown():
    # Stop any running stream and shutdown the app process
    try:
        gr_stop_stream()
    except Exception:
        pass
    # Give UI time to respond before exit
    def _bye():
        global _cap
        try:
            if _cap is not None:
                _cap.release()
        except Exception:
            pass
        os._exit(0)
    threading.Timer(0.2, _bye).start()
    return "Shutting down..."


def gr_reconnect(camera_choice, backend_name, dshow_name, width, height, fps, convert_rgb, force_fourcc, gentle_mode):
    # Explicitly re-open the device with current settings
    try:
        settings = _settings_from_inputs(camera_choice, backend_name, dshow_name, width, height, fps, convert_rgb, force_fourcc)
        _ensure_capture(settings, gentle_mode=gentle_mode, force_reopen=True)
        return gr.update(value=None), "Reconnected"
    except Exception as e:
        return gr.update(value=None), f"Reconnect failed: {e}"


def main() -> None:
    parser = argparse.ArgumentParser("Webcam Deep Swap (FaceFusion reuse)")
    parser.add_argument("--gui", action="store_true", help="Launch Gradio GUI")
    parser.add_argument("--list-cams", action="store_true", help="List available camera indices and exit")
    parser.add_argument("--headless", action="store_true", help="Run without GUI, process pipeline and (optionally) output to virtual camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Deep swapper model id from FaceFusion (e.g., 'iperov/james_carrey_224')",
    )
    parser.add_argument("--no-occlusion", action="store_true", help="Disable occlusion masking")
    parser.add_argument("--morph", type=int, default=None, help="Morph value [0-100] for deep swapper")
    parser.add_argument("--width", type=int, default=None, help="Capture width")
    parser.add_argument("--height", type=int, default=None, help="Capture height")
    # Extended CLI for UI parity
    parser.add_argument("--ep", dest="ep", type=str, help="Execution provider key (cuda/directml/cpu)")
    parser.add_argument("--device", dest="device", type=str, help="Execution device id (e.g. 0)")
    parser.add_argument("--video-mem", dest="video_mem", type=str, choices=["strict","moderate","relaxed"], help="Video memory strategy")
    # Camera and capture
    parser.add_argument("--backend", type=str, help="Camera backend (Media Foundation/DirectShow/OpenCV)")
    parser.add_argument("--resolution", type=str, help="Resolution preset like 1280x720")
    parser.add_argument("--fps", type=float, help="Target FPS")
    parser.add_argument("--dshow-name", dest="dshow_name", type=str, help="DirectShow device name")
    parser.add_argument("--convert-rgb", action="store_true", help="Force convert RGB at capture")
    parser.add_argument("--fourcc", type=str, help="Force FOURCC (e.g., MJPG/YUY2)")
    parser.add_argument("--retry-black", type=int, help="Retry count when frames are black")
    parser.add_argument("--gentle", action="store_true", help="Gentle mode for reopen/retry")
    parser.add_argument("--auto-repair", action="store_true", help="Enable auto-repair stream")
    parser.add_argument("--color-mode", type=str, help="Color mode (Auto (BGR->RGB) / Assume RGB [...])")
    parser.add_argument("--lock-exposure", action="store_true", help="Lock exposure")
    parser.add_argument("--exposure", type=float, help="Exposure value when locked")
    parser.add_argument("--lock-wb", action="store_true", help="Lock white balance")
    parser.add_argument("--wb-temp", type=int, help="White balance temperature")
    parser.add_argument("--show-overlay", action="store_true", help="Show detection overlay")
    parser.add_argument("--debug-logs", action="store_true", help="Enable debug logs")
    parser.add_argument("--show-boxes", action="store_true", help="Show detection boxes")
    parser.add_argument("--show-native", action="store_true", help="Show native window")
    parser.add_argument("--virtual-cam", action="store_true", help="Enable virtual camera output")
    # Swapping
    parser.add_argument("--swap-mode", type=str, choices=["none","deep","face"], help="Swap mode")
    parser.add_argument("--face-swapper-model", dest="face_swapper_model", type=str, help="Face swapper model")
    parser.add_argument("--face-swapper-pixel", dest="face_swapper_pixel", type=str, help="Face swapper pixel boost")
    parser.add_argument("--face-swapper-weight", dest="face_swapper_weight", type=float, help="Face swapper weight")
    parser.add_argument("--source", action="append", dest="source_paths", help="Face Swapper source image (repeat)")
    # Detection
    parser.add_argument("--detector-model", type=str, help="Detector model")
    parser.add_argument("--detector-size", type=str, help="Detector input size WxH")
    parser.add_argument("--detector-score", type=float, help="Detector score threshold")
    parser.add_argument("--selector-mode", type=str, choices=["one","many"], help="Selector mode")
    parser.add_argument("--auto-fallback", action="store_true", help="Enable auto fallback for detector")
    parser.add_argument("--landmarker-model", type=str, help="Landmarker model")
    parser.add_argument("--landmarker-score", type=float, help="Landmarker score threshold")
    parser.add_argument("--occluder-model", type=str, help="Occluder model")
    parser.add_argument("--parser-model", type=str, help="Parser model")
    # Enhancing & processors toggles and settings
    parser.add_argument("--face-enhancer", action="store_true", help="Enable face enhancer")
    parser.add_argument("--face-enhancer-model", type=str, help="Face enhancer model")
    parser.add_argument("--face-enhancer-blend", type=int, help="Face enhancer blend")
    parser.add_argument("--face-enhancer-weight", type=float, help="Face enhancer weight")
    parser.add_argument("--frame-enhancer", action="store_true", help="Enable frame enhancer")
    parser.add_argument("--frame-enhancer-model", type=str, help="Frame enhancer model")
    parser.add_argument("--frame-enhancer-blend", type=int, help="Frame enhancer blend")
    parser.add_argument("--enhance-async", action="store_true", help="Enable async enhance")
    parser.add_argument("--colorizer", action="store_true", help="Enable frame colorizer")
    parser.add_argument("--colorizer-model", type=str, help="Colorizer model")
    parser.add_argument("--colorizer-size", type=str, help="Colorizer size")
    parser.add_argument("--colorizer-blend", type=int, help="Colorizer blend")
    parser.add_argument("--expr", action="store_true", help="Enable expression restorer")
    parser.add_argument("--expr-model", type=str, help="Expression restorer model")
    parser.add_argument("--expr-factor", type=int, help="Expression restorer factor")
    parser.add_argument("--expr-area", action="append", dest="expr_areas", help="Expression restorer area (repeat)")
    parser.add_argument("--age-mod", action="store_true", help="Enable age modifier")
    parser.add_argument("--age-model", type=str, help="Age modifier model")
    parser.add_argument("--age-direction", type=int, help="Age modifier direction")
    parser.add_argument("--editor", action="store_true", help="Enable face editor")
    parser.add_argument("--editor-model", type=str, help="Face editor model")
    parser.add_argument("--fe-eyebrow-dir", type=float, help="Face editor eyebrow direction")
    parser.add_argument("--fe-eye-h", type=float, help="Face editor eye gaze horizontal")
    parser.add_argument("--fe-eye-v", type=float, help="Face editor eye gaze vertical")
    parser.add_argument("--fe-eye-open", type=float, help="Face editor eye open ratio")
    parser.add_argument("--fe-lip-open", type=float, help="Face editor lip open ratio")
    parser.add_argument("--fe-smile", type=float, help="Face editor mouth smile")
    parser.add_argument("--fe-head-pitch", type=float, help="Face editor head pitch")
    parser.add_argument("--fe-head-yaw", type=float, help="Face editor head yaw")
    parser.add_argument("--fe-head-roll", type=float, help="Face editor head roll")
    parser.add_argument("--face-debugger", action="store_true", help="Enable face debugger")
    parser.add_argument("--face-debugger-item", action="append", dest="face_debugger_items", help="Debugger item (repeat)")
    parser.add_argument("--lip-syncer", action="store_true", help="Enable lip syncer")
    parser.add_argument("--lip-model", type=str, help="Lip syncer model")
    parser.add_argument("--lip-weight", type=float, help="Lip syncer weight")

    args = parser.parse_args()

    # Apply CLI overrides into preferences prior to GUI build
    def _apply_cli_overrides():
        ov = {}
        # General
        if args.ep: ov['execution_providers'] = [args.ep]
        if args.device: ov['execution_device_ids'] = [str(args.device)]
        if args.video_mem: ov['video_memory_strategy'] = args.video_mem
        # Camera
        if args.backend: ov['backend'] = args.backend
        if args.resolution: ov['resolution_preset'] = args.resolution
        if args.width is not None: ov['width'] = int(args.width)
        if args.height is not None: ov['height'] = int(args.height)
        if args.fps is not None: ov['fps'] = float(args.fps)
        if args.dshow_name: ov['dshow_name_device'] = args.dshow_name
        if args.convert_rgb: ov['convert_rgb'] = True
        if args.fourcc: ov['force_fourcc'] = args.fourcc
        if args.retry_black is not None: ov['retry_black'] = int(args.retry_black)
        if args.gentle: ov['gentle_mode'] = True
        if args.auto_repair: ov['auto_repair'] = True
        if args.color_mode: ov['color_mode'] = args.color_mode
        if args.lock_exposure: ov['lock_exposure'] = True
        if args.exposure is not None: ov['exposure_value'] = float(args.exposure)
        if args.lock_wb: ov['lock_wb'] = True
        if args.wb_temp is not None: ov['wb_temperature'] = int(args.wb_temp)
        if args.show_overlay: ov['show_overlay'] = True
        if args.debug_logs: ov['debug_logs'] = True
        if args.show_boxes: ov['show_boxes'] = True
        if args.show_native: ov['show_native'] = True
        if args.virtual_cam: ov['virtual_cam_enabled'] = True
        # Swapping
        if args.swap_mode: ov['swap_mode'] = args.swap_mode
        if args.face_swapper_model: ov['face_swapper_model'] = args.face_swapper_model
        if args.face_swapper_pixel: ov['face_swapper_pixel_boost'] = args.face_swapper_pixel
        if args.face_swapper_weight is not None: ov['face_swapper_weight'] = float(args.face_swapper_weight)
        if args.model: ov['deep_swapper_model'] = args.model
        if args.morph is not None: ov['morph'] = int(args.morph)
        if getattr(args, 'source_paths', None): ov['source_paths'] = list(args.source_paths)
        # Detection
        if args.detector_model: ov['detector_model'] = args.detector_model
        if args.detector_size: ov['detector_size'] = args.detector_size
        if args.detector_score is not None: ov['detector_score'] = float(args.detector_score)
        if args.selector_mode: ov['selector_mode'] = args.selector_mode
        if args.auto_fallback: ov['auto_fallback'] = True
        if args.landmarker_model: ov['landmarker_model'] = args.landmarker_model
        if args.landmarker_score is not None: ov['landmarker_score'] = float(args.landmarker_score)
        if args.occluder_model: ov['occluder_model'] = args.occluder_model
        if args.parser_model: ov['parser_model'] = args.parser_model
        # Enhancers/processors
        if args.face_enhancer: ov['face_enhancer_enabled'] = True
        if args.face_enhancer_model: ov['face_enhancer_model'] = args.face_enhancer_model
        if args.face_enhancer_blend is not None: ov['face_enhancer_blend'] = int(args.face_enhancer_blend)
        if args.face_enhancer_weight is not None: ov['face_enhancer_weight'] = float(args.face_enhancer_weight)
        if args.frame_enhancer: ov['frame_enhancer_enabled'] = True
        if args.frame_enhancer_model: ov['frame_enhancer_model'] = args.frame_enhancer_model
        if args.frame_enhancer_blend is not None: ov['frame_enhancer_blend'] = int(args.frame_enhancer_blend)
        if args.enhance_async: ov['enhance_async'] = True
        if args.colorizer: ov['frame_colorizer_enabled'] = True
        if args.colorizer_model: ov['frame_colorizer_model'] = args.colorizer_model
        if args.colorizer_size: ov['frame_colorizer_size'] = args.colorizer_size
        if args.colorizer_blend is not None: ov['frame_colorizer_blend'] = int(args.colorizer_blend)
        if args.expr: ov['expression_restorer_enabled'] = True
        if args.expr_model: ov['expression_restorer_model'] = args.expr_model
        if args.expr_factor is not None: ov['expression_restorer_factor'] = int(args.expr_factor)
        if getattr(args, 'expr_areas', None): ov['expression_restorer_areas'] = list(args.expr_areas)
        if args.age_mod: ov['age_modifier_enabled'] = True
        if args.age_model: ov['age_modifier_model'] = args.age_model
        if args.age_direction is not None: ov['age_modifier_direction'] = int(args.age_direction)
        if args.editor: ov['face_editor_enabled'] = True
        if args.editor_model: ov['face_editor_model'] = args.editor_model
        for k, v in {
            'face_editor_eyebrow_direction': args.fe_eyebrow_dir,
            'face_editor_eye_gaze_horizontal': args.fe_eye_h,
            'face_editor_eye_gaze_vertical': args.fe_eye_v,
            'face_editor_eye_open_ratio': args.fe_eye_open,
            'face_editor_lip_open_ratio': args.fe_lip_open,
            'face_editor_mouth_smile': args.fe_smile,
            'face_editor_head_pitch': args.fe_head_pitch,
            'face_editor_head_yaw': args.fe_head_yaw,
            'face_editor_head_roll': args.fe_head_roll,
        }.items():
            if v is not None:
                ov[k] = float(v)
        if args.face_debugger: ov['face_debugger_enabled'] = True
        if getattr(args, 'face_debugger_items', None): ov['face_debugger_items'] = list(args.face_debugger_items)
        if args.lip_syncer: ov['lip_syncer_enabled'] = True
        if args.lip_model: ov['lip_syncer_model'] = args.lip_model
        if args.lip_weight is not None: ov['lip_syncer_weight'] = float(args.lip_weight)
        # Occlusion (default true); handled via --no-occlusion from existing flag
        if args.no_occlusion: ov['use_occlusion'] = False
        # Persist overrides
        if ov:
            prefs = _load_prefs().copy()
            prefs.update(ov)
            _save_prefs(prefs)
            for k, v in ov.items():
                try:
                    state_manager.set_item(k, v)
                except Exception:
                    pass

    _apply_cli_overrides()

    ensure_models_downloaded()

    # Headless mode (no GUI): run processing loop and optionally output to virtual camera
    if args.headless and not args.list_cams:
        try:
            prefs = _load_prefs().copy()
            # Minimal state priming from prefs
            for k in [
                'execution_providers','execution_device_ids','video_memory_strategy',
                'detector_model','detector_size','detector_score','selector_mode','auto_fallback',
                'landmarker_model','landmarker_score','occluder_model','parser_model',
                'swap_mode','deep_swapper_model','morph','face_swapper_model','face_swapper_pixel_boost','face_swapper_weight',
                'frame_colorizer_enabled','frame_colorizer_model','frame_colorizer_size','frame_colorizer_blend',
                'expression_restorer_enabled','expression_restorer_model','expression_restorer_factor','expression_restorer_areas',
                'age_modifier_enabled','age_modifier_model','age_modifier_direction',
                'face_editor_enabled','face_editor_model',
                'face_debugger_enabled','face_debugger_items',
                'lip_syncer_enabled','lip_syncer_model','lip_syncer_weight',
                'face_enhancer_enabled','face_enhancer_model','face_enhancer_blend','face_enhancer_weight',
                'frame_enhancer_enabled','frame_enhancer_model','frame_enhancer_blend','enhance_async',
            ]:
                if k in prefs:
                    try: state_manager.set_item(k, prefs[k])
                    except Exception: pass
            # Gather inputs for gr_stream from prefs
            camera_choice = prefs.get('camera_choice', "[0] Camera 0")
            width = int(prefs.get('width', 1920))
            height = int(prefs.get('height', 1080))
            use_occlusion = bool(prefs.get('use_occlusion', True))
            fps = float(prefs.get('fps', 15.0))
            backend = prefs.get('backend', 'DirectShow' if os.name == 'nt' else 'Auto')
            dshow_name = prefs.get('dshow_name_device', prefs.get('dshow_name_text', None))
            convert_rgb = bool(prefs.get('convert_rgb', True))
            fourcc = prefs.get('force_fourcc', 'Auto')
            retry_black = int(prefs.get('retry_black', 3))
            gentle_mode = bool(prefs.get('gentle_mode', True))
            auto_repair = bool(prefs.get('auto_repair', True))
            color_mode = prefs.get('color_mode', 'Auto (BGR->RGB)')
            lock_exp = bool(prefs.get('lock_exposure', False))
            exposure_val = float(prefs.get('exposure_value', -6.0))
            lock_wb = bool(prefs.get('lock_wb', False))
            wb_temp = int(prefs.get('wb_temperature', 4500))
            show_overlay = bool(prefs.get('show_overlay', False))
            debug_logs = bool(prefs.get('debug_logs', False))
            show_boxes = bool(prefs.get('show_boxes', False))
            show_native = bool(prefs.get('show_native', False))
            virtual_cam_enabled = bool(prefs.get('virtual_cam_enabled', False))
            # Downstream flags
            selector_mode_v = prefs.get('selector_mode', 'one')
            auto_fallback_v = bool(prefs.get('auto_fallback', True))
            exec_provider_key = (prefs.get('execution_providers') or ['cpu'])[0]
            exec_device_id = (prefs.get('execution_device_ids') or ['0'])[0]
            video_mem_val = prefs.get('video_memory_strategy', 'moderate')
            d_model_v = prefs.get('detector_model', 'retinaface')
            d_size_v = prefs.get('detector_size', '160x160')
            d_score_v = float(prefs.get('detector_score', 0.5))
            l_model_v = prefs.get('landmarker_model', 'many')
            l_score_v = float(prefs.get('landmarker_score', 0.5))
            o_model_v = prefs.get('occluder_model', 'xseg_2')
            p_model_v = prefs.get('parser_model', None)
            swap_mode_v = prefs.get('swap_mode', 'deep')
            ds_model_v = prefs.get('deep_swapper_model', 'iperov/james_carrey_224')
            morph_v = int(prefs.get('morph', 100))
            fs_model_v = prefs.get('face_swapper_model', None)
            fs_pixel_v = prefs.get('face_swapper_pixel_boost', None)
            fs_weight_v = float(prefs.get('face_swapper_weight', 0.5))
            face_enh_enabled_v = bool(prefs.get('face_enhancer_enabled', False))
            face_enh_model_v = prefs.get('face_enhancer_model', None)
            face_enh_bl_v = int(prefs.get('face_enhancer_blend', 80))
            face_enh_w_v = float(prefs.get('face_enhancer_weight', 0.5))
            frame_enh_enabled_v = bool(prefs.get('frame_enhancer_enabled', False))
            frame_enh_model_v = prefs.get('frame_enhancer_model', None)
            frame_enh_bl_v = int(prefs.get('frame_enhancer_blend', 80))
            enhance_async_v = bool(prefs.get('enhance_async', True))
            colorizer_enabled_v = bool(prefs.get('frame_colorizer_enabled', False))
            colorizer_model_v = prefs.get('frame_colorizer_model', None)
            colorizer_size_v = prefs.get('frame_colorizer_size', None)
            colorizer_blend_v = int(prefs.get('frame_colorizer_blend', 100))
            expr_enabled_v = bool(prefs.get('expression_restorer_enabled', False))
            expr_model_v = prefs.get('expression_restorer_model', None)
            expr_factor_v = int(prefs.get('expression_restorer_factor', 80))
            expr_areas_v = prefs.get('expression_restorer_areas', [])
            age_enabled_v = bool(prefs.get('age_modifier_enabled', False))
            age_model_v = prefs.get('age_modifier_model', None)
            age_direction_v = int(prefs.get('age_modifier_direction', 0))
            editor_enabled_v = bool(prefs.get('face_editor_enabled', False))
            editor_model_v = prefs.get('face_editor_model', None)
            # Face editor sliders default 0.0
            fe_defaults = lambda k: float(prefs.get(k, 0.0))
            fe_eyebrow_dir_v = fe_defaults('face_editor_eyebrow_direction')
            fe_eye_h_v = fe_defaults('face_editor_eye_gaze_horizontal')
            fe_eye_v_v = fe_defaults('face_editor_eye_gaze_vertical')
            fe_eye_open_v = fe_defaults('face_editor_eye_open_ratio')
            fe_lip_open_v = fe_defaults('face_editor_lip_open_ratio')
            fe_mouth_smile_v = fe_defaults('face_editor_mouth_smile')
            fe_head_pitch_v = fe_defaults('face_editor_head_pitch')
            fe_head_yaw_v = fe_defaults('face_editor_head_yaw')
            fe_head_roll_v = fe_defaults('face_editor_head_roll')
            dbg_enabled_v = bool(prefs.get('face_debugger_enabled', False))
            dbg_items_v = prefs.get('face_debugger_items', [])
            lip_enabled_v = bool(prefs.get('lip_syncer_enabled', False))
            lip_model_v = prefs.get('lip_syncer_model', None)
            lip_weight_v = float(prefs.get('lip_syncer_weight', 0.5))

            # Iterate frames from gr_stream; virtual camera sending is handled inside gr_stream when enabled
            for _ in gr_stream(
                camera_choice, width, height,
                use_occlusion, fps, backend, dshow_name or "", convert_rgb, fourcc, retry_black, gentle_mode, auto_repair, color_mode,
                lock_exp, exposure_val, lock_wb, wb_temp,
                show_overlay, debug_logs, show_boxes,
                show_native, virtual_cam_enabled,
                colorizer_enabled_v, expr_enabled_v, age_enabled_v, editor_enabled_v, dbg_enabled_v, lip_enabled_v,
                selector_mode_v, auto_fallback_v,
                exec_provider_key, str(exec_device_id), video_mem_val,
                d_model_v, d_size_v, d_score_v, l_model_v, l_score_v, o_model_v, p_model_v,
                swap_mode_v, [], ds_model_v, morph_v,
                fs_model_v, fs_pixel_v, fs_weight_v,
                face_enh_enabled_v, face_enh_model_v, face_enh_bl_v, face_enh_w_v,
                frame_enh_enabled_v, frame_enh_model_v, frame_enh_bl_v,
                enhance_async_v
            ):
                # In headless mode we don't yield anywhere; sleep is inside gr_stream
                pass
        except KeyboardInterrupt:
            try: gr_stop_stream()
            except Exception: pass
        return

    if args.gui:
        choices = get_model_choices()
        detector_models = choices["detector_models"]
        detector_sizes_map = choices["detector_sizes_map"]
        landmarker_models = choices["landmarker_models"]
        occluder_models = choices["occluder_models"]
        parser_models = choices["parser_models"]
        deep_models = choices["deep_models"]
        face_swapper_models = choices["face_swapper_models"]
        frame_enhancer_models = choices["frame_enhancer_models"]
        face_enhancer_models = choices["face_enhancer_models"]

        with gr.Blocks() as demo:
            gr.Markdown("# Webcam Deep Swap")

            # Keep the controls used during a call visible regardless of which
            # settings tab is open.
            cam_choices = get_camera_choices(10, backend_name="DirectShow")
            saved_camera = _get_pref(
                'camera_choice',
                cam_choices[0] if cam_choices else f"[{args.camera}] Camera {args.camera}",
            )
            preferred_camera_name = _get_pref(
                'dshow_name_device',
                _camera_name_from_choice(saved_camera),
            )
            if cam_choices:
                saved_camera = next(
                    (
                        choice for choice in cam_choices
                        if _camera_name_from_choice(choice) == preferred_camera_name
                    ),
                    saved_camera if saved_camera in cam_choices else cam_choices[0],
                )
            initial_camera_name = _camera_name_from_choice(saved_camera)
            initial_detected_modes = detect_camera_capabilities(initial_camera_name)
            initial_modes = initial_detected_modes or dict(_fallback_camera_capabilities)
            res_presets = _sorted_camera_resolutions(initial_modes)
            saved_resolution = _get_pref('resolution_preset', "1920x1080")
            initial_resolution = saved_resolution if saved_resolution in initial_modes else res_presets[-1]
            initial_width, initial_height = (int(part) for part in initial_resolution.split("x", 1))
            initial_max_fps = max(1.0, float(initial_modes[initial_resolution]))
            initial_fps = min(max(1.0, float(_get_pref('fps', 30))), initial_max_fps)
            camera_display_choices = [
                (_camera_name_from_choice(choice), choice) for choice in cam_choices
            ]

            with gr.Group():
                gr.Markdown("## Live camera")
                with gr.Row():
                    camera = gr.Dropdown(
                        choices=camera_display_choices,
                        value=saved_camera,
                        label="Camera",
                        scale=3,
                    )
                    fps = gr.Slider(
                        minimum=1,
                        maximum=initial_max_fps,
                        step=1,
                        value=initial_fps,
                        label=f"Camera FPS (detected max {initial_max_fps:g})",
                        scale=3,
                    )
                    detect_modes_btn = gr.Button("Detect Modes")
                    refresh_btn = gr.Button("Refresh Cameras")
                res = gr.Radio(
                    choices=res_presets,
                    value=initial_resolution,
                    label="Quick resolution switch",
                )
                camera_mode_info = gr.Markdown(
                    _camera_mode_status(initial_camera_name, initial_detected_modes, initial_resolution)
                )
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop")
                    reconnect_btn = gr.Button("Reconnect")
                    test_btn = gr.Button("Test Capture")
                    benchmark_btn = gr.Button("Run 10s Benchmark")
                    show_native = gr.Checkbox(value=bool(_get_pref('show_native', False)), label="Show native window")
                    virtual_cam = gr.Checkbox(value=bool(_get_pref('virtual_cam_enabled', False)), label="Enable Virtual Camera" if HAS_PYVIRTUALCAM else "Enable Virtual Camera (install pyvirtualcam)")
                    fast_live_analysis = gr.Checkbox(
                        value=bool(_get_pref('realtime_fast_analysis', True)),
                        label="Fast live analysis",
                    )
                    auto_start = gr.Checkbox(value=True, label="Auto-start stream")
                state_manager.set_item('realtime_fast_analysis', bool(_get_pref('realtime_fast_analysis', True)))

            # Filled automatically by the camera and quick-resolution controls.
            backend = gr.Dropdown(
                choices=list(BACKEND_MAP.keys()),
                value="DirectShow" if os.name == "nt" else "Auto",
                visible=False,
            )
            width = gr.Number(value=initial_width, precision=0, visible=False)
            height = gr.Number(value=initial_height, precision=0, visible=False)
            dshow_names = _ffmpeg_camera_names()
            dshow_name_drop = gr.Dropdown(
                choices=dshow_names,
                value=initial_camera_name if initial_camera_name in dshow_names else None,
                visible=False,
            )
            dshow_name = gr.Textbox(value="", visible=False)

            with gr.Row():
                # Execution provider & device selection
                ep_keys = _available_execution_provider_keys()
                # default from prefs; fall back to available set
                pref_ep = (_get_pref('execution_providers', ['cpu']) or ['cpu'])[0]
                if pref_ep not in ep_keys:
                    pref_ep = 'cuda' if 'cuda' in ep_keys else ('directml' if 'directml' in ep_keys else 'cpu')
                exec_provider = gr.Dropdown(choices=ep_keys, value=pref_ep, label="Execution Provider")
                pref_dev = (_get_pref('execution_device_ids', ['0']) or ['0'])[0]
                exec_device = gr.Textbox(value=str(pref_dev), label="Execution Device ID")
                video_mem = gr.Dropdown(choices=["strict","moderate","relaxed"], value=_get_pref('video_memory_strategy', "moderate"), label="Video Memory Strategy")
                shutdown_btn = gr.Button("Shutdown App", variant="stop")
                fast_startup = gr.Checkbox(value=bool(_get_pref('fast_startup', True)), label="Fast Startup (skip model checks)")
            with gr.Tabs():
                with gr.Tab("Swapping"):
                    with gr.Row("Face Swapper"):
                        swap_mode = gr.Dropdown(choices=["none","deep","face"], value=_get_pref('swap_mode', "deep"), label="Swap Mode")
                        source_files = gr.Files(label="Source Photos (for Face Swapper)", file_types=["image"], type="filepath")
                    with gr.Row():
                        fs_model = gr.Dropdown(choices=face_swapper_models, value=_get_pref('face_swapper_model', (face_swapper_models[0] if face_swapper_models else None)), label="Face Swapper Model")
                        fs_pixel_default = proc_choices.face_swapper_set.get((fs_model.value if fs_model.value else 'inswapper_128'), ["256x256"]) or ["256x256"]
                        fs_pixel = gr.Dropdown(choices=fs_pixel_default , value=_get_pref('face_swapper_pixel_boost', fs_pixel_default[0]), label="Face Swapper Pixel Boost")
                        fs_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=float(_get_pref('face_swapper_weight', 0.5)), label="Face Swapper Weight")
                    with gr.Row():
                        ds_model = gr.Dropdown(choices=deep_models, value=_get_pref('deep_swapper_model', "iperov/james_carrey_224"), label="Deep Swapper Model")
                        morph = gr.Slider(minimum=0, maximum=100, step=1, value=int(_get_pref('morph', 100)), label="Morph")
                    
                with gr.Tab("Camera"):
                    gr.Markdown("Capture tuning and diagnostics. Everyday camera controls stay above the tabs.")
                    with gr.Accordion("Advanced camera controls", open=False):
                        with gr.Row():
                            convert_rgb = gr.Checkbox(value=bool(_get_pref('convert_rgb', True)), label="Convert RGB in driver")
                            force_fourcc = gr.Dropdown(choices=["Auto","MJPG","YUY2","H264","NV12"], value=_get_pref('force_fourcc', "Auto"), label="Force FOURCC")
                            retry_black = gr.Slider(minimum=0, maximum=10, step=1, value=int(_get_pref('retry_black', 3)), label="Retry on black frames")
                            gentle_mode = gr.Checkbox(value=bool(_get_pref('gentle_mode', True)), label="Gentle Mode")
                            auto_repair = gr.Checkbox(value=bool(_get_pref('auto_repair', True)), label="Auto-repair capture")
                            color_mode = gr.Dropdown(choices=["Auto (BGR->RGB)", "Assume RGB (no swap)",], value=_get_pref('color_mode', "Auto (BGR->RGB)"), label="Color mode")
                        with gr.Row():
                            lock_exposure = gr.Checkbox(value=bool(_get_pref('lock_exposure', False)), label="Lock Exposure")
                            exposure_value = gr.Slider(minimum=-13.0, maximum=-1.0, step=0.5, value=float(_get_pref('exposure_value', -6.0)), label="Exposure (log scale)")
                            lock_wb = gr.Checkbox(value=bool(_get_pref('lock_wb', False)), label="Lock White Balance")
                            wb_temperature = gr.Slider(minimum=2800, maximum=6500, step=100, value=int(_get_pref('wb_temperature', 4500)), label="WB Temperature (K)")
                            show_overlay = gr.Checkbox(value=bool(_get_pref('show_overlay', False)), label="Show detection overlay")
                            debug_logs = gr.Checkbox(value=bool(_get_pref('debug_logs', False)), label="Debug logs to console")
                            show_boxes = gr.Checkbox(value=bool(_get_pref('show_boxes', False)), label="Show detection boxes")
                        
           
                with gr.Tab("Detection"):
                    with gr.Row():
                        d_model = gr.Dropdown(choices=detector_models, value=_get_pref('detector_model', "retinaface"), label="Detector Model")
                        d_size = gr.Dropdown(choices=detector_sizes_map.get("retinaface", ["640x640"]), value=_get_pref('detector_size', detector_sizes_map.get("retinaface", ["640x640"])[0]), label="Detector Size")
                        d_score = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=float(_get_pref('detector_score', 0.5)), label="Detector Score")
                        selector_mode_dd = gr.Dropdown(choices=ff_choices.face_selector_modes, value=_get_pref('selector_mode', "one"), label="Selector Mode")
                        auto_fallback = gr.Checkbox(value=bool(_get_pref('auto_fallback', True)), label="Auto fallback detector if no faces")
                    with gr.Row():
                        l_model = gr.Dropdown(choices=landmarker_models, value=_get_pref('landmarker_model', "many"), label="Landmarker Model")
                        l_score = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=float(_get_pref('landmarker_score', 0.5)), label="Landmarker Score")
                    with gr.Row():
                        o_model = gr.Dropdown(choices=occluder_models, value=_get_pref('occluder_model', "xseg_1"), label="Occluder Model")
                        p_model = gr.Dropdown(choices=parser_models, value=(parser_models[0] if parser_models else None), label="Parser Model")
                        occl_enabled = gr.Checkbox(value=bool(_get_pref('use_occlusion', True)), label="Use Occlusion Mask")
                
                with gr.Tab("Enhancing"):
                    with gr.Row():
                        face_enh_enabled = gr.Checkbox(value=bool(_get_pref('face_enhancer_enabled', False)), label="Enable Face Enhancer")
                        face_enh_model = gr.Dropdown(choices=face_enhancer_models, value=_get_pref('face_enhancer_model', ("gfpgan_1.4" if "gfpgan_1.4" in face_enhancer_models else (face_enhancer_models[0] if face_enhancer_models else None))), label="Face Enhancer Model")
                        face_enh_blend = gr.Slider(minimum=0, maximum=100, step=1, value=int(_get_pref('face_enhancer_blend', 80)), label="Face Enhancer Blend")
                        face_enh_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=float(_get_pref('face_enhancer_weight', 0.5)), label="Face Enhancer Weight")
                    with gr.Row():
                        frame_enh_enabled = gr.Checkbox(value=bool(_get_pref('frame_enhancer_enabled', False)), label="Enable Frame Enhancer")
                        frame_enh_model = gr.Dropdown(choices=frame_enhancer_models, value=_get_pref('frame_enhancer_model', ("span_kendata_x4" if "span_kendata_x4" in frame_enhancer_models else (frame_enhancer_models[0] if frame_enhancer_models else None))), label="Frame Enhancer Model")
                        frame_enh_blend = gr.Slider(minimum=0, maximum=100, step=1, value=int(_get_pref('frame_enhancer_blend', 80)), label="Frame Enhancer Blend")
                    with gr.Row():
                        enhance_async = gr.Checkbox(value=bool(_get_pref('enhance_async', True)), label="Async Enhance (background enhancers)")
                with gr.Tab("Processors"):
                    with gr.Row():
                        colorizer_enabled = gr.Checkbox(value=bool(_get_pref('frame_colorizer_enabled', False)), label="Enable Frame Colorizer")
                        colorizer_model = gr.Dropdown(choices=frame_colorizer_choices.frame_colorizer_models, value=_get_pref('frame_colorizer_model', (frame_colorizer_choices.frame_colorizer_models[0] if frame_colorizer_choices.frame_colorizer_models else None)), label="Colorizer Model")
                        colorizer_size = gr.Dropdown(choices=frame_colorizer_choices.frame_colorizer_sizes, value=_get_pref('frame_colorizer_size', (frame_colorizer_choices.frame_colorizer_sizes[0] if frame_colorizer_choices.frame_colorizer_sizes else None)), label="Colorizer Size")
                        colorizer_blend = gr.Slider(minimum=min(frame_colorizer_choices.frame_colorizer_blend_range), maximum=max(frame_colorizer_choices.frame_colorizer_blend_range), step=1, value=int(_get_pref('frame_colorizer_blend', 100)), label="Colorizer Blend")
                    with gr.Row():
                        expr_enabled = gr.Checkbox(value=bool(_get_pref('expression_restorer_enabled', False)), label="Enable Expression Restorer")
                        expr_model = gr.Dropdown(choices=expression_restorer_choices.expression_restorer_models, value=_get_pref('expression_restorer_model', (expression_restorer_choices.expression_restorer_models[0] if expression_restorer_choices.expression_restorer_models else None)), label="Expr Model")
                        expr_factor = gr.Slider(minimum=min(expression_restorer_choices.expression_restorer_factor_range), maximum=max(expression_restorer_choices.expression_restorer_factor_range), step=1, value=int(_get_pref('expression_restorer_factor', 80)), label="Expr Factor")
                        expr_areas = gr.CheckboxGroup(choices=expression_restorer_choices.expression_restorer_areas, value=_get_pref('expression_restorer_areas', expression_restorer_choices.expression_restorer_areas), label="Expr Areas")
                    with gr.Row():
                        age_enabled = gr.Checkbox(value=bool(_get_pref('age_modifier_enabled', False)), label="Enable Age Modifier")
                        age_model = gr.Dropdown(choices=age_modifier_choices.age_modifier_models, value=_get_pref('age_modifier_model', (age_modifier_choices.age_modifier_models[0] if age_modifier_choices.age_modifier_models else None)), label="Age Model")
                        age_direction = gr.Slider(minimum=min(age_modifier_choices.age_modifier_direction_range), maximum=max(age_modifier_choices.age_modifier_direction_range), step=1, value=int(_get_pref('age_modifier_direction', 0)), label="Age Direction")
                    with gr.Row():
                        editor_enabled = gr.Checkbox(value=bool(_get_pref('face_editor_enabled', False)), label="Enable Face Editor")
                        editor_model = gr.Dropdown(choices=face_editor_choices.face_editor_models, value=_get_pref('face_editor_model', (face_editor_choices.face_editor_models[0] if face_editor_choices.face_editor_models else None)), label="Editor Model")
                    with gr.Row():
                        fe_eyebrow_dir = gr.Slider(minimum=min(face_editor_choices.face_editor_eyebrow_direction_range), maximum=max(face_editor_choices.face_editor_eyebrow_direction_range), step=0.1, value=float(_get_pref('face_editor_eyebrow_direction', 0.0)), label="Eyebrow Direction")
                        fe_eye_h = gr.Slider(minimum=min(face_editor_choices.face_editor_eye_gaze_horizontal_range), maximum=max(face_editor_choices.face_editor_eye_gaze_horizontal_range), step=0.1, value=float(_get_pref('face_editor_eye_gaze_horizontal', 0.0)), label="Eye Gaze H")
                        fe_eye_v = gr.Slider(minimum=min(face_editor_choices.face_editor_eye_gaze_vertical_range), maximum=max(face_editor_choices.face_editor_eye_gaze_vertical_range), step=0.1, value=float(_get_pref('face_editor_eye_gaze_vertical', 0.0)), label="Eye Gaze V")
                        fe_eye_open = gr.Slider(minimum=min(face_editor_choices.face_editor_eye_open_ratio_range), maximum=max(face_editor_choices.face_editor_eye_open_ratio_range), step=0.1, value=float(_get_pref('face_editor_eye_open_ratio', 0.0)), label="Eye Open")
                    with gr.Row():
                        fe_lip_open = gr.Slider(minimum=min(face_editor_choices.face_editor_lip_open_ratio_range), maximum=max(face_editor_choices.face_editor_lip_open_ratio_range), step=0.1, value=float(_get_pref('face_editor_lip_open_ratio', 0.0)), label="Lip Open")
                        fe_mouth_smile = gr.Slider(minimum=min(face_editor_choices.face_editor_mouth_smile_range), maximum=max(face_editor_choices.face_editor_mouth_smile_range), step=0.1, value=float(_get_pref('face_editor_mouth_smile', 0.0)), label="Smile")
                        fe_head_pitch = gr.Slider(minimum=min(face_editor_choices.face_editor_head_pitch_range), maximum=max(face_editor_choices.face_editor_head_pitch_range), step=0.1, value=float(_get_pref('face_editor_head_pitch', 0.0)), label="Head Pitch")
                    with gr.Row():
                        fe_mouth_grim = gr.Slider(minimum=min(face_editor_choices.face_editor_mouth_grim_range), maximum=max(face_editor_choices.face_editor_mouth_grim_range), step=0.1, value=float(_get_pref('face_editor_mouth_grim', 0.0)), label="Mouth Grim")
                        fe_mouth_pout = gr.Slider(minimum=min(face_editor_choices.face_editor_mouth_pout_range), maximum=max(face_editor_choices.face_editor_mouth_pout_range), step=0.1, value=float(_get_pref('face_editor_mouth_pout', 0.0)), label="Mouth Pout")
                        fe_mouth_purse = gr.Slider(minimum=min(face_editor_choices.face_editor_mouth_purse_range), maximum=max(face_editor_choices.face_editor_mouth_purse_range), step=0.1, value=float(_get_pref('face_editor_mouth_purse', 0.0)), label="Mouth Purse")
                    with gr.Row():
                        fe_mouth_pos_h = gr.Slider(minimum=min(face_editor_choices.face_editor_mouth_position_horizontal_range), maximum=max(face_editor_choices.face_editor_mouth_position_horizontal_range), step=0.1, value=float(_get_pref('face_editor_mouth_position_horizontal', 0.0)), label="Mouth Pos H")
                        fe_mouth_pos_v = gr.Slider(minimum=min(face_editor_choices.face_editor_mouth_position_vertical_range), maximum=max(face_editor_choices.face_editor_mouth_position_vertical_range), step=0.1, value=float(_get_pref('face_editor_mouth_position_vertical', 0.0)), label="Mouth Pos V")
                    with gr.Row():
                        fe_head_yaw = gr.Slider(minimum=min(face_editor_choices.face_editor_head_yaw_range), maximum=max(face_editor_choices.face_editor_head_yaw_range), step=0.1, value=float(_get_pref('face_editor_head_yaw', 0.0)), label="Head Yaw")
                        fe_head_roll = gr.Slider(minimum=min(face_editor_choices.face_editor_head_roll_range), maximum=max(face_editor_choices.face_editor_head_roll_range), step=0.1, value=float(_get_pref('face_editor_head_roll', 0.0)), label="Head Roll")
                    with gr.Row():
                        debugger_enabled = gr.Checkbox(value=bool(_get_pref('face_debugger_enabled', False)), label="Enable Face Debugger")
                        dbg_items = gr.CheckboxGroup(choices=face_debugger_choices.face_debugger_items, value=_get_pref('face_debugger_items', ['face-landmark-5/68','face-mask']), label="Debugger Items")
                    with gr.Row():
                        lip_enabled = gr.Checkbox(value=bool(_get_pref('lip_syncer_enabled', False)), label="Enable Lip Syncer")
                        lip_model = gr.Dropdown(choices=lip_syncer_choices.lip_syncer_models, value=_get_pref('lip_syncer_model', (lip_syncer_choices.lip_syncer_models[0] if lip_syncer_choices.lip_syncer_models else None)), label="Lip Model")
                        lip_weight = gr.Slider(minimum=min(lip_syncer_choices.lip_syncer_weight_range), maximum=max(lip_syncer_choices.lip_syncer_weight_range), step=0.05, value=float(_get_pref('lip_syncer_weight', 0.5)), label="Lip Weight")
                with gr.Tab("Execution Routing"):
                    gr.Markdown(
                        "Choose where every model runs. **CUDA is the measured sweet spot.** "
                        "With Fast live analysis enabled, Landmarker, Recognizer and Classifier are bypassed."
                    )
                    with gr.Row():
                        provider_detector = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_detector', 'cuda'), label="Detector")
                        provider_landmarker = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_landmarker', 'cuda'), label="Landmarker (bypassed in Fast live)")
                        provider_recognizer = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_recognizer', 'cuda'), label="Recognizer (bypassed in Fast live)")
                        provider_classifier = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_classifier', 'cuda'), label="Classifier (bypassed in Fast live)")
                    with gr.Row():
                        provider_masker = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_masker', 'cuda'), label="Occlusion / parser masks")
                        provider_deep_swapper = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_deep_swapper', 'cuda'), label="Deep swapper")
                        provider_face_swapper = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_face_swapper', 'cuda'), label="Face swapper")
                        provider_content = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_content_analyser', 'cuda'), label="Content analyser")
                    with gr.Row():
                        provider_face_enhancer = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_face_enhancer', 'cuda'), label="Face enhancer")
                        provider_frame_enhancer = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_frame_enhancer', 'cuda'), label="Frame enhancer")
                        provider_colorizer = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_colorizer', 'cuda'), label="Colorizer")
                        provider_expression = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_expression_restorer', 'cuda'), label="Expression restorer")
                    with gr.Row():
                        provider_age = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_age_modifier', 'cuda'), label="Age modifier")
                        provider_editor = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_face_editor', 'cuda'), label="Face editor")
                        provider_lip = gr.Dropdown(choices=ep_keys, value=_get_pref('provider_lip_syncer', 'cuda'), label="Lip syncer")
                    apply_routing_btn = gr.Button("Apply routing", variant="primary")
                    routing_info = gr.Markdown(_module_routing_summary())
                    module_provider_components = [
                        provider_detector,
                        provider_landmarker,
                        provider_recognizer,
                        provider_classifier,
                        provider_masker,
                        provider_deep_swapper,
                        provider_face_swapper,
                        provider_face_enhancer,
                        provider_frame_enhancer,
                        provider_colorizer,
                        provider_expression,
                        provider_age,
                        provider_editor,
                        provider_lip,
                        provider_content,
                    ]
            out = gr.Image(label="Live preview", streaming=True)
            performance_info = gr.Markdown(
                "**Live performance:** Start the stream to measure processed/displayed FPS and latency."
            )
            diag = gr.Textbox(label="Diagnostics", lines=6)

            # CLI Preview tab
            with gr.Tab("CLI"):
                cli_box = gr.Textbox(label="CLI Command", lines=6, interactive=False, show_copy_button=True)

            def _build_cli_command(
                exec_provider_key, exec_device_id, video_mem_val,
                backend_name, camera_choice_v, res_v, width_v, height_v, fps_v,
                dshow_dev, dshow_name_text_v, convert_rgb_v, fourcc_v, retry_black_v, gentle_v, auto_repair_v, color_mode_v,
                lock_exp_v, exposure_v, lock_wb_v, wb_temp_v, show_overlay_v, debug_logs_v, show_boxes_v, show_native_v, virtual_cam_v,
                swap_mode_v, fs_model_v, fs_pixel_v, fs_weight_v, ds_model_v, morph_v,
                d_model_v, d_size_v, d_score_v, selector_mode_v, auto_fallback_v, l_model_v, l_score_v, o_model_v, p_model_v,
                use_occlusion_v,
                face_enh_en, face_enh_mod, face_enh_bl, face_enh_w,
                frame_enh_en, frame_enh_mod, frame_enh_bl, enhance_async_v,
                colorizer_en, colorizer_mod, colorizer_sz, colorizer_bl,
                expr_en, expr_mod, expr_factor_v, expr_areas_v,
                age_en, age_mod, age_dir_v,
                editor_en, editor_mod,
                fe_eyebrow_dir_v, fe_eye_h_v, fe_eye_v_v, fe_eye_open_v,
                fe_lip_open_v, fe_mouth_smile_v, fe_head_pitch_v,
                fe_head_yaw_v, fe_head_roll_v,
                dbg_en, dbg_items_v,
                lip_en, lip_mod, lip_weight_v,
                source_files_v,
            ):
                parts = ["python", os.path.basename(__file__)]
                # General/exec
                if exec_provider_key: parts += ["--ep", str(exec_provider_key)]
                if exec_device_id: parts += ["--device", str(exec_device_id)]
                if video_mem_val: parts += ["--video-mem", str(video_mem_val)]
                # Camera
                if backend_name: parts += ["--backend", f"{backend_name}"]
                if camera_choice_v and camera_choice_v.startswith("["):
                    try:
                        cam_index = int(camera_choice_v.split(']')[0][1:])
                        parts += ["--camera", str(cam_index)]
                    except Exception:
                        pass
                if res_v and res_v.lower() != "custom": parts += ["--resolution", res_v]
                if width_v: parts += ["--width", str(int(width_v))]
                if height_v: parts += ["--height", str(int(height_v))]
                if fps_v: parts += ["--fps", str(int(float(fps_v)))]
                if dshow_dev: parts += ["--dshow-name", f"{dshow_dev}"]
                if dshow_name_text_v: parts += ["--dshow-name", f"{dshow_name_text_v}"]
                if convert_rgb_v: parts.append("--convert-rgb")
                if fourcc_v and str(fourcc_v).lower() != "auto": parts += ["--fourcc", str(fourcc_v)]
                if retry_black_v: parts += ["--retry-black", str(int(retry_black_v))]
                if gentle_v: parts.append("--gentle")
                if auto_repair_v: parts.append("--auto-repair")
                if color_mode_v: parts += ["--color-mode", f"{color_mode_v}"]
                if lock_exp_v: parts.append("--lock-exposure")
                if exposure_v is not None: parts += ["--exposure", str(float(exposure_v))]
                if lock_wb_v: parts.append("--lock-wb")
                if wb_temp_v: parts += ["--wb-temp", str(int(wb_temp_v))]
                if show_overlay_v: parts.append("--show-overlay")
                if debug_logs_v: parts.append("--debug-logs")
                if show_boxes_v: parts.append("--show-boxes")
                if show_native_v: parts.append("--show-native")
                if virtual_cam_v: parts.append("--virtual-cam")
                # Swapping
                if swap_mode_v: parts += ["--swap-mode", swap_mode_v]
                if fs_model_v: parts += ["--face-swapper-model", fs_model_v]
                if fs_pixel_v: parts += ["--face-swapper-pixel", fs_pixel_v]
                if fs_weight_v is not None: parts += ["--face-swapper-weight", str(float(fs_weight_v))]
                if ds_model_v: parts += ["--model", ds_model_v]
                if morph_v is not None: parts += ["--morph", str(int(morph_v))]
                if source_files_v:
                    try:
                        for p in (source_files_v or []):
                            if isinstance(p, str) and p:
                                parts += ["--source", p]
                    except Exception:
                        pass
                # Detection
                if d_model_v: parts += ["--detector-model", d_model_v]
                if d_size_v: parts += ["--detector-size", d_size_v]
                if d_score_v is not None: parts += ["--detector-score", str(float(d_score_v))]
                if selector_mode_v: parts += ["--selector-mode", selector_mode_v]
                if auto_fallback_v: parts.append("--auto-fallback")
                if l_model_v: parts += ["--landmarker-model", l_model_v]
                if l_score_v is not None: parts += ["--landmarker-score", str(float(l_score_v))]
                if o_model_v: parts += ["--occluder-model", o_model_v]
                if p_model_v: parts += ["--parser-model", p_model_v]
                # Use occlusion defaults to true; add --no-occlusion when false
                if isinstance(use_occlusion_v, bool) and not use_occlusion_v:
                    parts.append("--no-occlusion")
                # Enhancing / processors
                if face_enh_en: parts.append("--face-enhancer")
                if face_enh_mod: parts += ["--face-enhancer-model", face_enh_mod]
                if face_enh_bl is not None: parts += ["--face-enhancer-blend", str(int(face_enh_bl))]
                if face_enh_w is not None: parts += ["--face-enhancer-weight", str(float(face_enh_w))]
                if frame_enh_en: parts.append("--frame-enhancer")
                if frame_enh_mod: parts += ["--frame-enhancer-model", frame_enh_mod]
                if frame_enh_bl is not None: parts += ["--frame-enhancer-blend", str(int(frame_enh_bl))]
                if enhance_async_v: parts.append("--enhance-async")
                if colorizer_en: parts.append("--colorizer")
                if colorizer_mod: parts += ["--colorizer-model", colorizer_mod]
                if colorizer_sz: parts += ["--colorizer-size", colorizer_sz]
                if colorizer_bl is not None: parts += ["--colorizer-blend", str(int(colorizer_bl))]
                if expr_en: parts.append("--expr")
                if expr_mod: parts += ["--expr-model", expr_mod]
                if expr_factor_v is not None: parts += ["--expr-factor", str(int(expr_factor_v))]
                if expr_areas_v: 
                    for a in (expr_areas_v or []):
                        parts += ["--expr-area", a]
                if age_en: parts.append("--age-mod")
                if age_mod: parts += ["--age-model", age_mod]
                if age_dir_v is not None: parts += ["--age-direction", str(int(age_dir_v))]
                if editor_en: parts.append("--editor")
                if editor_mod: parts += ["--editor-model", editor_mod]
                # Face editor sliders (emit only when non-zero to keep concise)
                def add_slider(flag, val):
                    try:
                        v = float(val)
                        if abs(v) > 1e-9:
                            parts += [flag, str(v)]
                    except Exception:
                        pass
                add_slider("--fe-eyebrow-dir", fe_eyebrow_dir_v)
                add_slider("--fe-eye-h", fe_eye_h_v)
                add_slider("--fe-eye-v", fe_eye_v_v)
                add_slider("--fe-eye-open", fe_eye_open_v)
                add_slider("--fe-lip-open", fe_lip_open_v)
                add_slider("--fe-smile", fe_mouth_smile_v)
                add_slider("--fe-head-pitch", fe_head_pitch_v)
                add_slider("--fe-head-yaw", fe_head_yaw_v)
                add_slider("--fe-head-roll", fe_head_roll_v)
                if dbg_en: parts.append("--face-debugger")
                if dbg_items_v:
                    for it in (dbg_items_v or []):
                        parts += ["--face-debugger-item", it]
                if lip_en: parts.append("--lip-syncer")
                if lip_mod: parts += ["--lip-model", lip_mod]
                if lip_weight_v is not None: parts += ["--lip-weight", str(float(lip_weight_v))]
                return " ".join(parts)

            # Wire CLI preview updates
            cli_inputs = [
                exec_provider, exec_device, video_mem,
                backend, camera, res, width, height, fps,
                dshow_name_drop, dshow_name, convert_rgb, force_fourcc, retry_black, gentle_mode, auto_repair, color_mode,
                lock_exposure, exposure_value, lock_wb, wb_temperature, show_overlay, debug_logs, show_boxes, show_native, virtual_cam,
                swap_mode, fs_model, fs_pixel, fs_weight, ds_model, morph,
                d_model, d_size, d_score, selector_mode_dd, auto_fallback, l_model, l_score, o_model, p_model,
                occl_enabled,
                face_enh_enabled, face_enh_model, face_enh_blend, face_enh_weight,
                frame_enh_enabled, frame_enh_model, frame_enh_blend, enhance_async,
                colorizer_enabled, colorizer_model, colorizer_size, colorizer_blend,
                expr_enabled, expr_model, expr_factor, expr_areas,
                age_enabled, age_model, age_direction,
                editor_enabled, editor_model,
                fe_eyebrow_dir, fe_eye_h, fe_eye_v, fe_eye_open,
                fe_lip_open, fe_mouth_smile, fe_head_pitch,
                fe_head_yaw, fe_head_roll,
                debugger_enabled, dbg_items,
                lip_enabled, lip_model, lip_weight,
                source_files,
            ]
            for comp in cli_inputs:
                try:
                    comp.change(
                        _build_cli_command,
                        inputs=cli_inputs,
                        outputs=cli_box,
                    )
                except Exception:
                    pass
            # Initialize CLI preview on load
            demo.load(_build_cli_command, inputs=cli_inputs, outputs=cli_box)

            def on_detector_change(model):
                sizes = detector_sizes_map.get(model, ["640x640"])
                return gr.update(choices=sizes, value=sizes[0])

            d_model.change(on_detector_change, inputs=d_model, outputs=d_size)

            def _persist_camera_mode(
                camera_choice: str,
                camera_name: str,
                resolution: str,
                selected_width: int,
                selected_height: int,
                selected_fps: float,
            ) -> None:
                values = {
                    'backend': 'DirectShow' if os.name == 'nt' else 'Auto',
                    'camera_choice': camera_choice,
                    'dshow_name_device': camera_name,
                    'resolution_preset': resolution,
                    'width': int(selected_width),
                    'height': int(selected_height),
                    'fps': float(selected_fps),
                }
                prefs = _load_prefs().copy()
                prefs.update(values)
                _save_prefs(prefs)
                for key, value in values.items():
                    try:
                        state_manager.set_item(key, value)
                    except Exception:
                        pass

            def _camera_mode_updates(camera_choice, current_resolution, current_fps, force_refresh=False):
                camera_name = _camera_name_from_choice(camera_choice)
                detected_modes = detect_camera_capabilities(camera_name, force_refresh=force_refresh)
                modes = detected_modes or dict(_fallback_camera_capabilities)
                resolutions = _sorted_camera_resolutions(modes)
                selected_resolution = current_resolution if current_resolution in modes else resolutions[-1]
                selected_width, selected_height = (
                    int(part) for part in selected_resolution.split("x", 1)
                )
                maximum_fps = max(1.0, float(modes[selected_resolution]))
                try:
                    selected_fps = min(max(1.0, float(current_fps)), maximum_fps)
                except Exception:
                    selected_fps = min(30.0, maximum_fps)
                names = _ffmpeg_camera_names()
                _persist_camera_mode(
                    camera_choice,
                    camera_name,
                    selected_resolution,
                    selected_width,
                    selected_height,
                    selected_fps,
                )
                return (
                    gr.update(choices=resolutions, value=selected_resolution),
                    gr.update(value=selected_width),
                    gr.update(value=selected_height),
                    gr.update(
                        minimum=1,
                        maximum=maximum_fps,
                        value=selected_fps,
                        label=f"Camera FPS (detected max {maximum_fps:g})",
                    ),
                    gr.update(choices=names, value=camera_name if camera_name in names else None),
                    _camera_mode_status(camera_name, detected_modes, selected_resolution),
                )

            def on_camera_change(camera_choice, current_resolution, current_fps):
                return _camera_mode_updates(camera_choice, current_resolution, current_fps)

            def on_detect_modes(camera_choice, current_resolution, current_fps):
                return _camera_mode_updates(camera_choice, current_resolution, current_fps, force_refresh=True)

            def on_refresh(current_camera, current_resolution, current_fps):
                _camera_capability_cache.clear()
                cams = get_camera_choices(10, backend_name="DirectShow")
                if not cams:
                    cams = ["[0] Camera 0"]
                current_name = _camera_name_from_choice(current_camera)
                selected_camera = next(
                    (choice for choice in cams if _camera_name_from_choice(choice) == current_name),
                    cams[0],
                )
                mode_updates = _camera_mode_updates(
                    selected_camera,
                    current_resolution,
                    current_fps,
                    force_refresh=True,
                )
                display_choices = [(_camera_name_from_choice(choice), choice) for choice in cams]
                return (gr.update(choices=display_choices, value=selected_camera), *mode_updates)

            def on_res_change(preset, camera_choice, current_fps):
                # End the current infinite generator in the same event that
                # changes the hidden width/height values. This lets the chained
                # stream start immediately instead of waiting behind the old
                # camera mode.
                gr_stop_stream()
                camera_name = _camera_name_from_choice(camera_choice)
                detected_modes = detect_camera_capabilities(camera_name)
                modes = detected_modes or dict(_fallback_camera_capabilities)
                if preset not in modes:
                    preset = _sorted_camera_resolutions(modes)[-1]
                selected_width, selected_height = (int(part) for part in preset.split("x", 1))
                maximum_fps = max(1.0, float(modes[preset]))
                try:
                    selected_fps = min(max(1.0, float(current_fps)), maximum_fps)
                except Exception:
                    selected_fps = min(30.0, maximum_fps)
                _persist_camera_mode(
                    camera_choice,
                    camera_name,
                    preset,
                    selected_width,
                    selected_height,
                    selected_fps,
                )
                return (
                    gr.update(value=selected_width),
                    gr.update(value=selected_height),
                    gr.update(
                        minimum=1,
                        maximum=maximum_fps,
                        value=selected_fps,
                        label=f"Camera FPS (detected max {maximum_fps:g})",
                    ),
                    _camera_mode_status(camera_name, detected_modes, preset),
                )

            camera.change(
                on_camera_change,
                inputs=[camera, res, fps],
                outputs=[res, width, height, fps, dshow_name_drop, camera_mode_info],
            )
            detect_modes_btn.click(
                on_detect_modes,
                inputs=[camera, res, fps],
                outputs=[res, width, height, fps, dshow_name_drop, camera_mode_info],
            )
            refresh_btn.click(
                on_refresh,
                inputs=[camera, res, fps],
                outputs=[camera, res, width, height, fps, dshow_name_drop, camera_mode_info],
            )
            resolution_change_event = res.change(
                on_res_change,
                inputs=[res, camera, fps],
                outputs=[width, height, fps, camera_mode_info],
            )

            # Live-change handlers (apply immediately while streaming)
            def on_detector_model_change(model: str):
                try:
                    sizes = detector_sizes_map.get(model, ["640x640"]) or ["640x640"]
                    state_manager.set_item("face_detector_model", model)
                    state_manager.set_item("face_detector_size", sizes[0])
                    _cleanup_inference()
                    return gr.update(choices=sizes, value=sizes[0])
                except Exception:
                    return gr.update()

            def on_detector_size_change(size: str):
                try:
                    state_manager.set_item("face_detector_size", size)
                    _cleanup_inference()
                except Exception:
                    pass

            def on_detector_score_change(score: float):
                try:
                    state_manager.set_item("face_detector_score", float(score))
                except Exception:
                    pass

            def on_landmarker_model_change(model: str):
                try:
                    state_manager.set_item("face_landmarker_model", model)
                    _cleanup_inference()
                except Exception:
                    pass

            def on_landmarker_score_change(score: float):
                try:
                    state_manager.set_item("face_landmarker_score", float(score))
                except Exception:
                    pass

            def on_selector_mode_change(mode: str):
                try:
                    state_manager.set_item("face_selector_mode", mode)
                except Exception:
                    pass

            def on_deep_model_change(model: str):
                try:
                    state_manager.set_item("deep_swapper_model", model)
                    # ensure files present and rebuild inference
                    try:
                        deep_swapper.pre_check()
                    except Exception:
                        pass
                    _cleanup_inference()
                except Exception:
                    pass

            def on_face_swapper_model_change(model: str):
                try:
                    state_manager.set_item("face_swapper_model", model)
                    if model:
                        try:
                            # clear pool to force reload
                            ff_face_swapper.clear_inference_pool()
                        except Exception:
                            pass
                    _cleanup_inference()
                except Exception:
                    pass

            def on_morph_change(val: int):
                try:
                    state_manager.set_item("deep_swapper_morph", int(val))
                except Exception:
                    pass

            def on_exec_provider_change(key: str):
                try:
                    ep_key = key if key in ff_choices.execution_providers else 'cpu'
                    state_manager.set_item("execution_providers", [ep_key])
                    _cleanup_inference()
                except Exception:
                    pass

            def on_exec_device_change(dev_id: str):
                try:
                    state_manager.set_item("execution_device_ids", [str(dev_id)])
                    _cleanup_inference()
                except Exception:
                    pass

            def on_face_enh_weight_change(v: float):
                try:
                    state_manager.set_item("face_enhancer_weight", float(v))
                except Exception:
                    pass

            def on_enhance_async_change(flag: bool):
                try:
                    state_manager.set_item("enhance_async", bool(flag))
                except Exception:
                    pass

            def on_face_enh_toggle(flag: bool):
                try:
                    state_manager.set_item("face_enhancer_enabled", bool(flag))
                    if flag:
                        try:
                            ff_face_enhancer.pre_check()
                        except Exception:
                            pass
                except Exception:
                    pass

            def on_frame_enh_toggle(flag: bool):
                try:
                    state_manager.set_item("frame_enhancer_enabled", bool(flag))
                    if flag:
                        try:
                            ff_frame_enhancer.pre_check()
                        except Exception:
                            pass
                except Exception:
                    pass

            face_enh_enabled.change(on_face_enh_toggle, inputs=face_enh_enabled, outputs=[])
            frame_enh_enabled.change(on_frame_enh_toggle, inputs=frame_enh_enabled, outputs=[])
            enhance_async.change(on_enhance_async_change, inputs=enhance_async, outputs=[])

            # Processor option live handlers
            def on_colorizer_model_change(v):
                try:
                    state_manager.set_item('frame_colorizer_model', v)
                    ff_frame_colorizer.pre_check()
                except Exception:
                    pass
            def on_colorizer_size_change(v):
                try:
                    state_manager.set_item('frame_colorizer_size', v)
                except Exception:
                    pass
            def on_colorizer_blend_change(v):
                try:
                    state_manager.set_item('frame_colorizer_blend', int(v))
                except Exception:
                    pass
            colorizer_model.change(on_colorizer_model_change, inputs=colorizer_model, outputs=[])
            colorizer_size.change(on_colorizer_size_change, inputs=colorizer_size, outputs=[])
            colorizer_blend.change(on_colorizer_blend_change, inputs=colorizer_blend, outputs=[])

            def on_colorizer_toggle(flag: bool):
                try:
                    state_manager.set_item('frame_colorizer_enabled', bool(flag))
                    if flag:
                        try:
                            ff_frame_colorizer.pre_check()
                        except Exception:
                            pass
                except Exception:
                    pass
            colorizer_enabled.change(on_colorizer_toggle, inputs=colorizer_enabled, outputs=[])

            def on_expr_model_change(v):
                try:
                    state_manager.set_item('expression_restorer_model', v)
                    ff_expr_restorer.pre_check()
                except Exception:
                    pass
            def on_expr_factor_change(v):
                try:
                    state_manager.set_item('expression_restorer_factor', int(v))
                except Exception:
                    pass
            def on_expr_areas_change(v):
                try:
                    state_manager.set_item('expression_restorer_areas', v)
                except Exception:
                    pass
            expr_model.change(on_expr_model_change, inputs=expr_model, outputs=[])
            expr_factor.change(on_expr_factor_change, inputs=expr_factor, outputs=[])
            expr_areas.change(on_expr_areas_change, inputs=expr_areas, outputs=[])

            def on_expr_toggle(flag: bool):
                try:
                    state_manager.set_item('expression_restorer_enabled', bool(flag))
                    if flag:
                        try:
                            ff_expr_restorer.pre_check()
                        except Exception:
                            pass
                except Exception:
                    pass
            expr_enabled.change(on_expr_toggle, inputs=expr_enabled, outputs=[])

            def on_age_model_change(v):
                try:
                    state_manager.set_item('age_modifier_model', v)
                    ff_age_modifier.pre_check()
                except Exception:
                    pass
            def on_age_direction_change(v):
                try:
                    state_manager.set_item('age_modifier_direction', int(v))
                except Exception:
                    pass
            age_model.change(on_age_model_change, inputs=age_model, outputs=[])
            age_direction.change(on_age_direction_change, inputs=age_direction, outputs=[])

            def on_age_toggle(flag: bool):
                try:
                    state_manager.set_item('age_modifier_enabled', bool(flag))
                    if flag:
                        try:
                            ff_age_modifier.pre_check()
                        except Exception:
                            pass
                except Exception:
                    pass
            age_enabled.change(on_age_toggle, inputs=age_enabled, outputs=[])

            def on_editor_model_change(v):
                try:
                    state_manager.set_item('face_editor_model', v)
                    ff_face_editor.pre_check()
                except Exception:
                    pass
            editor_model.change(on_editor_model_change, inputs=editor_model, outputs=[])
            def _set_float(key, v):
                try:
                    state_manager.set_item(key, float(v))
                except Exception:
                    pass
            fe_eyebrow_dir.change(lambda v: _set_float('face_editor_eyebrow_direction', v), inputs=fe_eyebrow_dir, outputs=[])
            fe_eye_h.change(lambda v: _set_float('face_editor_eye_gaze_horizontal', v), inputs=fe_eye_h, outputs=[])
            fe_eye_v.change(lambda v: _set_float('face_editor_eye_gaze_vertical', v), inputs=fe_eye_v, outputs=[])
            fe_eye_open.change(lambda v: _set_float('face_editor_eye_open_ratio', v), inputs=fe_eye_open, outputs=[])
            fe_lip_open.change(lambda v: _set_float('face_editor_lip_open_ratio', v), inputs=fe_lip_open, outputs=[])
            fe_mouth_smile.change(lambda v: _set_float('face_editor_mouth_smile', v), inputs=fe_mouth_smile, outputs=[])
            fe_mouth_grim.change(lambda v: _set_float('face_editor_mouth_grim', v), inputs=fe_mouth_grim, outputs=[])
            fe_mouth_pout.change(lambda v: _set_float('face_editor_mouth_pout', v), inputs=fe_mouth_pout, outputs=[])
            fe_mouth_purse.change(lambda v: _set_float('face_editor_mouth_purse', v), inputs=fe_mouth_purse, outputs=[])
            fe_mouth_pos_h.change(lambda v: _set_float('face_editor_mouth_position_horizontal', v), inputs=fe_mouth_pos_h, outputs=[])
            fe_mouth_pos_v.change(lambda v: _set_float('face_editor_mouth_position_vertical', v), inputs=fe_mouth_pos_v, outputs=[])
            fe_head_pitch.change(lambda v: _set_float('face_editor_head_pitch', v), inputs=fe_head_pitch, outputs=[])
            fe_head_yaw.change(lambda v: _set_float('face_editor_head_yaw', v), inputs=fe_head_yaw, outputs=[])
            fe_head_roll.change(lambda v: _set_float('face_editor_head_roll', v), inputs=fe_head_roll, outputs=[])

            def on_dbg_items_change(v):
                try:
                    state_manager.set_item('face_debugger_items', v)
                except Exception:
                    pass
            dbg_items.change(on_dbg_items_change, inputs=dbg_items, outputs=[])

            def on_debugger_toggle(flag: bool):
                try:
                    state_manager.set_item('face_debugger_enabled', bool(flag))
                except Exception:
                    pass
            debugger_enabled.change(on_debugger_toggle, inputs=debugger_enabled, outputs=[])

            def on_lip_model_change(v):
                try:
                    state_manager.set_item('lip_syncer_model', v)
                    ff_lip_syncer.pre_check()
                except Exception:
                    pass
            def on_lip_weight_change(v):
                try:
                    state_manager.set_item('lip_syncer_weight', float(v))
                except Exception:
                    pass
            lip_model.change(on_lip_model_change, inputs=lip_model, outputs=[])
            lip_weight.change(on_lip_weight_change, inputs=lip_weight, outputs=[])

            def on_lip_toggle(flag: bool):
                try:
                    state_manager.set_item('lip_syncer_enabled', bool(flag))
                    if flag:
                        try:
                            ff_lip_syncer.pre_check()
                        except Exception:
                            pass
                except Exception:
                    pass
            lip_enabled.change(on_lip_toggle, inputs=lip_enabled, outputs=[])

            def on_video_mem_change(v: str):
                try:
                    if v in ("strict","moderate","relaxed"):
                        state_manager.set_item('video_memory_strategy', v)
                        _cleanup_inference()
                        _persist_state()
                except Exception:
                    pass
            video_mem.change(on_video_mem_change, inputs=video_mem, outputs=[])

            def apply_execution_routing(*providers):
                gr_stop_stream()
                # Let an in-flight frame release its sessions before replacing
                # all inference pools with the newly selected providers.
                time.sleep(0.35)
                available = _available_execution_provider_keys()
                for (state_key, _label, _module), provider in zip(_MODULE_PROVIDER_ROUTES, providers):
                    selected = provider if provider in available else _normalize_module_provider(None)
                    state_manager.set_item(state_key, selected)
                _persist_state()
                _cleanup_inference()
                return _module_routing_summary() + "\n\nRouting applied. Press **Start** to resume."

            apply_routing_btn.click(
                apply_execution_routing,
                inputs=module_provider_components,
                outputs=routing_info,
            )

            # Live toggles for key flags used per-frame

            def on_occl_enabled_change(flag: bool):
                try:
                    state_manager.set_item('use_occlusion', bool(flag))
                    mask_types = ['box']
                    if flag:
                        mask_types.append('occlusion')
                    state_manager.set_item('face_mask_types', mask_types)
                    _persist_state()
                except Exception:
                    pass
            occl_enabled.change(on_occl_enabled_change, inputs=occl_enabled, outputs=[])

            def on_show_boxes_change(flag: bool):
                try:
                    state_manager.set_item('show_boxes', bool(flag))
                    _persist_state()
                except Exception:
                    pass
            show_boxes.change(on_show_boxes_change, inputs=show_boxes, outputs=[])

            def on_show_overlay_change(flag: bool):
                try:
                    state_manager.set_item('show_overlay', bool(flag))
                    _persist_state()
                except Exception:
                    pass
            show_overlay.change(on_show_overlay_change, inputs=show_overlay, outputs=[])

            def on_debug_logs_change(flag: bool):
                try:
                    state_manager.set_item('debug_logs', bool(flag))
                    _persist_state()
                except Exception:
                    pass
            debug_logs.change(on_debug_logs_change, inputs=debug_logs, outputs=[])

            # Additional live-change handlers for swapper/enhancers
            def on_swap_mode_change(mode: str):
                try:
                    state_manager.set_item("swap_mode", mode)
                    return gr.update()
                except Exception:
                    return gr.update()

            swap_mode.change(on_swap_mode_change, inputs=swap_mode, outputs=[])
            # Persist swap mode immediately
            def _persist_swap_mode(mode: str):
                try:
                    state_manager.set_item('swap_mode', mode)
                    _persist_state()
                except Exception:
                    pass
            swap_mode.change(_persist_swap_mode, inputs=swap_mode, outputs=[])

            def on_source_files_change(files):
                try:
                    paths = [f.name if hasattr(f, 'name') else f for f in (files or [])]
                    state_manager.set_item('source_paths', paths)
                    # cache to numpy
                    global _source_imgs
                    _source_imgs = []
                    for p in paths:
                        try:
                            img = cv2.imread(p)
                            if isinstance(img, np.ndarray) and getattr(img, 'size', 0) > 0:
                                _source_imgs.append(img)
                        except Exception:
                            continue
                    _persist_state()
                except Exception:
                    pass
            source_files.change(on_source_files_change, inputs=source_files, outputs=[])

            # Fast startup handler
            def on_fast_startup_toggle(flag: bool):
                try:
                    state_manager.set_item('fast_startup', bool(flag))
                    _persist_state()
                except Exception:
                    pass
            fast_startup.change(on_fast_startup_toggle, inputs=fast_startup, outputs=[])

            # Face Editor enable/disable in real time
            def on_editor_toggle(flag: bool):
                try:
                    state_manager.set_item('face_editor_enabled', bool(flag))
                    # Optional: warm up when enabling (unless fast startup)
                    if flag and not state_manager.get_item('fast_startup'):
                        try:
                            if ff_face_editor.pre_check():
                                state_manager.set_item('face_editor_ready', True)
                        except Exception:
                            pass
                    _persist_state()
                except Exception:
                    pass
            editor_enabled.change(on_editor_toggle, inputs=editor_enabled, outputs=[])

            # Camera preference handlers (persist on change)
            def _set_and_persist(key, val):
                try:
                    state_manager.set_item(key, val)
                    _persist_state()
                except Exception:
                    pass

            backend.change(lambda v: _set_and_persist('backend', v), inputs=backend, outputs=[])
            camera.change(lambda v: _set_and_persist('camera_choice', v), inputs=camera, outputs=[])
            res.change(lambda v: _set_and_persist('resolution_preset', v), inputs=res, outputs=[])
            width.change(lambda v: _set_and_persist('width', int(v)), inputs=width, outputs=[])
            height.change(lambda v: _set_and_persist('height', int(v)), inputs=height, outputs=[])
            fps.change(lambda v: _set_and_persist('fps', float(v)), inputs=fps, outputs=[])
            dshow_name_drop.change(lambda v: _set_and_persist('dshow_name_mode', v), inputs=dshow_name_drop, outputs=[])
            dshow_name.change(lambda v: _set_and_persist('dshow_name_text', v), inputs=dshow_name, outputs=[])
            dshow_name_drop.change(lambda v: _set_and_persist('dshow_name_device', v), inputs=dshow_name_drop, outputs=[])
            convert_rgb.change(lambda v: _set_and_persist('convert_rgb', bool(v)), inputs=convert_rgb, outputs=[])
            force_fourcc.change(lambda v: _set_and_persist('force_fourcc', v), inputs=force_fourcc, outputs=[])
            retry_black.change(lambda v: _set_and_persist('retry_black', int(v)), inputs=retry_black, outputs=[])
            gentle_mode.change(lambda v: _set_and_persist('gentle_mode', bool(v)), inputs=gentle_mode, outputs=[])
            auto_repair.change(lambda v: _set_and_persist('auto_repair', bool(v)), inputs=auto_repair, outputs=[])
            color_mode.change(lambda v: _set_and_persist('color_mode', v), inputs=color_mode, outputs=[])
            lock_exposure.change(lambda v: _set_and_persist('lock_exposure', bool(v)), inputs=lock_exposure, outputs=[])
            exposure_value.change(lambda v: _set_and_persist('exposure_value', float(v)), inputs=exposure_value, outputs=[])
            lock_wb.change(lambda v: _set_and_persist('lock_wb', bool(v)), inputs=lock_wb, outputs=[])
            wb_temperature.change(lambda v: _set_and_persist('wb_temperature', int(v)), inputs=wb_temperature, outputs=[])
            show_native.change(lambda v: _set_and_persist('show_native', bool(v)), inputs=show_native, outputs=[])
            virtual_cam.change(lambda v: _set_and_persist('virtual_cam_enabled', bool(v)), inputs=virtual_cam, outputs=[])
            fast_live_analysis.change(
                lambda v: _set_and_persist('realtime_fast_analysis', bool(v)),
                inputs=fast_live_analysis,
                outputs=[],
            )

            # Enhancing and Processor persistence handlers
            face_enh_enabled.change(lambda v: _set_and_persist('face_enhancer_enabled', bool(v)), inputs=face_enh_enabled, outputs=[])
            face_enh_model.change(lambda v: _set_and_persist('face_enhancer_model', v), inputs=face_enh_model, outputs=[])
            face_enh_blend.change(lambda v: _set_and_persist('face_enhancer_blend', int(v)), inputs=face_enh_blend, outputs=[])
            face_enh_weight.change(lambda v: _set_and_persist('face_enhancer_weight', float(v)), inputs=face_enh_weight, outputs=[])
            frame_enh_enabled.change(lambda v: _set_and_persist('frame_enhancer_enabled', bool(v)), inputs=frame_enh_enabled, outputs=[])
            frame_enh_model.change(lambda v: _set_and_persist('frame_enhancer_model', v), inputs=frame_enh_model, outputs=[])
            frame_enh_blend.change(lambda v: _set_and_persist('frame_enhancer_blend', int(v)), inputs=frame_enh_blend, outputs=[])
            enhance_async.change(lambda v: _set_and_persist('enhance_async', bool(v)), inputs=enhance_async, outputs=[])

            colorizer_enabled.change(lambda v: _set_and_persist('frame_colorizer_enabled', bool(v)), inputs=colorizer_enabled, outputs=[])
            colorizer_model.change(lambda v: _set_and_persist('frame_colorizer_model', v), inputs=colorizer_model, outputs=[])
            colorizer_size.change(lambda v: _set_and_persist('frame_colorizer_size', v), inputs=colorizer_size, outputs=[])
            colorizer_blend.change(lambda v: _set_and_persist('frame_colorizer_blend', int(v)), inputs=colorizer_blend, outputs=[])

            expr_enabled.change(lambda v: _set_and_persist('expression_restorer_enabled', bool(v)), inputs=expr_enabled, outputs=[])
            expr_model.change(lambda v: _set_and_persist('expression_restorer_model', v), inputs=expr_model, outputs=[])
            expr_factor.change(lambda v: _set_and_persist('expression_restorer_factor', int(v)), inputs=expr_factor, outputs=[])
            expr_areas.change(lambda v: _set_and_persist('expression_restorer_areas', v), inputs=expr_areas, outputs=[])

            debugger_enabled.change(lambda v: _set_and_persist('face_debugger_enabled', bool(v)), inputs=debugger_enabled, outputs=[])
            dbg_items.change(lambda v: _set_and_persist('face_debugger_items', v), inputs=dbg_items, outputs=[])

            # Execution provider/device persistence
            def _persist_exec_provider(v: str):
                try:
                    state_manager.set_item('execution_providers', [v])
                    _persist_state()
                except Exception:
                    pass
            exec_provider.change(_persist_exec_provider, inputs=exec_provider, outputs=[])

            def _persist_exec_device(v: str):
                try:
                    state_manager.set_item('execution_device_ids', [str(v)])
                    _persist_state()
                except Exception:
                    pass
            exec_device.change(_persist_exec_device, inputs=exec_device, outputs=[])

            # Detection tab persistence
            d_model.change(lambda v: _set_and_persist('detector_model', v), inputs=d_model, outputs=[])
            d_size.change(lambda v: _set_and_persist('detector_size', v), inputs=d_size, outputs=[])
            d_score.change(lambda v: _set_and_persist('detector_score', float(v)), inputs=d_score, outputs=[])
            selector_mode_dd.change(lambda v: _set_and_persist('selector_mode', v), inputs=selector_mode_dd, outputs=[])
            auto_fallback.change(lambda v: _set_and_persist('auto_fallback', bool(v)), inputs=auto_fallback, outputs=[])
            l_model.change(lambda v: _set_and_persist('landmarker_model', v), inputs=l_model, outputs=[])
            l_score.change(lambda v: _set_and_persist('landmarker_score', float(v)), inputs=l_score, outputs=[])

            # Face Swapper Model persistence & cleanup
            fs_model.change(on_face_swapper_model_change, inputs=fs_model, outputs=[])
            fs_model.change(lambda v: _set_and_persist('face_swapper_model', v), inputs=fs_model, outputs=[])
            fs_pixel.change(lambda v: _set_and_persist('face_swapper_pixel_boost', v), inputs=fs_pixel, outputs=[])
            fs_weight.change(lambda v: _set_and_persist('face_swapper_weight', float(v)), inputs=fs_weight, outputs=[])
            ds_model.change(lambda v: _set_and_persist('deep_swapper_model', v), inputs=ds_model, outputs=[])
            morph.change(lambda v: _set_and_persist('morph', int(v)), inputs=morph, outputs=[])
            o_model.change(lambda v: _set_and_persist('occluder_model', v), inputs=o_model, outputs=[])
            p_model.change(lambda v: _set_and_persist('parser_model', v), inputs=p_model, outputs=[])

            # Face swapper live handlers
            def on_face_swapper_model_change(model: str):
                try:
                    state_manager.set_item("face_swapper_model", model)
                    choices = proc_choices.face_swapper_set.get(model, ["256x256"]) or ["256x256"]
                    _cleanup_inference()
                    _persist_state()
                    return gr.update(choices=choices, value=choices[0])
                except Exception:
                    return gr.update()
            fs_model.change(on_face_swapper_model_change, inputs=fs_model, outputs=fs_pixel)

            def on_face_swapper_pixel_change(pixel: str):
                try:
                    state_manager.set_item("face_swapper_pixel_boost", pixel)
                    _persist_state()
                except Exception:
                    pass
            fs_pixel.change(on_face_swapper_pixel_change, inputs=fs_pixel, outputs=[])

            def on_face_swapper_weight_change(w: float):
                try:
                    state_manager.set_item("face_swapper_weight", float(w))
                    _persist_state()
                except Exception:
                    pass
            fs_weight.change(on_face_swapper_weight_change, inputs=fs_weight, outputs=[])

            def on_face_swapper_model_change(model: str):
                try:
                    state_manager.set_item("face_swapper_model", model)
                    choices = proc_choices.face_swapper_set.get(model, ["256x256"]) or ["256x256"]
                    _cleanup_inference()
                    return gr.update(choices=choices, value=choices[0])
                except Exception:
                    return gr.update()

            def on_face_swapper_pixel_change(px: str):
                try:
                    state_manager.set_item("face_swapper_pixel_boost", px)
                    _cleanup_inference()
                except Exception:
                    pass

            def on_face_swapper_weight_change(w: float):
                try:
                    state_manager.set_item("face_swapper_weight", float(w))
                except Exception:
                    pass

            def on_face_enh_toggle(flag: bool):
                try:
                    state_manager.set_item("face_enhancer_enabled", bool(flag))
                except Exception:
                    pass

            def on_face_enh_model_change(model: str):
                try:
                    state_manager.set_item("face_enhancer_model", model)
                    _cleanup_inference()
                except Exception:
                    pass

            def on_face_enh_blend_change(v: int):
                try:
                    state_manager.set_item("face_enhancer_blend", int(v))
                except Exception:
                    pass

            def on_frame_enh_toggle(flag: bool):
                try:
                    state_manager.set_item("frame_enhancer_enabled", bool(flag))
                except Exception:
                    pass

            def on_frame_enh_model_change(model: str):
                try:
                    state_manager.set_item("frame_enhancer_model", model)
                    _cleanup_inference()
                except Exception:
                    pass

            def on_frame_enh_blend_change(v: int):
                try:
                    state_manager.set_item("frame_enhancer_blend", int(v))
                except Exception:
                    pass

            d_model.change(on_detector_model_change, inputs=d_model, outputs=d_size)
            d_size.change(on_detector_size_change, inputs=d_size, outputs=[])
            d_score.change(on_detector_score_change, inputs=d_score, outputs=[])
            l_model.change(on_landmarker_model_change, inputs=l_model, outputs=[])
            l_score.change(on_landmarker_score_change, inputs=l_score, outputs=[])
            selector_mode_dd.change(on_selector_mode_change, inputs=selector_mode_dd, outputs=[])
            ds_model.change(on_deep_model_change, inputs=ds_model, outputs=[])

            # VRAM free action
            def gr_free_vram():
                try:
                    _cleanup_inference()
                    return "Freed inference pools and requested GC."
                except Exception as e:
                    return f"Cleanup error: {e}"

            stream_inputs = [
                camera, width, height,
                occl_enabled, fps, backend, dshow_name_drop, convert_rgb, force_fourcc, retry_black, gentle_mode, auto_repair, color_mode,
                lock_exposure, exposure_value, lock_wb, wb_temperature,
                show_overlay, debug_logs, show_boxes,
                show_native, virtual_cam,
                colorizer_enabled, expr_enabled, age_enabled, editor_enabled, debugger_enabled, lip_enabled,
                selector_mode_dd, auto_fallback,
                exec_provider, exec_device, video_mem,
                d_model, d_size, d_score, l_model, l_score, o_model, p_model,
                swap_mode, source_files, ds_model, morph,
                fs_model, fs_pixel, fs_weight,
                face_enh_enabled, face_enh_model, face_enh_blend, face_enh_weight,
                frame_enh_enabled, frame_enh_model, frame_enh_blend,
                enhance_async,
            ]
            start_event = start_btn.click(
                fn=gr_stream,
                inputs=stream_inputs,
                outputs=[out, performance_info],
            )

            def _prepare_quick_resolution_switch():
                # Let the old reader finish its current frame before the new
                # mode reuses the camera handle.
                time.sleep(0.25)
                return "**Switching camera resolution...**"

            resolution_change_event.then(
                fn=_prepare_quick_resolution_switch,
                inputs=None,
                outputs=performance_info,
            ).then(
                fn=gr_stream,
                inputs=stream_inputs,
                outputs=[out, performance_info],
            )
            stop_btn.click(gr_stop_stream, inputs=None, outputs=out)
            benchmark_btn.click(request_realtime_benchmark, inputs=None, outputs=performance_info)
            test_btn.click(
                fn=test_capture,
                inputs=[camera, backend, dshow_name_drop, width, height, fps, convert_rgb, force_fourcc],
                outputs=[out, diag],
            )
            reconnect_btn.click(
                fn=gr_reconnect,
                inputs=[camera, backend, dshow_name_drop, width, height, fps, convert_rgb, force_fourcc, gentle_mode],
                outputs=[out, diag],
            )
            gr.Button("Free VRAM").click(gr_free_vram, inputs=None, outputs=diag)
            shutdown_btn.click(gr_shutdown, inputs=None, outputs=out)

            # Auto-start on UI load
            def _gr_autostart(
                camera_choice, width, height, occl_enabled, fps, backend, dshow_name_drop, convert_rgb,
                force_fourcc, retry_black, gentle_mode, auto_repair, color_mode, lock_exposure, exposure_value, lock_wb,
                wb_temperature, show_overlay, debug_logs, show_boxes, show_native_flag, virtual_cam_flag, fast_startup_flag,
                colorizer_flag, expr_flag, age_flag, editor_flag, debugger_flag, lip_flag,
                colorizer_model_v, colorizer_size_v, colorizer_blend_v,
                expr_model_v, expr_factor_v, expr_areas_v,
                age_model_v, age_direction_v,
                editor_model_v, fe_eyebrow_dir_v, fe_eye_h_v, fe_eye_v_v, fe_eye_open_v, fe_lip_open_v, fe_mouth_smile_v, fe_head_pitch_v, fe_head_yaw_v, fe_head_roll_v,
                dbg_items_v,
                lip_model_v, lip_weight_v,
                selector_mode_dd, auto_fallback, exec_provider,
                exec_device, video_mem,
                d_model, d_size, d_score, l_model, l_score, o_model, p_model, swap_mode, source_files,
                ds_model, morph, fs_model, fs_pixel, fs_weight, face_enh_enabled, face_enh_model, face_enh_blend,
                face_enh_weight, frame_enh_enabled, frame_enh_model, frame_enh_blend, enhance_async, auto_start_flag
            ):
                if not auto_start_flag:
                    return None
                try:
                    state_manager.set_item('fast_startup', bool(fast_startup_flag))
                except Exception:
                    pass
                # Load persisted source_paths into cache so face swapper can use them without reselect
                try:
                    paths = (_load_prefs() or {}).get('source_paths') or []
                    if paths:
                        state_manager.set_item('source_paths', paths)
                        global _source_imgs
                        _source_imgs = []
                        for p in paths:
                            try:
                                img = cv2.imread(p)
                                if isinstance(img, np.ndarray) and getattr(img, 'size', 0) > 0:
                                    _source_imgs.append(img)
                            except Exception:
                                continue
                except Exception:
                    pass
                # Apply initial defaults for processors so they work before any change
                try:
                    state_manager.set_item('frame_colorizer_model', colorizer_model_v)
                    state_manager.set_item('frame_colorizer_size', colorizer_size_v)
                    state_manager.set_item('frame_colorizer_blend', int(colorizer_blend_v))
                except Exception:
                    pass
                try:
                    state_manager.set_item('expression_restorer_model', expr_model_v)
                    state_manager.set_item('expression_restorer_factor', int(expr_factor_v))
                    state_manager.set_item('expression_restorer_areas', expr_areas_v)
                except Exception:
                    pass
                try:
                    state_manager.set_item('age_modifier_model', age_model_v)
                    state_manager.set_item('age_modifier_direction', int(age_direction_v))
                except Exception:
                    pass
                try:
                    state_manager.set_item('face_editor_model', editor_model_v)
                    state_manager.set_item('face_editor_eyebrow_direction', float(fe_eyebrow_dir_v))
                    state_manager.set_item('face_editor_eye_gaze_horizontal', float(fe_eye_h_v))
                    state_manager.set_item('face_editor_eye_gaze_vertical', float(fe_eye_v_v))
                    state_manager.set_item('face_editor_eye_open_ratio', float(fe_eye_open_v))
                    state_manager.set_item('face_editor_lip_open_ratio', float(fe_lip_open_v))
                    state_manager.set_item('face_editor_mouth_smile', float(fe_mouth_smile_v))
                    state_manager.set_item('face_editor_mouth_grim', float(state_manager.get_item('face_editor_mouth_grim') or 0.0))
                    state_manager.set_item('face_editor_mouth_pout', float(state_manager.get_item('face_editor_mouth_pout') or 0.0))
                    state_manager.set_item('face_editor_mouth_purse', float(state_manager.get_item('face_editor_mouth_purse') or 0.0))
                    state_manager.set_item('face_editor_mouth_position_horizontal', float(state_manager.get_item('face_editor_mouth_position_horizontal') or 0.0))
                    state_manager.set_item('face_editor_mouth_position_vertical', float(state_manager.get_item('face_editor_mouth_position_vertical') or 0.0))
                    state_manager.set_item('face_editor_head_pitch', float(fe_head_pitch_v))
                    state_manager.set_item('face_editor_head_yaw', float(fe_head_yaw_v))
                    state_manager.set_item('face_editor_head_roll', float(fe_head_roll_v))
                except Exception:
                    pass
                try:
                    state_manager.set_item('face_debugger_items', dbg_items_v)
                except Exception:
                    pass
                try:
                    state_manager.set_item('lip_syncer_model', lip_model_v)
                    state_manager.set_item('lip_syncer_weight', float(lip_weight_v))
                except Exception:
                    pass
                yield from gr_stream(
                    camera_choice, width, height, occl_enabled, fps, backend, dshow_name_drop, convert_rgb,
                    force_fourcc, retry_black, gentle_mode, auto_repair, color_mode, lock_exposure, exposure_value,
                    lock_wb, wb_temperature, show_overlay, debug_logs, show_boxes, show_native_flag, virtual_cam_flag,
                    colorizer_flag, expr_flag, age_flag, editor_flag, debugger_flag, lip_flag,
                    selector_mode_dd, auto_fallback,
                    exec_provider, exec_device, video_mem,
                    d_model, d_size, d_score, l_model, l_score, o_model, p_model, swap_mode,
                    source_files, ds_model, morph, fs_model, fs_pixel, fs_weight, face_enh_enabled, face_enh_model,
                    face_enh_blend, face_enh_weight, frame_enh_enabled, frame_enh_model, frame_enh_blend, enhance_async
                )

            # Preferences load-sync (minimal) so refresh restores key UI state
            def _prefs_load_sync():
                try:
                    prefs = _load_prefs() or {}
                    sm = prefs.get('swap_mode', 'deep')
                    exec_prov = (prefs.get('execution_providers') or ['cpu'])[0]
                    exec_dev = (prefs.get('execution_device_ids') or ['0'])[0]
                    video_mem_val = prefs.get('video_memory_strategy', 'moderate')
                    vc_enabled = bool(prefs.get('virtual_cam_enabled', False))
                    LOGGER.info(f"[prefs_load_sync] swap_mode={sm} exec_prov={exec_prov} exec_dev={exec_dev} video_mem={video_mem_val} virtual_cam={vc_enabled}")
                    return (
                        gr.update(value=exec_prov),
                        gr.update(value=str(exec_dev)),
                        gr.update(value=video_mem_val),
                        gr.update(value=sm),
                        gr.update(value=vc_enabled),
                    )
                except Exception as e:
                    LOGGER.exception(f"[prefs_load_sync] failed: {e}")
                    return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

            demo.load(
                fn=_prefs_load_sync,
                inputs=[],
                outputs=[exec_provider, exec_device, video_mem, swap_mode, virtual_cam],
            )

            # Launch the GUI (auto-open browser)
            demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True, inbrowser=True, share=False)
            return

def test_capture(camera_choice, backend_name, dshow_name_text, width, height, target_fps, convert_rgb_flag, fourcc_name):
    # Simple one-shot capture with diagnostics
    info = []
    cap = None
    try:
        if not camera_choice.startswith("["):
            return None, "Invalid camera selection"
        settings = _settings_from_inputs(
            camera_choice,
            backend_name,
            dshow_name_text or None,
            width,
            height,
            target_fps,
            convert_rgb_flag,
            fourcc_name,
        )
        cap, effective, attempts = _open_configured_capture(settings, gentle_mode=False)
        if cap is None:
            return None, "Capture failed: " + "; ".join(attempts[-8:])
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, "Read failed"
        mean_val = float(frame.mean())
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps_r = cap.get(cv2.CAP_PROP_FPS)
        fourcc_str = _fourcc_string(cap)
        info.append(
            f"size={size} fps={fps_r:.1f} fourcc={fourcc_str} mean={mean_val:.2f} "
            f"backend={effective.get('backend')}"
        )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), '\n'.join(info)
    except Exception as e:
        return None, f"Diag error: {e}"
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
