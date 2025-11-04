import argparse
import sys
import time
from typing import Optional, Tuple, Dict, List
import subprocess
import os
import threading
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FF_ROOT = os.path.join(BASE_DIR, "facefusion_mrg")
if FF_ROOT not in sys.path:
    sys.path.insert(0, FF_ROOT)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import cv2
import numpy as np
import gradio as gr
import logging
import onnxruntime as ort
import gc
from concurrent.futures import ThreadPoolExecutor, Future

from facefusion import state_manager
from facefusion.types import VisionFrame
from facefusion.face_selector import select_faces
from facefusion.face_analyser import scale_face
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
from facefusion.filesystem import resolve_file_paths, get_file_name
from facefusion.download import conditional_download_hashes, conditional_download_sources

LOGGER = logging.getLogger("webcam_deep_swap")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_models_downloaded = False


def ensure_models_downloaded() -> None:
    global _models_downloaded
    if _models_downloaded:
        return
    try:
        LOGGER.info("Preparing FaceFusion models (first run may download files)...")
        # Seed essential state so pre_check() knows what to fetch
        try:
            if state_manager.get_item("download_scope") is None:
                state_manager.set_item("download_scope", "full")
            if state_manager.get_item("download_providers") is None:
                state_manager.set_item("download_providers", ["huggingface"])
            if not state_manager.get_item("face_detector_model"):
                state_manager.set_item("face_detector_model", "many")
            if not state_manager.get_item("face_detector_size"):
                sizes = ff_choices.face_detector_set.get("retinaface", ["640x640"]) or ["640x640"]
                state_manager.set_item("face_detector_size", sizes[0])
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
            if capture:
                if "DirectShow audio devices" in line:
                    break
                if "]  \"" in line:
                    # lines look like: "  [dshow @ ...]  "FaceTime HD Camera""
                    name = line.split('"')
                    if len(name) >= 2:
                        names.append(name[1])
        return names
    except Exception:
        return []


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
        if lock_exposure:
            # Try to force manual exposure
            # OpenCV/DSHOW commonly uses 0.25 = manual, 0.75 = auto
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            # Some drivers use 0/1
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            # Apply exposure value (note: many drivers expect negative log-scale)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_value))
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
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
    for idx in range(max_index):
        label = names[idx] if idx < len(names) and names[idx] else f"Camera {idx}"
        choices.append(f"[{idx}] {label}")
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
    state_manager.set_item("face_detector_angles", [0, 90, 180, 270])
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


def process_frame(frame: VisionFrame, do_swap: bool, show_debug: bool = False, debug_log: bool = False, show_boxes: bool = False) -> VisionFrame:
    if frame is None:
        return None
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
        return frame

    # Build inputs
    temp = frame.copy() if hasattr(frame, "copy") else frame
    inputs = {
        "reference_vision_frame": frame,  # unused when selector_mode='one'
        "target_vision_frame": frame,
        "temp_vision_frame": temp,
    }

    try:
        # Ensure faces are selected in state for this frame; if nothing found, skip swapping
        try:
            faces_res = select_faces(reference_vision_frame=frame, target_vision_frame=frame)
            faces = faces_res[0] if isinstance(faces_res, tuple) else faces_res
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
            if not faces:
                if debug_log:
                    LOGGER.info("[process_frame] skip swap: faces=0")
                return frame
        except Exception as e:
            # If selection fails, fall back without swapping
            if debug_log:
                LOGGER.exception(f"[process_frame] select_faces failed; skipping swap: {e}")
            return frame
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
        # Choose swapper
        swap_mode = (state_manager.get_item("swap_mode") or "deep")
        out = None
        if swap_mode == "face":
            # Prepare source frames from uploaded images if any
            try:
                cached_face = _get_cached_source_face()
                if cached_face is not None and faces:
                    out = temp
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
                            "temp_vision_frame": temp,
                        }
                    else:
                        fs_inputs = {
                            "reference_vision_frame": frame,
                            "source_vision_frames": None,
                            "target_vision_frame": frame,
                            "temp_vision_frame": temp,
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
            # Ensure deep swapper model/session exists; attempt lazy init if missing
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
            out = deep_swapper.process_frame(inputs)
            # deep_swapper returns (frame, mask); extract frame for display
            try:
                if isinstance(out, tuple) and len(out) > 0:
                    out = out[0]
            except Exception:
                pass
        # Accept only non-empty numpy image outputs; otherwise, fall back
        if isinstance(out, np.ndarray) and getattr(out, 'size', 0) > 0:
            # Optional post processors that run in real-time
            try:
                if state_manager.get_item('frame_colorizer_enabled'):
                    try:
                        ff_frame_colorizer.pre_check()
                    except Exception:
                        pass
                    out = ff_frame_colorizer.colorize_frame(out)
                if state_manager.get_item('expression_restorer_enabled'):
                    try:
                        ff_expr_restorer.pre_check()
                    except Exception:
                        pass
                    try:
                        res = ff_expr_restorer.process_frame({
                            'reference_vision_frame': frame,
                            'target_vision_frame': frame,
                            'temp_vision_frame': out,
                        })
                        if isinstance(res, tuple) and len(res) > 0:
                            out = res[0]
                        elif isinstance(res, np.ndarray):
                            out = res
                    except Exception:
                        pass
                if state_manager.get_item('age_modifier_enabled'):
                    try:
                        ff_age_modifier.pre_check()
                    except Exception:
                        pass
                    try:
                        res = ff_age_modifier.process_frame({
                            'reference_vision_frame': frame,
                            'target_vision_frame': frame,
                            'temp_vision_frame': out,
                        })
                        if isinstance(res, tuple) and len(res) > 0:
                            out = res[0]
                        elif isinstance(res, np.ndarray):
                            out = res
                    except Exception:
                        pass
                if state_manager.get_item('face_editor_enabled'):
                    try:
                        ff_face_editor.pre_check()
                    except Exception:
                        pass
                    try:
                        res = ff_face_editor.process_frame({
                            'reference_vision_frame': frame,
                            'target_vision_frame': frame,
                            'temp_vision_frame': out,
                        })
                        if isinstance(res, tuple) and len(res) > 0:
                            out = res[0]
                        elif isinstance(res, np.ndarray):
                            out = res
                    except Exception:
                        pass
                if state_manager.get_item('lip_syncer_enabled'):
                    try:
                        ff_lip_syncer.pre_check()
                    except Exception:
                        pass
                    try:
                        res = ff_lip_syncer.process_frame({
                            'reference_vision_frame': frame,
                            'source_voice_frame': None,
                            'target_vision_frame': frame,
                            'temp_vision_frame': out,
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
        if v in ort_avail:
            avail_keys.append(k)
    # Always include cpu as fallback
    if 'cpu' not in avail_keys:
        avail_keys.append('cpu')
    return avail_keys


_stop_stream = False
_cap: Optional[cv2.VideoCapture] = None
_cap_settings: Dict[str, object] = {}
_last_faces_count: int = 0
_zero_face_streak: int = 0
_source_imgs: List[np.ndarray] = []
_cached_source_face = None
_cached_source_key: Optional[tuple] = None


def _invalidate_source_cache():
    global _cached_source_face, _cached_source_key
    _cached_source_face = None
    _cached_source_key = None


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
    try:
        deep_swapper.clear_inference_pool()
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
    return {
        "cam_index": _parse_cam_index(camera_choice),
        "backend": backend_name,
        "dshow_name": dshow_name or None,
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "fps": float(target_fps) if target_fps else None,
        "convert_rgb": bool(convert_rgb),
        "fourcc": force_fourcc or "Auto",
    }


def _settings_equal(a: Dict[str, object], b: Dict[str, object]) -> bool:
    return all(a.get(k) == b.get(k) for k in ("cam_index","backend","dshow_name","width","height","fps","convert_rgb","fourcc"))


def _ensure_capture(settings: Dict[str, object], gentle_mode: bool = True, force_reopen: bool = False,
                    lock_exposure: bool = False, exposure_value: float = -6.0,
                    lock_wb: bool = False, wb_temperature: int = 4500) -> cv2.VideoCapture:
    global _cap, _cap_settings
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
    # Open new
    cap = _open_capture(int(settings["cam_index"]), str(settings["backend"]), settings.get("dshow_name"))
    if cap is None:
        raise gr.Error(f"Failed to open camera index {settings['cam_index']}")
    # Apply properties conservatively
    fourcc = str(settings.get("fourcc", "Auto"))
    # Some drivers require FOURCC first
    if not gentle_mode or (fourcc and fourcc != "Auto"):
        _apply_fourcc(cap, fourcc)
    if settings.get("width"):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(settings["width"]))
    if settings.get("height"):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(settings["height"]))
    if settings.get("fps"):
        cap.set(cv2.CAP_PROP_FPS, float(settings["fps"]))
    # If HD and FOURCC is Auto, prefer MJPG proactively (common requirement for 1080p)
    try:
        w_req = int(settings.get("width") or 0)
        h_req = int(settings.get("height") or 0)
        if (w_req >= 1280 or h_req >= 720) and (not gentle_mode) or (fourcc == "Auto"):
            _apply_fourcc(cap, "MJPG")
    except Exception:
        pass
    _set_convert_rgb(cap, bool(settings.get("convert_rgb", True)))
    # Apply camera controls after basic setup
    _apply_camera_controls(cap, lock_exposure, exposure_value, lock_wb, wb_temperature)
    # Gentle warmup and validation
    def _valid_frame(frm: Optional[np.ndarray]) -> bool:
        return (
            isinstance(frm, np.ndarray)
            and frm.ndim == 3
            and frm.shape[2] == 3
            and getattr(frm, 'size', 0) > 0
        )
    warmup_reads = 1 if gentle_mode else 5
    ok, test = False, None
    for _ in range(warmup_reads):
        try:
            ok, test = cap.read()
        except Exception:
            ok, test = False, None
        if ok and _valid_frame(test):
            break
    if not ok or not _valid_frame(test):
        # Try to repair common issues for unsupported resolution combos
        # 1) Clear FPS influence by setting again after a read cycle
        try:
            cap.set(cv2.CAP_PROP_FPS, 0)
        except Exception:
            pass
        for _ in range(2):
            try:
                cap.read()
            except Exception:
                pass
        try:
            ok, test = cap.read()
        except Exception:
            ok, test = False, None
        if not ok or not _valid_frame(test):
            # 2) Toggle convert_rgb and try common FOURCCs
            _set_convert_rgb(cap, True)
            for fmt in ([fourcc] if (fourcc and fourcc != "Auto") else []) + ["MJPG", "YUY2", "H264", "NV12"]:
                if fmt and fmt != "Auto":
                    _apply_fourcc(cap, fmt)
                for _ in range(2):
                    try:
                        cap.read()
                    except Exception:
                        pass
                try:
                    ok, test = cap.read()
                except Exception:
                    ok, test = False, None
                if ok and _valid_frame(test):
                    break
            # 2b) Reduce FPS for HD
            if not ok or not _valid_frame(test):
                try:
                    cap.set(cv2.CAP_PROP_FPS, 15.0)
                    for _ in range(2):
                        try:
                            cap.read()
                        except Exception:
                            pass
                    ok, test = cap.read()
                except Exception:
                    ok, test = False, None
            # 3) Fallback to 640x480 which most drivers support
            if not ok or not _valid_frame(test):
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    for _ in range(2):
                        try:
                            cap.read()
                        except Exception:
                            pass
                    try:
                        ok, test = cap.read()
                    except Exception:
                        ok, test = False, None
                except Exception:
                    ok = False
    if not ok or not _valid_frame(test):
        cap.release()
        raise gr.Error("Camera failed to deliver a valid frame for the selected resolution/FPS. Try MJPG/YUY2, enable Convert RGB, or use 640x480.")
    _cap = cap
    _cap_settings = dict(settings)
    return cap


def gr_stream(
    camera_choice: str,
    width: int,
    height: int,
    do_deep_swap: bool,
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
        global _stop_stream
        _stop_stream = False
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

        # Pre-download/init models now to avoid None sessions at first frame
        try:
            if debug_logs:
                LOGGER.info(f"[model] preparing deep swapper '{deep_model}' (scope={state_manager.get_item('download_scope')})")
            if deep_swapper.pre_check():
                deep_swapper.clear_inference_pool()  # rebuild with fresh files and selected EP/device
            if (swap_mode or 'deep') == 'face':
                try:
                    LOGGER.info(f"[model] preparing face swapper '{state_manager.get_item('face_swapper_model')}'")
                    if ff_face_swapper.pre_check():
                        ff_face_swapper.clear_inference_pool()
                except Exception:
                    pass
            if state_manager.get_item('face_enhancer_enabled'):
                try:
                    LOGGER.info(f"[model] preparing face enhancer '{state_manager.get_item('face_enhancer_model')}'")
                    if ff_face_enhancer.pre_check():
                        ff_face_enhancer.clear_inference_pool()
                except Exception:
                    pass
            if state_manager.get_item('frame_enhancer_enabled'):
                try:
                    LOGGER.info(f"[model] preparing frame enhancer '{state_manager.get_item('frame_enhancer_model')}'")
                    if ff_frame_enhancer.pre_check():
                        ff_frame_enhancer.clear_inference_pool()
                except Exception:
                    pass
        except Exception:
            pass

        settings = _settings_from_inputs(camera_choice, backend_name, dshow_name, width, height, target_fps, convert_rgb, force_fourcc)
        cap = _ensure_capture(settings, gentle_mode=gentle_mode, force_reopen=False,
                              lock_exposure=lock_exposure, exposure_value=exposure_value,
                              lock_wb=lock_wb, wb_temperature=wb_temperature)
        frame_interval = (1.0 / float(target_fps)) if target_fps and target_fps > 0 else 0.0
        last_time = time.time()
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
        while not _stop_stream:
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, 'size', 0) == 0:
                raise gr.Error(
                    "Failed to read frame from camera (empty frame). Reduce resolution/FPS, change FOURCC, or try a different camera entry."
                )
            frame_count += 1
            if debug_logs and (frame_count % 15 == 0):
                try:
                    LOGGER.info(f"[captured] shape={frame.shape} mean={frame.mean():.2f}")
                except Exception:
                    pass
            # Detect black frame and optionally retry a few cycles
            try:
                if frame is not None and retry_black > 0 and not gentle_mode:
                    if frame.mean() < 1.0:  # almost black
                        retries = retry_black
                        # Try toggling convert_rgb or fourcc if set to Auto
                        while retries > 0 and (frame is None or frame.mean() < 1.0):
                            retries -= 1
                            _set_convert_rgb(cap, True)
                            if force_fourcc == "Auto":
                                for fmt in ("MJPG", "YUY2", "H264", "NV12"):
                                    _apply_fourcc(cap, fmt)
                                    ok, frame = cap.read()
                                    if ok and frame is not None and frame.mean() >= 1.0:
                                        break
                            ok, frame = cap.read()
                            if not ok:
                                break
                        if frame is None or frame.mean() < 1.0:
                            raise gr.Error("Camera returns black frames. Try DirectShow backend, Convert RGB, and MJPG.")
            except Exception:
                pass

            # Auto-repair for corrupted frames (horizontal bands, wrong stride)
            try:
                if auto_repair and frame is not None and getattr(frame, 'size', 0) > 0:
                    h, w = frame.shape[:2]
                    # Heuristics: very low vertical variance or too few unique rows indicates corruption
                    row_means = frame.mean(axis=1)
                    vstd = float(row_means.std()) if row_means.size > 0 else 0.0
                    if vstd < 1.0 or h < 40:  # tiny effective height or almost constant rows
                        # Attempt repairs: ensure convert RGB, try FOURCCs, optional backend flip once
                        _set_convert_rgb(cap, True)
                        formats = ([force_fourcc] if (force_fourcc and force_fourcc != "Auto") else []) + ["MJPG", "YUY2", "H264", "NV12"]
                        repaired = False
                        for fmt in formats:
                            if fmt:
                                _apply_fourcc(cap, fmt)
                            # give driver a couple reads to settle
                            for _ in range(2):
                                ok, frame = cap.read()
                                if not ok:
                                    break
                            if ok and frame is not None:
                                h2, w2 = frame.shape[:2]
                                row_means2 = frame.mean(axis=1)
                                vstd2 = float(row_means2.std()) if row_means2.size > 0 else 0.0
                                if vstd2 >= 1.0 and h2 >= 40:
                                    repaired = True
                                    break
                        if not repaired and not switched_backend_once:
                            # Flip backend once (DSHOW <-> MSMF)
                            new_backend = "DirectShow" if backend_name == "Media Foundation" else "Media Foundation"
                            settings["backend"] = new_backend
                            cap = _ensure_capture(settings, gentle_mode=gentle_mode, force_reopen=True)
                            switched_backend_once = True
                            # settle
                            for _ in range(2):
                                cap.read()
            except Exception:
                pass

            processed = process_frame(frame, do_deep_swap, show_overlay, debug_logs, show_boxes)
            # If async enhancement is enabled, offload enhancers and display latest completed
            frame_out = processed if (isinstance(processed, np.ndarray) and getattr(processed, 'size', 0) > 0) else frame
            try:
                if state_manager.get_item('enhance_async') and (state_manager.get_item('face_enhancer_enabled') or state_manager.get_item('frame_enhancer_enabled')):
                    # collect finished job
                    if pending is not None and pending.done():
                        res = pending.result()
                        if isinstance(res, np.ndarray) and getattr(res, 'size', 0) > 0:
                            latest_enhanced = res
                        pending = None
                    # submit new job if idle
                    if pending is None and isinstance(frame_out, np.ndarray) and getattr(frame_out, 'size', 0) > 0:
                        pending = executor.submit(_run_enhancers, frame_out.copy())
                    # prefer newest enhanced frame if available
                    if isinstance(latest_enhanced, np.ndarray) and getattr(latest_enhanced, 'size', 0) > 0:
                        frame_out = latest_enhanced
            except Exception:
                pass
            if frame_out is None or getattr(frame_out, 'size', 0) == 0:
                raise gr.Error("Failed to obtain a valid frame. Try different backend/FOURCC or lower resolution.")
            # Mirror to native window if requested
            try:
                if show_native_window:
                    if isinstance(color_mode, str) and color_mode.startswith("Assume RGB"):
                        disp = frame_out
                    else:
                        disp = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                    cv2.namedWindow(native_window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(native_window_name, disp)
                    cv2.waitKey(1)
            except Exception:
                pass
            # Stream to Gradio output
            if isinstance(color_mode, str) and color_mode.startswith("Assume RGB"):
                yield frame_out
            else:
                yield cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            # Optional auto-fallback to other detectors if no faces for a while
            if auto_fallback:
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
                        if debug_logs:
                            LOGGER.info(f"[fallback] switching detector to {nxt} size={sizes[0]}")
                        _zero_face_streak = 0
                except Exception:
                    pass

            if frame_interval > 0:
                now = time.time()
                sleep_dur = frame_interval - (now - last_time)
                if sleep_dur > 0:
                    time.sleep(sleep_dur)
                last_time = now
        # Keep device open for reuse to avoid LED blinking
        pass
    except gr.Error as e:
        # Re-raise to show a visible banner in Gradio
        raise e
    except Exception as e:
        raise gr.Error(f"Start failed: {e}")
    finally:
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


def gr_stop_stream():
    global _stop_stream
    _stop_stream = True
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
    ensure_models_downloaded()
    parser = argparse.ArgumentParser("Webcam Deep Swap (FaceFusion reuse)")
    parser.add_argument("--gui", action="store_true", help="Launch Gradio GUI")
    parser.add_argument("--list-cams", action="store_true", help="List available camera indices and exit")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open")
    parser.add_argument(
        "--model",
        type=str,
        default="iperov/james_carrey_224",
        help="Deep swapper model id from FaceFusion (e.g., 'iperov/james_carrey_224')",
    )
    parser.add_argument("--no-occlusion", action="store_true", help="Disable occlusion masking")
    parser.add_argument("--morph", type=int, default=100, help="Morph value [0-100] for deep swapper")
    parser.add_argument("--width", type=int, default=1920, help="Capture width")
    parser.add_argument("--height", type=int, default=1080, help="Capture height")

    args = parser.parse_args()

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
            
            with gr.Row():
                # Execution provider & device selection
                ep_keys = _available_execution_provider_keys()
                # prefer cuda if available
                ep_default = 'cuda' if 'cuda' in ep_keys else ('directml' if 'directml' in ep_keys else 'cpu')
                exec_provider = gr.Dropdown(choices=ep_keys, value=ep_default, label="Execution Provider")
                exec_device = gr.Textbox(value="0", label="Execution Device ID")
                video_mem = gr.Dropdown(choices=["strict","moderate","relaxed"], value="moderate", label="Video Memory Strategy")
                shutdown_btn = gr.Button("Shutdown App", variant="stop")
            with gr.Tabs():
                with gr.Tab("Swapping"):
                    with gr.Row("Face Swapper"):
                        swap_mode = gr.Dropdown(choices=["deep","face"], value="deep", label="Swap Mode")
                        source_files = gr.Files(label="Source Photos (for Face Swapper)", file_types=["image"], type="filepath")
                    with gr.Row():
                        fs_model = gr.Dropdown(choices=face_swapper_models, value=(face_swapper_models[0] if face_swapper_models else None), label="Face Swapper Model")
                        fs_pixel = gr.Dropdown(choices=proc_choices.face_swapper_set.get((face_swapper_models[0] if face_swapper_models else 'inswapper_128'), ["256x256"]) , value=(proc_choices.face_swapper_set.get((face_swapper_models[0] if face_swapper_models else 'inswapper_128'), ["256x256"]) [0]), label="Face Swapper Pixel Boost")
                        fs_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Face Swapper Weight")
                    with gr.Row():
                        deep_enabled = gr.Checkbox(value=True, label="Enable Deep Swap")
                        ds_model = gr.Dropdown(choices=deep_models, value="iperov/james_carrey_224", label="Deep Swapper Model")
                        morph = gr.Slider(minimum=0, maximum=100, step=1, value=args.morph, label="Morph")
                    
                with gr.Tab("Camera"):
                    with gr.Row():
                        backend = gr.Dropdown(choices=list(BACKEND_MAP.keys()), value="Media Foundation", label="Backend")
                        cam_choices = get_camera_choices(5, backend_name="Auto")
                        camera = gr.Dropdown(choices=cam_choices, value=(cam_choices[0] if cam_choices else f"[{args.camera}] Camera {args.camera}"), label="Camera")
                        res_presets = ["640x480", "1280x720", "1920x1080", "Custom"]
                        res = gr.Dropdown(choices=res_presets, value="1280x720", label="Resolution")
                        width = gr.Number(value=args.width, label="Width", precision=0)
                        height = gr.Number(value=args.height, label="Height", precision=0)
                        fps = gr.Slider(minimum=5, maximum=60, step=1, value=5, label="Target FPS")
                        
                    with gr.Row():
                        dshow_names = _ffmpeg_camera_names()
                        dshow_name_drop = gr.Dropdown(choices=dshow_names, value=(dshow_names[0] if dshow_names else None), label="DirectShow device name")
                        dshow_name = gr.Textbox(value="", label="DShow name (manual)")
                        refresh_btn = gr.Button("Refresh Cameras")                        
                    with gr.Row():
                        convert_rgb = gr.Checkbox(value=True, label="Convert RGB in driver")
                        force_fourcc = gr.Dropdown(choices=["Auto","MJPG","YUY2","H264","NV12"], value="Auto", label="Force FOURCC")
                        retry_black = gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Retry on black frames")
                        gentle_mode = gr.Checkbox(value=True, label="Gentle Mode (minimize camera re-inits)")
                        auto_repair = gr.Checkbox(value=True, label="Auto-repair capture (try formats/backend)")
                        color_mode = gr.Dropdown(choices=["Auto (BGR->RGB)", "Assume RGB (no swap)",], value="Auto (BGR->RGB)", label="Color mode")
                    with gr.Row():
                        lock_exposure = gr.Checkbox(value=True, label="Lock Exposure")
                        exposure_value = gr.Slider(minimum=-13.0, maximum=-1.0, step=0.5, value=-6.0, label="Exposure (log scale)")
                        lock_wb = gr.Checkbox(value=True, label="Lock White Balance")
                        wb_temperature = gr.Slider(minimum=2800, maximum=6500, step=100, value=4500, label="WB Temperature (K)")
                        show_overlay = gr.Checkbox(value=False, label="Show detection overlay")
                        debug_logs = gr.Checkbox(value=True, label="Debug logs to console")
                        show_boxes = gr.Checkbox(value=False, label="Show detection boxes")
           
                with gr.Tab("Detection"):
                    with gr.Row():
                        d_model = gr.Dropdown(choices=detector_models, value="retinaface", label="Detector Model")
                        d_size = gr.Dropdown(choices=detector_sizes_map.get("retinaface", ["640x640"]), value=detector_sizes_map.get("retinaface", ["640x640"])[0], label="Detector Size")
                        d_score = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Detector Score")
                        selector_mode_dd = gr.Dropdown(choices=ff_choices.face_selector_modes, value="one", label="Selector Mode")
                        auto_fallback = gr.Checkbox(value=True, label="Auto fallback detector if no faces")
                    with gr.Row():
                        l_model = gr.Dropdown(choices=landmarker_models, value="many", label="Landmarker Model")
                        l_score = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Landmarker Score")
                    with gr.Row():
                        o_model = gr.Dropdown(choices=occluder_models, value="xseg_2", label="Occluder Model")
                        p_model = gr.Dropdown(choices=parser_models, value=(parser_models[0] if parser_models else None), label="Parser Model")
                        occl_enabled = gr.Checkbox(value=True, label="Use Occlusion Mask")
                
                with gr.Tab("Enhancing"):
                    with gr.Row():
                        face_enh_enabled = gr.Checkbox(value=False, label="Enable Face Enhancer")
                        face_enh_model = gr.Dropdown(choices=face_enhancer_models, value=("gfpgan_1.4" if "gfpgan_1.4" in face_enhancer_models else (face_enhancer_models[0] if face_enhancer_models else None)), label="Face Enhancer Model")
                        face_enh_blend = gr.Slider(minimum=0, maximum=100, step=1, value=80, label="Face Enhancer Blend")
                        face_enh_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Face Enhancer Weight")
                    with gr.Row():
                        frame_enh_enabled = gr.Checkbox(value=False, label="Enable Frame Enhancer")
                        frame_enh_model = gr.Dropdown(choices=frame_enhancer_models, value=("span_kendata_x4" if "span_kendata_x4" in frame_enhancer_models else (frame_enhancer_models[0] if frame_enhancer_models else None)), label="Frame Enhancer Model")
                        frame_enh_blend = gr.Slider(minimum=0, maximum=100, step=1, value=80, label="Frame Enhancer Blend")
                    with gr.Row():
                        enhance_async = gr.Checkbox(value=True, label="Async Enhance (background enhancers)")
                with gr.Tab("Processors"):
                    with gr.Row():
                        colorizer_enabled = gr.Checkbox(value=False, label="Enable Frame Colorizer")
                        colorizer_model = gr.Dropdown(choices=frame_colorizer_choices.frame_colorizer_models, value=(frame_colorizer_choices.frame_colorizer_models[0] if frame_colorizer_choices.frame_colorizer_models else None), label="Colorizer Model")
                        colorizer_size = gr.Dropdown(choices=frame_colorizer_choices.frame_colorizer_sizes, value=(frame_colorizer_choices.frame_colorizer_sizes[0] if frame_colorizer_choices.frame_colorizer_sizes else None), label="Colorizer Size")
                        colorizer_blend = gr.Slider(minimum=min(frame_colorizer_choices.frame_colorizer_blend_range), maximum=max(frame_colorizer_choices.frame_colorizer_blend_range), step=1, value=100, label="Colorizer Blend")
                    with gr.Row():
                        expr_enabled = gr.Checkbox(value=False, label="Enable Expression Restorer")
                        expr_model = gr.Dropdown(choices=expression_restorer_choices.expression_restorer_models, value=(expression_restorer_choices.expression_restorer_models[0] if expression_restorer_choices.expression_restorer_models else None), label="Expr Model")
                        expr_factor = gr.Slider(minimum=min(expression_restorer_choices.expression_restorer_factor_range), maximum=max(expression_restorer_choices.expression_restorer_factor_range), step=1, value=80, label="Expr Factor")
                        expr_areas = gr.CheckboxGroup(choices=expression_restorer_choices.expression_restorer_areas, value=expression_restorer_choices.expression_restorer_areas, label="Expr Areas")
                    with gr.Row():
                        age_enabled = gr.Checkbox(value=False, label="Enable Age Modifier")
                        age_model = gr.Dropdown(choices=age_modifier_choices.age_modifier_models, value=(age_modifier_choices.age_modifier_models[0] if age_modifier_choices.age_modifier_models else None), label="Age Model")
                        age_direction = gr.Slider(minimum=min(age_modifier_choices.age_modifier_direction_range), maximum=max(age_modifier_choices.age_modifier_direction_range), step=1, value=0, label="Age Direction")
                    with gr.Row():
                        editor_enabled = gr.Checkbox(value=False, label="Enable Face Editor")
                        editor_model = gr.Dropdown(choices=face_editor_choices.face_editor_models, value=(face_editor_choices.face_editor_models[0] if face_editor_choices.face_editor_models else None), label="Editor Model")
                    with gr.Row():
                        fe_eyebrow_dir = gr.Slider(minimum=min(face_editor_choices.face_editor_eyebrow_direction_range), maximum=max(face_editor_choices.face_editor_eyebrow_direction_range), step=0.1, value=0.0, label="Eyebrow Direction")
                        fe_eye_h = gr.Slider(minimum=min(face_editor_choices.face_editor_eye_gaze_horizontal_range), maximum=max(face_editor_choices.face_editor_eye_gaze_horizontal_range), step=0.1, value=0.0, label="Eye Gaze H")
                        fe_eye_v = gr.Slider(minimum=min(face_editor_choices.face_editor_eye_gaze_vertical_range), maximum=max(face_editor_choices.face_editor_eye_gaze_vertical_range), step=0.1, value=0.0, label="Eye Gaze V")
                        fe_eye_open = gr.Slider(minimum=min(face_editor_choices.face_editor_eye_open_ratio_range), maximum=max(face_editor_choices.face_editor_eye_open_ratio_range), step=0.1, value=0.0, label="Eye Open")
                    with gr.Row():
                        fe_lip_open = gr.Slider(minimum=min(face_editor_choices.face_editor_lip_open_ratio_range), maximum=max(face_editor_choices.face_editor_lip_open_ratio_range), step=0.1, value=0.0, label="Lip Open")
                        fe_mouth_smile = gr.Slider(minimum=min(face_editor_choices.face_editor_mouth_smile_range), maximum=max(face_editor_choices.face_editor_mouth_smile_range), step=0.1, value=0.0, label="Smile")
                        fe_head_pitch = gr.Slider(minimum=min(face_editor_choices.face_editor_head_pitch_range), maximum=max(face_editor_choices.face_editor_head_pitch_range), step=0.1, value=0.0, label="Head Pitch")
                    with gr.Row():
                        fe_head_yaw = gr.Slider(minimum=min(face_editor_choices.face_editor_head_yaw_range), maximum=max(face_editor_choices.face_editor_head_yaw_range), step=0.1, value=0.0, label="Head Yaw")
                        fe_head_roll = gr.Slider(minimum=min(face_editor_choices.face_editor_head_roll_range), maximum=max(face_editor_choices.face_editor_head_roll_range), step=0.1, value=0.0, label="Head Roll")
                    with gr.Row():
                        debugger_enabled = gr.Checkbox(value=False, label="Enable Face Debugger")
                        dbg_items = gr.CheckboxGroup(choices=face_debugger_choices.face_debugger_items, value=['face-landmark-5/68','face-mask'], label="Debugger Items")
                    with gr.Row():
                        lip_enabled = gr.Checkbox(value=False, label="Enable Lip Syncer")
                        lip_model = gr.Dropdown(choices=lip_syncer_choices.lip_syncer_models, value=(lip_syncer_choices.lip_syncer_models[0] if lip_syncer_choices.lip_syncer_models else None), label="Lip Model")
                        lip_weight = gr.Slider(minimum=min(lip_syncer_choices.lip_syncer_weight_range), maximum=max(lip_syncer_choices.lip_syncer_weight_range), step=0.05, value=0.5, label="Lip Weight")
            with gr.Row():
                start_btn = gr.Button("Start")
                stop_btn = gr.Button("Stop")
                test_btn = gr.Button("Test Capture")
                reconnect_btn = gr.Button("Reconnect")
                show_native = gr.Checkbox(value=True, label="Show native window")
                auto_start = gr.Checkbox(value=True, label="Auto-start stream")
            out = gr.Image(label="Output", streaming=True)
            diag = gr.Textbox(label="Diagnostics", lines=6)

            def on_detector_change(model):
                sizes = detector_sizes_map.get(model, ["640x640"])
                return gr.update(choices=sizes, value=sizes[0])

            d_model.change(on_detector_change, inputs=d_model, outputs=d_size)

            def on_refresh(backend_name, dshow_name_text):
                cams = get_camera_choices(5, backend_name=backend_name, dshow_name_hint=(dshow_name_text or None))
                if not cams:
                    cams = ["[0] Camera 0"]
                names_ff = _ffmpeg_camera_names()
                return gr.update(choices=cams, value=cams[0]), gr.update(choices=names_ff, value=(names_ff[0] if names_ff else None))

            backend.change(on_refresh, inputs=[backend, dshow_name], outputs=[camera, dshow_name_drop])
            refresh_btn.click(on_refresh, inputs=[backend, dshow_name], outputs=[camera, dshow_name_drop])

            def on_res_change(preset):
                if preset == "Custom":
                    return gr.update(), gr.update()
                w, h = [int(x) for x in preset.split("x")] 
                return gr.update(value=w), gr.update(value=h)

            res.change(on_res_change, inputs=res, outputs=[width, height])

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
            fe_head_pitch.change(lambda v: _set_float('face_editor_head_pitch', v), inputs=fe_head_pitch, outputs=[])
            fe_head_yaw.change(lambda v: _set_float('face_editor_head_yaw', v), inputs=fe_head_yaw, outputs=[])
            fe_head_roll.change(lambda v: _set_float('face_editor_head_roll', v), inputs=fe_head_roll, outputs=[])

            def on_dbg_items_change(v):
                try:
                    state_manager.set_item('face_debugger_items', v)
                except Exception:
                    pass
            dbg_items.change(on_dbg_items_change, inputs=dbg_items, outputs=[])

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

            # Additional live-change handlers for swapper/enhancers
            def on_swap_mode_change(mode: str):
                try:
                    state_manager.set_item("swap_mode", mode)
                    # If user selects 'face', disable deep swap; if 'deep', enable it
                    return gr.update(value=(mode == 'deep'))
                except Exception:
                    return gr.update()

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
                            pass
                except Exception:
                    pass

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

            start_btn.click(
                fn=gr_stream,
                inputs=[
                    camera, width, height,
                    deep_enabled, occl_enabled, fps, backend, dshow_name_drop, convert_rgb, force_fourcc, retry_black, gentle_mode, auto_repair, color_mode,
                    lock_exposure, exposure_value, lock_wb, wb_temperature,
                    show_overlay, debug_logs, show_boxes,
                    show_native,
                    colorizer_enabled, expr_enabled, age_enabled, editor_enabled, debugger_enabled, lip_enabled,
                    selector_mode_dd, auto_fallback,
                    exec_provider, exec_device, video_mem,
                    d_model, d_size, d_score, l_model, l_score, o_model, p_model,
                    swap_mode, source_files, ds_model, morph,
                    fs_model, fs_pixel, fs_weight,
                    face_enh_enabled, face_enh_model, face_enh_blend, face_enh_weight,
                    frame_enh_enabled, frame_enh_model, frame_enh_blend,
                    enhance_async
                ],
                outputs=out,
            )
            stop_btn.click(gr_stop_stream, inputs=None, outputs=out)
            gr.Button("Free VRAM").click(gr_free_vram, inputs=None, outputs=diag)
            shutdown_btn.click(gr_shutdown, inputs=None, outputs=out)

            # Auto-start on UI load
            def _gr_autostart(
                camera_choice, width, height, deep_enabled, occl_enabled, fps, backend, dshow_name_drop, convert_rgb,
                force_fourcc, retry_black, gentle_mode, auto_repair, color_mode, lock_exposure, exposure_value, lock_wb,
                wb_temperature, show_overlay, debug_logs, show_boxes, show_native_flag,
                colorizer_flag, expr_flag, age_flag, editor_flag, debugger_flag, lip_flag,
                selector_mode_dd, auto_fallback, exec_provider,
                exec_device, video_mem,
                d_model, d_size, d_score, l_model, l_score, o_model, p_model, swap_mode, source_files,
                ds_model, morph, fs_model, fs_pixel, fs_weight, face_enh_enabled, face_enh_model, face_enh_blend,
                face_enh_weight, frame_enh_enabled, frame_enh_model, frame_enh_blend, enhance_async, auto_start_flag
            ):
                if not auto_start_flag:
                    return None
                yield from gr_stream(
                    camera_choice, width, height, deep_enabled, occl_enabled, fps, backend, dshow_name_drop, convert_rgb,
                    force_fourcc, retry_black, gentle_mode, auto_repair, color_mode, lock_exposure, exposure_value,
                    lock_wb, wb_temperature, show_overlay, debug_logs, show_boxes, show_native_flag,
                    colorizer_flag, expr_flag, age_flag, editor_flag, debugger_flag, lip_flag,
                    selector_mode_dd, auto_fallback,
                    exec_provider, exec_device, video_mem,
                    d_model, d_size, d_score, l_model, l_score, o_model, p_model, swap_mode,
                    source_files, ds_model, morph, fs_model, fs_pixel, fs_weight, face_enh_enabled, face_enh_model,
                    face_enh_blend, face_enh_weight, frame_enh_enabled, frame_enh_model, frame_enh_blend, enhance_async
                )

            demo.load(
                fn=_gr_autostart,
                inputs=[
                    camera, width, height, deep_enabled, occl_enabled, fps, backend, dshow_name_drop, convert_rgb,
                    force_fourcc, retry_black, gentle_mode, auto_repair, color_mode, lock_exposure, exposure_value, lock_wb,
                    wb_temperature, show_overlay, debug_logs, show_boxes, show_native,
                    colorizer_enabled, expr_enabled, age_enabled, editor_enabled, debugger_enabled, lip_enabled,
                    selector_mode_dd, auto_fallback, exec_provider, exec_device, video_mem,
                    d_model, d_size, d_score, l_model, l_score, o_model, p_model, swap_mode, source_files,
                    ds_model, morph, fs_model, fs_pixel, fs_weight, face_enh_enabled, face_enh_model, face_enh_blend,
                    face_enh_weight, frame_enh_enabled, frame_enh_model, frame_enh_blend, enhance_async, auto_start
                ],
                outputs=out,
            )

            def test_capture(camera_choice, backend_name, dshow_name_text, width, height, convert_rgb_flag, fourcc_name):
                # Simple one-shot capture with diagnostics
                info = []
                try:
                    if not camera_choice.startswith("["):
                        return None, "Invalid camera selection"
                    cam_index = int(camera_choice.split(']')[0][1:])
                    cap = _open_capture(cam_index, backend_name, dshow_name_text or None)
                    if cap is None:
                        return None, f"Open failed (backend={backend_name}, name={dshow_name_text})"
                    if width:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                    if height:
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                    _set_convert_rgb(cap, convert_rgb_flag)
                    _apply_fourcc(cap, fourcc_name)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        cap.release()
                        return None, "Read failed"
                    mean_val = float(frame.mean())
                    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    fps_r = cap.get(cv2.CAP_PROP_FPS)
                    fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
                    fourcc_str = ''.join([chr((fourcc_val >> 8*i) & 0xFF) for i in range(4)])
                    info.append(f"size={size} fps={fps_r:.1f} fourcc={fourcc_str} mean={mean_val:.2f}")
                    cap.release()
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), '\n'.join(info)
                except Exception as e:
                    return None, f"Diag error: {e}"

            test_btn.click(
                fn=test_capture,
                inputs=[camera, backend, dshow_name_drop, width, height, convert_rgb, force_fourcc],
                outputs=[out, diag],
            )
            reconnect_btn.click(
                fn=gr_reconnect,
                inputs=[camera, backend, dshow_name_drop, width, height, fps, convert_rgb, force_fourcc, gentle_mode],
                outputs=[out, diag],
            )

        demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True, inbrowser=False, share=False)
        return

    if args.list_cams:
        list_cameras(12)
        return

    init_state(model_id=args.model, use_occlusion=not args.no_occlusion, morph=args.morph)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.camera}")
        sys.exit(1)

    # Try to set resolution
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Webcam (raw)"
    out_name = "Webcam (deep swap)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(out_name, cv2.WINDOW_NORMAL)

    running = False
    do_deep_swap = True

    draw_help = True
    last_toggle = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame from camera. Exiting...")
            break

        display = frame.copy()

        # Show controls overlay
        if draw_help:
            draw_info(display, "Controls: [SPACE]=Start/Stop  [F]=Toggle DeepSwap  [H]=Help  [Q]=Quit", (10, 24))
            draw_info(display, f"Status: running={running} deep_swap={do_deep_swap}", (10, 50))

        cv2.imshow(window_name, display)

        if running:
            processed = process_frame(frame, do_deep_swap)
            cv2.imshow(out_name, processed)
        else:
            # If not running, mirror raw into out window for convenience
            cv2.imshow(out_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):
            running = not running
        elif key in (ord('f'), ord('F')):
            # de-bounce
            now = time.time()
            if now - last_toggle > 0.2:
                do_deep_swap = not do_deep_swap
                last_toggle = now
        elif key in (ord('h'), ord('H')):
            draw_help = not draw_help

    cap.release()
    cv2.destroyAllWindows()

    # Cleanup heavy caches/pools
    try:
        deep_swapper.post_process()
    except Exception:
        pass


if __name__ == "__main__":
    main()
