from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
import zlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

import webcam_deep_swap as engine
from dfm_training import DfmTrainingError, DfmTrainingManager
from facefusion import face_analyser, face_store


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "webui" / "dist"
UPLOAD_DIR = BASE_DIR / ".assets" / "uploads"
BENCHMARK_PATH = BASE_DIR / "benchmarks" / "realtime_benchmark_latest.json"
PROVIDER_TIMING_PATH = BASE_DIR / "benchmarks" / "provider_timing_latest.json"
MODEL_THUMBNAIL_DIR = BASE_DIR / ".assets" / "model_thumbnails"
CUSTOM_MODEL_DIR = BASE_DIR / "facefusion_mrg" / ".assets" / "models" / "custom"
IDENTITY_PROFILE_DIR = BASE_DIR / ".assets" / "identities"
IDENTITY_SESSION_DIR = BASE_DIR / ".assets" / "identity_sessions"
DFM_TRAINING_DIR = BASE_DIR / ".assets" / "dfm_training"
COMMUNITY_MODEL_REPO = "dimanchkek/Deepfacelive-DFM-Models"
MODEL_THUMBNAIL_SEMAPHORE = threading.Semaphore(2)
DEEP_MODEL_HANDOFF_LOCK = threading.Lock()


FEMALE_MODEL_IDENTITIES = {
    "adrianne_palicki", "agnetha_falskog", "alicia_vikander", "amber_midthunder",
    "angelina_jolie", "anne_hathaway", "anya_chalotra", "brie_larson",
    "catherine_blanchett", "cobie_smulders", "elisabeth_shue", "elizabeth_olsen",
    "emily_blunt", "emma_stone", "emma_watson", "erin_moriarty", "eva_green",
    "florence_pugh", "freya_allan", "gigi_hadid", "jennifer_connelly",
    "kate_beckinsale", "lili_reinhart", "margaret_qualley", "mary_winstead",
    "melina_juergens", "millie_bobby_brown", "rachel_weisz", "rebecca_ferguson",
    "scarlett_johansson", "shannen_doherty", "zoe_saldana", "emma_roberts",
    "ivanka_trump", "lize_dzjabrailova", "sidney_sweeney", "winona_ryder",
    "alexandra_daddario", "amber_heard", "dilraba_dilmurat", "emilia_clarke",
    "margot_robbie", "natalie_dormer", "angelica_trae", "ella_freya",
    "emma_myers", "evie_pickerill", "kang_hyewon", "maddie_mead",
    "nicole_turnbull", "alica_schmidt", "ashley_alexiss", "billie_eilish",
    "cara_delevingne", "carolin_kebekus", "chelsea_clinton", "claire_boucher",
    "corinna_kopf", "hillary_clinton", "jenna_fischer", "kim_jisoo",
    "mica_suarez", "shailene_woodley", "shraddha_kapoor", "yu_jimin",
    "alison_brie", "aubrey_plaza", "bridget_regan", "deborah_woll", "dua_lipa",
    "hailee_steinfeld", "hilary_duff", "jessica_alba", "jessica_biel",
    "kim_kardashian", "kristen_bell", "lucy_liu", "megan_fox", "meghan_markle",
    "natalie_portman", "nicki_minaj", "olivia_wilde", "shay_mitchell",
    "sophie_turner", "taylor_swift",
}

MODEL_NAME_ALIASES = {
    "agnetha_falskog": "Agnetha Fältskog",
    "benjamin_affleck": "Ben Affleck",
    "benjamin_stiller": "Ben Stiller",
    "bradley_pitt": "Brad Pitt",
    "catherine_blanchett": "Cate Blanchett",
    "christopher_hemsworth": "Chris Hemsworth",
    "ewan_mcgregor": "Ewan McGregor",
    "james_carrey": "Jim Carrey",
    "james_mcavoy": "James McAvoy",
    "james_varney": "Jim Varney",
    "jimmy_donaldson": "Jimmy Donaldson (MrBeast)",
    "michael_fox": "Michael J. Fox",
    "mary_winstead": "Mary Elizabeth Winstead",
    "nicolas_coppola": "Nicolas Cage",
    "sidney_sweeney": "Sydney Sweeney",
    "seth_macfarlane": "Seth MacFarlane",
    "thomas_cruise": "Tom Cruise",
    "thomas_hanks": "Tom Hanks",
    "thomas_holland": "Tom Holland",
    "william_murray": "Bill Murray",
}

COMMUNITY_MODELS: Dict[str, Dict[str, Any]] = {
    "custom/eminem_288": {"name": "Eminem", "gender": "male", "resolution": 288, "size": 575698005, "path": "DFM-RTM/AMP/288/Eminem_AMP_288.dfm"},
    "custom/will_smith_288": {"name": "Will Smith", "gender": "male", "resolution": 288, "size": 314901343, "path": "DFM-RTM/AMP/288/SlapSmith(Will_Smith)_AMP_288.dfm"},
    "custom/george_clooney_320": {"name": "George Clooney", "gender": "male", "resolution": 320, "size": 1252187656, "path": "DFM-RTM/AMP/320/George_Clooney_AMP_320.dfm"},
    "custom/bruce_willis_224": {"name": "Bruce Willis", "gender": "male", "resolution": 224, "size": 718525559, "path": "DFM-RTM/SAEHD/224/Bruce Willis - 224 res - LIAE-UDT Bruce Willis - 224 res - LIAE-UDT/Bruce Willis.dfm"},
    "custom/george_carlin_224": {"name": "George Carlin", "gender": "male", "resolution": 224, "size": 718525559, "path": "DFM-RTM/SAEHD/224/George_Carlin_224.dfm"},
    "custom/justin_bieber_224": {"name": "Justin Bieber", "gender": "male", "resolution": 224, "size": 718525559, "path": "DFM-RTM/SAEHD/224/Justin Bieber 1.6 Million (FF).dfm"},
    "custom/saul_goodman_224": {"name": "Saul Goodman", "gender": "male", "resolution": 224, "size": 718525559, "path": "DFM-RTM/SAEHD/224/Saul_Goodman/Saul_Goodman_224.dfm"},
    "custom/liam_neeson_320": {"name": "Liam Neeson", "gender": "male", "resolution": 320, "size": 1039807607, "path": "DFM-RTM/SAEHD/320/Liam_Neeson_320.dfm"},
    "custom/barack_obama": {"name": "Barack Obama", "gender": "male", "resolution": None, "size": 718525559, "path": "DFM-RTM/Unidentified/Barak_Obama_907k.dfm"},
    "custom/joe_rogan": {"name": "Joe Rogan", "gender": "male", "resolution": None, "size": 718525559, "path": "DFM-RTM/Unidentified/Joe_Rogan_190k.dfm"},
    "custom/ana_de_armas_320": {"name": "Ana de Armas", "gender": "female", "resolution": 320, "size": 1039807607, "path": "DFM-RTM/SAEHD/320/Ana_de_Armas-320RTM-V1_320.dfm"},
    "custom/gal_gadot_352": {"name": "Gal Gadot", "gender": "female", "resolution": 352, "size": 1641290687, "path": "DFM-RTM/SAEHD/352/Gal_Gadot_352.dfm"},
    "custom/tricia_helfer_256": {"name": "Tricia Helfer", "gender": "female", "resolution": 256, "size": 813020279, "path": "DFM-RTM/SAEHD/256/Tricia Helfer_256_GAN Final.dfm"},
    "custom/kathryn_newton_224": {"name": "Kathryn Newton", "gender": "female", "resolution": 224, "size": 502674246, "path": "DFM-RTM/SAEHD/224/KathrynNewton_224.dfm"},
    "custom/anya_taylor_joy": {"name": "Anya Taylor-Joy", "gender": "female", "resolution": None, "size": 502674248, "path": "DFM-RTM/Unidentified/Anya_Taylor_Joy.dfm"},
    "custom/natalia_dyer": {"name": "Natalia Dyer", "gender": "female", "resolution": None, "size": 718525559, "path": "DFM-RTM/Unidentified/Natalia_Dyer.dfm"},
    "custom/tinashe_384": {"name": "Tinashe", "gender": "female", "resolution": 384, "size": 1987255412, "path": "DFM-RTM/SAEHD/384/Tinashe_Kachingwe_TdeepR_384.dfm"},
    "custom/eden_sher_384": {"name": "Eden Sher", "gender": "female", "resolution": 384, "size": 1987255441, "path": "DFM-RTM/SAEHD/384/Eden_Cher_DFM_384/Eden_Cher_TdeepR_384.dfm"},
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "backend": "DirectShow" if os.name == "nt" else "Auto",
    "camera_choice": "[0] Camera 0",
    "resolution_preset": "1920x1080",
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "dshow_name_device": "",
    "convert_rgb": True,
    "force_fourcc": "Auto",
    "retry_black": 3,
    "gentle_mode": True,
    "auto_repair": True,
    "color_mode": "Auto (BGR->RGB)",
    "lock_exposure": False,
    "exposure_value": -6.0,
    "lock_wb": False,
    "wb_temperature": 4500,
    "show_overlay": False,
    "debug_logs": False,
    "show_boxes": False,
    "show_native": False,
    "virtual_cam_enabled": False,
    "demand_capture_enabled": True,
    "tray_start_mode": "last",
    "realtime_fast_analysis": True,
    "gpu_pipeline_enabled": False,
    "swap_mode": "deep",
    "deep_swapper_model": "iperov/keanu_reeves_320",
    "morph": 100,
    "face_swapper_model": "hyperswap_1c_256",
    "face_swapper_pixel_boost": "256x256",
    "face_swapper_weight": 0.5,
    "source_paths": [],
    "selector_mode": "one",
    "auto_fallback": False,
    "detector_model": "retinaface",
    "detector_size": "320x320",
    "detector_score": 0.35,
    "landmarker_model": "many",
    "landmarker_score": 0.5,
    "occluder_model": "xseg_1",
    "parser_model": "bisenet_resnet_18",
    "use_occlusion": True,
    "temporal_occlusion_reuse": True,
    "temporal_occlusion_interval": 3,
    "temporal_face_tracking": False,
    "temporal_face_interval": 3,
    "face_enhancer_enabled": False,
    "face_enhancer_model": "gfpgan_1.4",
    "face_enhancer_blend": 80,
    "face_enhancer_weight": 0.5,
    "frame_enhancer_enabled": False,
    "frame_enhancer_model": "span_kendata_x4",
    "frame_enhancer_blend": 80,
    "enhance_async": True,
    "frame_colorizer_enabled": False,
    "frame_colorizer_model": "ddcolor",
    "frame_colorizer_size": "192x192",
    "frame_colorizer_blend": 100,
    "expression_restorer_enabled": False,
    "expression_restorer_model": "live_portrait",
    "expression_restorer_factor": 80,
    "expression_restorer_areas": ["upper-face", "lower-face"],
    "age_modifier_enabled": False,
    "age_modifier_model": "styleganex_age",
    "age_modifier_direction": 0,
    "face_editor_enabled": False,
    "face_editor_model": "live_portrait",
    "face_debugger_enabled": False,
    "lip_syncer_enabled": False,
    "lip_syncer_model": "edtalk_256",
    "lip_syncer_weight": 0.5,
    "execution_providers": ["cuda"],
    "execution_device_ids": ["0"],
    "video_memory_strategy": "strict",
    "fast_startup": True,
}

available_default_providers = engine._available_execution_provider_keys()
for route_key, _route_label, _module in engine._MODULE_PROVIDER_ROUTES:
    DEFAULT_CONFIG[route_key] = "tensorrt" if route_key == "provider_deep_swapper" and "tensorrt" in available_default_providers else "cuda"

ALLOWED_CONFIG_KEYS = set(DEFAULT_CONFIG) | {
    "face_editor_eyebrow_direction",
    "face_editor_eye_gaze_horizontal",
    "face_editor_eye_gaze_vertical",
    "face_editor_eye_open_ratio",
    "face_editor_lip_open_ratio",
    "face_editor_mouth_smile",
    "face_editor_head_pitch",
    "face_editor_head_yaw",
    "face_editor_head_roll",
}

# Only settings that change how the physical device is opened may reconnect it.
# Every detector, model, mask, enhancer, provider and output option is applied
# between processed frames while the existing capture handle stays alive.
CAPTURE_SOURCE_KEYS = {
    "backend",
    "camera_choice",
    "resolution_preset",
    "width",
    "height",
    "fps",
    "dshow_name_device",
    "convert_rgb",
    "force_fourcc",
    "retry_black",
    "gentle_mode",
    "auto_repair",
    "color_mode",
    "lock_exposure",
    "exposure_value",
    "lock_wb",
    "wb_temperature",
}

INFERENCE_RELOAD_KEYS = {
    "deep_swapper_model",
    "face_swapper_model",
    "face_swapper_pixel_boost",
    "face_enhancer_model",
    "frame_enhancer_model",
    "frame_colorizer_model",
    "expression_restorer_model",
    "age_modifier_model",
    "face_editor_model",
    "lip_syncer_model",
    "detector_model",
    "landmarker_model",
    "occluder_model",
    "parser_model",
    "execution_providers",
    "execution_device_ids",
    "video_memory_strategy",
    *(route_key for route_key, _label, _module in engine._MODULE_PROVIDER_ROUTES),
}


class ConfigPatch(BaseModel):
    values: Dict[str, Any] = Field(default_factory=dict)
    restart: bool = False
    clear_models: bool = False


class RuntimeAction(BaseModel):
    clear_models: bool = False


class ModelInstallRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str


class IdentitySessionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    consent_confirmed: bool = False
    kind: str = Field(default="instant", pattern="^(instant|dfm)$")


class IdentityCaptureRequest(BaseModel):
    prompt_id: str
    duration_seconds: float = Field(default=1.15, ge=0.5, le=20.0)


class DfmSetupRequest(BaseModel):
    distro: str
    engine_root: str


class DfmPrepareRequest(BaseModel):
    profile_id: str
    distro: str
    engine_root: str
    base_workspace: str
    target_iterations: int = Field(default=1_000_000, ge=25_000, le=5_000_000)


IDENTITY_CAPTURE_PROMPTS = [
    {
        "id": "front-neutral",
        "title": "Look straight ahead",
        "instruction": "Relax your face and look directly into the lens.",
    },
    {
        "id": "turn-left",
        "title": "Turn left",
        "instruction": "Turn your head about 30 degrees to your left; keep your eyes on the lens.",
    },
    {
        "id": "turn-right",
        "title": "Turn right",
        "instruction": "Turn your head about 30 degrees to your right; keep your eyes on the lens.",
    },
    {
        "id": "look-up",
        "title": "Look slightly up",
        "instruction": "Raise your chin a little without leaning away from the camera.",
    },
    {
        "id": "look-down",
        "title": "Look slightly down",
        "instruction": "Lower your chin a little while keeping your whole face visible.",
    },
    {
        "id": "smile",
        "title": "Smile",
        "instruction": "Give a natural smile and hold it for a moment.",
    },
    {
        "id": "blink",
        "title": "Blink naturally",
        "instruction": "Blink two or three times while facing the camera.",
    },
    {
        "id": "talk",
        "title": "Speak a short sentence",
        "instruction": "Say your name or count from one to five at a normal pace.",
    },
]
IDENTITY_PROMPT_IDS = {prompt["id"] for prompt in IDENTITY_CAPTURE_PROMPTS}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _model_identity(model_id: str) -> str:
    model_name = model_id.split("/", 1)[-1]
    return re.sub(r"_\d+$", "", model_name).lower()


def _model_display_name(model_id: str) -> str:
    identity = _model_identity(model_id)
    if identity in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[identity]
    return " ".join(word.capitalize() for word in identity.split("_") if word)


def _model_gender(model_id: str) -> str:
    scope = model_id.split("/", 1)[0]
    identity = _model_identity(model_id)
    if scope in {"edel", "jen", "mats"}:
        return "female"
    if scope == "rumateus":
        return "male" if identity == "john_cena" else "female"
    if scope == "custom" and model_id not in COMMUNITY_MODELS:
        return "other"
    return "female" if identity in FEMALE_MODEL_IDENTITIES else "male"


def _model_resolution(model_id: str) -> Optional[int]:
    match = re.search(r"_(\d+)$", model_id)
    return int(match.group(1)) if match else None


def _model_catalog() -> list[Dict[str, Any]]:
    known_ids = list(dict.fromkeys(engine.proc_choices.deep_swapper_models))
    for model_id in COMMUNITY_MODELS:
        if model_id not in known_ids:
            known_ids.append(model_id)

    catalog: list[Dict[str, Any]] = []
    for model_id in known_ids:
        community = COMMUNITY_MODELS.get(model_id)
        scope, model_name = model_id.split("/", 1)
        if community:
            model_path = CUSTOM_MODEL_DIR / f"{model_name}.dfm"
            display_name = str(community["name"])
            gender = str(community["gender"])
            resolution = community.get("resolution")
            source = "Hugging Face community"
            size_bytes = int(community["size"])
            hub_url = f"https://huggingface.co/datasets/{COMMUNITY_MODEL_REPO}/blob/main/{urllib.parse.quote(str(community['path']), safe='/()_-.')}"
        else:
            model_path = BASE_DIR / "facefusion_mrg" / ".assets" / "models" / scope / f"{model_name}.dfm"
            display_name = _model_display_name(model_id)
            gender = _model_gender(model_id)
            resolution = _model_resolution(model_id)
            source = scope
            size_bytes = model_path.stat().st_size if model_path.exists() else None
            hub_url = f"https://huggingface.co/facefusion/deepfacelive-models-{scope}/blob/main/{model_name}.dfm" if scope != "custom" else None
        catalog.append({
            "id": model_id,
            "name": display_name,
            "gender": gender,
            "resolution": resolution,
            "creator": source,
            "installed": model_path.exists(),
            "downloadable": bool(community and not model_path.exists()),
            "size_bytes": size_bytes,
            "thumbnail_url": f"/api/models/thumbnail?model_id={urllib.parse.quote(model_id, safe='')}&v=2",
            "hub_url": hub_url,
        })
    return sorted(catalog, key=lambda item: (item["gender"], item["name"], item["resolution"] or 0, item["id"]))


def _model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    return next((model for model in _model_catalog() if model["id"] == model_id), None)


def _placeholder_thumbnail(name: str) -> bytes:
    color_seed = zlib.crc32(name.encode("utf-8"))
    background = (
        44 + color_seed % 52,
        38 + (color_seed >> 8) % 58,
        54 + (color_seed >> 16) % 62,
    )
    image = np.full((280, 240, 3), background, dtype=np.uint8)
    initials = "".join(word[0] for word in name.split()[:2] if word).upper() or "?"
    font = cv2.FONT_HERSHEY_SIMPLEX
    size, _baseline = cv2.getTextSize(initials, font, 1.8, 3)
    cv2.putText(image, initials, ((240 - size[0]) // 2, (280 + size[1]) // 2), font, 1.8, (232, 230, 242), 3, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return encoded.tobytes() if ok else b""


def _fetch_model_thumbnail(model: Dict[str, Any], destination: Path) -> None:
    user_agent = "FaceFlow/1.0 (local FaceFusion model browser)"
    image_url: Optional[str] = None
    try:
        title = str(model["name"]).replace(" (MrBeast)", "")
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}"
        request = urllib.request.Request(summary_url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=8) as response:
            summary = json.loads(response.read().decode("utf-8"))
            image_url = (summary.get("thumbnail") or {}).get("source")
    except Exception:
        image_url = None
    if not image_url:
        try:
            params = urllib.parse.urlencode({
                "action": "query",
                "generator": "search",
                "gsrsearch": str(model["name"]),
                "gsrlimit": 1,
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": 360,
                "format": "json",
                "origin": "*",
            })
            request = urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{params}", headers={"User-Agent": user_agent})
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
                pages = list((payload.get("query") or {}).get("pages", {}).values())
                image_url = ((pages[0].get("thumbnail") or {}).get("source")) if pages else None
        except Exception:
            image_url = None

    thumbnail = None
    if image_url:
        try:
            request = urllib.request.Request(image_url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(request, timeout=10) as response:
                raw_image = response.read(5 * 1024 * 1024)
            decoded = cv2.imdecode(np.frombuffer(raw_image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if isinstance(decoded, np.ndarray) and decoded.size:
                height, width = decoded.shape[:2]
                side = min(width, height)
                left = max(0, (width - side) // 2)
                top = max(0, (height - side) // 2)
                cropped = decoded[top:top + side, left:left + side]
                resized = cv2.resize(cropped, (240, 280), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 88])
                thumbnail = encoded.tobytes() if ok else None
        except Exception:
            thumbnail = None

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(f".{threading.get_ident()}.tmp")
    temp_path.write_bytes(thumbnail or _placeholder_thumbnail(str(model["name"])))
    os.replace(temp_path, destination)


class CommunityModelInstaller:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._status: Dict[str, Any] = {
            "running": False,
            "model_id": None,
            "progress": 0.0,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "error": None,
            "complete": False,
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start(self, model_id: str) -> Dict[str, Any]:
        if model_id not in COMMUNITY_MODELS:
            raise HTTPException(status_code=404, detail="This model is not in the curated community catalog")
        with self._lock:
            if self._status["running"]:
                if self._status["model_id"] != model_id:
                    raise HTTPException(status_code=409, detail="Another model download is already running")
                return dict(self._status)
            target = CUSTOM_MODEL_DIR / f"{model_id.split('/', 1)[1]}.dfm"
            if target.exists():
                self._register_model(model_id)
                self._status = {**self._status, "running": False, "model_id": model_id, "progress": 1.0, "downloaded_bytes": target.stat().st_size, "total_bytes": target.stat().st_size, "error": None, "complete": True}
                return dict(self._status)
            metadata = COMMUNITY_MODELS[model_id]
            self._status = {"running": True, "model_id": model_id, "progress": 0.0, "downloaded_bytes": 0, "total_bytes": int(metadata["size"]), "error": None, "complete": False}
            self._thread = threading.Thread(target=self._download, args=(model_id,), name="community-model-download", daemon=True)
            self._thread.start()
            return dict(self._status)

    def _register_model(self, model_id: str) -> None:
        if model_id not in engine.proc_choices.deep_swapper_models:
            engine.proc_choices.deep_swapper_models.append(model_id)
        try:
            engine.deep_swapper.create_static_model_set.cache_clear()
        except Exception:
            pass

    def _download(self, model_id: str) -> None:
        metadata = COMMUNITY_MODELS[model_id]
        model_name = model_id.split("/", 1)[1]
        destination = CUSTOM_MODEL_DIR / f"{model_name}.dfm"
        partial = CUSTOM_MODEL_DIR / f"{model_name}.dfm.part"
        url_path = urllib.parse.quote(str(metadata["path"]), safe="/()_-.")
        url = f"https://huggingface.co/datasets/{COMMUNITY_MODEL_REPO}/resolve/main/{url_path}"
        checksum = 0
        downloaded = 0
        try:
            CUSTOM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(url, headers={"User-Agent": "FaceFlow/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as output:
                total = int(response.headers.get("Content-Length") or metadata["size"])
                while True:
                    chunk = response.read(2 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    checksum = zlib.crc32(chunk, checksum)
                    downloaded += len(chunk)
                    with self._lock:
                        self._status["downloaded_bytes"] = downloaded
                        self._status["total_bytes"] = total
                        self._status["progress"] = min(0.995, downloaded / max(1, total))
            if downloaded < int(metadata["size"]):
                raise RuntimeError(f"Download ended early ({downloaded} of {metadata['size']} bytes)")
            os.replace(partial, destination)
            destination.with_suffix(".hash").write_text(format(checksum & 0xFFFFFFFF, "08x"), encoding="utf-8")
            self._register_model(model_id)
            with self._lock:
                self._status.update({"running": False, "progress": 1.0, "downloaded_bytes": downloaded, "error": None, "complete": True})
        except Exception as exc:
            try:
                if partial.exists():
                    partial.unlink()
            except Exception:
                pass
            with self._lock:
                self._status.update({"running": False, "error": str(exc) or type(exc).__name__, "complete": False})
        finally:
            with self._lock:
                self._thread = None


community_model_installer = CommunityModelInstaller()


def _identity_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:48] or "identity"


def _identity_profile_path(profile_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", profile_id):
        raise HTTPException(status_code=404, detail="Unknown identity profile")
    path = IDENTITY_PROFILE_DIR / profile_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Unknown identity profile")
    return path


def _read_identity_profile(profile_id: str) -> Dict[str, Any]:
    profile_path = _identity_profile_path(profile_id)
    manifest_path = profile_path / "profile.json"
    try:
        profile = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Identity profile is damaged: {exc}") from exc
    return profile


def _identity_profile_summary(profile: Dict[str, Any], active_paths: set[str]) -> Dict[str, Any]:
    source_paths = [str(path) for path in profile.get("source_paths") or []]
    profile_id = str(profile.get("id") or "")
    return {
        "id": profile_id,
        "name": str(profile.get("name") or profile_id),
        "created_at": float(profile.get("created_at") or 0.0),
        "frame_count": int(profile.get("frame_count") or len(profile.get("frames") or [])),
        "kind": str(profile.get("kind") or "averaged-face-source"),
        "dfm_ready": int(profile.get("frame_count") or len(profile.get("frames") or [])) >= 400,
        "source_count": len(source_paths),
        "prompts_completed": list(profile.get("prompts_completed") or []),
        "thumbnail_url": f"/api/identities/{urllib.parse.quote(profile_id)}/thumbnail",
        "dataset_url": f"/api/identities/{urllib.parse.quote(profile_id)}/dataset.zip",
        "active": bool(source_paths and set(source_paths) == active_paths),
    }


class IdentityTrainer:
    """Guided capture that builds an averaged, reusable FaceFusion source identity."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def start(self, name: str, consent_confirmed: bool, kind: str = "instant") -> Dict[str, Any]:
        if not consent_confirmed:
            raise HTTPException(
                status_code=422,
                detail="Confirm that the person agreed to be recorded and used for this identity profile.",
            )
        clean_name = name.strip()
        if not clean_name:
            raise HTTPException(status_code=422, detail="Enter a name for this identity")
        session_id = uuid.uuid4().hex[:16]
        session_path = IDENTITY_SESSION_DIR / session_id
        (session_path / "frames").mkdir(parents=True, exist_ok=False)
        session = {
            "id": session_id,
            "name": clean_name,
            "consent_confirmed": True,
            "consent_confirmed_at": time.time(),
            "created_at": time.time(),
            "path": str(session_path),
            "captures": [],
            "capturing": False,
            "kind": "dfm" if kind == "dfm" else "instant",
        }
        with self._lock:
            self._sessions[session_id] = session
        # Identity capture consumes unprocessed frames. When capture was off,
        # open the camera without loading the selected DFM or enhancer stack.
        if not runtime.status().get("running"):
            runtime.start(reason="trainer", processing_mode="none")
        return self.status(session_id)

    def _get(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Capture session is no longer available")
        return session

    @staticmethod
    def _analyse_frame(frame: np.ndarray) -> Dict[str, Any]:
        with engine._live_processing_lock:
            faces = face_analyser.get_many_faces([frame])
        if not faces:
            return {"accepted": False, "reason": "No face detected", "faces": 0}
        faces = sorted(
            faces,
            key=lambda face: float(max(0.0, face.bounding_box[2] - face.bounding_box[0]))
            * float(max(0.0, face.bounding_box[3] - face.bounding_box[1])),
            reverse=True,
        )
        face = faces[0]
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(value))) for value in face.bounding_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        face_width, face_height = max(0, x2 - x1), max(0, y2 - y1)
        if face_width < 2 or face_height < 2:
            return {"accepted": False, "reason": "Face crop is invalid", "faces": len(faces)}
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        face_ratio = float((face_width * face_height) / max(1, width * height))
        detector_score = float(face.score_set.get("detector") or 0.0)
        size_score = float(np.clip(face_ratio / 0.075, 0.0, 1.0))
        focus_score = float(np.clip(blur_score / 180.0, 0.0, 1.0))
        exposure_score = float(np.clip(1.0 - abs(brightness - 128.0) / 118.0, 0.0, 1.0))
        quality = 100.0 * (
            0.42 * np.clip(detector_score, 0.0, 1.0)
            + 0.25 * focus_score
            + 0.20 * size_score
            + 0.13 * exposure_score
        )
        reason = None
        if face_width < 120 or face_height < 120:
            reason = "Move closer to the camera"
        elif blur_score < 22.0:
            reason = "Hold still; the frame is blurry"
        elif brightness < 24.0:
            reason = "The face is too dark"
        elif brightness > 235.0:
            reason = "The face is overexposed"
        landmarks = face.landmark_set.get("5/68")
        yaw = 0.0
        if isinstance(landmarks, np.ndarray) and len(landmarks) >= 3:
            eye_midpoint = (landmarks[0] + landmarks[1]) * 0.5
            eye_distance = float(np.linalg.norm(landmarks[1] - landmarks[0]))
            if eye_distance > 1e-6:
                yaw = float((landmarks[2][0] - eye_midpoint[0]) / eye_distance)
        return {
            "accepted": reason is None,
            "reason": reason,
            "faces": len(faces),
            "quality": round(float(quality), 1),
            "blur": round(blur_score, 1),
            "brightness": round(brightness, 1),
            "face_ratio": round(face_ratio, 4),
            "detector_score": round(detector_score, 4),
            "yaw": round(yaw, 3),
            "bounding_box": [x1, y1, x2, y2],
        }

    def capture(self, session_id: str, prompt_id: str, duration_seconds: float) -> Dict[str, Any]:
        if prompt_id not in IDENTITY_PROMPT_IDS:
            raise HTTPException(status_code=422, detail="Unknown guided capture action")
        session = self._get(session_id)
        camera_deadline = time.monotonic() + 15.0
        while not runtime.status().get("capture_active") and time.monotonic() < camera_deadline:
            if runtime.status().get("error"):
                break
            time.sleep(0.1)
        if not runtime.status().get("capture_active"):
            detail = runtime.status().get("error") or "The webcam did not become ready in time"
            raise HTTPException(status_code=409, detail=detail)
        with self._lock:
            if session.get("capturing"):
                raise HTTPException(status_code=409, detail="A guided capture is already running")
            session["capturing"] = True
        frames: list[np.ndarray] = []
        sequence = -1
        deadline = time.perf_counter() + float(duration_seconds)
        try:
            if session.get("kind") == "dfm":
                return self._capture_dfm_burst(session, session_id, prompt_id, deadline)
            while time.perf_counter() < deadline and len(frames) < 8:
                remaining = max(0.05, deadline - time.perf_counter())
                frame, sequence, _captured_at = engine.get_latest_raw_frame(sequence, min(0.4, remaining))
                if isinstance(frame, np.ndarray) and frame.size:
                    frames.append(frame)
            if not frames:
                raise HTTPException(status_code=409, detail="No camera frames were available")

            results: list[Dict[str, Any]] = []
            accepted: list[Dict[str, Any]] = []
            frame_dir = Path(str(session["path"])) / "frames"
            for index, frame in enumerate(frames):
                metrics = self._analyse_frame(frame)
                metrics.update({"prompt_id": prompt_id, "captured_at": time.time()})
                if metrics.get("accepted"):
                    file_name = f"{prompt_id}_{int(time.time() * 1000)}_{index:02d}.jpg"
                    destination = frame_dir / file_name
                    if cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                        metrics["path"] = str(destination)
                        accepted.append(metrics)
                results.append(metrics)
            with self._lock:
                session["captures"].extend(accepted)
            best = max(results, key=lambda result: float(result.get("quality") or 0.0))
            return {
                "session": self.status(session_id),
                "prompt_id": prompt_id,
                "sampled": len(results),
                "accepted": len(accepted),
                "best_quality": best.get("quality"),
                "feedback": best.get("reason") or "Good capture",
            }
        finally:
            with self._lock:
                session["capturing"] = False

    def _capture_dfm_burst(
        self,
        session: Dict[str, Any],
        session_id: str,
        prompt_id: str,
        deadline: float,
    ) -> Dict[str, Any]:
        """Save a dense, quality-gated burst for later DeepFaceLab extraction."""
        frame_dir = Path(str(session["path"])) / "frames"
        accepted: list[Dict[str, Any]] = []
        sampled = 0
        sequence = -1
        last_saved_at = 0.0
        latest_face_metrics: Optional[Dict[str, Any]] = None
        target_interval = 1.0 / 12.0
        best_quality = 0.0
        feedback = "No clear face frames were available"

        while time.perf_counter() < deadline and len(accepted) < 240:
            remaining = max(0.05, deadline - time.perf_counter())
            frame, sequence, captured_at = engine.get_latest_raw_frame(sequence, min(0.35, remaining))
            if not isinstance(frame, np.ndarray) or not frame.size:
                continue
            timestamp = float(captured_at or time.perf_counter())
            if timestamp - last_saved_at < target_interval:
                continue
            last_saved_at = timestamp
            sampled += 1
            height, width = frame.shape[:2]
            if width > 1280:
                frame = cv2.resize(frame, (1280, max(1, int(height * 1280 / width))), interpolation=cv2.INTER_AREA)

            if latest_face_metrics is None or sampled % 12 == 1:
                latest_face_metrics = self._analyse_frame(frame)
            metrics = dict(latest_face_metrics or {})
            if not metrics.get("accepted"):
                feedback = str(metrics.get("reason") or "Keep one face clearly visible")
                continue

            x1, y1, x2, y2 = [int(value) for value in metrics.get("bounding_box") or [0, 0, 0, 0]]
            frame_height, frame_width = frame.shape[:2]
            x1, y1 = max(0, min(frame_width - 1, x1)), max(0, min(frame_height - 1, y1))
            x2, y2 = max(x1 + 1, min(frame_width, x2)), max(y1 + 1, min(frame_height, y2))
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())
            if focus < 16.0:
                feedback = "Hold the pose a little steadier"
                continue
            if brightness < 22.0 or brightness > 238.0:
                feedback = "Improve the lighting on the face"
                continue

            quality = float(metrics.get("quality") or 0.0)
            file_name = f"{prompt_id}_{int(time.time() * 1000)}_{sampled:04d}.jpg"
            destination = frame_dir / file_name
            if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 93]):
                continue
            stored = {
                **metrics,
                "accepted": True,
                "reason": None,
                "quality": round(quality, 1),
                "blur": round(focus, 1),
                "brightness": round(brightness, 1),
                "prompt_id": prompt_id,
                "captured_at": time.time(),
                "path": str(destination),
            }
            accepted.append(stored)
            best_quality = max(best_quality, quality)
            feedback = "Good DFM training burst"

        if not accepted:
            return {
                "session": self.status(session_id),
                "prompt_id": prompt_id,
                "sampled": sampled,
                "accepted": 0,
                "best_quality": None,
                "feedback": feedback,
            }
        with self._lock:
            session["captures"].extend(accepted)
        return {
            "session": self.status(session_id),
            "prompt_id": prompt_id,
            "sampled": sampled,
            "accepted": len(accepted),
            "best_quality": round(best_quality, 1),
            "feedback": feedback,
        }

    def status(self, session_id: str) -> Dict[str, Any]:
        session = self._get(session_id)
        captures = list(session.get("captures") or [])
        counts = {
            prompt["id"]: sum(1 for capture in captures if capture.get("prompt_id") == prompt["id"])
            for prompt in IDENTITY_CAPTURE_PROMPTS
        }
        return {
            "id": session_id,
            "name": session["name"],
            "created_at": session["created_at"],
            "consent_confirmed": True,
            "capturing": bool(session.get("capturing")),
            "kind": str(session.get("kind") or "instant"),
            "accepted_frames": len(captures),
            "recommended_frames": 2000 if session.get("kind") == "dfm" else 16,
            "prompt_counts": counts,
            "prompts": IDENTITY_CAPTURE_PROMPTS,
        }

    @staticmethod
    def _create_thumbnail(frame_path: Path, bounding_box: list[int], destination: Path) -> None:
        frame = cv2.imread(str(frame_path))
        if not isinstance(frame, np.ndarray) or not frame.size:
            return
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bounding_box
        center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        side = max(x2 - x1, y2 - y1) * 1.65
        left = max(0, int(center_x - side * 0.5))
        top = max(0, int(center_y - side * 0.55))
        right = min(width, int(left + side))
        bottom = min(height, int(top + side))
        crop = frame[top:bottom, left:right]
        if crop.size:
            crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])

    def finish(self, session_id: str) -> Dict[str, Any]:
        session = self._get(session_id)
        captures = list(session.get("captures") or [])
        completed_prompts = sorted({str(capture.get("prompt_id")) for capture in captures})
        is_dfm = session.get("kind") == "dfm"
        minimum_frames = 400 if is_dfm else 6
        minimum_prompts = 5 if is_dfm else 3
        if len(captures) < minimum_frames or len(completed_prompts) < minimum_prompts:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Capture at least 400 clear frames across five actions for a DFM dataset. "
                    "Around 2,000 frames produces a much healthier training set."
                    if is_dfm
                    else "Capture at least six clear frames across three guided actions before creating the identity."
                ),
            )
        selected: list[Dict[str, Any]] = []
        for prompt in IDENTITY_CAPTURE_PROMPTS:
            prompt_frames = [capture for capture in captures if capture.get("prompt_id") == prompt["id"]]
            prompt_frames.sort(key=lambda capture: float(capture.get("quality") or 0.0), reverse=True)
            selected.extend(prompt_frames[:2])
        selected = sorted(selected, key=lambda capture: float(capture.get("quality") or 0.0), reverse=True)[:16]

        profile_id = f"{_identity_slug(str(session['name']))}-{int(time.time())}"
        profile_path = IDENTITY_PROFILE_DIR / profile_id
        frames_path = profile_path / "frames"
        frames_path.mkdir(parents=True, exist_ok=False)
        copied_by_source: Dict[str, str] = {}
        profile_frames: list[Dict[str, Any]] = []
        for capture in captures:
            source = Path(str(capture["path"]))
            destination = frames_path / source.name
            shutil.copy2(source, destination)
            copied_by_source[str(source)] = str(destination)
            stored = dict(capture)
            stored["path"] = str(destination)
            stored["selected"] = any(str(item.get("path")) == str(source) for item in selected)
            profile_frames.append(stored)
        source_paths = [copied_by_source[str(capture["path"])] for capture in selected]
        best_front = [capture for capture in selected if capture.get("prompt_id") == "front-neutral"]
        best_capture = max(best_front or selected, key=lambda capture: float(capture.get("quality") or 0.0))
        self._create_thumbnail(
            Path(copied_by_source[str(best_capture["path"])]),
            list(best_capture.get("bounding_box") or [0, 0, 1, 1]),
            profile_path / "thumbnail.jpg",
        )
        profile = {
            "id": profile_id,
            "name": session["name"],
            "kind": "dfm-training-dataset" if is_dfm else "averaged-face-source",
            "created_at": time.time(),
            "consent_confirmed": True,
            "consent_confirmed_at": session["consent_confirmed_at"],
            "frame_count": len(profile_frames),
            "prompts_completed": completed_prompts,
            "source_paths": source_paths,
            "frames": profile_frames,
        }
        (profile_path / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
        (profile_path / "README.txt").write_text(
            "Guided FaceFlow identity dataset. Selected frames can be averaged by the realtime "
            "FaceFusion face-swapper. The complete frames folder is also the source faceset for "
            "a consent-compliant DeepFaceLab/DFM training workflow.\n",
            encoding="utf-8",
        )
        shutil.rmtree(Path(str(session["path"])), ignore_errors=True)
        with self._lock:
            self._sessions.pop(session_id, None)
        return profile

    def cancel(self, session_id: str) -> Dict[str, Any]:
        session = self._get(session_id)
        if session.get("capturing"):
            raise HTTPException(status_code=409, detail="Wait for the current guided capture to finish")
        shutil.rmtree(Path(str(session["path"])), ignore_errors=True)
        with self._lock:
            self._sessions.pop(session_id, None)
        if runtime.status().get("start_reason") == "trainer":
            runtime.stop()
        return {"ok": True, "runtime": runtime.status()}

    def profiles(self, active_paths: set[str]) -> list[Dict[str, Any]]:
        IDENTITY_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        profiles: list[Dict[str, Any]] = []
        for manifest_path in IDENTITY_PROFILE_DIR.glob("*/profile.json"):
            try:
                profile = json.loads(manifest_path.read_text(encoding="utf-8"))
                profiles.append(_identity_profile_summary(profile, active_paths))
            except Exception:
                continue
        return sorted(profiles, key=lambda profile: float(profile["created_at"]), reverse=True)


identity_trainer = IdentityTrainer()


def _register_trained_dfm(model_id: str) -> None:
    community_model_installer._register_model(model_id)


dfm_training = DfmTrainingManager(
    DFM_TRAINING_DIR,
    CUSTOM_MODEL_DIR,
    _read_identity_profile,
    _register_trained_dfm,
)


def get_config() -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update(engine._load_prefs())
    if config.get("colorizer_enabled") is not None and "frame_colorizer_enabled" not in config:
        config["frame_colorizer_enabled"] = bool(config["colorizer_enabled"])
    return _json_safe(config)


def _set_engine_state(config: Dict[str, Any]) -> None:
    for key, value in config.items():
        if key in ALLOWED_CONFIG_KEYS:
            try:
                engine.state_manager.set_item(key, value)
            except Exception:
                pass
    if config.get("morph") is not None:
        engine.state_manager.set_item("deep_swapper_morph", int(config["morph"]))
    engine._install_module_provider_routing(prefer_current_state=True)


def save_config(values: Dict[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(values) - ALLOWED_CONFIG_KEYS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown settings: {', '.join(unknown)}")

    config = get_config()
    clean_values = dict(values)
    resolution = clean_values.get("resolution_preset")
    if isinstance(resolution, str):
        match = re.fullmatch(r"(\d+)x(\d+)", resolution)
        if match:
            clean_values["width"] = int(match.group(1))
            clean_values["height"] = int(match.group(2))

    camera_choice = clean_values.get("camera_choice")
    if isinstance(camera_choice, str):
        clean_values["dshow_name_device"] = engine._camera_name_from_choice(camera_choice)

    config.update(clean_values)
    engine._save_prefs(config)
    _set_engine_state(config)
    engine.configure_virtual_camera(
        bool(config.get("virtual_cam_enabled")),
        int(config.get("width") or 1280),
        int(config.get("height") or 720),
        float(config.get("fps") or 30.0),
    )
    return config


def _clear_model_sessions(module: Any, model_ids: list[str]) -> None:
    for model_id in dict.fromkeys(model_id for model_id in model_ids if model_id):
        try:
            engine.ff_inference_manager.clear_inference_pool(module.__name__, [model_id])
        except Exception:
            pass


def _warm_deep_model_session(session: Any) -> None:
    """Run one neutral frame so engine construction is finished before handoff."""
    inputs: Dict[str, np.ndarray] = {}
    for model_input in session.get_inputs():
        if model_input.name == "in_face:0":
            shape = list(model_input.shape)
            height = int(shape[1]) if len(shape) > 1 and isinstance(shape[1], int) else 224
            width = int(shape[2]) if len(shape) > 2 and isinstance(shape[2], int) else height
            inputs[model_input.name] = np.zeros((1, height, width, 3), dtype=np.float32)
        elif model_input.name == "morph_value:0":
            inputs[model_input.name] = np.ones((1,), dtype=np.float32)
        else:
            raise RuntimeError(f"Unsupported DFM input while warming model: {model_input.name}")
    session.run(None, inputs)


def _prepare_deep_model_sessions(model_id: str) -> Dict[str, Any]:
    """Create and warm future DFM sessions without changing the active model."""
    model_options = engine.deep_swapper.create_static_model_set("full").get(model_id)
    if not model_options:
        raise HTTPException(status_code=422, detail=f"Unknown face model: {model_id}")

    model_sources = model_options.get("sources") or {}
    model_source = model_sources.get("deep_swapper") or {}
    model_path = Path(str(model_source.get("path") or ""))
    if not model_path.is_file():
        raise HTTPException(status_code=409, detail="Install this face model before selecting it")

    execution_device_ids = engine.state_manager.get_item("execution_device_ids") or ["0"]
    execution_providers = engine.ff_inference_manager.resolve_execution_providers(engine.deep_swapper.__name__)
    prepared: Dict[str, Any] = {}
    for execution_device_id in execution_device_ids:
        device_id = str(execution_device_id)
        inference_context = engine.ff_inference_manager.get_inference_context(
            engine.deep_swapper.__name__,
            [model_id],
            device_id,
            execution_providers,
        )
        inference_pool = engine.ff_inference_manager.create_inference_pool(
            engine.deep_swapper.__name__,
            model_sources,
            device_id,
            execution_providers,
        )
        session = inference_pool.get("deep_swapper")
        if session is None:
            raise RuntimeError(f"Could not create an inference session for {model_id}")
        _warm_deep_model_session(session)
        prepared[inference_context] = inference_pool
    return prepared


def _install_deep_model_sessions(prepared: Dict[str, Any]) -> None:
    for app_context in ("cli", "ui"):
        engine.ff_inference_manager.INFERENCE_POOL_SET[app_context].update(prepared)


def _remove_deep_model_sessions(model_id: str) -> None:
    execution_device_ids = engine.state_manager.get_item("execution_device_ids") or ["0"]
    execution_providers = engine.ff_inference_manager.resolve_execution_providers(engine.deep_swapper.__name__)
    for execution_device_id in execution_device_ids:
        inference_context = engine.ff_inference_manager.get_inference_context(
            engine.deep_swapper.__name__,
            [model_id],
            str(execution_device_id),
            execution_providers,
        )
        for app_context in ("cli", "ui"):
            engine.ff_inference_manager.INFERENCE_POOL_SET[app_context].pop(inference_context, None)


def save_seamless_deep_model_config(values: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the old DFM live while the replacement session is prepared."""
    old_model_id = str(previous["deep_swapper_model"])
    new_model_id = str(values["deep_swapper_model"])
    with DEEP_MODEL_HANDOFF_LOCK:
        prepared = _prepare_deep_model_sessions(new_model_id)
        # Frame processing takes the same lock. The state and prepared cache
        # therefore become visible together between two processed frames.
        with engine._live_processing_lock:
            _install_deep_model_sessions(prepared)
            try:
                config = save_config(values)
                _apply_processing_config(config)
            except Exception:
                _remove_deep_model_sessions(new_model_id)
                raise
            _remove_deep_model_sessions(old_model_id)
        return config


def save_live_config(
    values: Dict[str, Any],
    previous: Dict[str, Any],
    clear_models: bool = False,
) -> Dict[str, Any]:
    """Apply non-capture settings between frames without reopening the webcam."""
    with engine._live_processing_lock:
        config = save_config(values)
        _apply_processing_config(config)
        if "source_paths" in values:
            engine.set_source_paths(config.get("source_paths") or [])
        changed_keys = {key for key, value in values.items() if previous.get(key) != value}
        if clear_models or changed_keys & INFERENCE_RELOAD_KEYS:
            engine._cleanup_inference()
        if "virtual_cam_enabled" in values and not config.get("virtual_cam_enabled"):
            engine.close_virtual_camera()
        return config


# Backward-compatible name for callers from earlier builds.
save_live_model_config = save_live_config


def _parse_metrics(status_text: str) -> Dict[str, Any]:
    patterns = {
        "processed_fps": r"([\d.]+) processed FPS",
        "processing_ms": r"([\d.]+) ms processing",
        "end_to_end_ms": r"([\d.]+) ms end-to-end",
        "camera_frames_skipped_percent": r"([\d.]+)% camera frames skipped",
        "preview_fps": r"UI preview ([\d.]+) FPS",
        "virtual_camera_fps": r"virtual cam \d+x\d+ at ([\d.]+) FPS",
    }
    metrics: Dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, status_text)
        if match:
            metrics[key] = float(match.group(1))
    resolution = re.search(r"virtual cam (\d+x\d+)", status_text)
    if resolution:
        metrics["virtual_camera_resolution"] = resolution.group(1)
    backend = re.search(r"via ([\w ._-]+?)(?: \||$)", status_text)
    if backend:
        metrics["virtual_camera_backend"] = backend.group(1).strip()
    metrics["benchmark_running"] = "benchmark" in status_text and "remaining" in status_text
    remaining = re.search(r"benchmark ([\d.]+)s remaining", status_text)
    if remaining:
        metrics["benchmark_remaining_seconds"] = float(remaining.group(1))
    return metrics


class RuntimeController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._frame_ready = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._latest_jpeg: Optional[bytes] = None
        self._preview_sequence = 0
        self._status_text = "Camera off"
        self._error: Optional[str] = None
        self._running = False
        self._started_at: Optional[float] = None
        self._start_reason: Optional[str] = None
        self._processing_mode = "last"
        self._preview_clients = 0
        self._demand_monitor_enabled = False
        self._external_demand = False

    def _stream_args(self, config: Dict[str, Any]) -> Dict[str, Any]:
        providers = config.get("execution_providers") or ["cuda"]
        devices = config.get("execution_device_ids") or ["0"]
        return {
            "camera_choice": str(config["camera_choice"]),
            "width": int(config["width"]),
            "height": int(config["height"]),
            "use_occlusion": bool(config["use_occlusion"]),
            "target_fps": float(config["fps"]),
            "backend_name": str(config["backend"]),
            "dshow_name": str(config.get("dshow_name_device") or ""),
            "convert_rgb": bool(config["convert_rgb"]),
            "force_fourcc": str(config["force_fourcc"]),
            "retry_black": int(config["retry_black"]),
            "gentle_mode": bool(config["gentle_mode"]),
            "auto_repair": bool(config["auto_repair"]),
            "color_mode": str(config["color_mode"]),
            "lock_exposure": bool(config["lock_exposure"]),
            "exposure_value": float(config["exposure_value"]),
            "lock_wb": bool(config["lock_wb"]),
            "wb_temperature": int(config["wb_temperature"]),
            "show_overlay": bool(config["show_overlay"]),
            "debug_logs": bool(config["debug_logs"]),
            "show_boxes": bool(config["show_boxes"]),
            "show_native_window": bool(config["show_native"]),
            "virtual_cam_enabled": bool(config["virtual_cam_enabled"]),
            "colorizer_enabled": bool(config["frame_colorizer_enabled"]),
            "expr_enabled": bool(config["expression_restorer_enabled"]),
            "age_enabled": bool(config["age_modifier_enabled"]),
            "editor_enabled": bool(config["face_editor_enabled"]),
            "debugger_enabled": bool(config["face_debugger_enabled"]),
            "lip_enabled": bool(config["lip_syncer_enabled"]),
            "selector_mode": str(config["selector_mode"]),
            "auto_fallback": bool(config["auto_fallback"]),
            "exec_provider_key": str(providers[0]),
            "exec_device_id": str(devices[0]),
            "video_memory_strategy": str(config["video_memory_strategy"]),
            "detector_model": str(config["detector_model"]),
            "detector_size": str(config["detector_size"]),
            "detector_score": float(config["detector_score"]),
            "landmarker_model": str(config["landmarker_model"]),
            "landmarker_score": float(config["landmarker_score"]),
            "occluder_model": str(config["occluder_model"]),
            "parser_model": str(config["parser_model"]),
            "swap_mode": str(config["swap_mode"]),
            "source_files": config.get("source_paths") or [],
            "deep_model": str(config["deep_swapper_model"]),
            "morph": int(config["morph"]),
            "face_swapper_model": str(config["face_swapper_model"]),
            "face_swapper_pixel_boost": str(config["face_swapper_pixel_boost"]),
            "face_swapper_weight": float(config["face_swapper_weight"]),
            "face_enhancer_enabled": bool(config["face_enhancer_enabled"]),
            "face_enhancer_model": str(config["face_enhancer_model"]),
            "face_enhancer_blend": int(config["face_enhancer_blend"]),
            "face_enhancer_weight": float(config["face_enhancer_weight"]),
            "frame_enhancer_enabled": bool(config["frame_enhancer_enabled"]),
            "frame_enhancer_model": str(config["frame_enhancer_model"]),
            "frame_enhancer_blend": int(config["frame_enhancer_blend"]),
            "enhance_async": bool(config["enhance_async"]),
        }

    def _run(self) -> None:
        try:
            config = dict(get_config())
            with self._lock:
                processing_mode = self._processing_mode
            passthrough = processing_mode == "none"
            if passthrough:
                config.update(
                    {
                        "swap_mode": "none",
                        "realtime_fast_analysis": True,
                        "use_occlusion": False,
                        "face_enhancer_enabled": False,
                        "frame_enhancer_enabled": False,
                        "frame_colorizer_enabled": False,
                        "expression_restorer_enabled": False,
                        "age_modifier_enabled": False,
                        "face_editor_enabled": False,
                        "face_debugger_enabled": False,
                        "lip_syncer_enabled": False,
                    }
                )
            _set_engine_state(config)
            engine.state_manager.set_item("realtime_passthrough", passthrough)
            stream = engine.gr_stream(**self._stream_args(config))
            for preview_rgb, status_text in stream:
                if not isinstance(preview_rgb, np.ndarray) or not preview_rgb.size:
                    continue
                with self._lock:
                    preview_requested = self._preview_clients > 0
                jpeg = None
                if preview_requested:
                    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
                    encoded, buffer = cv2.imencode(".jpg", preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 84])
                    if encoded:
                        jpeg = buffer.tobytes()
                with self._frame_ready:
                    if jpeg is not None:
                        self._latest_jpeg = jpeg
                        self._preview_sequence += 1
                    self._status_text = str(status_text).replace("**", "")
                    self._error = None
                    self._frame_ready.notify_all()
        except Exception as exc:
            with self._frame_ready:
                self._error = str(exc) or type(exc).__name__
                self._status_text = f"Stream stopped: {self._error}"
                self._frame_ready.notify_all()
        finally:
            engine.release_camera_capture()
            engine.release_virtual_camera_source()
            engine.state_manager.set_item("realtime_passthrough", False)
            with self._frame_ready:
                self._running = False
                self._thread = None
                self._started_at = None
                self._start_reason = None
                self._latest_jpeg = None
                self._preview_sequence += 1
                if self._error is None:
                    self._status_text = "Camera off"
                self._frame_ready.notify_all()

    def start(self, reason: str = "manual", processing_mode: str = "last") -> Dict[str, Any]:
        processing_mode = "none" if processing_mode == "none" else "last"
        with self._lock:
            if self._running:
                if reason == "manual":
                    self._start_reason = "manual"
                return self.status()
            self._running = True
            self._error = None
            self._status_text = "Starting camera and models…"
            self._started_at = time.time()
            self._start_reason = reason
            self._processing_mode = processing_mode
            self._latest_jpeg = None
            self._preview_sequence += 1
            self._thread = threading.Thread(target=self._run, name="web-api-stream", daemon=True)
            self._thread.start()
            return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            thread = self._thread
            self._status_text = "Stopping…"
        engine.gr_stop_stream()
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=10.0)
        with self._frame_ready:
            still_stopping = bool(thread and thread.is_alive())
            self._running = still_stopping
            self._thread = thread if still_stopping else None
            self._started_at = self._started_at if still_stopping else None
            self._start_reason = self._start_reason if still_stopping else None
            self._latest_jpeg = None
            self._preview_sequence += 1
            self._status_text = "Camera off; finishing current frame..." if still_stopping else "Camera off"
            self._frame_ready.notify_all()
        return self.status()

    def restart(self, clear_models: bool = False) -> Dict[str, Any]:
        self.stop()
        if clear_models:
            engine._cleanup_inference()
        return self.start()

    def set_processing_mode(self, processing_mode: str) -> Dict[str, Any]:
        """Switch a running passthrough stream to its saved processing setup."""
        next_mode = "none" if processing_mode == "none" else "last"
        with self._lock:
            self._processing_mode = next_mode
            if next_mode == "last" and self._start_reason == "trainer":
                self._start_reason = "manual"
        engine.state_manager.set_item("realtime_passthrough", next_mode == "none")
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            capture_active = engine.is_camera_capture_active()
            virtual_status = engine.virtual_camera_status()
            result = {
                "running": self._running,
                "capture_active": capture_active,
                "status": self._status_text,
                "error": self._error,
                "preview_sequence": self._preview_sequence,
                "has_preview": self._latest_jpeg is not None,
                "started_at": self._started_at,
                "start_reason": self._start_reason,
                "processing_mode": self._processing_mode if self._running else None,
                "preview_clients": self._preview_clients,
                "demand_monitor_enabled": self._demand_monitor_enabled,
                "external_camera_demand": self._external_demand,
                "virtual_camera_advertised": bool(virtual_status.get("advertised")),
                "virtual_camera_streaming": bool(virtual_status.get("source_active")),
                "virtual_camera_error": virtual_status.get("error"),
                "virtual_camera_backend": virtual_status.get("backend"),
                "virtual_camera_resolution": (
                    f"{virtual_status.get('width')}x{virtual_status.get('height')}"
                ),
                "temporal_occlusion": engine.temporal_occlusion_status(),
                "temporal_face_tracking": engine.temporal_face_tracking_status(),
                "gpu_pipeline": engine.gpu_pipeline_status(),
                "uptime_seconds": round(time.time() - self._started_at, 1) if self._running and self._started_at else 0,
            }
            result.update(_parse_metrics(self._status_text))
            return result

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def mjpeg(self) -> Iterator[bytes]:
        delivered = -1
        with self._frame_ready:
            self._preview_clients += 1
            self._frame_ready.notify_all()
        try:
            while True:
                with self._frame_ready:
                    self._frame_ready.wait_for(
                        lambda: self._preview_sequence != delivered or self._error is not None,
                        timeout=10.0,
                    )
                    sequence = self._preview_sequence
                    jpeg = self._latest_jpeg
                if jpeg is None or sequence == delivered:
                    continue
                delivered = sequence
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n" + jpeg + b"\r\n"
        finally:
            with self._frame_ready:
                self._preview_clients = max(0, self._preview_clients - 1)
                if self._preview_clients == 0:
                    self._latest_jpeg = None
                    self._preview_sequence += 1
                self._frame_ready.notify_all()

    def set_demand_state(self, enabled: bool, active: bool) -> None:
        with self._lock:
            self._demand_monitor_enabled = bool(enabled)
            self._external_demand = bool(active)


def _apply_processing_config(config: Dict[str, Any], provider_override: Optional[str] = None) -> None:
    _set_engine_state(config)
    engine.apply_state_from_ui(
        str(config["detector_model"]),
        str(config["detector_size"]),
        float(config["detector_score"]),
        str(config["landmarker_model"]),
        float(config["landmarker_score"]),
        str(config["occluder_model"]),
        str(config["parser_model"]),
        str(config["deep_swapper_model"]),
        int(config["morph"]),
        bool(config["use_occlusion"]),
        str(config["selector_mode"]),
    )
    for key in (
        "swap_mode",
        "face_swapper_model",
        "face_swapper_pixel_boost",
        "face_swapper_weight",
        "source_paths",
        "face_enhancer_enabled",
        "face_enhancer_model",
        "face_enhancer_blend",
        "face_enhancer_weight",
        "frame_enhancer_enabled",
        "frame_enhancer_model",
        "frame_enhancer_blend",
        "enhance_async",
        "frame_colorizer_enabled",
        "frame_colorizer_model",
        "frame_colorizer_size",
        "frame_colorizer_blend",
        "expression_restorer_enabled",
        "expression_restorer_model",
        "expression_restorer_factor",
        "expression_restorer_areas",
        "age_modifier_enabled",
        "age_modifier_model",
        "age_modifier_direction",
        "face_editor_enabled",
        "face_editor_model",
        "face_debugger_enabled",
        "lip_syncer_enabled",
        "lip_syncer_model",
        "lip_syncer_weight",
        "video_memory_strategy",
        "execution_device_ids",
    ):
        try:
            engine.state_manager.set_item(key, config[key])
        except KeyError:
            pass
    if provider_override:
        engine.state_manager.set_item("execution_providers", [provider_override])
        for route_key, _label, _module in engine._MODULE_PROVIDER_ROUTES:
            engine.state_manager.set_item(route_key, provider_override)
    engine._install_module_provider_routing(prefer_current_state=True)


def _load_provider_benchmark_frames(limit: int = 7) -> list[np.ndarray]:
    # The original 10-second capture currently contains an empty chair. Prefer
    # saved webcam frames with a visible face so the face-analysis and active
    # swapping stages are included in the provider comparison.
    face_frame_paths = [
        BASE_DIR / "benchmarks" / "cpu_split_presence.jpg",
        BASE_DIR / "benchmarks" / "presence_check.jpg",
    ]
    face_frames = [cv2.imread(str(path)) for path in face_frame_paths if path.exists()]
    face_frames = [frame for frame in face_frames if isinstance(frame, np.ndarray) and frame.size]
    if face_frames:
        return [
            cv2.resize(face_frames[index % len(face_frames)], (1280, 720), interpolation=cv2.INTER_CUBIC)
            for index in range(limit)
        ]

    video_path = BASE_DIR / "benchmarks" / "webcam_sample_10s.avi"
    frames: list[np.ndarray] = []
    if video_path.exists():
        capture = cv2.VideoCapture(str(video_path))
        try:
            frame_total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            target_indices = [int((index + 1) * frame_total / (limit + 1)) for index in range(limit)]
            for target_index in target_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
                ok, frame = capture.read()
                if ok and isinstance(frame, np.ndarray) and frame.size:
                    frames.append(cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA))
        finally:
            capture.release()
    if not frames:
        fallback_path = BASE_DIR / "benchmarks" / "current_snapshot.jpg"
        fallback = cv2.imread(str(fallback_path)) if fallback_path.exists() else None
        if isinstance(fallback, np.ndarray) and fallback.size:
            resized = cv2.resize(fallback, (1280, 720), interpolation=cv2.INTER_AREA)
            frames = [resized.copy() for _ in range(limit)]
    if len(frames) < 3:
        raise RuntimeError("No recorded webcam frames are available for the provider benchmark")
    return frames


class ProviderTimingBenchmark:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._phase = "idle"
        self._progress = 0.0
        self._error: Optional[str] = None
        self._result: Optional[Dict[str, Any]] = None
        self._started_at: Optional[float] = None
        if PROVIDER_TIMING_PATH.exists():
            try:
                self._result = json.loads(PROVIDER_TIMING_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._result = None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "phase": self._phase,
                "progress": round(self._progress, 3),
                "error": self._error,
                "started_at": self._started_at,
                "result": self._result,
            }

    def start(self) -> Dict[str, Any]:
        with self._lock:
            if self._running:
                return self.status()
            self._running = True
            self._phase = "Preparing recorded webcam frames"
            self._progress = 0.01
            self._error = None
            self._started_at = time.time()
            self._thread = threading.Thread(target=self._run, name="provider-timing-benchmark", daemon=True)
            self._thread.start()
            return self.status()

    def _set_progress(self, phase: str, progress: float) -> None:
        with self._lock:
            self._phase = phase
            self._progress = progress

    def _run_provider(self, provider: str, frames: list[np.ndarray], config: Dict[str, Any], provider_index: int) -> Dict[str, Any]:
        self._set_progress(f"Loading {provider.upper()} inference sessions", provider_index * 0.5 + 0.04)
        _apply_processing_config(config, provider)
        engine.state_manager.set_item("realtime_fast_analysis", False)
        engine._cleanup_inference()
        face_store.clear_static_faces()

        # Session creation and graph optimization are excluded from the measured samples.
        engine.process_frame(frames[0].copy(), True, False, False, False)
        engine.reset_inference_timings()
        face_store.clear_static_faces()

        measured_frames = frames[1:]
        frame_latencies: list[float] = []
        for frame_index, frame in enumerate(measured_frames):
            self._set_progress(
                f"Profiling {provider.upper()} frame {frame_index + 1}/{len(measured_frames)}",
                provider_index * 0.5 + 0.1 + 0.34 * ((frame_index + 1) / len(measured_frames)),
            )
            face_store.clear_static_faces()
            started_at = time.perf_counter()
            engine.process_frame(frame.copy(), True, False, False, False)
            frame_latencies.append((time.perf_counter() - started_at) * 1000.0)

        summary = engine.get_inference_timing_summary()
        sorted_frames = sorted(frame_latencies)
        p95_index = max(0, min(len(sorted_frames) - 1, int(len(sorted_frames) * 0.95) - 1))
        return {
            "provider": provider,
            "frames": len(frame_latencies),
            "pipeline_mean_ms": round(sum(frame_latencies) / len(frame_latencies), 3),
            "pipeline_median_ms": round(sorted_frames[len(sorted_frames) // 2], 3),
            "pipeline_p95_ms": round(sorted_frames[p95_index], 3),
            "modules": summary.get("modules", {}),
        }

    def _run(self) -> None:
        saved_config = get_config()
        was_running = runtime.status()["running"]
        result: Optional[Dict[str, Any]] = None
        try:
            if was_running:
                runtime.stop()
            frames = _load_provider_benchmark_frames()
            benchmark_config = dict(saved_config)
            benchmark_config["realtime_fast_analysis"] = False
            benchmark_config["selector_mode"] = "one"
            available = engine._available_execution_provider_keys()
            providers = [provider for provider in ("cuda", "cpu") if provider in available]
            if "cpu" not in providers:
                raise RuntimeError("CPUExecutionProvider is not available")
            provider_results: Dict[str, Any] = {}
            for provider_index, provider in enumerate(providers):
                provider_results[provider] = self._run_provider(provider, frames, benchmark_config, provider_index)

            comparisons: Dict[str, Any] = {}
            for route_key, label, _module in engine._MODULE_PROVIDER_ROUTES:
                cuda_entry = provider_results.get("cuda", {}).get("modules", {}).get(route_key, {}).get("providers", {}).get("cuda")
                cpu_entry = provider_results.get("cpu", {}).get("modules", {}).get(route_key, {}).get("providers", {}).get("cpu")
                cuda_frame = (cuda_entry or {}).get("per_frame")
                cpu_frame = (cpu_entry or {}).get("per_frame")
                cuda_call = (cuda_entry or {}).get("per_call")
                cpu_call = (cpu_entry or {}).get("per_call")
                cuda_ms = (cuda_frame or cuda_call or {}).get("mean_ms")
                cpu_ms = (cpu_frame or cpu_call or {}).get("mean_ms")
                comparisons[route_key] = {
                    "label": label,
                    "cuda": cuda_entry,
                    "cpu": cpu_entry,
                    "cuda_mean_ms": cuda_ms,
                    "cpu_mean_ms": cpu_ms,
                    "cpu_to_cuda_ratio": round(cpu_ms / cuda_ms, 2) if cuda_ms and cpu_ms else None,
                    "active": bool(cuda_entry or cpu_entry),
                }
            result = {
                "created_at": time.time(),
                "machine": "AMD Ryzen 9 9950X3D + NVIDIA RTX 5080",
                "resolution": "1280x720 recorded webcam frames",
                "model": saved_config.get("deep_swapper_model"),
                "swap_mode": saved_config.get("swap_mode"),
                "forced_full_analysis": True,
                "frames_per_provider": max(0, len(frames) - 1),
                "providers": provider_results,
                "comparisons": comparisons,
            }
            PROVIDER_TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
            PROVIDER_TIMING_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
            with self._lock:
                self._result = result
                self._phase = "Complete"
                self._progress = 1.0
        except Exception as exc:
            with self._lock:
                self._error = str(exc) or type(exc).__name__
                self._phase = "Failed"
        finally:
            try:
                _apply_processing_config(saved_config)
                engine.state_manager.set_item("realtime_fast_analysis", bool(saved_config.get("realtime_fast_analysis", True)))
                engine._cleanup_inference()
                face_store.clear_static_faces()
                if was_running:
                    runtime.start()
            except Exception as restore_error:
                with self._lock:
                    self._error = self._error or f"Could not restore live stream: {restore_error}"
            with self._lock:
                self._running = False
                self._thread = None


provider_timing_benchmark = ProviderTimingBenchmark()


runtime = RuntimeController()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    runtime.stop()
    engine.release_camera_capture()
    engine.shutdown_virtual_camera()


app = FastAPI(
    title="Webcam FaceFusion API",
    version="1.0.0",
    description="Realtime camera, FaceFusion processing, benchmark and virtual-camera control API.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _camera_payload(force_refresh: bool = False, selected_override: Optional[str] = None) -> Dict[str, Any]:
    config = get_config()
    cameras = engine.get_camera_choices(
        max_index=6,
        backend_name=str(config["backend"]),
        dshow_name_hint=str(config.get("dshow_name_device") or ""),
    )
    selected = str(selected_override or config.get("camera_choice") or (cameras[0] if cameras else ""))
    if selected not in cameras and cameras:
        selected = cameras[0]
    camera_name = engine._camera_name_from_choice(selected)
    modes = engine.detect_camera_capabilities(camera_name, force_refresh=force_refresh) if camera_name else {}
    resolutions = engine._sorted_camera_resolutions(modes)
    return {
        "cameras": cameras,
        "selected": selected,
        "camera_name": camera_name,
        "modes": modes,
        "resolutions": resolutions,
        "max_fps": max(modes.values()) if modes else 30.0,
    }


def _options_payload() -> Dict[str, Any]:
    models = engine.get_model_choices()
    return _json_safe(
        {
            **models,
            "deep_model_catalog": _model_catalog(),
            "execution_providers": engine._available_execution_provider_keys(),
            "execution_routes": [
                {"key": key, "label": label} for key, label, _module in engine._MODULE_PROVIDER_ROUTES
            ],
            "selector_modes": engine.ff_choices.face_selector_modes,
            "frame_colorizer_models": engine.frame_colorizer_choices.frame_colorizer_models,
            "frame_colorizer_sizes": engine.frame_colorizer_choices.frame_colorizer_sizes,
            "expression_restorer_models": engine.expression_restorer_choices.expression_restorer_models,
            "expression_restorer_areas": engine.expression_restorer_choices.expression_restorer_areas,
            "age_modifier_models": engine.age_modifier_choices.age_modifier_models,
            "face_editor_models": engine.face_editor_choices.face_editor_models,
            "face_debugger_items": engine.face_debugger_choices.face_debugger_items,
            "lip_syncer_models": engine.lip_syncer_choices.lip_syncer_models,
            "face_swapper_sizes": engine.proc_choices.face_swapper_set,
            "backends": ["Auto", "DirectShow", "Media Foundation", "OpenCV"],
            "fourcc": ["Auto", "MJPG", "YUY2", "H264", "NV12"],
            "video_memory_strategies": ["strict", "moderate", "relaxed"],
        }
    )


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "api": "Webcam FaceFusion", "version": app.version}


@app.get("/api/bootstrap")
def bootstrap() -> Dict[str, Any]:
    return {"config": get_config(), "options": _options_payload(), "camera": _camera_payload(False), "runtime": runtime.status()}


@app.get("/api/config")
def read_config() -> Dict[str, Any]:
    return get_config()


@app.patch("/api/config")
def patch_config(payload: ConfigPatch) -> Dict[str, Any]:
    previous = get_config()
    runtime_before = runtime.status()
    changed_keys = {key for key, value in payload.values.items() if previous.get(key) != value}
    capture_changed = bool(changed_keys & CAPTURE_SOURCE_KEYS)
    inference_changed = bool(changed_keys & INFERENCE_RELOAD_KEYS)
    restart_required = bool(runtime_before["running"] and capture_changed)
    seamless_model_handoff = bool(
        runtime_before["running"]
        and changed_keys == {"deep_swapper_model"}
        and not capture_changed
    )

    if seamless_model_handoff:
        config = save_seamless_deep_model_config(payload.values, previous)
    elif runtime_before["running"] and not capture_changed:
        config = save_live_config(payload.values, previous, payload.clear_models)
    else:
        config = save_config(payload.values)
        # A model selected while capture is off used to leave every previously
        # used ONNX context resident. Clear once at the configuration boundary;
        # the selected pipeline will then remain warm across camera stops.
        if inference_changed or payload.clear_models:
            engine._cleanup_inference()

    if changed_keys & {"use_occlusion", "temporal_occlusion_reuse", "temporal_occlusion_interval"}:
        engine.reset_temporal_occlusion_cache()
    if changed_keys & {
        "temporal_face_tracking",
        "temporal_face_interval",
        "selector_mode",
        "detector_model",
        "detector_size",
        "landmarker_model",
    }:
        engine.reset_temporal_face_tracker()

    # The backend is the authority here: an old UI may still request a restart
    # for a detector or processor setting, but only capture-source changes are
    # allowed to release and reopen the physical webcam.
    runtime_state = runtime.restart(payload.clear_models) if restart_required else runtime.status()
    return {
        "config": config,
        "runtime": runtime_state,
        "capture_restarted": restart_required,
        "applied_live": bool(runtime_before["running"] and changed_keys and not capture_changed),
        "seamless_model_handoff": seamless_model_handoff,
    }


@app.get("/api/cameras")
def cameras(refresh: bool = False, camera_choice: Optional[str] = None) -> Dict[str, Any]:
    return _camera_payload(refresh, camera_choice)


@app.get("/api/models")
def models() -> Dict[str, Any]:
    return {"models": _model_catalog(), "install": community_model_installer.status()}


@app.get("/api/models/thumbnail")
def model_thumbnail(model_id: str) -> Response:
    model = _model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Unknown model")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_id).strip("_")
    cache_key = format(zlib.crc32(model_id.encode("utf-8")) & 0xFFFFFFFF, "08x")
    destination = MODEL_THUMBNAIL_DIR / f"{safe_name}_{cache_key}.jpg"
    needs_refresh = not destination.exists() or (destination.stat().st_size < 5000 and time.time() - destination.stat().st_mtime > 15)
    if needs_refresh:
        with MODEL_THUMBNAIL_SEMAPHORE:
            needs_refresh = not destination.exists() or (destination.stat().st_size < 5000 and time.time() - destination.stat().st_mtime > 15)
            if needs_refresh:
                _fetch_model_thumbnail(model, destination)
    cache_control = "no-store" if destination.stat().st_size < 5000 else "public, max-age=604800"
    return FileResponse(destination, media_type="image/jpeg", headers={"Cache-Control": cache_control})


@app.post("/api/models/install")
def install_model(payload: ModelInstallRequest) -> Dict[str, Any]:
    return community_model_installer.start(payload.model_id)


@app.post("/api/runtime/start")
def start_runtime() -> Dict[str, Any]:
    return runtime.start()


@app.post("/api/runtime/stop")
def stop_runtime() -> Dict[str, Any]:
    return runtime.stop()


@app.post("/api/capture/start")
def start_capture() -> Dict[str, Any]:
    """Start webcam capture and its realtime processing pipeline."""
    return runtime.start()


@app.post("/api/capture/stop")
def stop_capture() -> Dict[str, Any]:
    """Stop processing and release the physical webcam."""
    return runtime.stop()


@app.post("/api/runtime/restart")
def restart_runtime(action: RuntimeAction) -> Dict[str, Any]:
    return runtime.restart(action.clear_models)


@app.post("/api/runtime/free-vram")
def free_vram() -> Dict[str, Any]:
    was_running = runtime.status()["running"]
    if was_running:
        runtime.stop()
    engine._cleanup_inference()
    return {"ok": True, "runtime": runtime.start() if was_running else runtime.status()}


@app.get("/api/status")
def runtime_status() -> Dict[str, Any]:
    return runtime.status()


@app.get("/api/timings")
def inference_timings() -> Dict[str, Any]:
    return {
        "live": engine.get_inference_timing_summary(),
        "benchmark": provider_timing_benchmark.status(),
    }


@app.post("/api/timings/benchmark")
def start_provider_timing_benchmark() -> Dict[str, Any]:
    return provider_timing_benchmark.start()


@app.get("/api/preview.mjpg")
def preview_stream() -> StreamingResponse:
    return StreamingResponse(runtime.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


def _raw_preview_mjpeg() -> Iterator[bytes]:
    sequence = -1
    while runtime.status().get("running"):
        frame, sequence, _captured_at = engine.get_latest_raw_frame(sequence, timeout=2.0)
        if not isinstance(frame, np.ndarray) or not frame.size:
            continue
        height, width = frame.shape[:2]
        if width > 720:
            frame = cv2.resize(frame, (720, max(1, int(height * 720 / width))), interpolation=cv2.INTER_AREA)
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if encoded:
            payload = buffer.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"


@app.get("/api/raw-preview.mjpg")
def raw_preview_stream() -> StreamingResponse:
    return StreamingResponse(_raw_preview_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/preview.jpg")
def preview_frame() -> Response:
    jpeg = runtime.latest_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=404, detail="No preview frame is available yet")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/benchmark")
def start_benchmark() -> Dict[str, Any]:
    if not runtime.status()["running"]:
        raise HTTPException(status_code=409, detail="Start the stream before benchmarking")
    return {"ok": True, "message": engine.request_realtime_benchmark()}


@app.get("/api/benchmark/latest")
def latest_benchmark() -> Dict[str, Any]:
    if not BENCHMARK_PATH.exists():
        raise HTTPException(status_code=404, detail="No benchmark has completed yet")
    try:
        return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read benchmark: {exc}") from exc


@app.post("/api/source")
async def upload_source(file: UploadFile = File(...)) -> Dict[str, Any]:
    suffix = Path(file.filename or "source.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, WEBP or BMP image")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(file.filename or "source").stem).strip("_") or "source"
    destination = UPLOAD_DIR / f"{int(time.time() * 1000)}_{safe_stem}{suffix}"
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Source image must be smaller than 20 MB")
    destination.write_bytes(data)
    config = get_config()
    paths = list(config.get("source_paths") or [])
    paths.append(str(destination))
    save_config({"source_paths": paths})
    return {"ok": True, "path": str(destination), "source_paths": paths}


@app.delete("/api/source")
def clear_sources() -> Dict[str, Any]:
    engine.set_source_paths([])
    save_config({"source_paths": []})
    return {"ok": True, "source_paths": []}


def _activate_identity_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    source_paths = [str(path) for path in profile.get("source_paths") or [] if Path(str(path)).is_file()]
    if not source_paths:
        raise HTTPException(status_code=409, detail="This identity profile has no usable source frames")
    values = {"source_paths": source_paths, "swap_mode": "face"}
    previous = get_config()
    if runtime.status().get("running"):
        config = save_live_config(values, previous, clear_models=False)
        runtime.set_processing_mode("last")
    else:
        config = save_config(values)
        engine.set_source_paths(source_paths)
    return {
        "ok": True,
        "profile": _identity_profile_summary(profile, set(source_paths)),
        "config": config,
        "runtime": runtime.status(),
    }


@app.get("/api/identities")
def identity_profiles() -> Dict[str, Any]:
    active_paths = {str(path) for path in get_config().get("source_paths") or []}
    return {"profiles": identity_trainer.profiles(active_paths), "prompts": IDENTITY_CAPTURE_PROMPTS}


@app.post("/api/identities/session")
def start_identity_session(payload: IdentitySessionRequest) -> Dict[str, Any]:
    return identity_trainer.start(payload.name, payload.consent_confirmed, payload.kind)


@app.get("/api/identities/session/{session_id}")
def identity_session_status(session_id: str) -> Dict[str, Any]:
    return identity_trainer.status(session_id)


@app.post("/api/identities/session/{session_id}/capture")
def capture_identity_action(session_id: str, payload: IdentityCaptureRequest) -> Dict[str, Any]:
    return identity_trainer.capture(session_id, payload.prompt_id, payload.duration_seconds)


@app.delete("/api/identities/session/{session_id}")
def cancel_identity_session(session_id: str) -> Dict[str, Any]:
    return identity_trainer.cancel(session_id)


@app.post("/api/identities/session/{session_id}/finish")
def finish_identity_session(session_id: str) -> Dict[str, Any]:
    profile = identity_trainer.finish(session_id)
    if profile.get("kind") == "dfm-training-dataset":
        if runtime.status().get("start_reason") == "trainer":
            runtime.stop()
        active_paths = {str(path) for path in get_config().get("source_paths") or []}
        return {
            "ok": True,
            "profile": _identity_profile_summary(profile, active_paths),
            "config": get_config(),
            "runtime": runtime.status(),
        }
    return _activate_identity_profile(profile)


@app.post("/api/identities/{profile_id}/activate")
def activate_identity_profile(profile_id: str) -> Dict[str, Any]:
    return _activate_identity_profile(_read_identity_profile(profile_id))


@app.get("/api/identities/{profile_id}/thumbnail")
def identity_profile_thumbnail(profile_id: str) -> FileResponse:
    thumbnail = _identity_profile_path(profile_id) / "thumbnail.jpg"
    if not thumbnail.is_file():
        raise HTTPException(status_code=404, detail="This identity has no thumbnail")
    return FileResponse(thumbnail, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/identities/{profile_id}/dataset.zip")
def identity_profile_dataset(profile_id: str) -> FileResponse:
    profile_path = _identity_profile_path(profile_id)
    archive_path = profile_path / f"{profile_id}-dataset.zip"
    manifest_path = profile_path / "profile.json"
    if not archive_path.is_file() or archive_path.stat().st_mtime < manifest_path.stat().st_mtime:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source in profile_path.rglob("*"):
                if source.is_file() and source != archive_path:
                    archive.write(source, source.relative_to(profile_path))
    return FileResponse(archive_path, media_type="application/zip", filename=archive_path.name)


def _dfm_result(action: Any) -> Dict[str, Any]:
    try:
        return action()
    except DfmTrainingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/dfm")
def dfm_status(refresh: bool = False) -> Dict[str, Any]:
    return _dfm_result(lambda: dfm_training.status(refresh_environment=refresh))


@app.post("/api/dfm/setup")
def dfm_setup(payload: DfmSetupRequest) -> Dict[str, Any]:
    return _dfm_result(lambda: dfm_training.open_setup(payload.distro, payload.engine_root))


@app.post("/api/dfm/jobs")
def dfm_prepare(payload: DfmPrepareRequest) -> Dict[str, Any]:
    return _dfm_result(
        lambda: dfm_training.prepare(
            payload.profile_id,
            payload.distro,
            payload.engine_root,
            payload.base_workspace,
            payload.target_iterations,
        )
    )


@app.post("/api/dfm/jobs/{job_id}/extract")
def dfm_extract(job_id: str) -> Dict[str, Any]:
    if runtime.status().get("running"):
        runtime.stop()
    engine._cleanup_inference()
    return _dfm_result(lambda: dfm_training.extract(job_id))


@app.post("/api/dfm/jobs/{job_id}/train")
def dfm_train(job_id: str) -> Dict[str, Any]:
    if runtime.status().get("running"):
        runtime.stop()
    engine._cleanup_inference()
    return _dfm_result(lambda: dfm_training.train(job_id))


@app.post("/api/dfm/jobs/{job_id}/stop")
def dfm_stop(job_id: str) -> Dict[str, Any]:
    return _dfm_result(lambda: dfm_training.stop(job_id))


@app.post("/api/dfm/jobs/{job_id}/export")
def dfm_export(job_id: str) -> Dict[str, Any]:
    return _dfm_result(lambda: dfm_training.export(job_id))


@app.post("/api/dfm/jobs/{job_id}/activate")
def dfm_activate(job_id: str) -> Dict[str, Any]:
    job = _dfm_result(lambda: dfm_training.job(job_id))
    model_id = str(job.get("model_id") or "")
    model = _model_by_id(model_id)
    if not model or not model.get("installed"):
        raise HTTPException(status_code=409, detail="Export this training job before activating it")
    previous = get_config()
    values = {"swap_mode": "deep", "deep_swapper_model": model_id}
    if runtime.status().get("running"):
        config = save_seamless_deep_model_config(values, previous)
    else:
        config = save_config(values)
    return {"ok": True, "model": model, "config": config, "runtime": runtime.status(), "job": job}


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str) -> Response:
        candidate = (FRONTEND_DIST / path).resolve()
        try:
            candidate.relative_to(FRONTEND_DIST.resolve())
        except ValueError:
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    def frontend_missing() -> Dict[str, str]:
        return {"message": "React build missing. Run npm install && npm run build in webui/."}


def main() -> None:
    parser = argparse.ArgumentParser("Webcam FaceFusion web application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    target = "web_api:app" if args.reload else app
    uvicorn.run(target, host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
