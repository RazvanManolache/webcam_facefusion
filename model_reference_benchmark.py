"""Validate installed custom DeepFaceLive models against a recorded webcam face.

The script never opens the physical webcam and never changes saved UI settings.
It finds a clear detected face in the recorded webcam sample, then runs every
installed custom model through the same crop/mask/paste path used by live mode.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import web_api


def detect_largest_face(frame: np.ndarray) -> object | None:
    height, width = frame.shape[:2]
    detect_scale = max(1.0, width / 640.0)
    detect_frame = frame if detect_scale == 1.0 else cv2.resize(
        frame,
        (640, round(height / detect_scale)),
        interpolation=cv2.INTER_AREA,
    )
    faces = web_api.engine._select_live_faces_fast(detect_frame)
    if not faces:
        return None
    face = faces[0]
    if detect_scale != 1.0:
        landmarks = {
            key: value * detect_scale if isinstance(value, np.ndarray) else value
            for key, value in face.landmark_set.items()
        }
        face = face._replace(
            bounding_box=face.bounding_box * detect_scale,
            landmark_set=landmarks,
        )
    return face


def find_reference_frame(
    video_path: Path,
    fallback_image_path: Path,
    sample_every: int = 5,
) -> tuple[str, int | None, np.ndarray, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open reference video: {video_path}")

    best: tuple[float, int, np.ndarray, object] | None = None
    frame_index = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % sample_every:
                continue

            face = detect_largest_face(frame)
            if face is None:
                continue

            height, width = frame.shape[:2]
            x1, y1, x2, y2 = np.asarray(face.bounding_box, dtype=float)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            crop = frame[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))]
            sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
            score = area * max(1.0, sharpness)
            if best is None or score > best[0]:
                best = (score, frame_index, frame.copy(), face)
    finally:
        capture.release()

    if best is not None:
        return str(video_path.resolve()), best[1], best[2], best[3]

    fallback_frame = cv2.imread(str(fallback_image_path))
    if fallback_frame is None:
        raise RuntimeError(
            "No face was detected in the reference video and the saved webcam fallback is missing"
        )
    fallback_face = detect_largest_face(fallback_frame)
    if fallback_face is None:
        raise RuntimeError("No face was detected in the reference video or saved webcam fallback")
    print(
        f"No face found in {video_path}; using saved webcam frame {fallback_image_path}",
        flush=True,
    )
    return str(fallback_image_path.resolve()), None, fallback_frame, fallback_face


def configure_engine(provider: str) -> dict:
    engine = web_api.engine
    config = web_api.get_config()
    config.update(
        {
            "swap_mode": "deep",
            "realtime_fast_analysis": True,
            "detector_model": "retinaface",
            "detector_size": "640x640",
            "detector_score": 0.35,
            "selector_mode": "one",
            "use_occlusion": True,
            "face_enhancer_enabled": False,
            "frame_enhancer_enabled": False,
            "show_overlay": False,
            "show_boxes": False,
            "debug_logs": False,
            "execution_providers": [provider],
            "provider_detector": provider,
            "provider_landmarker": provider,
            "provider_recognizer": provider,
            "provider_classifier": provider,
            "provider_masker": provider,
            "provider_deep_swapper": provider,
        }
    )
    engine.state_manager.set_item("download_providers", list(engine.ff_choices.download_providers))
    engine.state_manager.set_item("download_scope", "full")
    engine.state_manager.set_item("log_level", "error")
    engine.state_manager.set_item("face_detector_margin", [0, 0, 0, 0])
    engine.state_manager.set_item("face_mask_blur", 0.5)
    engine.state_manager.set_item("face_mask_padding", (0, 0, 0, 0))
    web_api._apply_processing_config(config, provider)
    engine.state_manager.set_item("realtime_fast_analysis", True)
    return config


def validate_models(video_path: Path, fallback_image_path: Path, provider: str) -> dict:
    engine = web_api.engine
    config = configure_engine(provider)
    reference_source, frame_index, frame, target_face = find_reference_frame(video_path, fallback_image_path)
    custom_models = [
        model
        for model in web_api._model_catalog()
        if model["id"].startswith("custom/") and model["installed"]
    ]

    results = []
    for index, model in enumerate(custom_models, start=1):
        model_id = model["id"]
        engine._cleanup_inference()
        config["deep_swapper_model"] = model_id
        web_api._apply_processing_config(config, provider)
        started = time.perf_counter()
        try:
            model_size = tuple(int(value) for value in engine.deep_swapper.get_model_size())
            output = engine.deep_swapper.swap_face(target_face, frame.copy())
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            delta = np.abs(output.astype(np.int16) - frame.astype(np.int16))
            changed_percent = float((np.max(delta, axis=2) > 5).mean() * 100.0)
            mean_difference = float(delta.mean())
            passed = bool(output.shape == frame.shape and changed_percent >= 0.05)
            error = None if passed else "Model returned an effectively unchanged frame"
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            model_size = None
            changed_percent = 0.0
            mean_difference = 0.0
            passed = False
            error = f"{type(exc).__name__}: {exc}"

        result = {
            "id": model_id,
            "name": model["name"],
            "passed": passed,
            "model_size": model_size,
            "first_frame_ms": round(elapsed_ms, 1),
            "changed_pixels_percent": round(changed_percent, 3),
            "mean_absolute_difference": round(mean_difference, 4),
            "error": error,
        }
        results.append(result)
        print(f"[{index:02d}/{len(custom_models):02d}] {model['name']}: {'PASS' if passed else 'FAIL'}", flush=True)

    engine._cleanup_inference()
    return {
        "requested_reference_video": str(video_path.resolve()),
        "reference_source": reference_source,
        "reference_frame_index": frame_index,
        "provider": provider,
        "tested": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("benchmarks/webcam_sample_10s.avi"))
    parser.add_argument("--fallback-image", type=Path, default=Path("benchmarks/cpu_split_presence.jpg"))
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/model_reference_validation.json"))
    args = parser.parse_args()

    report = validate_models(args.video, args.fallback_image, args.provider)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("tested", "passed", "failed", "reference_frame_index")}), flush=True)
    print(f"Report: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
