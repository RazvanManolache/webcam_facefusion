"""CUDA-resident DeepFaceLive crop processing.

The webcam frame still originates on the CPU and the virtual-camera backend
ultimately consumes a CPU array.  This module keeps the expensive 384px face
crop, XSeg mask, DFM outputs, colour matching and mask cleanup on one CUDA
device between those two unavoidable boundaries.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as torch_functional
except Exception:  # pragma: no cover - exercised by the automatic CPU fallback
    torch = None
    torch_functional = None

from facefusion import face_masker, state_manager
from facefusion.processors.modules.deep_swapper import core as deep_swapper


LOGGER = logging.getLogger("gpu_face_pipeline")
_GPU_PROVIDERS = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
_BOX_MASK_CACHE: Dict[Tuple[Any, ...], Any] = {}
_GAUSSIAN_KERNEL_CACHE: Dict[Tuple[float, str], Any] = {}
_STATUS_LOCK = threading.RLock()
_RECENT_TIMINGS: deque[float] = deque(maxlen=120)
_LAST_ERROR: Optional[str] = None
_FAILURES = 0


def is_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _device() -> Any:
    device_ids = state_manager.get_item("execution_device_ids") or ["0"]
    try:
        device_index = int(device_ids[0])
    except (TypeError, ValueError):
        device_index = 0
    return torch.device(f"cuda:{device_index}")


def _session_supports_cuda(session: Any) -> bool:
    try:
        return bool(_GPU_PROVIDERS.intersection(session.get_providers()))
    except Exception:
        return False


def can_process() -> bool:
    if not is_available() or not bool(state_manager.get_item("gpu_pipeline_enabled")):
        return False
    try:
        deep_session = deep_swapper.get_inference_pool().get("deep_swapper")
        if not _session_supports_cuda(deep_session):
            return False
        return True
    except Exception:
        return False


def status() -> Dict[str, Any]:
    with _STATUS_LOCK:
        timings = list(_RECENT_TIMINGS)
        return {
            "available": is_available(),
            "enabled": bool(state_manager.get_item("gpu_pipeline_enabled")),
            "active": bool(timings) and can_process(),
            "mean_ms": round(sum(timings) / len(timings), 3) if timings else None,
            "samples": len(timings),
            "failures": _FAILURES,
            "last_error": _LAST_ERROR,
        }


def reset_status() -> None:
    global _LAST_ERROR, _FAILURES
    with _STATUS_LOCK:
        _RECENT_TIMINGS.clear()
        _LAST_ERROR = None
        _FAILURES = 0


def record_failure(exc: BaseException) -> None:
    global _LAST_ERROR, _FAILURES
    with _STATUS_LOCK:
        _FAILURES += 1
        _LAST_ERROR = str(exc) or type(exc).__name__


def _bind_cuda_session(session: Any, inputs: Dict[str, Any], outputs: Dict[str, Tuple[int, ...]]) -> Dict[str, Any]:
    """Run ORT directly against Torch-owned CUDA buffers."""
    device = _device()
    device_id = int(device.index or 0)
    binding = session.io_binding()
    contiguous_inputs: Dict[str, Any] = {}
    for name, value in inputs.items():
        tensor = value.contiguous()
        contiguous_inputs[name] = tensor
        binding.bind_input(name, "cuda", device_id, np.float32, tuple(tensor.shape), tensor.data_ptr())

    output_tensors: Dict[str, Any] = {}
    for name, shape in outputs.items():
        tensor = torch.empty(shape, device=device, dtype=torch.float32)
        output_tensors[name] = tensor
        binding.bind_output(name, "cuda", device_id, np.float32, shape, tensor.data_ptr())

    # PyTorch and ORT can own different CUDA streams. These two boundaries are
    # deliberately explicit; they prevent a model from reading an unfinished
    # preprocessing kernel or Torch from consuming an unfinished ORT output.
    torch.cuda.synchronize(device)
    session.run_with_iobinding(binding)
    torch.cuda.synchronize(device)
    return output_tensors


def _resize_hwc(image: Any, size: Tuple[int, int], mode: str) -> Any:
    width, height = size
    tensor = image.permute(2, 0, 1).unsqueeze(0)
    options: Dict[str, Any] = {"size": (height, width), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        options["align_corners"] = False
    resized = torch_functional.interpolate(tensor, **options)
    return resized[0].permute(1, 2, 0)


def _gaussian_kernel_1d(sigma: float, device: Any) -> Any:
    cache_key = (round(float(sigma), 4), str(device))
    kernel = _GAUSSIAN_KERNEL_CACHE.get(cache_key)
    if kernel is None:
        radius = max(1, int(math.ceil(float(sigma) * 3.0)))
        positions = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-(positions * positions) / (2.0 * float(sigma) * float(sigma)))
        kernel = kernel / kernel.sum()
        _GAUSSIAN_KERNEL_CACHE[cache_key] = kernel
    return kernel


def _gaussian_blur(image: Any, sigma: float) -> Any:
    original_dimensions = image.ndim
    if original_dimensions == 2:
        tensor = image.unsqueeze(0).unsqueeze(0)
    elif original_dimensions == 3:
        tensor = image.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported CUDA blur shape: {tuple(image.shape)}")

    channels = tensor.shape[1]
    kernel = _gaussian_kernel_1d(sigma, tensor.device)
    radius = kernel.numel() // 2
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    tensor = torch_functional.pad(tensor, (radius, radius, 0, 0), mode="reflect")
    tensor = torch_functional.conv2d(tensor, horizontal, groups=channels)
    tensor = torch_functional.pad(tensor, (0, 0, radius, radius), mode="reflect")
    tensor = torch_functional.conv2d(tensor, vertical, groups=channels)
    if original_dimensions == 2:
        return tensor[0, 0]
    return tensor[0].permute(1, 2, 0)


def _prepare_dfm_input(crop_bgr: Any) -> Any:
    crop_float = crop_bgr.to(dtype=torch.float32)
    blurred = _gaussian_blur(crop_float, 2.0)
    sharpened = crop_float * 1.75 - blurred * 0.75
    return sharpened.mul_(1.0 / 255.0).unsqueeze(0).contiguous()


def _bgr_histogram(image: Any) -> Any:
    bgr = image.clamp(0, 255).to(dtype=torch.float32).mul(1.0 / 255.0)
    blue, green, red = bgr.unbind(dim=2)
    maximum = torch.maximum(torch.maximum(red, green), blue)
    minimum = torch.minimum(torch.minimum(red, green), blue)
    delta = maximum - minimum
    safe_delta = torch.where(delta > 1e-7, delta, torch.ones_like(delta))
    hue = torch.zeros_like(maximum)
    red_mask = (maximum == red) & (delta > 1e-7)
    green_mask = (maximum == green) & (delta > 1e-7)
    blue_mask = (maximum == blue) & (delta > 1e-7)
    hue = torch.where(red_mask, torch.remainder((green - blue) / safe_delta, 6.0), hue)
    hue = torch.where(green_mask, (blue - red) / safe_delta + 2.0, hue)
    hue = torch.where(blue_mask, (red - green) / safe_delta + 4.0, hue)
    hue = hue * 30.0
    saturation = torch.where(maximum > 1e-7, delta / maximum.clamp_min(1e-7), torch.zeros_like(maximum)) * 255.0
    hue_bin = torch.floor(hue * (50.0 / 180.0)).to(torch.int64).clamp_(0, 49)
    saturation_bin = torch.floor(saturation * (60.0 / 256.0)).to(torch.int64).clamp_(0, 59)
    return torch.bincount((hue_bin * 60 + saturation_bin).reshape(-1), minlength=3000).to(torch.float32)


def _histogram_factor(source: Any, target: Any) -> Any:
    source_histogram = _bgr_histogram(source)
    target_histogram = _bgr_histogram(target)
    source_centered = source_histogram - source_histogram.mean()
    target_centered = target_histogram - target_histogram.mean()
    denominator = torch.sqrt((source_centered.square().sum()) * (target_centered.square().sum()))
    correlation = torch.where(
        denominator > 0,
        (source_centered * target_centered).sum() / denominator.clamp_min(1e-12),
        torch.ones((), device=source.device, dtype=torch.float32),
    )
    return ((correlation.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)


def _equalize_color(source: Any, target: Any, size: Tuple[int, int]) -> Any:
    source_small = _resize_hwc(source, size, "area")
    target_small = _resize_hwc(target, size, "area")
    difference = _resize_hwc(source_small - target_small, (target.shape[1], target.shape[0]), "bicubic")
    # NumPy's uint8 cast truncates. Keep the same quantization point after
    # every pyramid pass so repeated equalization remains visually close.
    return (target + difference).clamp(0, 255).to(torch.uint8).to(torch.float32)


def _match_color(source: Any, target: Any) -> Any:
    factor = _histogram_factor(source, target)
    matched_source = source
    height = int(target.shape[0])
    for index in range(3):
        raw_size = 16.0 + index * ((height - 16.0) / 3.0)
        normalized_size = int(round(raw_size / 2.0) * 2)
        matched_source = _equalize_color(matched_source, target, (normalized_size, normalized_size))
    matched_target = _equalize_color(matched_source, target, (int(target.shape[1]), height))
    return (target * (1.0 - factor) + matched_target * factor).clamp(0, 255)


def _erode_ellipse_3x3(mask: Any, iterations: int = 2) -> Any:
    result = mask
    for _ in range(iterations):
        padded = torch_functional.pad(result.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")[0, 0]
        height, width = result.shape
        result = torch.minimum(torch.minimum(padded[1:1 + height, 1:1 + width], padded[:height, 1:1 + width]), padded[2:2 + height, 1:1 + width])
        result = torch.minimum(result, padded[1:1 + height, :width])
        result = torch.minimum(result, padded[1:1 + height, 2:2 + width])
    return result


def _prepare_dfm_mask(source_mask: Any, target_mask: Any, model_size: Tuple[int, int]) -> Any:
    width, height = model_size
    mask = torch.minimum(source_mask, target_mask).reshape(height, width).clamp(0, 1)
    mask = _erode_ellipse_3x3(mask, 2)
    return _gaussian_blur(mask, 6.25)


def _box_mask_gpu(crop: np.ndarray, device: Any) -> Any:
    blur = float(state_manager.get_item("face_mask_blur") or 0.0)
    padding = tuple(float(value) for value in (state_manager.get_item("face_mask_padding") or (0, 0, 0, 0)))
    cache_key = (crop.shape[1], crop.shape[0], blur, padding, str(device))
    mask = _BOX_MASK_CACHE.get(cache_key)
    if mask is None:
        cpu_mask = deep_swapper.create_box_mask(crop, blur, padding)
        mask = torch.from_numpy(np.ascontiguousarray(cpu_mask)).to(device=device, dtype=torch.float32)
        _BOX_MASK_CACHE[cache_key] = mask
    return mask


def _occlusion_mask_gpu(crop_gpu: Any, crop_size: Tuple[int, int]) -> Any:
    configured_model = str(state_manager.get_item("face_occluder_model") or "xseg_1")
    model_names = ["xseg_1", "xseg_2", "xseg_3"] if configured_model == "many" else [configured_model]
    inference_pool = face_masker.get_inference_pool(model_names)
    masks = []
    for model_name in model_names:
        session = inference_pool.get(model_name)
        if not _session_supports_cuda(session):
            raise RuntimeError(f"{model_name} is not running on a CUDA-capable provider")
        model_width, model_height = face_masker.create_static_model_set("full")[model_name]["size"]
        prepared = _resize_hwc(crop_gpu.to(torch.float32), (model_width, model_height), "bilinear")
        prepared = prepared.mul_(1.0 / 255.0).unsqueeze(0).contiguous()
        outputs = _bind_cuda_session(session, {"input": prepared}, {"output": (1, model_height, model_width, 1)})
        mask = outputs["output"][0, :, :, 0].clamp(0, 1)
        if (model_width, model_height) != crop_size:
            mask = _resize_hwc(mask.unsqueeze(-1), crop_size, "bilinear")[:, :, 0]
        masks.append(mask)
    mask = torch.stack(masks).amin(dim=0)
    mask = _gaussian_blur(mask.clamp(0, 1), 5.0)
    return (mask.clamp(0.5, 1.0) - 0.5) * 2.0


def _run_dfm_gpu(crop_input: Any) -> Tuple[Any, Any, Any]:
    session = deep_swapper.get_inference_pool().get("deep_swapper")
    if not _session_supports_cuda(session):
        raise RuntimeError("The selected DFM is not running on a CUDA-capable provider")
    width, height = deep_swapper.get_model_size()
    morph = float(int(state_manager.get_item("deep_swapper_morph") or 0)) / 100.0
    inputs: Dict[str, Any] = {"in_face:0": crop_input}
    if any(model_input.name == "morph_value:0" for model_input in session.get_inputs()):
        inputs["morph_value:0"] = torch.tensor([morph], device=_device(), dtype=torch.float32)
    output_shapes = {
        "out_face_mask:0": (1, height, width, 1),
        "out_celeb_face:0": (1, height, width, 3),
        "out_celeb_face_mask:0": (1, height, width, 1),
    }
    outputs = _bind_cuda_session(session, inputs, output_shapes)
    return (
        outputs["out_celeb_face:0"][0],
        outputs["out_celeb_face_mask:0"][0],
        outputs["out_face_mask:0"][0],
    )


def swap_face(target_face: Any, temp_vision_frame: np.ndarray) -> np.ndarray:
    """Run the visually safe CUDA crop pipeline.

    The DFM and colour pyramid remain CUDA-resident. Mask resize, erosion and
    blur deliberately use the original OpenCV functions: small interpolation
    differences at a mask edge are much more visible than small RGB changes.
    """
    if not can_process():
        raise RuntimeError("CUDA face pipeline is not available for the selected providers")
    started_at = perf_counter()
    device = _device()
    model_options = deep_swapper.get_model_options()
    model_size = deep_swapper.get_model_size()
    crop, affine_matrix = deep_swapper.warp_face_by_face_landmark_5(
        temp_vision_frame,
        target_face.landmark_set.get("5/68"),
        model_options.get("template"),
        model_size,
    )
    crop_raw = crop.copy()
    mask_types = state_manager.get_item("face_mask_types") or []
    cpu_masks = [
        deep_swapper.create_box_mask(
            crop,
            state_manager.get_item("face_mask_blur"),
            state_manager.get_item("face_mask_padding"),
        )
    ]
    if "occlusion" in mask_types:
        cpu_masks.append(deep_swapper.create_occlusion_mask(crop))

    with torch.inference_mode():
        # Keep preprocessing byte-for-byte compatible because a small input
        # change is amplified by the DFM. Upload its finished tensor once.
        prepared_cpu = deep_swapper.prepare_crop_frame(crop.copy())
        prepared_crop = torch.from_numpy(np.ascontiguousarray(prepared_cpu)).to(device=device, dtype=torch.float32)
        generated_face, source_mask, target_mask = _run_dfm_gpu(prepared_crop)
        generated_face = generated_face.mul(255.0).clamp(0, 255)
        crop_raw_gpu = torch.from_numpy(np.ascontiguousarray(crop_raw)).to(device=device, dtype=torch.float32)
        generated_face = _match_color(crop_raw_gpu, generated_face)
        generated_cpu = generated_face.to(torch.uint8).cpu().numpy()
        source_mask_cpu = source_mask.cpu().numpy()
        target_mask_cpu = target_mask.cpu().numpy()

    cpu_masks.append(deep_swapper.prepare_crop_mask(source_mask_cpu, target_mask_cpu))
    if "area" in mask_types:
        face_landmark_68 = cv2.transform(target_face.landmark_set.get("68").reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
        cpu_masks.append(deep_swapper.create_area_mask(crop_raw, face_landmark_68, state_manager.get_item("face_mask_areas")))
    if "region" in mask_types:
        cpu_masks.append(deep_swapper.create_region_mask(crop_raw, state_manager.get_item("face_mask_regions")))
    mask_cpu = np.minimum.reduce(cpu_masks).clip(0, 1)

    result = deep_swapper.paste_back(temp_vision_frame, generated_cpu, mask_cpu, affine_matrix)
    torch.cuda.synchronize(device)
    elapsed_ms = (perf_counter() - started_at) * 1000.0
    with _STATUS_LOCK:
        _RECENT_TIMINGS.append(elapsed_ms)
    return result
