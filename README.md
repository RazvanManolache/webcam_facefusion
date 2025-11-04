# Webcam FaceFusion (Webcam Deep Swap)

A friendly, real-time webcam face swapper built for personal curiosity and fun. It wraps pieces of the FaceFusion stack with a simple Gradio UI, live camera controls, optional virtual camera output (so you can use it in video apps), and full CLI parity for automation.

- Developed on Windows.
- Tested with an NVIDIA GeForce RTX 5080.
- Uses Gradio for UI, OpenCV for capture, ONNX Runtime for inference, and pyvirtualcam for a virtual webcam.

## Key Features

- **Realtime webcam deep swap** with multiple swap modes (deep/classic).
- **Virtual camera output** via `pyvirtualcam` for apps like Teams/Zoom.
- **Full camera controls**: backend, resolution, FPS, FOURCC, exposure/WB locking, retry/repair, etc.
- **Detectors and processors**: detector/landmarker/occluder/parser models, enhancers, colorizer, expression restorer, lip sync, and face editor.
- **State persistence** in `user_prefs.json`, restored automatically on reload.
- **CLI tab** in the UI that renders a copy-ready command mirroring your current settings.
- **Headless mode** to run without a GUI (optionally to a virtual camera).

See the complete UI and preferences map in:

- [features_ui.md](./features_ui.md)

## Libraries and Runtime

- Python
- [Gradio](https://gradio.app/)
- [OpenCV](https://opencv.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- Pillow, Requests, tqdm, NumPy

All Python dependencies are listed in:

- `requirements.txt` (CPU/default)
- `requirements-gpu.txt` (optional GPU variants)

## Installation

1. Install Python 3.10+ (Windows recommended; project developed on Windows).
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
   - CPU path:
     ```bash
     pip install -r requirements.txt
     ```
   - (Optional) GPU/alternative providers: see `requirements-gpu.txt` and adjust ONNX Runtime to your setup.
4. Ensure your camera works in Windows (Media Foundation or DirectShow). Close other apps that may hold the camera.

## Quick Start (GUI)

Launch the Gradio UI (auto-opens your browser):

```bash
python webcam_deep_swap.py --gui
```

- Choose your **Execution Provider** and **Device**.
- Select **Camera**, **Resolution**, **FOURCC**, etc.
- Pick your **Swap Mode** and model(s).
- Toggle **Enable Virtual Camera** if you want to send processed frames to a system-wide virtual camera device.
- Use the **CLI** tab to copy a full command reflecting the current UI settings.

## Virtual Camera

- Enabling the virtual camera publishes the processed frames via `pyvirtualcam`.
- On first use, Windows may prompt for camera device permissions.
- Some conferencing apps need to be restarted to detect a newly created virtual camera.

## Headless Mode (No GUI)

Run using your saved preferences and/or CLI flags. This is ideal for automated setups or direct virtual camera output.

```bash
python webcam_deep_swap.py --headless \
  --backend "DirectShow" --camera 0 --width 1920 --height 1080 --fps 15 --fourcc MJPG \
  --ep cuda --device 0 --virtual-cam \
  --swap-mode deep --model iperov/james_carrey_224 --morph 80
```

- Use `Ctrl+C` to stop.
- You can also use classic Face Swapper mode with `--source` images (repeatable):

```bash
python webcam_deep_swap.py --headless \
  --swap-mode face --face-swapper-model inswapper_128 \
  --source "C:\\imgs\\person1.jpg" --source "C:\\imgs\\person2.jpg"
```

## Command-Line Usage

- Nearly every UI field has a corresponding CLI flag. The **CLI** tab in the UI shows your current configuration as a command you can copy.
- Common camera flags:
  - `--backend`, `--camera`, `--resolution`, `--width`, `--height`, `--fps`, `--fourcc`, `--color-mode`
  - `--lock-exposure`, `--exposure`, `--lock-wb`, `--wb-temp`
  - `--retry-black`, `--gentle`, `--auto-repair`, `--convert-rgb`
- Swapping and models:
  - `--swap-mode`, `--model`, `--morph`, `--face-swapper-model`, `--face-swapper-pixel`, `--face-swapper-weight`, `--source` (repeat)
- Detection & processing options:
  - `--detector-model`, `--detector-size`, `--detector-score`, `--selector-mode`, `--auto-fallback`
  - `--landmarker-model`, `--landmarker-score`, `--occluder-model`, `--parser-model`
  - Enhancers: `--face-enhancer`, `--face-enhancer-model`, `--face-enhancer-blend`, `--face-enhancer-weight`
  - Frame enhancer: `--frame-enhancer`, `--frame-enhancer-model`, `--frame-enhancer-blend`, `--enhance-async`
  - Colorizer: `--colorizer`, `--colorizer-model`, `--colorizer-size`, `--colorizer-blend`
  - Expression restorer: `--expr`, `--expr-model`, `--expr-factor`, `--expr-area` (repeat)
  - Lip syncer: `--lip-syncer`, `--lip-model`, `--lip-weight`
  - Face editor: `--editor`, `--editor-model`, and sliders like `--fe-eye-h`, `--fe-head-yaw`, etc.
- Execution provider and device:
  - `--ep` (e.g., `cuda`, `directml`, `cpu`), `--device` (e.g., `0`)

For a full map of flags to preference keys and defaults, see:

- [features_ui.md](./features_ui.md)

## Tips & Troubleshooting

- **Camera busy or not found**: Close other apps using the webcam. Try switching backends (Media Foundation vs DirectShow), changing FOURCC, lowering resolution/FPS, or enabling `Convert RGB`.
- **Black frames**: Increase `--retry-black` or enable `--auto-repair`.
- **Virtual camera not listed**: Restart the target app, or toggle the virtual camera off/on and try again. Ensure `pyvirtualcam` is installed.
- **Performance**: Use GPU (e.g., `--ep cuda --device 0`), reduce resolution/FPS, or adjust model sizes.

## Motivation

This project was created purely for personal curiosity and fun—to explore real-time face swapping with a simple, approachable UI and the flexibility to run headless with a virtual camera.

Enjoy experimenting!
