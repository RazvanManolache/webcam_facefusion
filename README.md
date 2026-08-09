# Webcam FaceFusion (Webcam Deep Swap)

A friendly, real-time webcam face swapper built for personal curiosity and fun. It wraps pieces of the FaceFusion stack with a dedicated React interface and FastAPI backend, live camera controls, optional virtual camera output (so you can use it in video apps), and a documented local API for automation.

- Developed on Windows.
- Tested with an NVIDIA GeForce RTX 5080.
- Uses React for UI, FastAPI for the local API, OpenCV for capture, ONNX Runtime for inference, and pyvirtualcam for a virtual webcam.

## Key Features

- **Realtime webcam deep swap** with multiple swap modes (deep/classic).
- **Virtual camera output** via `pyvirtualcam` for apps like Teams/Zoom.
- **Windows tray mode** with browser/API shortcuts, capture controls, and automatic start/idle-stop from Windows camera-use signals.
- **Full camera controls**: automatic camera/mode detection, fast resolution switching, FPS, FOURCC, exposure/WB locking, retry/repair, etc.
- **Dedicated web app**: responsive React control surface, optional live preview, realtime FPS/latency cards, and live processing changes that keep the camera connected.
- **Local API**: lifecycle, camera discovery, configuration, MJPEG preview, benchmarking, source upload, virtual camera, and interactive OpenAPI documentation.
- **Per-module routing**: independently choose CUDA, TensorRT, or CPU for detectors, swappers, masks, enhancers and optional processors.
- **Detectors and processors**: detector/landmarker/occluder/parser models, enhancers, colorizer, expression restorer, lip sync, and face editor.
- **State persistence** in `user_prefs.json`, restored automatically on reload.
- **Local-only personal data**: camera preferences, captured identities, logs, benchmark media and generated model caches are excluded from Git.
- **CLI tab** in the UI that renders a copy-ready command mirroring your current settings.
- **Headless mode** to run without a GUI (optionally to a virtual camera).

See the complete UI and preferences map in:

- [features_ui.md](./features_ui.md)
- [PRIVACY.md](./PRIVACY.md) for local-data locations and the public-release checklist.

## Libraries and Runtime

- Python
- [Gradio](https://gradio.app/)
- [OpenCV](https://opencv.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- [pystray](https://github.com/moses-palmer/pystray)
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
   - NVIDIA GPU + TensorRT:
     ```bash
     pip install -r requirements-gpu.txt
     ```
     The pinned TensorRT 10.9 CUDA 12 package is compatible with this app's ONNX Runtime build and can coexist with a CUDA 13 system toolkit.
4. Ensure your camera works in Windows (Media Foundation or DirectShow). Close other apps that may hold the camera.

## Quick Start (React + FastAPI)

Double-click `start_gui.bat`, or launch the server directly:

```bash
python web_api.py --port 7862
```

Open [http://127.0.0.1:7862](http://127.0.0.1:7862). Interactive API documentation is available at [http://127.0.0.1:7862/docs](http://127.0.0.1:7862/docs).

The launcher builds the React frontend automatically the first time. For frontend development, use `npm install` followed by `npm run dev` inside `webui/`; Vite proxies `/api` to port 7862.

- Choose your **Execution Provider** and **Device**.
- Select a **Camera**. The available **Resolution** choices and maximum **FPS** are detected automatically from its Windows driver.
- Use **Detect Modes** to re-scan the selected camera or **Refresh Cameras** after connecting a new device. Manual codec and image controls are under **Advanced camera controls**.
- The live preview reports processed/displayed FPS, processing latency, end-to-end latency, and how many stale camera frames were skipped.
- Pick your **Swap Mode** and model(s).
- With **Fast live analysis** off, optional **Temporal face tracking** can track landmarks between configurable refresh frames; it is experimental and defaults off for maximum detection reliability. **Temporal mask reuse** avoids rerunning XSeg on stable aligned crops and refreshes immediately when crop motion exceeds its safety threshold.
- The stable NVIDIA sweet spot keeps XSeg masks on CUDA for clean face edges and routes DFM face models through TensorRT FP32. The first use of each DFM model builds a disk-cached engine; later frames and app starts reuse it. XSeg TensorRT FP16 remains selectable for experiments, but is not the default because it can create visible mask-edge artifacts. DFM FP16 is disabled because it changes these models' face and mask output substantially.
- Toggle **Virtual camera** in the top bar if you want to send processed frames to a system-wide virtual camera device.
- Open **System → API docs** to inspect and call the same endpoints used by the React interface.

### Windows tray mode

Double-click `start_tray.bat` to run FaceFlow without a console or an always-open browser. The tray menu provides:

- **Open FaceFlow controls** and **Open API options**.
- Manual **Start camera** / **Stop camera**.
- **Enable/disable virtual-camera streaming** and **Start on app demand** toggles.
- **Camera start mode** with **No processing (passthrough)** or **Last processing setup**.

Tray mode keeps the web server resident and continuously advertises the OBS virtual camera at the saved resolution and frame rate. While physical capture is off it sends a low-cost FaceFlow standby frame, so Teams, Zoom, browsers, and other applications can discover and open the device. Opening the OBS virtual-camera shared-memory queue starts FaceFlow; a demand-started session stops eight seconds after the final consumer closes while the standby stream remains available. Manual starts remain running until manually stopped.

The selected ONNX pipeline remains warm when capture stops, avoiding model reloads on the next demand start. Changing a model or execution provider clears the previous inference contexts once, so inactive model sessions do not accumulate in RAM or VRAM. **System -> Free VRAM** remains available when a fully cold idle state is preferred.

Demand detection inspects the OBS virtual camera's live shared-memory handles, including ordinary desktop DirectShow clients that are absent from the Windows privacy registry. If another `pyvirtualcam` backend is selected because OBS is unavailable, FaceFlow falls back to Windows' broader camera-use signal.

### Train a custom DFM

Open **Identity trainer** and choose **Train a DFM**. This is the long-training path for creating a reusable DeepFaceLab model from a consenting person's guided webcam recording:

1. Enter the person's name, confirm consent, and record at least five guided actions. FaceFlow requires 400 clear frames and recommends roughly 2,000 varied frames.
2. In **DFM training engine**, use **Open compatible setup** to install the RTX 5000/Blackwell DeepFaceLab fork in an Ubuntu 24.04 WSL2 distribution. Setup opens in a visible terminal and may request the Linux password; FaceFlow does not silently install it.
3. Select the saved DFM dataset and enter an RTT/pretrained SAEHD workspace path. The base workspace must contain `data_dst/aligned` and `model` so training does not start from an unusable empty destination set.
4. Prepare the workspace, extract aligned source faces, start or resume training, then stop at a useful checkpoint and export the `.dfm`.
5. Choose **Use in FaceFlow** on the completed job. The exported model is copied into the custom model collection and selected without reopening the physical webcam.

Training stops physical capture and releases FaceFlow's inference sessions so DeepFaceLab can use the full GPU. The tray application continues advertising the virtual camera with its standby frame. The iteration choice is a progress marker, not an automatic stop; DeepFaceLab saves checkpoints continuously and training can be stopped and resumed from the same job.

### Legacy Gradio fallback

The previous interface remains available during the migration:

```bash
start_gradio_gui.bat
```

## Virtual Camera

- Enabling the virtual camera publishes processed frames via `pyvirtualcam`; tray mode keeps a standby frame flowing even when the physical camera is off.
- It can be enabled or disabled while capture is running without reopening the physical camera.
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
- **Performance**: Install `requirements-gpu.txt` and use the default TensorRT route for the DFM model. The first engine build is slow but cached. XSeg uses CUDA by default because its faster FP16 route can harm mask quality.

## Motivation

This project was created purely for personal curiosity and fun—to explore real-time face swapping with a simple, approachable UI and the flexibility to run headless with a virtual camera.

Enjoy experimenting!
