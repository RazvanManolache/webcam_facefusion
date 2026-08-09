# FaceFlow UI and runtime reference

This document describes the current React interface served by `web_api.py`. It distinguishes capture-source settings, which reopen the physical camera, from inference and output settings, which apply while the source remains connected.

Defaults below come from `DEFAULT_CONFIG` in `web_api.py`. A saved `user_prefs.json` overrides them on later starts. Dynamic model and provider choices depend on the installed runtime.

## Global controls

These controls remain visible above every tab.

| Control | Default | Preference key | Behavior |
|---|---:|---|---|
| Camera source | `[0] Camera 0` | `camera_choice` | Selects a detected physical source. Changing it while running reopens capture. |
| Resolution cards | `1920x1080` | `resolution_preset`, `width`, `height` | Lists driver-detected modes. Each card shows that mode's reported maximum FPS. Changing it reopens capture. |
| Capture rate | `30` FPS | `fps` | Clamped by the selected mode's detected limit. Changing it reopens capture. |
| Detect cameras and modes | action | - | Rescans cameras and their resolutions/FPS. Available only while capture is stopped. |
| Start/Stop camera | stopped | - | Starts or stops the physical webcam without shutting down the web app or necessarily unloading models. |
| Virtual camera | off | `virtual_cam_enabled` | Starts/stops output publication without reopening the physical webcam. Tray mode can continue advertising a standby frame. |
| API docs | action | - | Opens the FastAPI explorer at `/docs`. |

Camera selection automatically chooses a valid resolution if the previously selected one is not supported. The maximum reported resolution is used as the fallback, and FPS is reduced to the new mode's limit.

## Preview and live metrics

| Element | Meaning |
|---|---|
| Processed preview | MJPEG output from `/api/preview.mjpg`. |
| Raw identity preview | Unprocessed MJPEG from `/api/raw-preview.mjpg`, shown during identity work. |
| Hide/Show preview | Stops rendering the browser image without stopping processing or virtual-camera output. |
| Processed FPS | Frames actually completed by the processing pipeline per second. This is not camera capture FPS. |
| Processing latency | Time spent processing a selected frame. |
| End-to-end latency | Approximate age from capture through the displayed result. |
| Virtual output | Published virtual-camera FPS and backend. |
| Skipped/stale frames | Frames replaced in the newest-frame slot because capture was faster than processing. |

The old frame-embedded FPS overlay is optional. The compact metric cards are the primary live counters.

## Face & output

### Face model

| Control | Default | Preference key | Runtime effect |
|---|---:|---|---|
| Swap engine | `deep` | `swap_mode` | `deep` uses a DFM, `face` uses a source image and FaceFusion swapper, `none` is passthrough. Applies without reopening capture. |
| Deep model browser | `iperov/keanu_reeves_320` | `deep_swapper_model` | Searchable, gender-filtered installed/Hub catalog with square thumbnails, names, resolution, creator, install, and select actions. A new model is prepared before handoff so the previous face can continue during the switch. |
| Morph strength | `100%` | `morph` | Controls DFM morph blend. Applies live. |
| Face swapper model | `hyperswap_1c_256` | `face_swapper_model` | Selects the classic swapper. Reloads inference, not capture. |
| Pixel boost | `256x256` | `face_swapper_pixel_boost` | Sets classic swap inference size. Applies without reopening capture. |
| Add source face | none | `source_paths` | Uploads a local source image for classic swapping. |
| Clear source faces | action | `source_paths` | Removes the current source selection from the session/config. |

Installed model files and generated thumbnails are local assets and are not stored in Git.

### Realtime behavior

| Control | Default | Preference key | Notes |
|---|---:|---|---|
| GPU colour pipeline | off | `gpu_pipeline_enabled` | Keeps supported DFM buffers/colour matching in VRAM. Falls back automatically when the GPU path cannot safely handle an operation. Validate quality before enabling permanently. |
| Fast live analysis | on | `realtime_fast_analysis` | Bypasses analysis modules that live swapping does not require. This is the default high-throughput path. |
| Temporal face tracking | off | `temporal_face_tracking` | Experimental landmark tracking between full analysis frames. Visible only when fast analysis is off. Leave disabled if detection or alignment flickers. |
| Face analysis interval | `3` frames | `temporal_face_interval` | Full-analysis refresh interval, range 1-4. |
| Detection overlay | off | `show_overlay` | Draws processing information into the output frame. |
| Face boxes | off | `show_boxes` | Draws detected face boundaries. |
| Occlusion mask | on | `use_occlusion` | Preserves objects crossing in front of the face. |
| Temporal mask reuse | on | `temporal_occlusion_reuse` | Reuses a stable mask across adjacent aligned crops and refreshes immediately when the crop changes. |
| Mask refresh interval | `3` frames | `temporal_occlusion_interval` | Scheduled mask refresh, range 1-4. |

Quality baseline: route XSeg masking through CUDA. TensorRT/FP16 masking remains experimental because it can create coloured or torn face-edge artifacts. DFM TensorRT uses FP32 for output stability.

## Identity trainer

The trainer always requires a person's name and an explicit consent checkbox before recording begins. Guided recordings use the raw preview.

### Recording actions

1. Look straight ahead.
2. Turn about 30 degrees left.
3. Turn about 30 degrees right.
4. Look slightly up.
5. Look slightly down.
6. Smile naturally.
7. Blink naturally.
8. Speak a short sentence.

Only clear, sufficiently exposed frames with a usable face are accepted.

| Mode | Minimum to save | Recommended | Capture behavior | Result |
|---|---|---|---|---|
| Instant identity | 6 frames across 3 actions | complete all 8 actions | short guided bursts | averaged reusable FaceFusion source profile plus retained source dataset |
| DFM dataset | 400 frames across 5 actions | about 2,000 varied frames | 12 seconds per action, up to 12 clear frames/second | DeepFaceLab-ready source dataset |

### Session actions

| Action | Behavior |
|---|---|
| Begin guided capture | Starts an instant session after name and consent validation. |
| Begin DFM dataset capture | Starts the longer DFM session. |
| Record / Again | Captures or repeats one prompted action. |
| Discard capture | Cancels the active session and its temporary data. |
| Create and use this identity | Saves an instant identity and activates it. |
| Save DFM training dataset | Saves the larger training dataset. |
| Use saved identity | Switches the active identity without reopening the physical camera. |
| Download dataset | Explicitly downloads a saved profile's source dataset. |

### DFM training engine

The local DFM path runs DeepFaceLab SAEHD separately under WSL2 so training can own the GPU.

| Control/action | Purpose |
|---|---|
| DeepFaceLab root inside WSL | Location of the compatible RTX 5000/Blackwell fork. |
| Open compatible setup | Opens a visible Ubuntu 24.04 setup terminal. It may request the Linux password and does not silently modify an older distribution. |
| Refresh | Rechecks WSL distribution, GPU, engine, and job status. |
| Source identity | Selects a saved DFM-ready capture. |
| RTT base workspace | Pretrained SAEHD workspace; must contain `data_dst/aligned` and `model`. |
| Iteration progress marker | 250k, 500k, 1m, or 2m marker. It is not an automatic stop. |
| Prepare DFM workspace | Creates a resumable training job from the source and base workspace. |
| Extract faces | Runs source face extraction. |
| Start/Resume training | Starts SAEHD or resumes its saved checkpoint. |
| Stop safely | Requests a checkpoint-safe stop. |
| Export DFM | Exports a checkpoint with completed iterations to `.dfm`. |
| Use this DFM | Copies the export into FaceFlow's custom model collection and activates it. |

Preparing/training stops physical capture and releases FaceFlow inference sessions to make VRAM available. In tray mode the virtual camera remains discoverable through its standby publisher.

## Camera tab

### Capture pipeline

| Control | Default | Preference key | Capture restart? |
|---|---:|---|---:|
| Backend | `DirectShow` on Windows | `backend` | yes |
| Codec / FOURCC | `Auto` | `force_fourcc` | yes |
| Color handling | `Auto (BGR->RGB)` | `color_mode` | yes |
| Black-frame retries | `3` | `retry_black` | yes |
| Driver RGB conversion | on | `convert_rgb` | yes |
| Gentle reconnect | on | `gentle_mode` | yes |
| Automatic capture repair | on | `auto_repair` | yes |

### Camera image controls

| Control | Default | Preference key | Capture restart? |
|---|---:|---|---:|
| Lock exposure | off | `lock_exposure` | yes |
| Exposure | `-6.0` | `exposure_value` | yes |
| Lock white balance | off | `lock_wb` | yes |
| White balance | `4500 K` | `wb_temperature` | yes |
| Native OpenCV window | off | `show_native` | no |

Automatic exposure is recommended unless the driver is known to handle manual mode cleanly; some USB cameras produce edge halos after manual exposure negotiation.

## Processing tab

### Face detection

| Control | Default | Preference key | Notes |
|---|---:|---|---|
| Detector | `retinaface` | `detector_model` | Model change rebuilds its inference session, not capture. |
| Detector size | `320x320` | `detector_size` | Larger input can improve difficult detections at a throughput cost. |
| Landmarker | `many` | `landmarker_model` | Landmark model used by the non-fast path and dependent processors. |
| Selector mode | `one` | `selector_mode` | Policy for selecting faces from the frame. |
| Occluder | `xseg_1` | `occluder_model` | Mask model used when occlusion is enabled. |
| Parser | `bisenet_resnet_18` | `parser_model` | Face parsing model for dependent processors. |
| Detection confidence | `0.35` | `detector_score` | Lower accepts weaker detections; higher reduces false positives. |
| Detector fallback | off | `auto_fallback` | Tries another detector after an extended no-face streak. |

The default landmarker confidence is `0.5` (`landmarker_score`). It is retained in preferences/backend configuration even though the current React panel does not expose a separate slider.

### Enhancers

| Control | Default | Preference key | Notes |
|---|---:|---|---|
| Face enhancer | off | `face_enhancer_enabled` | Enables face-only enhancement. |
| Face enhancer model | `gfpgan_1.4` | `face_enhancer_model` | Visible while enabled. |
| Frame enhancer | off | `frame_enhancer_enabled` | Enables whole-frame enhancement. |
| Frame enhancer model | `span_kendata_x4` | `frame_enhancer_model` | Visible while enabled. |
| Asynchronous enhancement | on | `enhance_async` | Visible when either enhancer is enabled. |

Backend/CLI defaults also include `face_enhancer_blend=80`, `face_enhancer_weight=0.5`, and `frame_enhancer_blend=80`. These detailed values are not separate controls in the current React panel.

### Optional processors

| Toggle | Default | Preference key | Backend default model/settings |
|---|---:|---|---|
| Frame colorizer | off | `frame_colorizer_enabled` | `ddcolor`, `192x192`, blend 100 |
| Expression restorer | off | `expression_restorer_enabled` | `live_portrait`, factor 80, upper/lower face |
| Age modifier | off | `age_modifier_enabled` | `styleganex_age`, direction 0 |
| Face editor | off | `face_editor_enabled` | `live_portrait` |
| Lip syncer | off | `lip_syncer_enabled` | `edtalk_256`, weight 0.5 |

The React panel exposes the processor toggles. More detailed model, blend, region, and face-editor controls remain available through saved preferences, the local API where allowed, command-line flags, or the legacy Gradio UI.

## Execution routing tab

### CPU vs CUDA profiler

The profiler runs the currently applied models on 1280x720 recorded webcam frames. Model loading and warm-up are excluded. Results show:

- whole-pipeline CUDA time
- whole-pipeline CPU time
- current model and frames measured per provider
- per-module CPU and CUDA inference cost where the module actually ran
- current live provider timing beside each route

Disabled and input-dependent modules are shown as unmeasured rather than assigned a fabricated zero.

### Module providers

Each route can select an available provider such as `cuda`, `tensorrt`, or `cpu`:

| UI label | Preference key |
|---|---|
| Detector | `provider_detector` |
| Landmarker | `provider_landmarker` |
| Recognizer | `provider_recognizer` |
| Classifier | `provider_classifier` |
| Masker | `provider_masker` |
| Deep swapper | `provider_deep_swapper` |
| Face swapper | `provider_face_swapper` |
| Face enhancer | `provider_face_enhancer` |
| Frame enhancer | `provider_frame_enhancer` |
| Colorizer | `provider_colorizer` |
| Expression restorer | `provider_expression_restorer` |
| Age modifier | `provider_age_modifier` |
| Face editor | `provider_face_editor` |
| Lip syncer | `provider_lip_syncer` |
| Content analyser | `provider_content_analyser` |

Default route is CUDA for every module except `provider_deep_swapper`, which defaults to TensorRT when TensorRT is available. Provider changes rebuild affected inference contexts but do not reopen the physical webcam.

The measured quality/performance baseline is CUDA masking plus TensorRT FP32 DFM. Route availability depends on the installed ONNX Runtime and TensorRT environment.

## System tab

### Execution runtime

| Control | Default | Preference key | Behavior |
|---|---:|---|---|
| Primary provider | `cuda` | `execution_providers[0]` | Global fallback/provider context. Changing it reloads inference, not capture. |
| Device ID | `0` | `execution_device_ids[0]` | Selects the provider device. |
| VRAM strategy | `strict` | `video_memory_strategy` | Controls inference-pool retention/release policy. Options are `strict`, `moderate`, and `relaxed`. |
| Fast startup | on | `fast_startup` | Skips redundant model checks when checkpoints are already local. |

### Maintenance

| Action | Behavior |
|---|---|
| Restart stream | Explicitly restarts the processing stream/source. |
| Free VRAM | Clears/rebuilds inference pools for a cold state. |
| Open API docs | Opens `/docs`. |

## Apply and restart semantics

Changes are staged until **Apply changes** is pressed. The action bar states whether the result will apply live or restart the source.

### Settings that reopen physical capture

- `backend`
- `camera_choice`
- `resolution_preset`, `width`, `height`
- `fps`
- `dshow_name_device`
- `convert_rgb`
- `force_fourcc`
- `retry_black`
- `gentle_mode`
- `auto_repair`
- `color_mode`
- `lock_exposure`, `exposure_value`
- `lock_wb`, `wb_temperature`

### Settings that do not reopen physical capture

- swap engine, model, morph, source identity
- fast analysis, temporal tracking, overlay, boxes, and masks
- detector, landmarker, selector, parser, enhancers, and optional processors
- per-module provider routing and global inference provider/device
- VRAM strategy and fast-startup preference
- virtual-camera enable/disable
- preview visibility

A deep-model switch uses a seamless handoff: the current model continues processing while the requested model is prepared, then the active reference changes. If loading fails, the previous model remains usable.

## Tray-only controls

The tray host keeps the web API resident and can keep an OBS virtual camera advertised with a standby frame while physical capture is stopped.

| Tray item | Preference/default | Behavior |
|---|---|---|
| Open FaceFlow controls | - | Opens the React UI. |
| Open API options | - | Opens FastAPI documentation. |
| Start/Stop camera | - | Manual physical capture control. |
| Enable virtual streaming | `virtual_cam_enabled=false` | Enables/disables virtual output while leaving the server resident. |
| Start on app demand | `demand_capture_enabled=true` | Watches supported virtual-camera consumer handles. |
| Start mode: No processing | `tray_start_mode=none` | Demand starts in passthrough. |
| Start mode: Last processing setup | `tray_start_mode=last` | Demand starts with saved processing. This is the default. |
| Exit | - | Stops demand monitoring, shuts down the virtual publisher, server, and tray process. |

A demand-started session stops after eight seconds without a consumer. A manually started session is not auto-stopped. The selected model can stay warm while capture is off for faster subsequent starts.

## API surface

The React interface calls the local FastAPI server. Use `/docs` for exact schemas. Endpoint families include:

- `/api/health` and `/api/bootstrap`
- configuration GET/PATCH
- camera list/mode detection
- model catalog, thumbnails, install, source upload/clear
- capture start/stop and runtime start/stop/restart/free-VRAM
- status and live timings
- CPU/CUDA provider benchmark state/results
- processed and raw preview streams/snapshots
- identity session/status/capture/cancel/finish/activate/thumbnail/dataset
- DFM environment/setup/jobs/extract/train/stop/export/activate

The API binds locally by default. Treat any deliberate non-loopback binding as access to camera controls and locally stored identity data.

## Command line and legacy UI

There is no CLI tab in the React application. Use:

```powershell
python webcam_deep_swap.py --help
```

Common headless categories include camera source/mode, provider/device, virtual output, deep/classic swap, source images, detector/mask configuration, enhancers, and optional processors. The saved preferences file supplies values not explicitly overridden by flags.

`start_gradio_gui.bat` launches the legacy interface, which exposes some detailed processor sliders that the streamlined React panel intentionally omits.

## Local persistence and privacy

`user_prefs.json` stores runtime choices and may include camera names and local paths. Captures, identities, DFM jobs, uploads, checkpoints, model caches, logs, benchmarks, and generated previews are local runtime data. They are excluded by `.gitignore` and must not be force-added.

See [PRIVACY.md](PRIVACY.md) before publishing changes.
