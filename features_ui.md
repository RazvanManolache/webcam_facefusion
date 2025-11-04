# Features UI Reference

Below is a reference of all visible UI fields, grouped by tab, with description, the corresponding preference key in `user_prefs.json`, and whether a command-line flag exists for it (and its name).

Note: If a field has no CLI flag, the app expects you to set it via the GUI. CLI flags currently cover a minimal subset used for quick starts.

## General

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Execution Provider | string | `cpu` (or `cuda`/`directml` if available) | ONNXRuntime provider selection | `execution_providers` (array, first used) | No | - |
| Execution Device ID | string | `0` | Device ordinal for provider (e.g., GPU id) | `execution_device_ids` (array, first used) | No | - |
| Video Memory Strategy | string | `moderate` | Memory policy for model/session reuse | `video_memory_strategy` | No | - |
| Fast Startup | bool | `true` | Skip model checks on startup | `fast_startup` | No | - |
| Shutdown App | action | - | Stops the app process | - | No | - |

## Swapping

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Swap Mode | string | `deep` | Off/Deep/Face modes | `swap_mode` | No | - |
| Source Photos | array<string:path> | `[]` | Image files used by Face Swapper | `source_paths` | No | - |
| Face Swapper Model | string | first available | Model for classical face swap | `face_swapper_model` | No | - |
| Face Swapper Pixel Boost | string | model-specific first | Inference patch/boost size | `face_swapper_pixel_boost` | No | - |
| Face Swapper Weight | float | `0.5` | Blend weight for face swap | `face_swapper_weight` | No | - |
| Deep Swapper Model | string | `iperov/james_carrey_224` | FaceFusion deep swap model id | `deep_swapper_model` | Yes | `--model` |
| Morph | int | `100` | Deep swap morph strength (0-100) | `morph` | Yes | `--morph` |

## Camera

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Backend | string | `Media Foundation` | Camera backend | `backend` | No | - |
| Camera | string | first available | Selected camera entry (label) | `camera_choice` | Yes (index only) | `--camera` |
| Resolution Preset | string | `1280x720` | Quick selector or Custom | `resolution_preset` | No | - |
| Width | int | `1920` | Capture width | `width` | Yes | `--width` |
| Height | int | `1080` | Capture height | `height` | Yes | `--height` |
| Target FPS | float | `15.0` | Desired frame rate | `fps` | No | - |
| DirectShow device name | string | empty | Exact device name (DShow) | `dshow_name_device` | No | - |
| DirectShow name (text) | string | empty | Manual device string override | `dshow_name_text` | No | - |
| Convert RGB | bool | `true` | Force RGB conversion at capture | `convert_rgb` | No | - |
| Force FOURCC | string | `Auto` | Force pixel format (MJPG/YUY2/Auto) | `force_fourcc` | No | - |
| Retry on black frames | int | `3` | Retry count if frames are black | `retry_black` | No | - |
| Gentle Mode | bool | `true` | Avoid aggressive reopen/retry | `gentle_mode` | No | - |
| Auto-repair Stream | bool | `true` | Try to recover if frames fail | `auto_repair` | No | - |
| Color Mode | string | `Auto (BGR->RGB)` | Assume input color vs convert | `color_mode` | No | - |
| Lock Exposure | bool | `true` | Lock exposure setting | `lock_exposure` | No | - |
| Exposure Value | float | `-6.0` | Exposure value when locked | `exposure_value` | No | - |
| Lock White Balance | bool | `true` | Lock WB setting | `lock_wb` | No | - |
| WB Temperature | int | `4500` | White balance temperature | `wb_temperature` | No | - |
| Show detection overlay | bool | `false` | Draw detector masks/overlays | `show_overlay` | No | - |
| Debug logs | bool | `true` | Extra logs to console | `debug_logs` | No | - |
| Show detection boxes | bool | `false` | Draw face boxes | `show_boxes` | No | - |
| Show native window | bool | `true` | Mirror to native OpenCV window | `show_native` | No | - |
| Enable Virtual Camera | bool | `false` | Publish frames to virtual camera | `virtual_cam_enabled` | No | - |

## Detection

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Detector Model | string | `retinaface` | Face detector backend | `detector_model` | No | - |
| Detector Size | string | `160x160` | Input size for detector | `detector_size` | No | - |
| Detector Score | float | `0.5` | Min confidence threshold | `detector_score` | No | - |
| Selector Mode | string | `one` | Face selection policy | `selector_mode` | No | - |
| Auto Fallback | bool | `true` | Rotate models after long no-face | `auto_fallback` | No | - |
| Landmarker Model | string | `many` | Landmark model | `landmarker_model` | No | - |
| Landmarker Score | float | `0.5` | Landmark confidence threshold | `landmarker_score` | No | - |
| Occluder Model | string | `xseg_2` | Occlusion mask model | `occluder_model` | No | - |
| Parser Model | string | first available | Face parser/segmentation model | `parser_model` | No | - |
| Use Occlusion Mask | bool | `true` | Enable occlusion use in pipeline | `use_occlusion` | Yes (negated) | `--no-occlusion` |

## Enhancing

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Face Enhancer | bool | `false` | Toggle face enhancer | `face_enhancer_enabled` | No | - |
| Face Enhancer Model | string | `gfpgan_1.4` (if available) | Model for face enhancement | `face_enhancer_model` | No | - |
| Face Enhancer Blend | int | `80` | Blend of enhanced face | `face_enhancer_blend` | No | - |
| Face Enhancer Weight | float | `0.5` | Weight in overall pipeline | `face_enhancer_weight` | No | - |
| Enable Frame Enhancer | bool | `false` | Toggle frame enhancer | `frame_enhancer_enabled` | No | - |
| Frame Enhancer Model | string | `span_kendata_x4` (if available) | Model for frame enhancement | `frame_enhancer_model` | No | - |
| Frame Enhancer Blend | int | `80` | Blend of enhanced frame | `frame_enhancer_blend` | No | - |
| Async Enhance | bool | `true` | Run enhancers asynchronously | `enhance_async` | No | - |

## Processors

### Colorizer

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Frame Colorizer | bool | `false` | Toggle frame colorizer | `frame_colorizer_enabled` | No | - |
| Colorizer Model | string | `ddcolor` (if available) | Model name | `frame_colorizer_model` | No | - |
| Colorizer Size | string | `192x192` | Model input size | `frame_colorizer_size` | No | - |
| Colorizer Blend | int | `100` | Blend intensity (0-100) | `frame_colorizer_blend` | No | - |

### Expression Restorer

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Expression Restorer | bool | `false` | Toggle | `expression_restorer_enabled` | No | - |
| Expr Model | string | `live_portrait` (if available) | Model name | `expression_restorer_model` | No | - |
| Expr Factor | int | `80` | Intensity factor | `expression_restorer_factor` | No | - |
| Expr Areas | array<string> | all available | Regions to restore | `expression_restorer_areas` | No | - |

### Age Modifier

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Age Modifier | bool | `false` | Toggle | `age_modifier_enabled` | No | - |
| Age Model | string | `styleganex_age` (if available) | Model name | `age_modifier_model` | No | - |
| Age Direction | int | `0` | Younger/Older value | `age_modifier_direction` | No | - |

### Face Editor

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Face Editor | bool | `false` | Toggle | `face_editor_enabled` | No | - |
| Editor Model | string | `live_portrait` (if available) | Model name | `face_editor_model` | No | - |
| Eyebrow Direction | float | `0.0` | Slider value | `face_editor_eyebrow_direction` | No | - |
| Eye Gaze H | float | `0.0` | Slider value | `face_editor_eye_gaze_horizontal` | No | - |
| Eye Gaze V | float | `0.0` | Slider value | `face_editor_eye_gaze_vertical` | No | - |
| Eye Open | float | `0.0` | Slider value | `face_editor_eye_open_ratio` | No | - |
| Lip Open | float | `0.0` | Slider value | `face_editor_lip_open_ratio` | No | - |
| Smile | float | `0.0` | Slider value | `face_editor_mouth_smile` | No | - |
| Mouth Grim | float | `0.0` | Slider value | `face_editor_mouth_grim` | No | - |
| Mouth Pout | float | `0.0` | Slider value | `face_editor_mouth_pout` | No | - |
| Mouth Purse | float | `0.0` | Slider value | `face_editor_mouth_purse` | No | - |
| Mouth Pos H | float | `0.0` | Slider value | `face_editor_mouth_position_horizontal` | No | - |
| Mouth Pos V | float | `0.0` | Slider value | `face_editor_mouth_position_vertical` | No | - |
| Head Pitch | float | `0.0` | Slider value | `face_editor_head_pitch` | No | - |
| Head Yaw | float | `0.0` | Slider value | `face_editor_head_yaw` | No | - |
| Head Roll | float | `0.0` | Slider value | `face_editor_head_roll` | No | - |

### Face Debugger

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Face Debugger | bool | `false` | Toggle drawing tools | `face_debugger_enabled` | No | - |
| Debugger Items | array<string> | `['face-landmark-5/68','face-mask']` | Which overlays to draw | `face_debugger_items` | No | - |

### Lip Syncer

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Lip Syncer | bool | `false` | Toggle | `lip_syncer_enabled` | No | - |
| Lip Model | string | `edtalk_256` (if available) | Model name | `lip_syncer_model` | No | - |
| Lip Weight | float | `0.5` | Blend/weight | `lip_syncer_weight` | No | - |

## Buttons / Actions (no prefs)

- **Start**: begins stream.
- **Stop**: stops stream.
- **Test Capture**: runs one-shot capture and shows diagnostics.
- **Reconnect**: reopens device with current settings.
- **Free VRAM**: clears inference pools.
- **Shutdown App**: quits the app.

## Notes

- CLI flags available (from `main()` in `webcam_deep_swap.py`):
  - `--gui`, `--list-cams`, `--camera`, `--model`, `--no-occlusion`, `--morph`, `--width`, `--height`.
- Many fields are dynamic and validated against model lists; if a saved value is unavailable, the app falls back to a valid default.
