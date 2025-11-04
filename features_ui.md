# Features UI Reference

Below is a reference of all visible UI fields, grouped by tab, with description, the corresponding preference key in `user_prefs.json`, and whether a command-line flag exists for it (and its name).

Note: If a field has no CLI flag, the app expects you to set it via the GUI. CLI flags currently cover a minimal subset used for quick starts.

## General

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Execution Provider | string | `cpu` (or `cuda`/`directml` if available) | ONNXRuntime provider selection | `execution_providers` (array, first used) | Yes | `--ep` |
| Execution Device ID | string | `0` | Device ordinal for provider (e.g., GPU id) | `execution_device_ids` (array, first used) | Yes | `--device` |
| Video Memory Strategy | string | `moderate` | Memory policy for model/session reuse | `video_memory_strategy` | Yes | `--video-mem` |
| Fast Startup | bool | `true` | Skip model checks on startup | `fast_startup` | No | - |
| Shutdown App | action | - | Stops the app process | - | No | - |

## Swapping

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Swap Mode | string | `deep` | Off/Deep/Face modes | `swap_mode` | No | - |
| Source Photos | array<string:path> | `[]` | Image files used by Face Swapper | `source_paths` | Yes | `--source (repeat)` |
| Face Swapper Model | string | first available | Model for classical face swap | `face_swapper_model` | Yes | `--face-swapper-model` |
| Face Swapper Pixel Boost | string | model-specific first | Inference patch/boost size | `face_swapper_pixel_boost` | Yes | `--face-swapper-pixel` |
| Face Swapper Weight | float | `0.5` | Blend weight for face swap | `face_swapper_weight` | Yes | `--face-swapper-weight` |
| Deep Swapper Model | string | `iperov/james_carrey_224` | FaceFusion deep swap model id | `deep_swapper_model` | Yes | `--model` |
| Morph | int | `100` | Deep swap morph strength (0-100) | `morph` | Yes | `--morph` |

## Camera

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Backend | string | `Media Foundation` | Camera backend | `backend` | Yes | `--backend` |
| Camera | string | first available | Selected camera entry (label) | `camera_choice` | Yes (index only) | `--camera` |
| Resolution Preset | string | `1280x720` | Quick selector or Custom | `resolution_preset` | Yes | `--resolution` |
| Width | int | `1920` | Capture width | `width` | Yes | `--width` |
| Height | int | `1080` | Capture height | `height` | Yes | `--height` |
| Target FPS | float | `15.0` | Desired frame rate | `fps` | Yes | `--fps` |
| DirectShow device name | string | empty | Exact device name (DShow) | `dshow_name_device` | Yes | `--dshow-name` |
| DirectShow name (text) | string | empty | Manual device string override | `dshow_name_text` | Yes | `--dshow-name` |
| Convert RGB | bool | `true` | Force RGB conversion at capture | `convert_rgb` | Yes | `--convert-rgb` |
| Force FOURCC | string | `Auto` | Force pixel format (MJPG/YUY2/Auto) | `force_fourcc` | Yes | `--fourcc` |
| Retry on black frames | int | `3` | Retry count if frames are black | `retry_black` | Yes | `--retry-black` |
| Gentle Mode | bool | `true` | Avoid aggressive reopen/retry | `gentle_mode` | Yes | `--gentle` |
| Auto-repair Stream | bool | `true` | Try to recover if frames fail | `auto_repair` | Yes | `--auto-repair` |
| Color Mode | string | `Auto (BGR->RGB)` | Assume input color vs convert | `color_mode` | Yes | `--color-mode` |
| Lock Exposure | bool | `true` | Lock exposure setting | `lock_exposure` | Yes | `--lock-exposure` |
| Exposure Value | float | `-6.0` | Exposure value when locked | `exposure_value` | Yes | `--exposure` |
| Lock White Balance | bool | `true` | Lock WB setting | `lock_wb` | Yes | `--lock-wb` |
| WB Temperature | int | `4500` | White balance temperature | `wb_temperature` | Yes | `--wb-temp` |
| Show detection overlay | bool | `false` | Draw detector masks/overlays | `show_overlay` | Yes | `--show-overlay` |
| Debug logs | bool | `true` | Extra logs to console | `debug_logs` | Yes | `--debug-logs` |
| Show detection boxes | bool | `false` | Draw face boxes | `show_boxes` | Yes | `--show-boxes` |
| Show native window | bool | `true` | Mirror to native OpenCV window | `show_native` | Yes | `--show-native` |
| Enable Virtual Camera | bool | `false` | Publish frames to virtual camera | `virtual_cam_enabled` | Yes | `--virtual-cam` |

## Detection

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Detector Model | string | `retinaface` | Face detector backend | `detector_model` | Yes | `--detector-model` |
| Detector Size | string | `160x160` | Input size for detector | `detector_size` | Yes | `--detector-size` |
| Detector Score | float | `0.5` | Min confidence threshold | `detector_score` | Yes | `--detector-score` |
| Selector Mode | string | `one` | Face selection policy | `selector_mode` | Yes | `--selector-mode` |
| Auto Fallback | bool | `true` | Rotate models after long no-face | `auto_fallback` | Yes | `--auto-fallback` |
| Landmarker Model | string | `many` | Landmark model | `landmarker_model` | Yes | `--landmarker-model` |
| Landmarker Score | float | `0.5` | Landmark confidence threshold | `landmarker_score` | Yes | `--landmarker-score` |
| Occluder Model | string | `xseg_2` | Occlusion mask model | `occluder_model` | Yes | `--occluder-model` |
| Parser Model | string | first available | Face parser/segmentation model | `parser_model` | Yes | `--parser-model` |
| Use Occlusion Mask | bool | `true` | Enable occlusion use in pipeline | `use_occlusion` | Yes (negated) | `--no-occlusion` |

## Enhancing

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Face Enhancer | bool | `false` | Toggle face enhancer | `face_enhancer_enabled` | Yes | `--face-enhancer` |
| Face Enhancer Model | string | `gfpgan_1.4` (if available) | Model for face enhancement | `face_enhancer_model` | Yes | `--face-enhancer-model` |
| Face Enhancer Blend | int | `80` | Blend of enhanced face | `face_enhancer_blend` | Yes | `--face-enhancer-blend` |
| Face Enhancer Weight | float | `0.5` | Weight in overall pipeline | `face_enhancer_weight` | Yes | `--face-enhancer-weight` |
| Enable Frame Enhancer | bool | `false` | Toggle frame enhancer | `frame_enhancer_enabled` | Yes | `--frame-enhancer` |
| Frame Enhancer Model | string | `span_kendata_x4` (if available) | Model for frame enhancement | `frame_enhancer_model` | Yes | `--frame-enhancer-model` |
| Frame Enhancer Blend | int | `80` | Blend of enhanced frame | `frame_enhancer_blend` | Yes | `--frame-enhancer-blend` |
| Async Enhance | bool | `true` | Run enhancers asynchronously | `enhance_async` | Yes | `--enhance-async` |

## Processors

### Colorizer

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Frame Colorizer | bool | `false` | Toggle frame colorizer | `frame_colorizer_enabled` | Yes | `--colorizer` |
| Colorizer Model | string | `ddcolor` (if available) | Model name | `frame_colorizer_model` | Yes | `--colorizer-model` |
| Colorizer Size | string | `192x192` | Model input size | `frame_colorizer_size` | Yes | `--colorizer-size` |
| Colorizer Blend | int | `100` | Blend intensity (0-100) | `frame_colorizer_blend` | Yes | `--colorizer-blend` |

### Expression Restorer

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Expression Restorer | bool | `false` | Toggle | `expression_restorer_enabled` | Yes | `--expr` |
| Expr Model | string | `live_portrait` (if available) | Model name | `expression_restorer_model` | Yes | `--expr-model` |
| Expr Factor | int | `80` | Intensity factor | `expression_restorer_factor` | Yes | `--expr-factor` |
| Expr Areas | array<string> | all available | Regions to restore | `expression_restorer_areas` | Yes | `--expr-area (repeat)` |

### Age Modifier

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Age Modifier | bool | `false` | Toggle | `age_modifier_enabled` | Yes | `--age-mod` |
| Age Model | string | `styleganex_age` (if available) | Model name | `age_modifier_model` | Yes | `--age-model` |
| Age Direction | int | `0` | Younger/Older value | `age_modifier_direction` | Yes | `--age-direction` |

### Face Editor

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Face Editor | bool | `false` | Toggle | `face_editor_enabled` | Yes | `--editor` |
| Editor Model | string | `live_portrait` (if available) | Model name | `face_editor_model` | Yes | `--editor-model` |
| Eyebrow Direction | float | `0.0` | Slider value | `face_editor_eyebrow_direction` | Yes | `--fe-eyebrow-dir` |
| Eye Gaze H | float | `0.0` | Slider value | `face_editor_eye_gaze_horizontal` | Yes | `--fe-eye-h` |
| Eye Gaze V | float | `0.0` | Slider value | `face_editor_eye_gaze_vertical` | Yes | `--fe-eye-v` |
| Eye Open | float | `0.0` | Slider value | `face_editor_eye_open_ratio` | Yes | `--fe-eye-open` |
| Lip Open | float | `0.0` | Slider value | `face_editor_lip_open_ratio` | Yes | `--fe-lip-open` |
| Smile | float | `0.0` | Slider value | `face_editor_mouth_smile` | Yes | `--fe-smile` |
| Mouth Grim | float | `0.0` | Slider value | `face_editor_mouth_grim` | No | - |
| Mouth Pout | float | `0.0` | Slider value | `face_editor_mouth_pout` | No | - |
| Mouth Purse | float | `0.0` | Slider value | `face_editor_mouth_purse` | No | - |
| Mouth Pos H | float | `0.0` | Slider value | `face_editor_mouth_position_horizontal` | No | - |
| Mouth Pos V | float | `0.0` | Slider value | `face_editor_mouth_position_vertical` | No | - |
| Head Pitch | float | `0.0` | Slider value | `face_editor_head_pitch` | Yes | `--fe-head-pitch` |
| Head Yaw | float | `0.0` | Slider value | `face_editor_head_yaw` | Yes | `--fe-head-yaw` |
| Head Roll | float | `0.0` | Slider value | `face_editor_head_roll` | Yes | `--fe-head-roll` |

### Face Debugger

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Face Debugger | bool | `false` | Toggle drawing tools | `face_debugger_enabled` | Yes | `--face-debugger` |
| Debugger Items | array<string> | `['face-landmark-5/68','face-mask']` | Which overlays to draw | `face_debugger_items` | Yes | `--face-debugger-item (repeat)` |

### Lip Syncer

| Field | Type | Default | Description | Prefs key | CLI | CLI flag |
|---|---|---|---|---|---|---|
| Enable Lip Syncer | bool | `false` | Toggle | `lip_syncer_enabled` | Yes | `--lip-syncer` |
| Lip Model | string | `edtalk_256` (if available) | Model name | `lip_syncer_model` | Yes | `--lip-model` |
| Lip Weight | float | `0.5` | Blend/weight | `lip_syncer_weight` | Yes | `--lip-weight` |

## Buttons / Actions (no prefs)

- **Start**: begins stream.
- **Stop**: stops stream.
- **Test Capture**: runs one-shot capture and shows diagnostics.
- **Reconnect**: reopens device with current settings.
- **Free VRAM**: clears inference pools.
- **Shutdown App**: quits the app.

## CLI-only and Utility

- **Headless mode**: run without GUI using prefs/flags; optional virtual camera output.
  - Flag: `--headless`
- **List cameras**: `--list-cams`
- **GUI**: `--gui` (GUI now auto-opens browser)

## Notes

- CLI flags available (from `main()` in `webcam_deep_swap.py`):
  - `--gui`, `--list-cams`, `--camera`, `--model`, `--no-occlusion`, `--morph`, `--width`, `--height`.
- Many fields are dynamic and validated against model lists; if a saved value is unavailable, the app falls back to a valid default.
