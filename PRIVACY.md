# Privacy and public-release safety

FaceFlow processes webcam frames locally. Runtime data is deliberately kept out of Git:

- `.assets/` contains uploads, identity datasets, generated thumbnails, DFM training jobs and other local runtime assets.
- `.caches/` contains downloaded models and generated inference caches.
- `logs/`, `.logs/` and `*.log` contain local diagnostics.
- `benchmarks/` contains captured webcam videos, frames and machine-specific timing reports.
- `user_prefs.json` contains camera names, local file paths and personal runtime choices.
- `webui/dist/` and TypeScript build-info files are generated locally.

These paths are excluded by `.gitignore`. Do not force-add them. Before publishing, review the exact staged file list and scan it for credentials, absolute home-directory paths, email addresses and captured media.

Guided identity and DFM capture requires explicit consent in the UI. Captured identity datasets remain under `.assets/` unless the operator intentionally exports them. They must never be committed to the source repository.

The public repository vendors the required `facefusion_mrg` source directly. This keeps the release self-contained and avoids publishing personal submodule history or depending on a private or disabled fork.
