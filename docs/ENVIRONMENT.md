# Environment — working dev setup

## Python

Verified interpreter (2026-07-03): the sibling clone's venv —
`/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip/.venv-train/bin/python3`
(Python 3.11.14; cv2 4.13, av, onnxruntime, pytest 9.0.3; torch/ultralytics installed for training but unused by the runtime).
No venv in this worktree. Alternative full-stack interpreters (unverified this session):
conda `tennis_env`, `/Users/ismaelrobles-razzaq/anaconda3/bin/python`.

Fresh setup (if you need your own env):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,desktop]"        # core + pytest + pywebview; add [train] for torch/ultralytics
pip install -e ".[e2e-ui]" && playwright install chromium   # browser e2e only
pip install -e ".[pack]"               # PyInstaller packaging only
```

From a checkout without editable install, prefix commands with `PYTHONPATH=src:tests`.

## Run commands

```bash
rallyclip --help                 # CLI (entry: cli:main)
rallyclip --video match.mp4      # segment a match
rallyclip gui                    # Flask dev UI in browser
rallyclip-desktop                # pywebview shell, WKWebView/WebView2 (entry: gui.desktop:main)
```

Local runtime config: `config.toml` (don't commit machine-specific paths).

## Model assets

- Packaged artifact (tracked): `models/rallyclip_v0.3.1/{model.onnx,scaler.json,manifest.json}`.
  The manifest is the contract source of truth (pipeline id, imgsz 960, fps 5, seq_len 100).
- YOLO pose weights: `yolov8n-pose.pt` resolved from `models/` or auto-downloaded by
  Ultralytics (gitignored).

## Env vars (names only)

- `RALLYCLIP_COURT_VIDEO_DIR`, `RALLYCLIP_YOLO_WEIGHTS` — court-e2e source data overrides; tests self-skip when absent.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` — for heavy e2e on this Mac (forces deterministic CPU-ish YOLO behavior).
- `QTWEBENGINE_REMOTE_DEBUGGING` — planned frozen-app UI testing via Playwright CDP.

## Local test data (outside this repo)

- Court-e2e source videos: `/Users/ismaelrobles-razzaq/cs_projects/RallyClip/data/raw_videos` (11 annotated 720p videos).
- Long real-video perf clips: `../RallyClip/raw_video/testing_app/` (see docs/perf/PLAN.md).
- Committed fixtures (no external data needed): `tests/fixtures/{court,golden_cli,quality}`.

## Gotchas

- Never write temp/scratch files into repo roots (GUI tests from source can drop
  `preferences.json` at root — bug pattern, see docs/runtime-config-refactor-plan.md).
- `.venv-train` carries duplicate FFmpeg dylibs (cv2 + av) → objc duplicate-class
  warnings on import; noisy but benign locally.
- Deps mostly unpinned; `opencv-python>=4.8,<5` pinned deliberately (court goldens
  validated against 4.x algorithms).
