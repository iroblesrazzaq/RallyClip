# Environment — working dev setup

## Python

Verified interpreter (2026-07-03): the sibling clone's venv —
`/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip/.venv-train/bin/python3`
(Python 3.11.14; cv2 4.13, av, onnxruntime, pytest 9.0.3; torch/ultralytics installed for training but unused by the runtime).
No venv in this worktree. Alternative full-stack interpreters (unverified this session):
conda `tennis_env`, `/Users/ismaelrobles-razzaq/anaconda3/bin/python`.

Fresh setup (uv is the supported installer):

```bash
uv sync --extra cpu --extra dev --extra desktop   # runtime + pytest + pywebview
uv sync --extra cpu --extra e2e-ui && uv run playwright install chromium   # browser e2e only
uv sync --extra cpu --extra pack                  # PyInstaller packaging only
uv sync --extra cpu --extra train                 # torch / ultralytics / wandb / h5py / sklearn, training only
uv run pytest -m "not slow and not e2e"
```

Desktop-only Mac/CPU (no pytest): `uv sync --extra cpu --extra desktop`.

### NVIDIA CUDA (Windows / Linux)

`cpu` and `gpu` extras are mutually exclusive (`onnxruntime` vs `onnxruntime-gpu` — same import name; the CPU wheel wins if both are present). `dev` no longer pulls a runtime wheel, so pick one:

```bash
uv sync --extra gpu --extra desktop   # NVIDIA
# or: uv sync --extra cpu --extra desktop
```

Verify:

```bash
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You should see `CUDAExecutionProvider`. Match `onnxruntime-gpu` to your installed CUDA toolkit/driver (mismatched versions drop the CUDA EP or fail session init). Mac packaging stays on `--extra cpu` / CoreML — do not use `--extra gpu` there.

## Run commands

```bash
uv run rallyclip                 # same as `uv run rallyclip gui`
uv run rallyclip gui             # Flask UI in the browser
uv run rallyclip --video match.mp4
uv run rallyclip --help
uv run rallyclip-desktop         # pywebview shell (needs --extra desktop)
```

Local runtime config: `config.toml` (don't commit machine-specific paths).

## Model assets

- Packaged artifact (tracked): `models/rallyclip_v0.5.0/{model.onnx,scaler.json,manifest.json}`.
  Classic LSTM fallback: `models/rallyclip_v0.4.0/`. The manifest is the contract
  source of truth (pipeline id, imgsz 960, fps 5, seq_len 100).
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
