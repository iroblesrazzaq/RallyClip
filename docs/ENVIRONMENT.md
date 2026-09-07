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
uv run python scripts/fetch_artifact.py            # ONNX into models/rallyclip_v0.5.0/
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

- Default inference dir: `models/rallyclip_v0.5.0/` (`runtime.defaults.DEFAULT_ARTIFACT_DIR`).
  Git tracks `manifest.json` + `SHA256SUMS`. ONNX / `scaler.json` come from GitHub
  Release `artifact-rallyclip_v0.5.0` via `python scripts/fetch_artifact.py`
  (CI and from-source). The Mac `.app` embeds that folder at PyInstaller time
  and does not fetch at launch.
- Classic LSTM contract: `models/rallyclip_v0.4.0/manifest.json` (weights not in git).
- YOLO pose weights used at runtime are the ONNX siblings in the v0.5.0 zip.
  `yolov8n-pose.pt` is gitignored (Ultralytics / `[train]` extra only).

## Env vars (names only)

- `RALLYCLIP_COURT_VIDEO_DIR`, `RALLYCLIP_YOLO_WEIGHTS` — court-e2e source data overrides; tests self-skip when absent.
- `RALLYCLIP_ARTIFACT_URL`, `RALLYCLIP_ARTIFACT_ZIP` — optional fetch overrides (`scripts/fetch_artifact.py`). `GITHUB_TOKEN` / `GH_TOKEN` used if set.
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
