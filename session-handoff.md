# Session Handoff

Updated: 2026-07-04 (Session 9)

## Repo/worktree layout — read this first

`RallyClip/` and `RallyClip-perf/` are **two worktrees of the same git repo**
(shared `.git`; a branch checked out in one cannot be checked out in the other):

- `RallyClip/` — parked on the **`docs` branch**. Harness files (this file,
  feature_list.json, claude-progress.md) and training data live here. Do not do
  feature work here.
- `RallyClip-perf/` — the **code worktree**, on `refactor/runtime-api-engine`.
  All recent engine/API/CI work happened here. Run tests from here.

Sibling independent repos: `YOLO-ONNX/` (torch-free ONNX pose runner + parity
harness) and `rallyclip-prod/` (Modal cloud experiments). Cross-repo context:
`../CLAUDE.md` (container level).

## GitHub state

- `main` == `feat/first-release` == `refactor/runtime-api-engine` == `33a961c`
  (content); merged via PRs #24/#25.
- `main` is protected (PR required). **CI is green** (first time ever, run
  28692435270) but not yet a required status check — enabling it is a pending
  quick win.
- v0.1.0 Mac app released, but built from pre-refactor code; rebuild + frozen
  smoke test is pending.
- Issue #21 (A/V drift): fixed and closed.

## Current in-progress feature: onnx-pose-runner-swap

Torch-free pose runner (`YOLO-ONNX/python/yolo_onnx_runner.py`) validated to
**byte-equal** end-to-end segments on the golden clip vs the Ultralytics/torch
path. Key contract facts: imgsz=960 from `models/rallyclip_v0.3.1/manifest.json`
(not 640/1920); rect letterbox => dynamic-axes ONNX export
(`YOLO-ONNX/parity_960/`, gitignored, regenerate via
`scripts/parity_v8n_960.py export`); NMS iou=0.7/max_det=300/conf-sorted.

Next steps, in order:
1. Finish/inspect the 11-video 60s-sample sweep (`YOLO-ONNX/scripts/sweep_e2e_onnx.py`
   -> `parity_960/sweep_report.json`). Decision rule: all byte-equal or <=0.2s
   boundary hops with equal counts => swap; any count mismatch => localize first.
2. Memory/perf benchmark at 960 (expect ~155MB vs ~530MB RSS class).
3. Integrate runner into RallyClip `src/` behind the RuntimeDeps.PoseExtractor
   seam; manifest gains the pose ONNX; demote torch/ultralytics to a dev extra;
   then opencv-python can go too.

## Other queued work (not started)

- Frozen-app e2e tiers: `--backend-only` dispatch + reuse L1 suite against the
  built binary; `QTWEBENGINE_REMOTE_DEBUGGING` + Playwright CDP for real-UI tests.
- CI dependency lockfile/constraints (OpenCV 5 + torch 2.12 both broke CI when
  released; `opencv-python<5` pin is in place).
- gui.app config-object injection (retire module-global monkeypatching — the
  PREFERENCES_PATH leak class).
- macos-native-app-storage, training-quality-harness (see feature_list.json).

## Test invocation (perf worktree)

    cd ../RallyClip-perf
    PYTHONPATH=src:tests ~/anaconda3/envs/tennis_env/bin/python -m pytest -m "not slow and not e2e" -q   # ~211 tests, <1 min
    # e2e: pip install .[dev,e2e-ui] && playwright install chromium; pytest -m e2e

Golden parity uses 0.25s tolerance cross-platform (one 0.2s hysteresis hop =
measured torch-version noise floor); byte-exact only on the generating machine.
