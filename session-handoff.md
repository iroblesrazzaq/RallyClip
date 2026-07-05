# Session Handoff

Updated: 2026-07-04 (Session 10)

## Repo/worktree layout — read this first

`RallyClip/` and `RallyClip-perf/` are **two worktrees of the same git repo**
(shared `.git`; a branch checked out in one cannot be checked out in the other):

- `RallyClip/` — parked on the **`docs` branch**. Harness files (this file,
  feature_list.json, claude-progress.md) and training data live here. Do not do
  feature work here.
- `RallyClip-perf/` — the **code worktree**, on `feat/pywebview-shell`
  (stacked on `feat/onnx-pose-runner`). Run tests from here. It has its own
  tiered agent harness: AGENTS.md -> features.json,
  docs/{REPO_MAP,PROGRESS,DECISIONS,ENVIRONMENT,testing}.md.

Sibling independent repos: `YOLO-ONNX/` (torch-free ONNX pose runner + parity
harness) and `rallyclip-prod/` (Modal cloud experiments, deprecated). Cross-repo
context: the sibling table in `RallyClip-perf/docs/REPO_MAP.md`.

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

Session 10 shipped TWO features, both CI-green and awaiting merge:

- **PR #26 — torch-free runtime**: ONNX pose runner integrated (17/17 sweep
  byte-equal; golden CLI byte-equal in a venv without torch);
  torch/ultralytics -> [train] extra.
- **PR #27 — system-webview shell** (stacked on #26): pywebview
  (WKWebView/WebView2) replaces QtWebEngine/PySide6; Qt native player +
  QWebChannel bridge deleted (system webview plays H.264/HEVC natively).
  Bundle 765MB -> 266MB .app / 113MB zipped (v0.1.0 dmg was 558MB); PyAV is
  the only bundled FFmpeg.

Next steps, in order:
1. User merges PR #26, then PR #27 (it auto-retargets to main).
2. Tag a release so release.yml (already updated) produces the torch-free,
   Qt-free dmg. Release QA: WebView2 presence on older Win10; confirm HTML5
   playback suffices, then delete gui/app.py playback-proxy endpoints.
3. Optional: videoio-less OpenCV wheel kills cv2's 75MB direct-linked FFmpeg
   (app ~190MB); build-infra chore, zero algorithm risk.

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
