# PROGRESS — overwrite me at every session end

_Last updated: 2026-07-04 (session: torch-free runtime + system-webview shell)._

## Repo state

- Branch `feat/pywebview-shell` (stacked on `feat/onnx-pose-runner`), clean, pushed.
- Open PRs, both CI-green (3 OS × test/e2e): **#26** torch-free ONNX runtime,
  **#27** pywebview shell. Merge #26 first, then #27; the user merges.
- Gates: fast suite 202 passed / 1 skipped; e2e 15 passed local; golden CLI parity
  byte-equal (incl. from the frozen bundle and a torch-free venv).

## What shipped this session

1. **Torch-free runtime (PR #26)**: pose + court-detection person inference run on
   `models/rallyclip_v0.3.1/yolov8n-pose-960-dynamic.onnx` via
   `src/extraction/yolo_onnx_runner.py` (onnxruntime+numpy+cv2). 17/17 sweep samples
   byte-equal vs torch; torch/ultralytics demoted to `[train]`; typed
   UnsupportedOnnxOutputShapeError on non-pose heads. Results:
   docs/onnx-pose-parity-plan.md.
2. **System-webview shell (PR #27)**: pywebview (WKWebView/WebView2) replaces
   QtWebEngine/PySide6; `gui/native_player.py` + QWebChannel bridge deleted (the
   system webview plays H.264/HEVC natively; the frontend's HTML5 fallback was
   already complete). Bundle 765MB → 266MB .app / 113MB zipped (v0.1.0 dmg: 558MB).
   PyAV is now the only bundled FFmpeg.

## Next steps (in order)

1. User merges #26 then #27; tag a release so release.yml (already updated) produces
   the torch-free, Qt-free dmg. Release QA: WebView2 presence on older Win10;
   confirm HTML5 playback suffices → then delete the gui/app.py playback-proxy
   endpoints.
2. Optional size follow-up: videoio-less OpenCV wheel (core+imgproc+calib3d+features2d)
   kills cv2's 75MB direct-linked FFmpeg → app ~190MB. Build-infra chore, zero
   algorithm risk.
3. Backlog: training-quality-harness, macos-native-app-storage (see
   ../RallyClip/feature_list.json), perf streaming loop (docs/perf/).
