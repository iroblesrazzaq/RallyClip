# RallyClip iOS

A native iOS port of the RallyClip desktop app: import a full tennis match,
segment it to point-only clips **entirely on-device** (no cloud inference), then
review / edit / export — feature-parity with the macOS desktop app, which is the
design template (`src/gui/frontend/`).

Everything the Python runtime does off-device is reproduced natively:

```
AVFoundation decode (5 fps sample)
  → OpenCV letterbox  ─┐
  → YOLOv8-pose ONNX (onnxruntime + CoreML EP)   [pose]
  → OpenCV court detection (Obj-C++ port of court_detector_impl.py)
  → court filter → player assignment → reference rescale   [preprocess]
  → FeatureSetV1 (362-dim)  →  StandardScaler (scaler.json)   [features]
  → TennisPointLSTM ONNX, windowed-average (seq 100 / overlap 50) + sigmoid   [inference]
  → gaussian smooth → hysteresis (high/low/min-dur) → segments   [postprocess]
  → AVMutableComposition export (point-only .mp4) + CSV
```

The macOS runtime is the source of truth. Ported files map 1:1 (see
`Pipeline/` headers for the `// Ports:` reference to the Python original).

## Architecture

Native SwiftUI (the desktop is a WKWebView over Flask; here the UI is rebuilt as
SwiftUI and the pipeline is called directly — no embedded HTTP server).

| Layer | Where | Notes |
|---|---|---|
| UI | `RallyClip/UI/*.swift` | SwiftUI mirror of `frontend/` (library, viewer, edit mode, upload, progress, welcome). Design tokens in `Theme.swift` mirror `styles.css`. |
| Engine | `RallyClip/Engine/*.swift` | `AnalysisJob` orchestrates the pipeline off the main actor and emits `ProgressEvent`s (same 5 stages: pose/preprocess/feature/inference/output). `MatchStore` = on-device library (`Application Support/RallyClip`). |
| Pipeline (pure Swift) | `RallyClip/Pipeline/*.swift` | Deterministic ports: `FeatureSetV1`, `PlayerAssigner`, `Preprocessor` filter/merge, `StandardScaler`, windowing, `gaussianFilter1d`, `hysteresisThreshold`, `extractSegments`. |
| Inference | `RallyClip/Pipeline/PoseRunner.swift`, `LSTMRunner.swift` | onnxruntime-objc sessions. Pose picks CoreML EP (static 544×960 export) with CPU fallback, exactly like `pose_extractor._resolve_onnx_session`. |
| Image ops / court (Obj-C++) | `RallyClip/Vision/*.{h,mm}` | OpenCV. `RCImageOps` = letterbox parity with `yolo_onnx_runner.letterbox_exact`; `RCCourtDetector` = port of `court_detector_impl.py`. |
| Video | `RallyClip/Video/*.swift` | `VideoFrameReader` (AVAssetReader, 5 fps sampling → OpenCV `Mat`), `ClipExporter` (AVMutableComposition, point-only concat). |

## Model assets (bundled)

Copied from `models/rallyclip_v0.3.1/` at build time (see `project.yml`):
`model.onnx` (LSTM), `scaler.json`, `manifest.json` (the **contract** — imgsz 960,
conf 0.25, fps 5, seq_len 100, overlap 50, hysteresis high .7/low .45/min-dur 1s),
`yolov8n-pose-544x960-static.onnx` (CoreML), `yolov8n-pose-960-dynamic.onnx` (CPU
parity fallback), and `default_court_mask.png`. Contract values are read from
`manifest.json` at runtime — never hardcoded (AGENTS.md invariant).

## Build

Prereqs: Xcode 15+ (built with 26.2), XcodeGen (`brew install xcodegen`).

```bash
cd ios && xcodegen generate && open RallyClip.xcodeproj
```

Both native deps resolve over SPM on first build — no manual downloads:
`onnxruntime` (`onnxruntime-swift-package-manager`, module `OnnxRuntimeBindings`)
and OpenCV 4.13 (`yeatse/opencv-spm`, product `OpenCV`, framework `opencv2`).
Select a device/simulator and Run. CoreML EP + Neural Engine require a **real
device**; the simulator falls back to CPU automatically. Deployment target iOS 17.

CLI build (what CI / the authoring sandbox used):

```bash
cd ios && xcodegen generate --spec project.yml --project . \
  && xcodebuild -project RallyClip.xcodeproj -scheme RallyClip \
       -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' build
```

## Verification status

- [x] Compiles for the iOS Simulator (Xcode 26.2, clean build, no app-code warnings).
- [ ] Runs on a physical device with the CoreML EP (simulator = CPU only).
- [ ] Numerical parity vs. desktop golden CLI run (`tests/fixtures/golden_cli`):
      pose decode, feature vector, LSTM probs, final segments. **Must be checked
      on-device before trusting output.** Parity risks are marked `// PARITY:` in code.
- [ ] Court detection parity vs. `tests/test_court_detection_deterministic.py`
      fixtures (the Obj-C++ port mirrors cv2 calls 1:1, but ORB/RANSAC and some
      cv2 defaults can differ across OpenCV builds).
