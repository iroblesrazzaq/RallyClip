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

## Tests (`RallyClipTests`, Swift Testing)

```bash
cd ios && xcodegen generate --spec project.yml --project .
# Fast, deterministic suites only (skip the heavy end-to-end run):
xcodebuild test -project RallyClip.xcodeproj -scheme RallyClip \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -skip-testing:RallyClipTests/EndToEndParityTests
# Everything, including end-to-end parity:
xcodebuild test -project RallyClip.xcodeproj -scheme RallyClip \
  -destination 'platform=iOS Simulator,name=iPhone 17'
```

- **`PipelineMathTests`** — deterministic ports vs. the Python originals:
  windowing, gaussian, hysteresis, segment extraction, scaler, the 362-dim
  feature layout + velocity, near/far assignment + court filter, pose decode + NMS.
- **`ModelRuntimeTests`** — bundled assets on the CPU path: manifest contract,
  scaler, LSTM produces valid probabilities, pose runs, court default mask +
  synthetic detect.
- **`ExportTests`** — the per-point export flows cut real clips out of the golden
  fixture: one clip per point in order (`points.zip`), and a highlight whose
  length is the sum of the selected points (`buildHighlight`).
- **`EndToEndParityTests`** (tag `.e2e`) — runs the whole pipeline on the committed
  `tests/fixtures/golden_cli/clip.mp4` and asserts the segments match
  `golden_segments.csv` (same rule as the desktop `test_cli_golden_parity`:
  identical count, boundaries within one 0.2 s hop).

## Verification status

- [x] Compiles for the iOS Simulator (Xcode 26.2, clean build, no app-code warnings).
- [x] Full test suite green on the simulator (CPU path): 26 deterministic/runtime
      tests + end-to-end parity — **segments match the desktop golden exactly**.
- [ ] Runs on a physical device with the CoreML EP (simulator = CPU only). The
      static-export + ANE path (`PoseDevice.coreml`) is not exercised by the
      simulator; verify a device run before trusting the CoreML numbers.
- [ ] Court detection asserted directly against `test_court_detection_deterministic`
      fixtures (currently only exercised transitively by the end-to-end run, which
      passes). Parity risks remain marked `// PARITY:` in code.
