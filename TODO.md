# TODO

## First macOS release scope

Included in the first Mac release:

- Packaged PySide6 desktop app with the local Flask backend embedded in the app.
- Saved-match library and replay-first startup path.
- Native QtMultimedia replay viewer for the packaged Mac app.
- Browser/WebM replay remains as a development fallback, not the production Mac path.
- Source-time playback timeline with point skipping, manual gap playback, hover controls,
  fullscreen, keyboard shortcuts, CSV, export, and visible point markers.
- Stable welcome/get-started persistence through the backend preferences endpoint, not
  fragile localhost `localStorage` alone.
- Replay startup does not import Torch, Ultralytics, PyAV, or Numpy.
- New Match warms the analysis runtime only when the user enters the upload flow.
- Analysis runs in a child worker process so inference memory can exit after completion,
  cancellation, or failure.
- Current playback strategy is native source playback with ready `playback_proxy.mp4`
  preferred when present, plus proxy fallback on media error.
- Native playback watchdog logging for RSS, frame heartbeat, media status, buffer status,
  and one guarded reload attempt on stall or rising memory.

Not included in the first Mac release:

- YOLO ONNX runtime replacement. The release still uses the existing Torch/Ultralytics
  analysis stack inside the isolated worker process.
- The larger engine/API rework where CLI, desktop, browser, and future mobile clients all
  call one explicit runtime API contract.
- A full GitHub Actions release pipeline with Apple signing, notarization, DMG creation,
  and release upload.
- MLX or CoreML inference runtime.
- iOS app support.
- Training dependency slimming. Training remains a developer workflow, not a runtime
  requirement for replaying saved matches.
- HLS/fMP4 local playback. That experiment was reverted because first-open transcoding
  blocked playback and overheated the machine.

## First macOS release checklist

1. Clean and commit the release branch.
2. Rebuild `dist/RallyClip.app` from the committed branch.
3. Fresh-user smoke test:
   - remove or move local RallyClip app data;
   - launch the app;
   - confirm welcome appears once;
   - click Get Started;
   - quit and reopen;
   - confirm the app opens straight to the library/home view.
4. Packaged replay smoke test:
   - open an existing saved match;
   - confirm native viewer opens rather than browser chunk playback;
   - play at least 15 minutes;
   - confirm memory stabilizes, controls remain responsive, fullscreen works, and the app
     does not steal foreground focus when inactive.
5. Packaged processing smoke test:
   - start New Match from the packaged app;
   - confirm runtime warmup appears only in the upload flow;
   - process one short valid video;
   - confirm progress updates, saved library output, CSV output, export, and worker exit.
6. Input-validation smoke test:
   - missing upload field returns a clear error;
   - non-video input returns a clear error before pose extraction;
   - unreadable/unsupported video returns a clear error before expensive inference.
7. Bundle audit:
   - confirm frontend assets are bundled;
   - confirm model artifacts are bundled or resolved from the app data root;
   - confirm QtWebEngine and QtMultimedia resources are bundled;
   - confirm ffmpeg/proxy-generation behavior is clear if ffmpeg is absent.
8. Apple release packaging:
   - sign app with Developer ID Application certificate;
   - enable hardened runtime with required entitlements;
   - notarize with Apple;
   - staple the notarization ticket;
   - package as a signed DMG or zip;
   - install and launch on a second Mac.

## Post-release product/architecture work

- Replace inference-time Ultralytics/PyTorch pose extraction with the sibling
  `../YOLO-ONNX` runner/wrapper and bundled YOLO ONNX weights, so the app runtime can
  shrink toward ONNX/runtime dependencies unless users explicitly install training extras.
- Rework the runtime engine behind an explicit API contract shared by CLI, desktop,
  browser dev UI, and future iOS/mobile clients.
- Keep CLI support as a first-class interface, but make it a client of the shared runtime
  API rather than a separate path that can drift from the app.
- Split install extras clearly:
  - replay/library runtime;
  - analysis runtime;
  - training/evaluation runtime.
- Add release CI/CD:
  - run tests;
  - build the macOS app on GitHub Actions;
  - sign and notarize with Apple credentials;
  - create DMG/zip artifacts;
  - publish GitHub Release assets.
- Add fresh-machine release tests for app launch, first-run preference persistence,
  replay, New Match, export, and CLI mode.
- Decide proxy policy after more data:
  - keep current lazy/preferred proxy behavior if stable;
  - optionally generate proxy after processing completes when the machine is idle;
  - avoid blocking first replay on full-video transcoding.
- Add durable playback diagnostics UI or exportable log bundle for user support.
- Confirm storage model and cleanup policy for copied source videos, CSV files, proxies,
  exports, logs, and preferences.

## Backlog

- Diagnose native Mac playback memory/stall issue. Forced first-open HLS generation
  was reverted because it blocked playback behind a long, hot full-video transcode.
  The native viewer now opens source video immediately again, keeps proxy fallback on
  media error, and logs watchdog memory/frame-heartbeat data for the next repro.
- Add W&B integration behind `wandb.enabled` (run metadata, metrics, artifacts).
- Add integration tests for the full pipeline on synthetic HDF5 inputs.
- Add CLI smoke tests for `train.py` and `visualize.py`.
- Add dataset sharding option for large feature sets.
- Expand dataset metadata tracking (court surface, indoor/outdoor, player info) into manifests.
- Add optional augmentation hooks for future data sources and label generation.
- Document inference-time downsampling strategy separate from training preproc.




- test YOLO preproc hyperparams, particularly yolo size on quality of model outputs


- test automating data collection with model self-dataset generation


- improve postprocessing: train another model with IOU?
- another LSTM?


- try LSTM with attention, other architectures, hyperparams. 

- optimize inference: push YOLO inference cost as far down as possible
- minimize downsampled frames
- introduce linear interpolation for features?

- fine tune YOLO-n on YOLO-L outputs, reduced imgsz

- optimize inference with batching, other stuff?
- video resolution: if we can reduce video res for smaller yolo inference, would be optimal: potentially reduce imgsz



- add aggresiveness slider for postprocessing tuning of to have more sensitive to inclusion vs not



- document failure cases for court detector, maybe make more robust?
