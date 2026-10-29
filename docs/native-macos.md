# Native macOS desktop

The desktop shell, player, editor, and exporter are SwiftUI/AppKit/AVFoundation.
Analysis uses the same manifest-driven, torch-free v0.6 Python/ONNX engine as the
CLI, shipped inside the app as `Resources/Backend/RallyClipBackend`. Users do not
install Python or download weights. macOS 14+, Apple Silicon; build with Xcode
15+ or its Swift command-line tools. CI uses Xcode on `macos-15`.

## Use and development

- Import one match or naturally ordered consecutive MP4 camera chunks.
- Analysis creates a 720p proxy and saves predictions; original files are
  referenced in place. Existing saved matches in Application Support are reused.
- Points Only is one continuous AVMutableComposition across cuts and source
  boundaries. Full Match and editing use the full original timeline. Playback
  is local AVPlayer, with native fullscreen and picture-in-picture controls.
- Edit start/end times, set boundaries at the playhead, add/delete points, Save
  Edits, Discard, or Restore Original. Navigation and quitting guard unsaved edits.
- Export all points, selected highlights, separate point videos, or timestamps.
  Video export stages files beside the destination before committing them;
  progress and cancellation are visible. Originals retain their source resolution.
- Analysis cancellation stops its worker process group. Quitting shuts down the
  private service; the service also watches for parent-process death.

```bash
uv sync --extra cpu --extra dev --extra pack
uv run python scripts/fetch_artifact.py
RALLYCLIP_NATIVE_PYTHON="$PWD/.venv/bin/python" bash scripts/native/run.sh

RALLYCLIP_BUILD_PYTHON="$PWD/.venv/bin/python" bash scripts/native/build.sh
open dist/native/RallyClip.app
```

The build verifies model checksums, freezes only the analysis backend, compiles
Swift in release mode, copies the existing logo and icon, and ad-hoc signs a
standalone app. `RALLYCLIP_REUSE_BACKEND=1` reuses the helper for UI-only rebuilds.
The packaged CLI is available at
`dist/native/RallyClip.app/Contents/Resources/Backend/RallyClipBackend --cli`.
`rallyclip-desktop` remains the legacy webview launcher for source installs.

## Boundaries and security

`RallyClipKit` contains the HTTP client, point/time mappings, composition builder,
and exporter. The Swift UI owns its view state. Saved-match and analysis mutations
go through the existing `/api/*` contracts; native media playback and native file
export run directly in AVFoundation. Python analysis/proxy decode remains PyAV.
Model inputs, geometry and decode parameters still come from the artifact manifest.

The helper binds only `127.0.0.1` on an ephemeral port. Every read and write
requires a random per-launch bearer token. Browser Origin requests and rebinding
hosts are refused. The port readiness file is created exclusively with mode 0600
inside a private temporary directory; it contains no token. Worker subprocesses
do not inherit the token. Local paths are exposed only to the authenticated
native client, allowing the player to read originals without an HTTP video proxy.

Application data and logs live in `~/Library/Application Support/RallyClip`.
`RALLYCLIP_NATIVE_DATA_DIR` provides an isolated library for QA; it should not
point at the user's library during tests. `native-backend.log` captures helper
startup and analysis errors. Camera originals must remain at their imported paths;
missing footage produces a reconnect message.
The first launch shows the intro (the legacy typewriter); `defaults delete
com.iroblesrazzaq.rallyclip welcomeSeen` shows it again.

## Verification and release

```bash
swift test --package-path macos
bash scripts/native/launch_smoke.sh dist/native/RallyClip.app
PYTHONPATH=src:tests uv run pytest -q tests/test_native_backend.py
uv run python scripts/native/smoke.py \
  --backend dist/native/RallyClip.app/Contents/Resources/Backend/RallyClipBackend
```

Swift media tests encode synthetic clips and verify exported frames across a
camera boundary. The packaged smoke test boots the frozen backend in a disposable
library, analyzes the committed tennis fixture, saves/restores edits, plays three
cut intervals while counting decoded frames and playback progress, exports the
exact expected duration, and deletes its test match. Interactive QA covers native
pickers, play/seek/mode changes, edits, saved-match navigation and export controls.

`.github/workflows/native-macos.yml` runs native/backend tests and the packaged
smoke on PRs. The existing **Release** workflow defaults to the native desktop
and reuses this workflow with the existing certificate/notarization secrets.
Manual `desktop: legacy` retains the old build path. Tagged releases become
drafts; merging and publishing the desktop release remain the user's steps.

```bash
gh workflow run release.yml --ref feat/native-macos -f desktop=native
```

No signing credentials are supplied to install, compile, or smoke-test steps.
Developer ID signing applies to the embedded executable and its native libraries
before the outer app; the DMG is notarized and stapled with the existing scripts.
The workflow uploads the DMG and SHA-256 checksum.

## Bundle size

Measured 2026-10-29 with a CI-like build venv (`pip install '.[cpu,dev,pack]'`:
onnxruntime 1.31.1, opencv-python 4.14.0.94, av 18.1.0), sizes in MiB:

| | App | Backend `_internal` | DMG (unsigned) |
|---|---|---|---|
| Before | 338.2 | 328.1 | 172.5 |
| After dropping the ORT C API dylib | 306.9 | 296.7 | 163.2 |

Largest `_internal` entries after the cut: cv2 109.1, models 81.0, av 42.5,
onnxruntime 38.6, python3.11 7.0, Python.framework 6.7, numpy 6.6.

What is and is not removable:

- **onnxruntime C API dylib (dropped, 31.3 MiB).** `libonnxruntime.<ver>.dylib`
  ships beside `onnxruntime_pybind11_state.so`, which links ORT statically; no
  bundled binary links it and the Python package never `dlopen`s it. The spec
  filters it out of `a.binaries`. Frozen `--cli` output on the golden clip is
  byte-identical before and after, on both `--yolo-device cpu` and `coreml`.
- **cv2's FFmpeg (kept).** `cv2.abi3.so` links six libav*/libswscale dylibs
  directly, and dyld loads all 93 dylibs in `cv2/.dylibs` on `import cv2`,
  so none can be excluded. `opencv-python-headless` 4.14.0.94 has the same
  FFmpeg tree and is slightly larger (79 vs 76 MiB of dylibs). The FFmpeg
  majors differ from PyAV's (avformat 61 vs 62), so the two cannot share one
  copy. The only route is a source build of OpenCV with `-DWITH_FFMPEG=OFF`
  (about 75 MiB), which would change the cv2 binary that court masking and
  homography run on and so needs its own parity check.
- **Both yolov8n-pose exports (kept).** `--yolo-device auto` resolves to CoreML
  on Apple Silicon, which runs `yolov8n-pose-544x960-static.onnx`; CPU, and the
  fallback when the CoreML EP is unavailable, run `yolov8n-pose-960-dynamic.onnx`.
  `verify_artifact_dir` requires both.
- **cv2 and the classic `CourtDetector` (kept).** `compute_court_mask` runs on
  every analysis, and `court_lines.py`/`yolo_onnx_runner.py` use cv2.

`court_lines.onnx` in fp16 (not done; about 28 MiB smaller) would need: an
fp16 conversion with fp32 I/O (`onnxconverter-common`, `keep_io_types=True`);
a check that the CPU EP, which `CourtLineModel` uses, does not slow down from
inserted Casts; court-point accuracy against the court labels, since fp16
heatmaps move the soft-argmax points; a deliberate golden regeneration if
segments change; and a new artifact version (manifest SHA, `SHA256SUMS`,
release zip, `ARTIFACT_TAG`).
