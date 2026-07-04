# Current State

| Field | Value |
| --- | --- |
| Repository root directory | `/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip` |
| Standard startup path | `./init.sh` |
| Standard evidence path | Feature-specific; define before marking a feature `passing` |
| Highest priority unfinished feature | `runtime-video-validation` |
| Current active branch work | `runtime-api-engine-refactor` in `/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf` |
| Current blocker | None for v0.1.0 packaging; GitHub-downloaded DMG install/open verified. Runtime/API/engine refactor is an uncommitted checkpoint; `main` has not been rebased yet. |

# Session Record

## Session 0 - 2026-06-28

Harness scaffolded. Feature-specific evidence definitions are intentionally deferred.

## Session 1 - 2026-06-28

Video UX work was moved away from the fragile browser/WebM chunk player for the packaged Mac app. The app now uses a native PySide6 QtMultimedia viewer with source-time playback, point-skip scheduling, hover controls, fullscreen, keyboard shortcuts, CSV/export actions, and a point-aware timeline overlay. Current manual testing shows the native viewer is working decently well for release use; the initial slow-mo behavior appears to have been buffering-related rather than a scheduler bug.

Still left: formal release-harness feature work has not started. The highest-priority tracked feature remains `runtime-video-validation`, and browser chunk playback remains a dev fallback rather than the production Mac playback path.

## Session 2 - 2026-06-28

Observed a release-blocking native playback memory/stall issue: after a few minutes of playback, video can freeze while audio continues, controls become laggy/unusable, and app memory can climb to roughly 750 MB. This is likely not caused by Python buffering decoded video frames; `NativeViewerWidget` uses `QMediaPlayer`/`QVideoWidget` directly against the stored source file. Current code-level suspects are full-resolution source playback stressing the native backend, translucent always-on-top overlay windows composited over the native video layer, and no watchdog for silent video-render stalls that do not emit a Qt media error.

Next debugging direction: add native playback memory/stall instrumentation, reproduce with the same saved match, compare source playback vs `playback_proxy.mp4`, and test a simplified in-widget overlay/no-opacity path. Proposed mitigation is to make bounded H.264/AAC proxy playback the normal Mac viewer path for high-risk sources or after any stall/memory threshold, instead of only falling back when Qt raises an explicit media error.

## Session 3 - 2026-06-28

Implemented replay-first startup isolation on `feat/first-release`: `gui.app` no longer imports Torch, Ultralytics, PyAV, or Numpy during library/replay startup; `/api/config/defaults` is now static; `/api/runtime/status` and `/api/runtime/warmup` expose analysis readiness; and upload jobs run through a child analysis worker process so inference memory exits after completion or cancellation. Evidence: direct import probe showed `gui.app` imports in 0.216s with the heavy modules unloaded; worker warmup reached `ready` while the parent process still had those modules unloaded; and targeted startup/default/upload/default-job tests passed.

Still left: rebuild and manually launch the Mac app to confirm the packaged path uses `--analysis-worker`, that New Match warmup does not steal replay startup performance, and that long native/proxy playback remains stable in the real app.

## Session 4 - 2026-06-30

First macOS release packaging reached release-ready state on `feat/first-release`. The release branch and tag `v0.1.0` were pushed to GitHub at commit `8848492 add manual update check`. The corrected release artifact is `dist/RallyClip-0.1.0-macOS-arm64.dmg` in the `RallyClip-perf` worktree, copied from the accepted/stapled resubmission DMG.

Release artifact evidence:

- DMG path: `/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf/dist/RallyClip-0.1.0-macOS-arm64.dmg`
- SHA256: `a89dfdadee10d304e34f313cae540923639a16e39abf783634a6f9a76199ad96`
- App signature inside corrected DMG: Developer ID Application `Ismael Robles-Razzaq (L9W8X6N9B9)`, Team ID `L9W8X6N9B9`, hardened runtime enabled.
- Apple notarization accepted submission `0fb928dc-2ce5-4dcf-b6c7-d385dbad3691` for `RallyClip-0.1.0-macOS-arm64-resubmit.dmg`.
- `xcrun stapler staple` and `xcrun stapler validate` passed.
- Mounted-DMG Gatekeeper assessment passed: `/Volumes/RallyClip 0.1.0/RallyClip.app: accepted`, `source=Notarized Developer ID`.

Important correction: the earlier accepted `RallyClip-0.1.0-macOS-arm64.dmg` submission was based on an ad-hoc signed app and should not be shipped. The final `RallyClip-0.1.0-macOS-arm64.dmg` file was overwritten with the notarized/stapled resubmission artifact and is the one to upload to GitHub Releases.

Manual acceptance: user confirmed DMG install/open worked, welcome behavior worked, source-only native replay completed successfully without proxy freeze, and inference/replay looked acceptable for first release. Automatic proxy and HLS playback are intentionally out of v0.1.0; source playback is the production path.

GitHub state: `feat/first-release` and tag `v0.1.0` are pushed. A CLI GitHub Release attempt created a draft without the DMG asset; continue via the GitHub website by targeting tag `v0.1.0`, uploading the notarized DMG, publishing as a normal release, and marking compatibility as Apple Silicon only. GitHub does not convert zips to DMGs.

Branch cleanup: `origin/perf/streaming-pipeline` is stale and can be deleted because it is an ancestor of `feat/first-release` and has `0` unique commits relative to the release branch.

Next highest-priority harness feature remains `runtime-video-validation`.

## Session 5 - 2026-06-30

Release blocker found after downloading/installing the DMG: the packaged app opened to a blank gray Qt window even though the backend was listening and serving `/`, `script.js`, and `styles.css`. Diagnostic launch with `QTWEBENGINE_REMOTE_DEBUGGING=9222` showed `V8 process OOM (Failed to reserve virtual memory for CodeRange)`. Root cause: QtWebEngine/Chromium V8 under hardened runtime needs JIT/executable-memory entitlements on the main executable and `QtWebEngineProcess.app`; Apple notarization accepted the old DMG but did not catch this runtime failure.

Fix staged in `RallyClip-perf`: added `packaging/macos/RallyClip.entitlements`, updated `RallyClip.spec` to use it, and added `scripts/release/sign_macos_app.sh` to explicitly sign both `Contents/MacOS/RallyClip` and the QtWebEngine helper with the entitlements before signing the app bundle. Evidence: the entitlement-signed `dist/RallyClip.app` launched and rendered the UI; the V8 OOM log disappeared; `codesign --verify --deep --strict` passed; both main executable and helper printed `com.apple.security.cs.allow-jit` and `com.apple.security.cs.allow-unsigned-executable-memory`.

New fixed local DMG was created but is not notarized yet:

- `/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf/dist/RallyClip-0.1.0-macOS-arm64-fixed.dmg`
- SHA256: `80d72a829a2a9b3008106743eb4179c4dc80b015256e519ad7c24663f5780ec7`
- Mounted app verifies locally and has the required entitlements.
- `spctl` reports `rejected source=Unnotarized Developer ID`, which is expected until this new DMG is submitted to Apple, accepted, and stapled.

Important correction: do not publish the previous notarized DMG. Build/sign/notarize from the entitlement-fixed app.

## Session 6 - 2026-06-30

The entitlement-fixed macOS DMG was accepted by Apple notarization, stapled, validated, renamed to the normal release filename, uploaded to GitHub Releases, downloaded from GitHub, installed, and opened successfully. User confirmed the GitHub-downloaded app works. Old local DMGs and old app bundles were deleted for a clean install test.

Final release artifact state:

- GitHub release asset name: `RallyClip-0.1.0-macOS-arm64.dmg`
- Final local artifact was removed after upload/download validation.
- Final SHA256 before deletion: `8370eaefd0a9f2bf2dedc7e1892667b3f55105175bf79dc618b80b4d4988fcf8`
- Release branch fix commit pushed: `80ba932 fix mac release webengine entitlements`

Packaged-app storage contract confirmed from code: app data is stored under `~/RallyClip`. Saved matches live in `~/RallyClip/library/<item-id>/`; each saved match stores the copied full source video as `source.mp4`, detected points as `segments.csv`, metadata as `meta.json`, and thumbnail as `thumb.jpg`. Analysis scratch jobs live under `~/RallyClip/jobs`, logs under `~/RallyClip/logs`, and lazy export/cache files may be created inside the relevant match folder.

## Session 7 - 2026-06-30

Added future feature tracking for standard macOS app storage. Current v0.1.0 behavior remains `~/RallyClip`, which is acceptable for the first release but should be migrated later to normal platform locations: saved matches/preferences under `~/Library/Application Support/RallyClip`, cache/scratch data under `~/Library/Caches/RallyClip`, and logs under `~/Library/Logs/RallyClip`.

Feature added: `macos-native-app-storage` with status `not_started`. Evidence required before passing: packaged path resolution uses the standard directories, existing `~/RallyClip` installs migrate without data loss, and old saved matches still open after migration.

## Session 8 - 2026-07-01

Started the runtime/API/engine split on the shipped release baseline in the `RallyClip-perf` worktree. Branch: `refactor/runtime-api-engine`, based on `feat/first-release` at `80ba932`. `main` remains untouched and should not be rebased until CLI and GUI/API parity are proven.

Implemented an uncommitted checkpoint:

- Added `rallyclip_core` for pure contracts, interval helpers, pipeline resolution, playback manifest payloads, and source-time scheduling.
- Added `rallyclip_engine` for model-object analysis execution. A pipeline now owns preprocessing, inference, postprocessing, and CSV/video-ready result output.
- Added `rallyclip_api` as a thin service facade. Flask currently delegates defaults, runtime status/warmup, library listing, and playback manifest through it.
- Moved native point-skip scheduler behavior into `rallyclip_core.playback.SourceTimelineScheduler`; the Qt native player now aliases that pure scheduler while retaining platform-specific rendering/control code.
- Added `ENVIRONMENT.md` and `docs/runtime-api-engine-refactor.md` documenting the branch architecture, test commands, current git state, and next steps.

Evidence:

- `PYTHONPATH=src:tests python3 -m compileall -q src tests` passed.
- Runtime/API/CLI/GUI/startup/native/video parity suite passed: `78 passed, 1 skipped`.
- GUI E2E passed: `15 passed, 3 skipped`.

Still left before merging/rebasing `main`: commit the checkpoint, run direct CLI output parity on a real or fixture video, move saved-match file resolution out of `gui.app`, wire job lifecycle/export through `RallyClipServices`, add stronger golden interval parity, then decide whether to rebase or fast-forward `main`.

## Session 9 - 2026-07-03/04

Completed and landed the runtime/API/engine refactor, achieved the first green CI in repo history, and validated the torch-free ONNX pose runner to byte-equality end-to-end.

Worktree/branch clarification (recorded because past notes were ambiguous): `RallyClip/` and `RallyClip-perf/` are two worktrees of the SAME git repo. `RallyClip/` stays parked on the `docs` branch (harness/progress files live here); `RallyClip-perf/` is where code work happens, on `refactor/runtime-api-engine`. A branch checked out in one worktree cannot be checked out in the other. As of this session `main` == `feat/first-release` == `refactor/runtime-api-engine` == `33a961c` content-wise.

Landed on main (PRs #24 fcda361, #25 ed39983):

- Full facade wiring: job lifecycle, export, library, playback through RallyClipServices; CLI --json; SavedMatchStore; golden CLI parity test on a committed 24s fixture.
- Issue #21 A/V sync drift fixed (per-interval seek+decode, sample-granular audio reconciliation; 336ms -> 0.3ms measured) and closed.
- PyAV is the single runtime video decoder (VideoFrameReader); OpenCV demoted to image ops (full removal blocked on ultralytics until the ONNX swap).
- First fully green CI (run 28692435270, 3 OSes x test/e2e). Root-cause fixes: OpenCV 5 HoughLinesP shape (+ <5 pin), OpenCV 4.13 Windows grayscale imread (H,W,1) normalization (production court-mask loader), e2e PREFERENCES_PATH isolation (tests were writing the real user prefs file + cross-test welcome bleed), macOS cold-boot 120s deadline, golden parity 0.25s cross-platform tolerance (torch 2.7->2.12 alone shifts boundaries one 0.2s hop), Windows path-separator test fixes, and a REAL jobs_lock race caught by the new concurrency hammer test (status reads/cancel writes now hold the lock across the whole access).

ONNX pose swap (YOLO-ONNX repo + docs/onnx-pose-parity-plan.md in the perf worktree):

- Audited the exact Ultralytics surface RallyClip uses: model.predict(source, conf, imgsz=960 from the v0.3.1 manifest (NOT 640/1920), device, batch) -> boxes.xyxy/conf + keypoints.xy/conf; NMS defaults iou=0.7, max_det=300, conf-sorted; rect letterbox (960x544 for 16:9) so exports need dynamic axes.
- yolo_onnx_runner (onnxruntime+numpy+cv2, no torch/ultralytics): preprocessing bitwise-equal to Ultralytics; decoded parity on 40 frames IoU >= 0.999997, kpt err <= 3.1e-4 px; stage-3 golden clip e2e (runner injected via RuntimeDeps, zero RallyClip changes) BYTE-EQUAL segments CSV vs torch.
- PoseExtractor imgsz fallback corrected 1920 -> 960 (bb72dd6).
- In flight: 11-video 60s-sample torch-vs-onnx segment sweep (scripts/sweep_e2e_onnx.py in YOLO-ONNX).

Also: container-level CLAUDE.md written at rallyclip_container/ covering all four sibling checkouts, GH state, and house rules.
