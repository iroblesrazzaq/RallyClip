# Current State

| Field | Value |
| --- | --- |
| Repository root directory | `/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip` |
| Standard startup path | `./init.sh` |
| Standard evidence path | Feature-specific; define before marking a feature `passing` |
| Highest priority unfinished feature | `runtime-video-validation` |
| Current blocker | None for v0.1.0 packaging; GitHub Release asset upload/publish may still need website completion |

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
