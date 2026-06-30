# Session Handoff

## Current State

`feat/first-release` is the unified release branch for the first macOS product release. The branch and tag `v0.1.0` have been pushed to `origin`, but the first notarized DMG must not be published.

Current blocker status: the previously notarized DMG opened to a blank gray Qt window on install. The backend was healthy, but QtWebEngine JavaScript did not start. Diagnostic launch showed `V8 process OOM (Failed to reserve virtual memory for CodeRange)`, caused by missing Chromium/V8 JIT entitlements under hardened runtime.

Superseded artifact, do not upload:

```text
/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf/dist/RallyClip-0.1.0-macOS-arm64.dmg
```

Fixed local artifact, pending Apple notarization/stapling:

```text
/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf/dist/RallyClip-0.1.0-macOS-arm64-fixed.dmg
```

Fixed DMG SHA256:

```text
80d72a829a2a9b3008106743eb4179c4dc80b015256e519ad7c24663f5780ec7
```

Compatibility: Apple Silicon only. The app is `Mach-O thin (arm64)`, so Intel Macs are not supported by this build.

## What Shipped

- Packaged PySide6 desktop app with embedded local Flask backend.
- Saved-match library and replay-first startup path.
- Native QtMultimedia replay viewer for packaged Mac playback.
- Browser/WebM replay remains a dev fallback only.
- Source-time replay timeline with point skipping, manual gap playback, hover controls, fullscreen, keyboard shortcuts, CSV/export actions, and point markers.
- First-run welcome persistence via backend preferences.
- Manual update check: latest GitHub Release check plus a button that opens GitHub Releases when a newer release exists.
- Replay/library startup avoids importing Torch, Ultralytics, PyAV, and Numpy.
- New Match warms analysis only from the upload flow.
- Analysis runs in a child worker process so inference memory exits after completion/cancel/error.
- Production Mac replay uses native source playback. Automatic proxy generation/playback and HLS playback are not part of v0.1.0.

## Release Evidence

- Source branch pushed: `feat/first-release`.
- Release tag pushed: `v0.1.0`.
- Current release commit/tag: `8848492 add manual update check`.
- Developer ID app signature verified with hardened runtime:
  - `Authority=Developer ID Application: Ismael Robles-Razzaq (L9W8X6N9B9)`
  - `TeamIdentifier=L9W8X6N9B9`
  - `flags=0x10000(runtime)`
- Original notary submission accepted, but superseded because the app blanked at runtime:
  - `RallyClip-0.1.0-macOS-arm64-resubmit.dmg`
  - submission id `0fb928dc-2ce5-4dcf-b6c7-d385dbad3691`
- Stapler output passed: `The staple and validate action worked!`
- `xcrun stapler validate` passed.
- Mounted-DMG Gatekeeper assessment passed:
  - `/Volumes/RallyClip 0.1.0/RallyClip.app: accepted`
  - `source=Notarized Developer ID`
- Later user-installed DMG reproduced a blank window. This invalidates the old artifact despite notarization passing.
- User manually verified source-only playback completed successfully after proxy removal/disable.
- User manually verified app launch, welcome behavior, inference, and replay were acceptable for first release.

Blank-screen fix evidence:

- Added tracked entitlement source: `packaging/macos/RallyClip.entitlements`.
- Added tracked signing helper: `scripts/release/sign_macos_app.sh`.
- `RallyClip.spec` now references the entitlement plist.
- `dist/RallyClip.app` was signed with the helper script.
- `codesign --verify --deep --strict --verbose=2 dist/RallyClip.app` passed.
- Entitlements on both `Contents/MacOS/RallyClip` and `QtWebEngineProcess.app` include:
  - `com.apple.security.cs.allow-jit`
  - `com.apple.security.cs.allow-unsigned-executable-memory`
- Diagnostic relaunch no longer logs the V8 CodeRange OOM.
- User confirmed the entitlement-signed `dist/RallyClip.app` renders.
- Fixed DMG mounted app verifies locally and carries the same entitlements.
- Fixed DMG currently reports `source=Unnotarized Developer ID`, which is expected until it is notarized and stapled.

## GitHub Release State

The CLI upload attempt created a draft GitHub Release but did not upload the DMG asset before interruption. Do not publish until the fixed DMG is notarized and stapled. Then use the GitHub website:

1. Delete the weird `untagged-...` draft if it is awkward, or edit it to target tag `v0.1.0`.
2. Upload the fixed, notarized, stapled DMG. Rename it to `RallyClip-0.1.0-macOS-arm64.dmg` only after notarization succeeds, or upload the fixed suffix explicitly.
3. Publish as a normal release, not a pre-release.
4. Release title: `RallyClip v0.1.0`.
5. Clearly state Apple Silicon only.

Do not upload a zip expecting GitHub to convert it. GitHub Releases hosts exactly the uploaded file. For macOS distribution, upload the notarized DMG.

## Release Notes Summary

Use the release notes already drafted in chat:

- First macOS release of RallyClip.
- Compatibility: macOS on Apple Silicon only, M1 or newer.
- Install: download DMG, open DMG, drag `RallyClip.app` to Applications.
- Included: native macOS app, saved-match library, New Match flow, native replay viewer, point-skipping playback, CSV/export, manual update check.
- Known limitations: large bundle, Apple Silicon only, manual updates, Torch/Ultralytics still bundled in isolated analysis worker.
- Deferred: YOLO ONNX runtime replacement, bundle slimming, universal/Intel build, CI/CD release pipeline, auto-update installer, CLI/app/API runtime restructuring, iOS.
- Include the fixed SHA256 above.

## Branch Cleanup

Remote branch `origin/perf/streaming-pipeline` is stale and safe to delete. It has no commits not already contained in `feat/first-release`.

Evidence:

```text
git rev-list --left-right --count feat/first-release...origin/perf/streaming-pipeline
24 0
```

Delete when ready:

```bash
git push origin --delete perf/streaming-pipeline
```

## Still Broken Or Deferred

- `runtime-video-validation` remains the highest-priority unfinished harness feature.
- First release still bundles a large Python/ML dependency stack.
- PyInstaller likely over-collects packages and duplicates some package/native-library content.
- Ultralytics/PyTorch replacement with sibling YOLO ONNX runner is deferred.
- Shared runtime/API restructuring is deferred.
- GitHub Actions signing/notarization/release automation is deferred.
- Universal/Intel macOS build is deferred.
- Manual update check is v1; automatic update install is deferred.

## Next Best Actions

1. Submit `dist/RallyClip-0.1.0-macOS-arm64-fixed.dmg` to Apple notarization.
2. Staple the accepted fixed DMG.
3. Mount the stapled fixed DMG, run `spctl --assess --type execute --verbose=4` on the mounted app, and launch it from the mounted volume.
4. Upload only the fixed notarized/stapled DMG to GitHub Releases.
5. Optionally delete stale `origin/perf/streaming-pipeline`.
6. Start `runtime-video-validation`: reject missing, unreadable, or unsupported videos before expensive pose extraction in CLI and GUI.
7. After v0.1.0 is public, plan v0.2 around ONNX runtime replacement and bundle slimming.
