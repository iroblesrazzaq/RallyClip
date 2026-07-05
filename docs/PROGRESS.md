# PROGRESS — overwrite me at every session end

_Last updated: 2026-07-05 (session: storage fix + direct playback + segment edit mode + CoreML spike)._

## Repo state

- `main` is the release line and everything is merged: **#28** app-data storage,
  **#29** viewer direct playback, **#30** CoreML spike docs, **#31** segment edit
  mode. No open PRs.
- Gates: CI green 3 OS × test/e2e on every merged PR; smoke 33 passed;
  playwright module 7/7 (includes a real mouse-drag edit-mode e2e).
- User is manually QA-ing a fresh DMG built from the merged code
  (266MB .app / 116MB dmg, drag-to-Applications layout). Welcome state was
  cleared (prefs + WebKit localStorage) to re-test first-launch.

## What shipped this session

1. **App-data storage (PR #28)**: frozen builds keep user data in the OS
   per-user app-data dir (`~/Library/Application Support/RallyClip` on macOS,
   `%APPDATA%` on Windows, XDG on Linux) with a one-time `shutil.move`
   migration from the v0.1.0 `~/RallyClip` location. Windows branch selected
   via `sys.platform` (pathlib breaks if you monkeypatch `os.name`).
2. **Viewer direct playback (PR #29)**: new `/api/library/<id>/source` route
   (Range/206) streams the original file; WKWebView decodes H.264 natively.
   Fixes the stuck-at-8s bug (WebM preview windows transcode at ~2.5× real
   time and starved playback). Window pipeline remains as automatic fallback.
   Review hardening: probe callbacks take a `directProbeSeq` ticket so stale
   canplay/error/timeout handlers are inert after item switches.
3. **Segment edit mode (PR #31)**: non-destructive point editing in the viewer
   — drag trim handles (Photos-style, live scrub), add/delete point, Reset to
   original. Edits live in `segments_edited.csv`; `segments.csv` is never
   written; the edited copy wins for playback manifest, `/segments`, CSV
   download, and lazy export (export.mp4 invalidated on edit/reset). Routes:
   `PUT /api/library/<id>/segments`, `DELETE /api/library/<id>/segments/edits`.
   Autosaves are serialized + generation-guarded so a stale PUT can't
   resurrect a reset; reset refuses to delete the edited copy if the original
   CSV is missing.
4. **CoreML spike (PR #30, docs only)**: static 544×960 re-export of the pose
   weights + CoreMLExecutionProvider (MLProgram, ALL) = 120.4 fps vs 15.6 fps
   shipping CPU path (~7.7×), confident-det divergence 1.2e-4. Dynamic-axes
   export blocks the ANE (that's why naive CoreML EP is only 1.2×). LSTM is
   slower on CoreML — stays CPU. Verdict: no MLX rewrite. Scripts + results:
   `docs/perf/coreml-spike/` (reproducible via `RALLYCLIP_SPIKE_SOURCE`).

## Next steps (in order)

1. **CoreML productionization** (feature `mlx-apple-silicon-backend`,
   in_progress): static 544×960 export into the model bundle; opt-in provider
   flag (CPU stays the byte-parity default); letterbox-pad fallback for
   non-16:9; golden-verify static export vs bundled dynamic ONNX.
2. After the user's manual QA: tag a release from main so release.yml ships
   the storage-fix + direct-playback + edit-mode build.
3. Distribution: Developer ID signing + notarization (user has an Apple dev
   account — feature `macos-app-distribution`), then iOS/subscription track.
4. Cleanup candidates: WebM preview-window pipeline + Qt-era proxy endpoints
   once direct playback is QA-confirmed; welcome-screen quirk (frontend trusts
   stale WebKit localStorage over server `welcome_seen: false`, script.js:311).
5. Backlog: training-quality-harness, model retrain on all data, perf
   streaming loop (docs/perf/).
