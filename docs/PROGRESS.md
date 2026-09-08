# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-08 (session: release GUI probe covers WKWebView)._

## Repo state

- `main` includes artifact-registry (#53). GitHub Releases latest app channel
  is still **v0.3.0**. Tag `v0.5.0` still points at the July docs commit —
  do not move it until this GUI-probe fix is on `main`.
- Artifact zip is **published** (not draft):
  https://github.com/iroblesrazzaq/RallyClip/releases/tag/artifact-rallyclip_v0.5.0
- Active PR: https://github.com/iroblesrazzaq/RallyClip/pull/54
  (`cursor/fix-gui-probe-5f28`). Release CI must boot the real packaged GUI
  (Flask first, then pywebview); `--backend-only` is not the probe.

## What shipped this session

1. GUI path starts Flask and waits for `/api/health` before importing
   pywebview, so a slow WKWebView cannot starve the backend (root cause of
   run 34176081009: `RallyClip backend failed to start.`).
2. Release probe launches `"$BIN"` (no `--backend-only`) and requires
   `window.events.loaded` to print `RallyClip desktop shell ready` (WKWebView
   loaded the Flask page), then `kill -0` plus defaults ONNX name.
3. `--backend-only` remains a debug flag. Dispatch tests stub Flask for the
   GUI path, fake webview for the ready-line contract, and import `gui.app`
   in a fresh interpreter with webview blocked.
4. Default gate: **302 passed, 6 skipped, 27 deselected, ~20s**.

## Next steps

1. Greptile 5/5 on PR #54, then user merges.
2. Move tag `v0.5.0` onto the new `main` and force-push the tag to re-run
   Release (do not re-upload `artifact-rallyclip_v0.5.0`).
