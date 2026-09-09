# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-09 (session: assisted DMG update, Greptile 3/5 fixes)._

## Repo state

- `main` has published app **v0.5.0**. Inference artifact stays
  `artifact-rallyclip_v0.5.0`.
- Active work: assisted update (`cursor/assisted-dmg-update-5f28`) as app **0.5.1**.
- PR: https://github.com/iroblesrazzaq/RallyClip/pull/58

## What shipped this session

1. Latest app release is the newest published `v*` tag from `/releases`, not
   GitHub `/releases/latest` (skips `artifact-rallyclip_*`).
2. Frozen app: `POST /api/update/download` stages the DMG, verifies SHA-256,
   then replaces `~/Downloads`. A failed retry keeps the previous installer.
3. Update button becomes Cancel during the transfer (AbortController).
4. Source/GUI: `POST /api/update/open` opens that release URL.
5. pyproject **0.5.1**; ONNX dir still `models/rallyclip_v0.5.0`.
6. Default gate: **315 passed, 6 skipped, 27 deselected**.

## Next steps

1. Greptile 5/5, then user merges.
2. Tag `v0.5.1` (do not retag `v0.5.0`; do not re-upload the ONNX zip).
