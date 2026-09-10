# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-09 (session: assisted DMG update, Greptile cancel + checksum)._

## Repo state

- `main` has published app **v0.5.0**. Inference artifact stays
  `artifact-rallyclip_v0.5.0`.
- Active work: assisted update (`cursor/assisted-dmg-update-5f28`) as app **0.5.1**.
- PR: https://github.com/iroblesrazzaq/RallyClip/pull/58

## What shipped this session

1. Latest app release is the newest published `v*` tag from `/releases`.
2. Frozen app downloads to a unique staging file, verifies SHA-256 (sidecar
   must name the DMG), then replaces `~/Downloads`.
3. `POST /api/update/cancel` stops the server-side transfer; the button
   becomes Cancel.
4. Source/GUI opens that release URL. App version **0.5.1**; ONNX stays
   `models/rallyclip_v0.5.0`.
5. Default gate: **321 passed, 6 skipped, 27 deselected**.

## Next steps

1. Greptile 5/5, then user merges.
2. Tag `v0.5.1` (do not retag `v0.5.0`; do not re-upload the ONNX zip).
