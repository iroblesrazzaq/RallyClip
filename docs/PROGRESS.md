# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-09 (session: assisted DMG update)._

## Repo state

- `main` has published app **v0.5.0**. Inference artifact stays
  `artifact-rallyclip_v0.5.0`.
- Active work: assisted update (`cursor/assisted-dmg-update-5f28`) as app **0.5.1**.

## What shipped this session

1. Latest release parse picks `RallyClip-*-macOS-arm64.dmg` + `.sha256`, ignoring
   `artifact-rallyclip_*`.
2. Frozen app: `POST /api/update/download` saves the DMG to `~/Downloads`,
   verifies SHA-256, opens it. Does not replace `/Applications`.
3. Source/GUI: `POST /api/update/open` opens that release URL, not the list.
4. pyproject **0.5.1**; ONNX dir still `models/rallyclip_v0.5.0`.

## Next steps

1. Greptile 5/5, then user merges.
2. Tag `v0.5.1` (do not retag `v0.5.0`; do not re-upload the ONNX zip).
