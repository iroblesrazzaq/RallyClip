# RallyClip Model Artifacts

Git tracks **manifests and checksums**, not ONNX. Weights live on GitHub Release
[`artifact-rallyclip_v0.8.0`](https://github.com/iroblesrazzaq/RallyClip/releases/tag/artifact-rallyclip_v0.8.0).

After clone:

```bash
python scripts/fetch_artifact.py
# or: PYTHONPATH=src python -m runtime.artifact fetch
```

That unpacks into `models/rallyclip_v0.8.0/` (gitignored binaries). Idempotent:
skips the network when SHA-256 already matches `SHA256SUMS`. Hash mismatch fails
closed.

## Layout

- `models/rallyclip_v0.8.0/` — **current default** (`DEFAULT_ARTIFACT_DIR`).
  Same TCN heatmap pipeline and 290-dim `v2_nearfar` features as v0.6.x, with a
  new classifier and scaler: retrained on the cleaned labels (300-epoch run,
  epoch 35). Decode merges two points less than 5 s apart (`merge_gap_sec`).
  Git: `manifest.json`, `SHA256SUMS`. After fetch: `model.onnx`, `scaler.json`,
  `court_lines.onnx` and three pose ONNX files: dynamic (CPU), static 544x960
  (CoreML), and the same static model at batch 2 (`-static-b2.onnx`), which runs
  each frame and its far-court crop in one CoreML call. The pose and court files
  are byte-identical to v0.6.1.
- `models/rallyclip_v0.6.1/` — the previous bundle: the v0.6.0 classifier plus
  the batch-2 pose file.
- `models/rallyclip_v0.6.0/` — an older bundle (no batch-2 pose file).
- `models/rallyclip_v0.5.0/manifest.json` — the previous default: v1 (362-dim)
  features, full-frame slots.
- `models/rallyclip_v0.4.0/manifest.json` — classic bidirectional LSTM +
  hysteresis. Kept so pipeline-resolution tests do not need the old LSTM ONNX.
- `models/rallyclip_v0.3.1/manifest.json` / `models/rallyclip_v0.1.0_legacy/manifest.json`
  — historical contracts only.

Pose ONNX lives **once**, only in the current (v0.8.0) zip.

The Mac DMG embeds `models/rallyclip_v0.8.0/` at build time
(`RallyClip.spec`). The packaged app does **not** download weights at launch.

To pack a new zip from a complete local folder:

```bash
scripts/release/pack_artifact.sh dist/
```
