# RallyClip Model Artifacts

Git tracks **manifests and checksums**, not ONNX. Weights live on GitHub Release
[`artifact-rallyclip_v0.6.0`](https://github.com/iroblesrazzaq/RallyClip/releases/tag/artifact-rallyclip_v0.6.0).

After clone:

```bash
python scripts/fetch_artifact.py
# or: PYTHONPATH=src python -m runtime.artifact fetch
```

That unpacks into `models/rallyclip_v0.6.0/` (gitignored binaries). Idempotent:
skips the network when SHA-256 already matches `SHA256SUMS`. Hash mismatch fails
closed.

## Layout

- `models/rallyclip_v0.6.0/` — **current default** (`DEFAULT_ARTIFACT_DIR`).
  Same TCN heatmap pipeline on 290-dim `v2_nearfar` features with net-line
  player slots: a court line model (`court_lines.onnx`) gives the net line and
  court homography, and a second pose pass on a fixed far-court crop feeds the
  far slot. Git: `manifest.json`, `SHA256SUMS`. After fetch: `model.onnx`,
  `scaler.json`, `court_lines.onnx`, both pose ONNX files.
- `models/rallyclip_v0.5.0/manifest.json` — the previous default: v1 (362-dim)
  features, full-frame slots.
- `models/rallyclip_v0.4.0/manifest.json` — classic bidirectional LSTM +
  hysteresis. Kept so pipeline-resolution tests do not need the old LSTM ONNX.
- `models/rallyclip_v0.3.1/manifest.json` / `models/rallyclip_v0.1.0_legacy/manifest.json`
  — historical contracts only.

Pose ONNX lives **once**, only in the current (v0.6.0) zip.

The Mac DMG embeds `models/rallyclip_v0.6.0/` at build time
(`RallyClip.spec`). The packaged app does **not** download weights at launch.

To pack a new zip from a complete local folder:

```bash
scripts/release/pack_artifact.sh dist/
```
