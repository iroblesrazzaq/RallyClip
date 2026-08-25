# PROGRESS — overwrite me at every session end

_Last updated: 2026-08-24 (session: ship TCN heatmap as default v0.5.0)._

## Repo state

- Worktree `hm-wt` on `feat/v0.5.0-tcn` (heatmap runtime + TCN export + default swap).
  Not merged to `main`. Heatmap runtime also sits on open PR #48
  (`feat/heatmap-runtime-v050`); original PR #45 is stale.
- Shipped default is now `models/rallyclip_v0.5.0/` (dilated TCN, 3 named logit
  heads, `frame_startend_heatmap` hybrid decode). Classic LSTM kept at
  `models/rallyclip_v0.4.0/`.
- GitHub Releases latest is still **v0.3.0**; tagging `v0.5.0` is the Mac DMG
  path once this branch is proven. Do not pretend v0.4.0 was a GitHub product
  release.
- Gates this session: default unit **267 passed, 48 deselected**; compile clean;
  golden CLI regenerated and passing; CLI smoke v0.4.0 vs v0.5.0 on the fixture
  clip (classic one 3.8–24.0s segment vs TCN two points 3.753–11.964 and
  13.040–23.449). L1 GUI e2e (default job) + Playwright new-match
  (upload → progress → library) passed.

## What shipped this session

1. **Heatmap runtime** (already on this branch): `HeatmapHybridModel`,
   `frame_startend_heatmap`, 3-head ONNX track ordering, hybrid decode knobs
   from the manifest.
2. **Champion TCN export**: `scripts/export_heatmap_model.py` +
   `src/training/models/heatmap_tcn.py`. Checkpoint
   `training_data/runs/20260724_tcn64_cos1e4/checkpoints/best.pth` →
   `models/rallyclip_v0.5.0/model.onnx` (opset 17, named
   `pointness_logit` / `start_heatmap_logit` / `end_heatmap_logit`). Torch vs
   ORT max abs 1.88e-06.
3. **Default swap**: CLI/GUI/defaults, `RallyClip.spec`, `release.yml`,
   pyproject **0.5.0**, golden CSV, packaging paths off the lagging v0.3.1
   bundle. GUI `_normalize_config` does not sticky-override `pipeline_id`
   from defaults (artifact manifest wins).

## Next steps (in order)

1. Open/land the v0.5.0 PR (do not commit to `main`).
2. Tag `v0.5.0` to fire the existing Mac DMG / notarization workflow.
3. Out of scope here: `rallyclip serve`, Win/Linux freeze, more TCN search,
   pair-DP decode.
