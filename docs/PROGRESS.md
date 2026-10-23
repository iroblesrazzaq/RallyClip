# PROGRESS — overwrite me at every session end

_Last updated: 2026-10-22 (session: clean v0.6 training PR replacing #61)._

## State

- `main` has published app **v0.5.0**; `artifact-rallyclip_v0.6.0` is a
  prerelease. The review gate (PR #65) and the README court update (PR #67)
  are merged.
- Active branch: `training/v060-tcn-netline` (worktree `training-v060-wt`),
  PR against main that supersedes #61. See `docs/DECISIONS.md` 2026-10-22.

## What the branch does

- v0.6 training code: TCN heatmap head (`train/heatmap_loop.py`,
  `train/heatmap_loss.py`), net-line slots + v2_nearfar features,
  normalize, manifest-backed pose extract, court anchors/clean frame under
  `src/training/courts/`.
- Dead arms removed (e2e_seg, LSTM/GRU heatmap, time loss, LR finder,
  24 configs). Shipped recipe: `configs/train/v060_netline.yaml`.
- Training eval now decodes with the runtime decode, so numbers may differ
  from older runs.
- Flip leakage fixed in `dataset/builder.py`.
- Torch-only tests skip when torch is missing.

## Gates (2026-10-22)

Default suite 471 passed / 17 skipped; torch blocked 427 / 27; golden CLI
parity 1 passed; court flights 24 passed; compileall clean; remaining ruff
findings on changed files are pre-existing on main.

## Next

- Update the external driver scripts in `../training_data_netline`
  (`cv_netline.py`, `final_train.py`, `decode_sweep.py`): they import
  `gt_segments_from_targets` from the deleted `eval.seg_evaluator` (now
  `eval.heatmap_evaluator`) and call `build_heatmap_model` positionally
  (now `build_heatmap_model(train_cfg, input_size)`).
- Merge #64 and close #59/#60; rebase #62/#63.
- Court-model training code and driver scripts are still outside git.
- Try a bigger YOLO pose model (yolov8s/m) after v0.6.0 ships.
