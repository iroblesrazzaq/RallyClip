# PROGRESS — overwrite me at every session end

_Last updated: 2026-11-01 (session: v0.8.0 model retrain and release)._

## Repo state

- `main` has published app **v0.8.0** (Latest; signed, notarized arm64 DMG from
  the native Release workflow). Inference artifact `artifact-rallyclip_v0.8.0`
  (prerelease) is `DEFAULT_ARTIFACT_DIR`; the DMG embeds it.
- No open PRs. #74 (native UI zoom, library rename/sort, accent), #75 (batch-2
  CoreML pose, artifact v0.6.1) and #77 (model v0.8.0) are merged.
- #76 (six-bin decent limits per boundary: start 2 s / 0.5 s, end 1 s / 3 s) was
  closed without a merge at the user's request. `main` still scores with the v1
  limits (1.5 s / 0.5 s both ends); the new limits are on branch
  `feat/clip-metric`. The v0.8.0 CV numbers used the new limits.

## Model v0.8.0

- Classifier and scaler retrained on cleaned labels (flagged windows removed),
  300-epoch cosine run, checkpoint at epoch 35 (best pooled CV F1). Pose, court
  model and features are the same as v0.6.1.
- Decode merges points less than 5 s apart (`merge_gap_sec` 5.0, chosen on CV).
- Held-out test (5 videos, scored once, merge gap 0): six-bin F1 0.432 vs
  v0.6.0 0.389. Do not score the test again for this model.
- Training runs and the merge-gap sweep are in
  `../training_data_netline/cv_nested/long300/`. The 300/700-epoch runs showed
  no double descent. See DECISIONS.md 2026-10-30.

## Next steps

1. README demo GIF: pose on black, point/start/end probabilities, the 3 heads.
2. Update the external driver scripts in `../training_data_netline`
   (`cv_netline.py`, `final_train.py`, `decode_sweep.py`): they import
   `gt_segments_from_targets` from the deleted `eval.seg_evaluator` (now
   `eval.heatmap_evaluator`) and call `build_heatmap_model` positionally
   (now `build_heatmap_model(train_cfg, input_size)`).
3. Court-model training code and driver scripts are still outside git.
4. Not covered by automation: clicking the folder picker in the desktop app.
5. Optional: find which change in #70 caused the small CV score drift (candidates:
   the court_detector_impl.py rewrite, the court detection time change).
6. Deferred by the user: Intel (non-Apple-silicon) Mac support; a bigger YOLO
   pose model (yolov8s/m).
