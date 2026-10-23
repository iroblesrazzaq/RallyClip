# PROGRESS — overwrite me at every session end

_Last updated: 2026-10-23 (session: review round cap)._

## Repo state

- `main` has published app **v0.5.0**; `artifact-rallyclip_v0.6.0` is a
  prerelease. Merged: review gate (#65), update cancel fix (#66), README court
  update (#67), v0.6 training code (#68). #59/#60 closed in favour of #64.
- Open: #64 (large-video import), #62 (runtime v0.6.0), #63 (native macOS).
  Active work: `chore/review-round-cap` (#69).

## What shipped this session

1. `scripts/agent/review.py` enforces the review round limit per branch: at most
   3 rounds; a round runs only if the last one had a confirmed high/P1 finding
   and HEAD moved (exit 4 = stop, write the digest). Each round needs a
   `triage.json` next to its `findings.json`.
2. Rejected findings from earlier rounds go into the Cursor prompt, and new
   findings near one are marked `repeat_of`. Skill, docs/testing.md and
   DECISIONS (2026-10-23) updated.
3. Default gate: **377 passed, 11 skipped, 49 deselected** (macOS, before
   merging #68 into this branch; see docs/testing.md for the latest).

## Next steps

1. #69 and #64: the user merges.
2. Rebase #62 onto main (it builds on #68), then #63.
3. Update the external driver scripts in `../training_data_netline`
   (`cv_netline.py`, `final_train.py`, `decode_sweep.py`): they import
   `gt_segments_from_targets` from the deleted `eval.seg_evaluator` (now
   `eval.heatmap_evaluator`) and call `build_heatmap_model` positionally
   (now `build_heatmap_model(train_cfg, input_size)`).
4. Court-model training code and driver scripts are still outside git.
5. Branches reviewed before the round cap have no round state; their next run
   counts as round 1.
6. Try a bigger YOLO pose model (yolov8s/m) after v0.6.0 ships.
