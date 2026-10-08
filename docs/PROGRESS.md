# PROGRESS — overwrite me at every session end

_Last updated: 2026-10-27 (session: native macOS app rebuilt on main after #64)._

## Repo state

- `main` has published app **v0.5.0**; inference artifact `artifact-rallyclip_v0.5.0`.
  `artifact-rallyclip_v0.6.0` is a prerelease.
- Branch `rebase/runtime-v060` is #62 (runtime v0.6.0) rebuilt on main after #64.
- This branch, `rebase/native-macos`, is #63 rebuilt on it: only the native app
  commit (0fbeb66) is cherry-picked. The old `fix/pr59-export-guards` merge is
  dropped; #64 replaced that code. The native helper uses #64's
  `_start_folder_analysis_job`, `_natural_filename_key` and `sources.json`
  (`path`, `start_s`, `duration_s`).
- Do not merge; the user merges. #62 lands first.

## Implementation

- Native SwiftUI sidebar/player/editor with original RallyClip logo/icon,
  system font, light/dark appearance, green accent, factual UI text.
- Single continuous AVFoundation composition across points/camera files;
  full-match mode, fullscreen/PiP, seek, point selection, timeline zoom.
- Save/discard/reset boundary edits; add/delete intervals; guard navigation,
  window close, quitting, active exports and analysis.
- Native all-point/highlight/separate-video and CSV export, staging,
  cancellation, progress. Audio mix renders silence at delayed-audio gaps.
- Authenticated loopback helper bundles the proven v0.6 torch-free engine and
  learned court detector. Native imports reference originals and analyze a
  720p proxy. Existing Application Support saved matches are reused.
- Local build/run scripts, Swift tests/media probe, packaged import/edit/media
  smoke. Release workflow defaults to native; retains legacy manual option;
  existing Apple secrets/scripts sign, notarize, staple and upload the DMG.

## Next steps

1. #62, then #63: the user merges.
2. Native release: run the Release workflow on this branch; verify the signed,
   notarized DMG; install it on this Mac and test it.
3. Packaged smoke and window-close guard QA on the final UI build.
4. Support Intel (non-Apple-silicon) Macs: today the native app, the frozen
   backend and the DMG are arm64 only.
5. Not covered by automation: clicking the folder picker in the desktop app.
6. Update the external driver scripts in `../training_data_netline`
   (`cv_netline.py`, `final_train.py`, `decode_sweep.py`): they import
   `gt_segments_from_targets` from the deleted `eval.seg_evaluator` (now
   `eval.heatmap_evaluator`) and call `build_heatmap_model` positionally
   (now `build_heatmap_model(train_cfg, input_size)`).
7. Court-model training code and driver scripts are still outside git.
8. Try a bigger YOLO pose model (yolov8s/m) after v0.6.0 ships.
