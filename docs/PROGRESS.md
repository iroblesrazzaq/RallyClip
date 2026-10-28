# PROGRESS — overwrite me at every session end

_Last updated: 2026-10-27 (session: runtime v0.6.0 rebuilt on main after #64)._

## Repo state

- `main` has published app **v0.5.0**; inference artifact `artifact-rallyclip_v0.5.0`.
  `artifact-rallyclip_v0.6.0` is a prerelease.
- #64 (large-video import) is merged. Branch `rebase/runtime-v060` is #62 rebuilt
  on main: the three runtime commits cherry-picked, the training merge commits
  dropped. Conflicts: training v2 features now re-export `features.v2`; the
  preprocessor uses `preprocessing.netline_slots`, which takes main's frame-size
  check and horizon guard.
- Open: #62 (runtime v0.6.0), #63 (native macOS, stacked on #62).

## What shipped this session

1. Upload has no 2GB cap (`MAX_CONTENT_LENGTH` removed, frontend check removed).
2. Folder import: `POST /api/folder/select` (pywebview dialog in the desktop
   shell, osascript on macOS browser mode) keeps the file list server-side behind
   a single-use token; `POST /api/folder/start` pops it under `jobs_lock`.
   Preflight: every clip >= 720p, same resolution and fps, total >= seq_len/fps.
3. `src/segmentation/proxy.py`: clips -> one constant-frame-rate 720p proxy at the
   manifest fps (the `proxy` job step). Clip spans come from decoded frames;
   gaps hold the previous frame; clips without audio get silence.
4. `segment.py`: one re-encode path (stream copy deleted), VideoToolbox with
   libx264 fallback, `segment_video_sources` cuts across clip boundaries;
   times are measured from the video stream's start_time.
5. Library: folder items store the proxy as `source.mp4` plus `sources.json`;
   exports are cut from the originals (404 if the drive is gone). Exports write
   `export.partial.mp4` then rename. `POST /api/library/<id>/export` +
   `GET .../export/status` run it in the background; the button shows progress.
6. Review round 1 fixes: AudioTrack keeps a 250 ms unencoded reserve so each
   cut's audio overrun is trimmed (it drifted up to ~60 ms over 15 cuts); a
   points save bumps an export generation instead of waiting on the export lock;
   one background export encodes at a time.
7. Review round 2 fixes: exports re-resolve the effective CSV on each pass (a
   save mid-encode creates segments_edited.csv); MPEG-TS export seeks are
   checked and backed off (they landed late or at EOF and dropped cut starts);
   folder preflight skips the length check when a clip reports no duration.
8. Cancel race: a cancel that lands before the worker handle is recorded now
   terminates the worker; a cancelled job dir is removed.
9. Greptile fixes after merging main with #69: export cuts hold the previous
   picture across timestamp gaps, also when a cut starts inside one (as the
   proxy does); the proxy encodes a clip's
   audio only up to its decoded video span; a background export reports ready
   only while export.mp4 exists (a save in between cuts again); the viewer
   export button shows the progress of the match open now.
10. Review gate round 1 (Cursor 4 lenses + Greptile): cut-starts-in-gap fixed;
    tests added for the folder pipeline handoff and `.MTS` discovery.
11. Review gate round 2: a cut past the last picture stops there (no frozen
    footage), and its audio stops with it (export audio waits until a picture
    covers it, as in the proxy); a clip shorter than half a proxy frame adds 0 s
    instead of aborting the import; the export button cuts again when a save
    cleared the finished export, and downloads only a ready one. Tests added for
    a cancelled folder picker, a failed background export and the encoder
    fallback. Deferred: single-file import still uploads over HTTP (folder
    import avoids it); a path-based single-file picker is a follow-up.
12. Gates: see docs/testing.md.

## Next steps

1. #62: the user merges. Then #63 (native macOS) on top.
2. Support Intel (non-Apple-silicon) Macs.
3. Not covered by automation: clicking the folder picker in the desktop app.
4. Update the external driver scripts in `../training_data_netline`
   (`cv_netline.py`, `final_train.py`, `decode_sweep.py`): they import
   `gt_segments_from_targets` from the deleted `eval.seg_evaluator` (now
   `eval.heatmap_evaluator`) and call `build_heatmap_model` positionally
   (now `build_heatmap_model(train_cfg, input_size)`).
5. Court-model training code and driver scripts are still outside git.
6. Branches reviewed before the round cap have no round state; their next run
   counts as round 1.
7. Try a bigger YOLO pose model (yolov8s/m) after v0.6.0 ships.
