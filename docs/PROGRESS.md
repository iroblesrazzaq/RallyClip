# PROGRESS — overwrite me at every session end

_Last updated: 2026-10-22 (session: lean large-video import, replaces #59/#60)._

## Repo state

- `main` has published app **v0.5.0**; inference artifact stays `artifact-rallyclip_v0.5.0`.
- Active work: `feat/large-video-import` — a from-scratch rewrite of PRs #59/#60
  (which stay open until the user closes them).

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
9. Gates: default **350 passed, 49 deselected**; golden parity 1
   passed; `test_gui_playwright.py` 11 passed.

## Next steps

1. Review round (Claude + Cursor/Greptile CLIs), PR, user merges; user closes #59/#60.
2. Rebase the Swift branch (#63) onto this: it uses `_start_folder_analysis_job`,
   `_natural_filename_key` and `sources.json`, all kept.
3. Not covered by automation: clicking the folder picker in the desktop app.
