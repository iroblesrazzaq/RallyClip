# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-26 (session: fix Greptile blockers on folder export)._

## Repo state

- Topic branch `fix/pr59-export-guards`, based on PR #59 (`fa2d55b`). Do not
  commit to `main`.
- App version remains **0.5.1** and the inference artifact remains
  `models/rallyclip_v0.5.0`.

## What changed this session

1. Added **Choose a folder of consecutive clips** to the new-match screen. The
   native macOS picker selects a directory once; MP4s are naturally sorted so
   names such as `clip_2.mp4` precede `clip_10.mp4`.
2. Folder clips are referenced in place instead of uploaded or copied. RallyClip
   creates one continuous 1280x720-or-smaller, 30 fps H.264/AAC proxy, then runs
   the existing 5 fps model pipeline against that proxy.
3. Saved folder matches retain `sources.json` with the original paths, durations,
   and exact continuous-timeline offsets. The proxy is the review source, so the
   existing point editor works across the whole match and file boundaries.
4. Final export maps global edited intervals back to local source timestamps and
   encodes directly from the original-resolution chunks into one output. A point
   beginning in one MP4 and ending in the next is preserved, with audio synced.
5. Folder progress includes a visible proxy-preparation stage. Inputs must share
   resolution/frame rate, and originals must remain available until export.
6. Fixed the first real folder-start crash: the lazy validation runtime now
   exposes `probe_video` and `MIN_HEIGHT`, which folder preflight requires.
7. Nominal frame-rate differences between camera chunks are accepted. Export
   re-times frames against source timestamps, so 59.94/60 and other mixed-rate
   chunks retain their real duration instead of playing fast or slow. Resolution
   mismatches still fail with both exact dimensions in the message.
8. Replaced the slow 4K software-decode proxy path on macOS with AVFoundation's
   native `avconvert` 720p30 preset, then packet-remuxes its per-file outputs into
   one timeline without a second encode. On the selected DJI footage, 60 source
   seconds took **9.06s** (6.6x realtime) with only 1.73s user CPU; joining two
   proxy minutes took **0.30s**. Cancellation terminates `avconvert` and cleans
   partial parts; PyAV remains the portable fallback.
9. Verification: targeted proxy/segmentation/GUI/API suite **84 passed**; full
   fast gate **338 passed, 5
   skipped, 27 deselected**; golden CLI parity **1 passed**; compile and JavaScript
   syntax gates passed. The updated screen was
   manually verified in Chrome against the running backend.
10. Lazy saved-match exports now report frame-based percentage progress through
   the status API, and the export button shows `Exporting N%` instead of only
   the preparing spinner. Focused API/segmentation verification: **20 passed**.
11. Saved-match export now tries keyframe-aligned packet remuxing first, with up
   to one second of context around each point. Sources without suitable end
   keyframes safely fall back to the existing exact re-encode. Focused
   verification: **80 passed**.

12. Greptile's three findings on PR #59 are fixed on `fix/pr59-export-guards`:
   overlapping keyframe pads and keyframe-overshoot ranges refuse stream copy
   and fall back to frame-accurate encode; mixed-audio folders fail in
   preflight; a new folder selection replaces the previous one and tokens older
   than 30 minutes are rejected.
13. Verification on this host: focused regression tests passed; default gate
   **338 passed, 11 skipped, 49 deselected** (the extra skips are missing
   `models/rallyclip_v0.5.0` weights, not these guards); compileall exit 0.

## Next steps

1. Iterate `greptile review --branch main --agent` to 5/5, then open the PR.
   Do not merge; the user merges.
2. Run one complete real-world import against a consecutive 4K camera folder,
   confirming source ordering, proxy wall time, boundary alignment, final
   dimensions, audio, and export quality.
