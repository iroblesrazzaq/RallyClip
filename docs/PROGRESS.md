# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-26 (session: dismiss folder tokens and drop avconvert)._

## Repo state

- Topic branch `fix/pr59-export-guards`, based on PR #59 (`fa2d55b`). Do not
  commit to `main`. The user merges.
- App version remains **0.5.1** and the inference artifact remains
  `models/rallyclip_v0.5.0`.

## What changed this session

1. Stream copy remuxes a video-only cut only when a keyframe already sits on
   the boundary, in decode order, and otherwise falls back to encode.
2. Folder preflight rejects mixed audio before analysis. Tokens expire after
   30 minutes, a ninth pending selection is refused, and the UI posts
   `/api/folder/dismiss` when the user removes, replaces, or re-chooses a folder.
3. Analysis proxies are created only with PyAV. Each clip's audio stays on
   that clip's timeline, including a late start. Multiple video or audio
   tracks are rejected.
4. Cancel signals the worker process group, deletes a partial
   `analysis_proxy.mp4`, and does not replace an installer if hashing is
   cancelled. A second export start does not spawn another thread.

## Next steps

1. Run `greptile review --branch main --agent` until Confidence is 5/5, then
   open the PR. Do not merge.
2. Run one complete real-world import against a consecutive 4K camera folder,
   confirming source ordering, proxy wall time, boundary alignment, final
   dimensions, audio, and export quality.
