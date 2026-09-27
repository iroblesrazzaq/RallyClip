# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-26 (session: dismiss folder tokens and drop avconvert)._

## Repo state

- Topic branch `fix/pr59-export-guards`, based on PR #59 (`fa2d55b`). Do not
  commit to `main`. The user merges.
- App version remains **0.5.1** and the inference artifact remains
  `models/rallyclip_v0.5.0`.

## What changed this session

1. Stream copy refuses overlapping pads and any packet that would display
   outside the selected cut, then falls back to frame-accurate encode.
2. Folder preflight rejects mixed audio before analysis. Tokens expire after
   30 minutes, a ninth pending selection is refused, and the UI posts
   `/api/folder/dismiss` when the user removes, replaces, or re-chooses a folder.
3. Analysis proxies are created only with PyAV (VideoToolbox, then libx264).
   `/usr/bin/avconvert` is no longer on the runtime path.
4. Cancel still signals the worker process group. Proxy silence stays chunked
   to one AAC frame.

## Next steps

1. Run `greptile review --branch main --agent` until Confidence is 5/5, then
   open the PR. Do not merge.
2. Run one complete real-world import against a consecutive 4K camera folder,
   confirming source ordering, proxy wall time, boundary alignment, final
   dimensions, audio, and export quality.
