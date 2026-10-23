# PROGRESS — overwrite me at every session end

_Last updated: 2026-10-23 (session: review round cap)._

## Repo state

- `main` has published app **v0.5.0**; `artifact-rallyclip_v0.6.0` is a
  prerelease. The review gate (#65) and the update cancel fix (#66) are merged.
- Open: #64 (large-video import), #68 (v0.6 training). Active work:
  `chore/review-round-cap`.

## What shipped this session

1. `scripts/agent/review.py` enforces the review round limit per branch: at most
   3 rounds; a round runs only if the last one had a confirmed high/P1 finding
   and HEAD moved (exit 4 = stop, write the digest). Each round needs a
   `triage.json` next to its `findings.json`.
2. Rejected findings from earlier rounds go into the Cursor prompt, and new
   findings near one are marked `repeat_of`. Skill, docs/testing.md and
   DECISIONS (2026-10-23) updated.
3. Default gate: **376 passed, 11 skipped, 49 deselected** (macOS).

## Next steps

1. PR open with a Review digest; the user merges.
2. Branches reviewed before this change have no round state; their next run
   counts as round 1.
