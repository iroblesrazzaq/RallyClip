# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-07 (session: fix scheduler e2e keyboard-skip expects)._

## Repo state

- Topic branch `cursor/macos-release-cicd-5f28`, PR #50 against `main`.
- Default artifact is `models/rallyclip_v0.5.0/`. GitHub Actions Developer ID /
  notary secrets are set. Do not tag `v0.5.0` until #50 is on `main`.
- GitHub Releases latest is still **v0.3.0**.

## What shipped this session

1. `test_ui_viewer_uses_source_timeline_scheduler` keyboard-skip expects were
   23/33s. Source time is window start 20 + `currentTime` 10 = 30, skip is 5s,
   so seeks are 25/35 — same as the skip-button block already asserted. CI on
   all three OS failed that one assert. Expectation updated; product code unchanged.

## Next steps

1. Wait for CI e2e green on this push, then merge #50 (user merges).
2. Dry-run: Actions → Release → Run workflow on this branch (not `main`) if a
   notarized DMG is needed before the tag.
3. After merge: tag `v0.5.0` (must match `pyproject.toml`).
