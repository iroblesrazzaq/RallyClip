# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-07 (session: make scheduler e2e skip checks deterministic)._

## Repo state

- Topic branch `cursor/macos-release-cicd-5f28`, PR #50 against `main`.
- Default artifact is `models/rallyclip_v0.5.0/`. GitHub Actions Developer ID /
  notary secrets are set. Do not tag `v0.5.0` until #50 is on `main`.
- GitHub Releases latest is still **v0.3.0**.

## What shipped this session

1. Scheduler e2e skip checks no longer read live `currentTime` / `paused`.
   Keyboard and skip-button blocks stub `getViewerSourceTime` at 30s and
   `paused` false, so seeks are 25/35 with autoplay true on every OS.
   Ubuntu was clamping `currentTime=10` inside the 8s preview window (23/33);
   macOS/Windows had a paused element (`autoplay: False`).

## Next steps

1. Wait for CI e2e green on this push, then merge #50 (user merges).
2. After merge: tag `v0.5.0` (must match `pyproject.toml`).
