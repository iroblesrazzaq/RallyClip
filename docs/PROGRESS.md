# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-07 (session: artifact-registry plan)._

## Repo state

- `main` includes merged macOS release CI (#50). GitHub Releases latest is still
  **v0.3.0**. Do not tag `v0.5.0` until the artifact zip exists if we have
  already dropped ONNX from git (see plan).
- Active feature: `artifact-registry` (`docs/artifact-registry-plan.md`).

## What shipped this session

1. Plan only: ONNX off git, GitHub `artifact-rallyclip_v0.5.0` zip, CI fetch,
   DMG still embeds `DEFAULT_ARTIFACT_DIR`. Frozen Mac app does not download
   weights at launch.

## Next steps

1. Review/merge the plan PR.
2. Phase 0: pack + publish the v0.5.0 zip while weights are still in git.
3. Then CI fetch, then delete binaries from `main`.
