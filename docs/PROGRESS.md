# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-07 (session: rebase macOS CI/CD onto v0.5.0 main)._

## Repo state

- Topic branch `cursor/macos-release-cicd-5f28` rebased onto `main` (`012bc7e`,
  PR #49 TCN heatmap default). Default artifact is `models/rallyclip_v0.5.0/`.
- GitHub Actions secrets for Developer ID / notary are **not set yet**. Tag
  pushes fail closed until they exist. `workflow_dispatch` still builds an
  unsigned DMG on GitHub-hosted `macos-latest`.
- GitHub Releases latest is still **v0.3.0**.

## What shipped this session

1. Rebased the release CI/CD PR onto v0.5.0 main. Spec, `DEFAULT_ARTIFACT_DIR`,
   and `release.yml` pack `models/rallyclip_v0.5.0` via `RallyClip.spec`.
2. Unsigned GitHub build path kept: Actions → Release → Run workflow.

## Next steps

1. Add the GitHub Actions secrets in README (optional until you want a
   notarized DMG).
2. `workflow_dispatch` the Release workflow on this branch to get a GitHub-built
   unsigned v0.5.0 DMG.
3. After secrets: tag `v0.5.0` (must match `pyproject.toml`).
