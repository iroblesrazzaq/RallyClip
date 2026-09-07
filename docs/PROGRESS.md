# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-07 (session: artifact-registry implementation)._

## Repo state

- `main` includes merged macOS release CI (#50). GitHub Releases latest app
  channel is still **v0.3.0**. Do not tag app `v0.5.0` until this lands on
  `main` (and you want the DMG).
- Artifact zip is **published** (not draft):
  https://github.com/iroblesrazzaq/RallyClip/releases/tag/artifact-rallyclip_v0.5.0
- Active feature: `artifact-registry` (`docs/artifact-registry-plan.md`).

## What shipped this session

1. `src/runtime/artifact.py` + `scripts/fetch_artifact.py` +
   `scripts/release/pack_artifact.sh`: pack/verify/fetch with SHA-256, zip-slip
   reject, idempotent no-op when checksums match. Fail closed on mismatch.
2. Published `artifact-rallyclip_v0.5.0` / `rallyclip_v0.5.0.zip` while weights
   were still in git.
3. `ci.yml` and `release.yml` fetch after install; release verify calls
   `verify_artifact_dir` before PyInstaller. `RallyClip.spec` unchanged — DMG
   still embeds `models/rallyclip_v0.5.0/`; frozen app does not fetch at launch.
4. Gitignore ONNX + `scaler.json` under `models/`; dropped old/duplicate
   weights from git; tests that only need pipeline id read `manifest.json`.
5. Default gate: **297 passed, 6 skipped, 27 deselected, ~20s**.

## Next steps

1. Merge this PR. Then tag app `v0.5.0` when ready (fetch same zip, existing
   sign/notary path).
2. In-app Skip/Later auto-update remains out of scope.
