# Artifact registry — keep git light, combine models into the DMG at build

Status: **plan only** (not started). Default inference stays `models/rallyclip_v0.5.0/`
(`frame_startend_heatmap`). Mac `.app` still ships that folder **inside the bundle**.

## Goal

`git clone --depth 1` is code + tiny JSON, not ~100MB of ONNX. The current
default weights live as a GitHub Release asset. Release CI downloads that zip,
unpacks it next to the spec, then PyInstaller/sign/DMG as today.

## Non-goals

- In-app auto-update UI (separate). Skip/Later is for app releases, not mixing
  new code with an old ONNX.
- Downloading weights on first launch of `/Applications/RallyClip.app`.
- Git LFS, DVC, history rewrite (`git filter-repo`). Full clones of old SHAs
  stay fat until a later optional purge.
- Sharing Application Support with the `.app`. Localhost library stays
  `RallyClipLibrary/` next to the checkout.
- Path-only / symlink library (desktop native picker is a later idea).

## Contract

| Place | Contents |
|---|---|
| Git | `models/rallyclip_v0.5.0/manifest.json` (and any other `manifest.json` we still test). `DEFAULT_ARTIFACT_DIR` in `runtime/defaults.py`. SHA-256 of each weight in the v0.5.0 manifest (or `SHA256SUMS` beside it). |
| GitHub Release `artifact-rallyclip_v0.5.0` | `rallyclip_v0.5.0.zip`: `model.onnx`, `scaler.json`, both pose ONNX files, `manifest.json` copy, checksums. |
| DMG | `RallyClip.app` with that folder under PyInstaller `_MEIPASS` (unchanged layout). |
| From-source cache | After fetch: `models/rallyclip_v0.5.0/*` in the checkout (gitignored binaries) **or** `~/Library/Caches/RallyClip/artifacts/rallyclip_v0.5.0/`. Prefer checkout path so existing `resolve_asset` / tests keep working with no frozen-app change. |

Old trees (`v0.4.0`, `v0.3.1`, `v0.1.0_legacy`, tracked `lstm_300_v0.1.pth`): drop weights from `main`. Keep `v0.4.0/manifest.json` so hysteresis resolution tests do not need a 3.5MB LSTM. Pose ONNX is **one** copy, only in the v0.5.0 zip (stop triplicating it).

App tag `v0.5.0` embeds artifact `rallyclip_v0.5.0`. Do not run newer code on an older zip.

## Mac runtime

**No behavior change** if CI places files at `models/rallyclip_v0.5.0/` before
`pyinstaller RallyClip.spec`. `candidate_roots()` / `_MEIPASS` already resolve
them. Do not add a first-launch download inside the notarized app.

From-source CLI/GUI: if `model.onnx` is missing, `scripts/fetch_artifact.py`
(or a thin `resolve_asset` hook) downloads the zip, verifies SHA-256, unpacks.
Tests that need real weights call the same helper or skip when offline (same
pattern as today’s “artifact missing” skips, but CI always fetches).

## CI/CD

**Phase 0 — publish the zip while weights are still in git**

1. `scripts/release/pack_artifact.sh` zips `DEFAULT_ARTIFACT_DIR` (required files
   only) + writes SHA-256.
2. One-shot: upload `rallyclip_v0.5.0.zip` to a **published** GitHub Release
   named `artifact-rallyclip_v0.5.0` (not a draft; `/releases/latest` is the
   *app* channel — use this dedicated tag so model and app versions can move
   separately).
3. `scripts/fetch_artifact.py` downloads that tag’s zip when the local folder is
   incomplete. Idempotent. Uses `gh` or `urllib` + checksum.

**Phase 1 — CI uses fetch; git still has files (safety)**

- `ci.yml` (unit + e2e) and `release.yml`: after checkout, `python scripts/fetch_artifact.py`
  (no-op if files already present).
- `release.yml` keep: verify required names under `DEFAULT_ARTIFACT_DIR`, then
  PyInstaller. Fail closed on hash mismatch.
- Tests: `test_release_packaging.py` still asserts those files exist **after**
  fetch in CI; locally, skip or fetch.

**Phase 2 — remove binaries from `main`**

- `.gitignore`: `models/**/*.onnx`, `models/**/*.pth` (keep `manifest.json`,
  maybe leave `scaler.json` only inside the zip so git stays tiny).
- Delete from the tree: all ONNX, the tracked `.pth`, extra pose copies, old
  `model.onnx` files. Keep `models/rallyclip_v0.5.0/manifest.json` and
  `models/rallyclip_v0.4.0/manifest.json`.
- Point tests at manifests, not missing ONNX:
  - `test_runtime_api_engine.py`: `resolve_pipeline_spec(..., manifest_path=...)`
    with the committed JSON; dummy `model.onnx` bytes if the helper requires a
    path.
  - `test_yolo_onnx_runner.py` / CoreML / CUDA pose: use
    `models/rallyclip_v0.5.0/` after fetch, or skip.
  - `test_runtime_feature_contract.py`: copy scaler from the fetched v0.5.0
    artifact or a tiny fixture JSON if the numbers must stay v0.3.1-shaped.
  - e2e / golden / playwright: fetch in CI; skip if still missing (offline).
- Update `AGENTS.md` (tracked ONNX exception goes away), `models/README.md`,
  `README.md` install (`clone --depth 1` + `pip install ".[cpu]"` +
  `python scripts/fetch_artifact.py`).
- `RallyClip.spec` unchanged: still packs `DEFAULT_ARTIFACT_DIR`.

**Phase 3 — app release**

Tag `v0.5.0` as already designed: tests → fetch artifact → PyInstaller → sign →
DMG → draft GitHub **app** Release with the DMG (and optionally attach the same
zip for from-source users). Publishing the app release is still a human click.

Later app-only bumps (no new weights): fetch the existing
`artifact-rallyclip_v0.5.0` zip. New weights: new dir `rallyclip_v0.6.0`, new
artifact tag, bump `DEFAULT_ARTIFACT_DIR`, then tag the app.

## Verification (definition of done)

1. Fresh `git clone --depth 1 --single-branch` has no `*.onnx` under `models/`.
2. `python scripts/fetch_artifact.py` then `rallyclip --cli --help` and golden
   CLI (or the synthetic e2e) run.
3. `ci.yml` green with fetch (weights not in the git tree).
4. `workflow_dispatch` Release: `.app` contains
   `models/rallyclip_v0.5.0/model.onnx`; unsigned or signed DMG as today.
5. Unit tests that only care about pipeline id pass without downloading ONNX.
6. Packaged app does not call GitHub at launch for weights.

## Order of PRs

1. Fetch/pack scripts + Phase 0 upload of `artifact-rallyclip_v0.5.0` (weights
   still in git).
2. Wire fetch into `ci.yml` / `release.yml`.
3. Gitignore + delete binaries + retarget tests + docs.
4. Tag `v0.5.0` when ready (app DMG).

Do not merge 3 before 1 is a **published** (non-draft) artifact release, or CI
and clones cannot get weights.
