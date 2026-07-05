# REPO_MAP — RallyClip (worktree: RallyClip-perf; branch: see docs/PROGRESS.md)

Built at commit `42da683` (2026-07-05). Staleness check:
`git diff --stat 42da683.. -- src tests scripts configs .github pyproject.toml`
If that shows changes not reflected here, update this map in the same commit as your work.

## Container context (one level up)

This checkout lives in `rallyclip_container/` beside three siblings (the container
dir is not a git repo; this table is the authoritative summary):

| Sibling | What | Touch it? |
|---|---|---|
| `../RallyClip/` | Primary clone of the SAME repo (shares `.git`), parked on `docs`. Also hosts the test venv (`.venv-train`). | No feature work there. |
| `../YOLO-ONNX/` | Separate repo: C++ ORT YOLO-pose runner + Ultralytics parity harness (`scripts/parity_v8n_960.py`). Prior art for the ONNX pose swap. | Read-only reference. |
| `../rallyclip-prod/` | Separate repo: Modal cloud worker experiments. Likely deprecated (on-device pivot). | Don't invest. |

## Top-level layout (this repo)

| Path | What |
|---|---|
| `src/` | The Python package (8 subpackages, see seams below). `package-dir = src`. |
| `tests/` | Pytest suites + `tests/fixtures/` (court goldens ~11MB, golden CLI clip, quality GT) + `tests/helpers/`. |
| `scripts/` | Training/data tooling + `scripts/perf/` benchmarks + `scripts/release/sign_macos_app.sh`. Not shipped. |
| `configs/` | Training YAMLs (`configs/train/base.yaml`, `configs/extract/*`). |
| `models/` | Tracked inference artifacts: `rallyclip_v0.3.1/{model.onnx,scaler.json,manifest.json}` (+ v0.1.0_legacy). Weights (`*.pt`, `*.pth`) present locally but gitignored. |
| `docs/` | This harness + plans (Tier 3) + `docs/perf/` (streaming-perf loop journal) + `docs/training.md`. |
| `packaging/`, `RallyClip.spec` | PyInstaller/macOS packaging. |
| `.github/workflows/` | `ci.yml` (3 OS × unit/e2e), `release.yml`. |
| `train.py`, `visualize.py` | Training-pipeline entry points (developer workflow, not runtime). |
| `config.toml` | Local runtime config for CLI runs. |
| `build/`, `logs/`, `src/rallyclip.egg-info/`, `__pycache__` | SKIP: generated, gitignored. |
| `README.md`, `PROJECT_SUMMARY.md`, `REFACTOR.md`, `TODO.md` | Tier 4: human-oriented background. TODO.md is the long-horizon idea pile, not current state (that's docs/PROGRESS.md). |

## By task — read Y, nothing else

- **Analysis pipeline behavior / segments wrong** → `src/rallyclip_engine/models.py` (AnalysisModel ABC + pipeline impls), `src/rallyclip_core/pipelines.py`, `models/rallyclip_v0.3.1/manifest.json`, then `src/rallyclip_core/intervals.py`.
- **CLI** → `src/cli/main.py` (single file; `main()` at :337), `tests/test_cli_config_contract.py`, `tests/test_cli_json_output.py`.
- **GUI/backend HTTP API** → `src/gui/app.py` (Flask, module-global config — see seams), `src/rallyclip_api/services.py`, `tests/test_api_route_contracts.py`, `tests/test_api_job_lifecycle.py`.
- **Desktop shell / playback** → `src/gui/desktop.py` (pywebview shell), `src/rallyclip_core/playback.py`, `tests/test_native_playback.py` (server-side proxy/descriptor only; the Qt native player was deleted 2026-07-04). Viewer streams `/api/library/<id>/source` directly (Range/206); WebM preview windows are the codec fallback.
- **Segment edit mode** → `src/gui/app.py` (`PUT .../segments`, `DELETE .../segments/edits`), `src/rallyclip_core/library.py` (`resolve_segments`: `segments_edited.csv` wins, original never written), `src/rallyclip_core/intervals.py` (`write_point_intervals`), frontend `src/gui/frontend/script.js` (edit-mode section), e2e `tests/test_gui_playwright.py::test_ui_segment_edit_mode_*`.
- **Pose extraction (ONNX runtime)** → `src/extraction/yolo_onnx_runner.py` + `src/extraction/pose_extractor.py` (dispatch on weights extension), `tests/test_yolo_onnx_runner.py`, `docs/onnx-pose-parity-plan.md`, `../YOLO-ONNX/scripts/parity_v8n_960.py`.
- **Court detection** → `src/preprocessing/court_detector_impl.py`, `tests/test_court_detection_deterministic.py`, `tests/helpers/court_fixtures.py`; regen fixtures with `scripts/court_fixtures_gen.py`.
- **Features/preprocessing (runtime)** → `src/features/feature_engineer.py`, `src/preprocessing/data_preprocessor.py`, contract tests `tests/test_runtime_*_contract.py`.
- **Training pipeline** → `docs/training.md`, `src/training/pipeline.py`, `configs/train/base.yaml`, `train.py`.
- **Packaging/release** → `RallyClip.spec`, `packaging/macos/`, `.github/workflows/release.yml`, `docs/cli-in-release-binary-plan.md`.
- **Perf** → `docs/perf/PLAN.md` + `docs/perf/JOURNAL.md`, `scripts/perf/bench_*.py`, baselines in `docs/perf/baseline/`.

## Key seams & entry points

- `src/rallyclip_core/` — pure contracts (`RunRequest`→`RunResult`, ProgressEvent, SavedMatchStore, playback scheduling). **Rule: no heavy imports here** (torch/ultralytics/av/cv2/numpy); `tests/test_gui_startup_imports.py` enforces it.
- `src/rallyclip_engine/runtime.py:27` — `RuntimeDeps` default binds `PoseExtractor`; the dependency-injection seam (the ONNX pose swap was validated through it before landing as the default).
- `src/rallyclip_core/pipelines.py:14` — `pipeline_id_from_manifest_values`: the model artifact's manifest (not code) selects the pipeline. Shipped: `frame_probability_hysteresis`; `start_end_attention_voting` is a stub.
- `models/rallyclip_v0.3.1/manifest.json` — the model contract: imgsz 960, conf 0.25, fps 5, seq_len 100. Don't hardcode these in code.
- `src/rallyclip_api/services.py:10` — `RallyClipServices` facade; CLI and Flask GUI are both thin clients of it. Desktop app = pywebview (WKWebView/WebView2) → local Flask; **all behavior is `/api/*` HTTP**, no private channel.
- `src/extraction/pose_extractor.py` `_flush_batch` — the predict surface (4 arrays per result) that `yolo_onnx_runner.YOLO` replicates; `.pt` weights still route to lazily-imported ultralytics (`[train]` extra).
- `src/gui/app.py:86-226` — config is module globals (`PREFERENCES_PATH`, `JOBS_DIR`, …). E2E harness `tests/helpers/e2e_backend.py` must redirect ALL of them; config-object refactor planned (`docs/runtime-config-refactor-plan.md`).
- Video decode is **PyAV everywhere** in runtime; OpenCV is image-ops only (court detector classical CV + the parity-critical letterbox resize; its bundled FFmpeg is dead weight — removable only via a custom videoio-less wheel).

## Deliberately NOT worth reading

- `build/` — stale PyInstaller output (includes a full copy of src as `build/lib/`; grep hits there are noise).
- `models/*/model.onnx`, weights, `tests/fixtures/**` binaries — data, not code.
- `docs/perf/baseline/*.json`, `docs/perf/iterations.jsonl` — machine-written bench records.
- `docs/*.icns|svg|png` — icon assets.
- Root `README.md`/`PROJECT_SUMMARY.md`/`REFACTOR.md` — background prose; superseded for orientation by this map.
