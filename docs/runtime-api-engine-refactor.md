# Runtime/API/Engine Refactor

**Branch:** `refactor/runtime-api-engine`  
**Base:** `feat/first-release` at `80ba932`  
**Status:** In progress, checkpoint committed at `5d266c2`  
**Last updated:** 2026-07-01

This branch starts the post-v0.1 runtime split. The shipped macOS release remains
stable on `feat/first-release`; `main` has not been rebased or fast-forwarded yet.

## Goal

Create a shared runtime architecture where CLI, desktop, browser development UI,
and future mobile clients call the same analysis and playback contracts instead of
each owning separate logic.

The split is intentionally conservative:

- **`rallyclip_core`** owns pure contracts and source-time helpers.
- **`rallyclip_engine`** owns analysis execution: preprocess, infer, postprocess,
  and output intervals/CSV-ready results.
- **`rallyclip_api`** is the application service facade used by Flask now and by
  future clients later.
- **GUI/native/browser clients** own UI and actual video rendering.

## Current Package Boundaries

### `rallyclip_core`

Added as a new internal package for pure, import-light behavior.

Current contents:

- `contracts.py`
  - `RunRequest`
  - `RunResult`
  - `PipelineSpec`
  - `ProgressEvent`
  - `SavedMatch`
  - `PlaybackManifest`
  - `RuntimeDeps`
  - shared user-facing errors
- `intervals.py`
  - frame-segment to source-time interval conversion
  - CSV interval reading
  - point-duration calculation
- `pipelines.py`
  - pipeline id constants
  - manifest-to-pipeline resolution
  - explicit override compatibility checks
- `playback.py`
  - source-time playback manifest payload builder
  - pure point-skip scheduler

`rallyclip_core` must stay free of Torch, Ultralytics, PyAV, Numpy, Flask,
PySide, Qt, and browser-specific code.

### `rallyclip_engine`

Added as the analysis runtime package. It is the only layer that should know how
to run a model pipeline end-to-end.

Current contents:

- `runner.py`
  - `run_analysis(request, deps=None, progress_callback=None, cancel_check=None)`
- `runtime.py`
  - lazy heavy-runtime dependency loader
- `models.py`
  - `AnalysisModel` base class
  - `FrameProbabilityHysteresisModel`
  - `StartEndAttentionVotingModel` placeholder
  - deterministic start/end voting decoder test helper

The important design decision: a pipeline is a full model object, not just a
postprocessor. It owns:

1. preprocessing
2. inference
3. postprocessing
4. writing CSV/video outputs through the shared result contract

The current shipped artifact resolves to `frame_probability_hysteresis`. Future
E2E start/end-head artifacts should declare their own pipeline id and implement a
separate `AnalysisModel`.

### `rallyclip_api`

Added as a thin service boundary. Flask still owns most route glue, but route
handlers now delegate selected operations through `RallyClipServices`.

Currently wired:

- defaults
- runtime status
- runtime warmup
- library listing (via `SavedMatchStore` fallback)
- playback manifest
- start job / job status / cancel job (None for unknown jobs)
- export match (lazy cut generation; raises FileNotFoundError/ValueError)
- analysis runs (`run_analysis` with lazy engine import; used by the CLI)
- saved-match storage resolution (`rallyclip_core.library.SavedMatchStore`)

JSON serialization for `RunResult` and `SavedMatch` lives in
`rallyclip_api.serialization` with exact-shape contract tests.

## Analysis Engine Contract

The engine should produce stable CSV/JSON-ready analysis output. It should not
know about Qt, HTML video, fullscreen, hover controls, or player events.

Current engine flow:

```text
RunRequest
  -> resolve PipelineSpec from artifact manifest plus optional override
  -> build AnalysisModel
  -> model.preprocess()
  -> model.infer()
  -> model.postprocess()
  -> RunResult(frame_segments, intervals_sec, csv_path, video_path, diagnostics)
```

Expected output shape:

```json
{
  "pipeline_id": "frame_probability_hysteresis",
  "intervals": [
    {"start_s": 12.4, "end_s": 18.9}
  ],
  "csv_path": "..."
}
```

`RunResult` currently stores intervals as tuples; JSON conversion belongs in API
or client-facing serialization, not inside the engine.

## Playback Contract

Playback is separate from analysis. It consumes saved match data:

- `source.mp4`
- `segments.csv`
- `meta.json`
- optional thumbnail/export/cache files

The shared playback layer should provide source-time data and scheduling rules.
Actual rendering stays platform-specific:

- native macOS: QtMultimedia + `QVideoWidget`
- browser dev fallback: HTML video/chunk path
- future iOS: AVPlayer

Current shared playback flow:

```text
segments.csv + source duration
  -> PlaybackManifest
  -> source-time point intervals
  -> SourceTimelineScheduler
  -> client-specific player seeks/plays
```

Current scheduler behavior:

- default starts at first detected point
- seeking inside a point plays to that point end, then jumps to the next point
- seeking before/between points plays continuously through the next point end,
  then resumes point skipping
- seeking after the last point plays to source end
- last point can continue into tail video when source remains
- large forward/backward seeks are absolute, not inferred from crossed points

## Pipeline Selection

Pipeline selection is manifest-first, with explicit override for experiments.

Current ids:

- `frame_probability_hysteresis`
- `start_end_attention_voting`

Current rules:

- existing `postprocess.method = "hysteresis"` resolves to
  `frame_probability_hysteresis`
- future `pipeline.id` manifests can select another pipeline
- explicit override fails before model execution when incompatible with the
  artifact output contract

## Import Isolation

The refactor preserves the release requirement that replay/library startup does
not load heavy analysis modules.

Expected import-light modules:

- `gui.app` for replay/library startup
- `rallyclip_core`
- `rallyclip_api`

Heavy imports should remain lazy inside analysis execution paths:

- Torch
- Ultralytics
- PyAV
- Numpy
- ONNX runtime where applicable

## Tests Run At This Checkpoint

```bash
PYTHONPATH=src:tests python3 -m compileall -q src tests
```

Passed.

```bash
PYTHONPATH=src:tests python3 -m pytest -q \
  tests/test_runtime_api_engine.py \
  tests/test_cli_config_contract.py \
  tests/test_cli_pipeline_smoke.py \
  tests/test_gui_smoke.py \
  tests/test_gui_startup_imports.py \
  tests/test_native_playback.py \
  tests/test_video_validation.py
```

Result:

```text
78 passed, 1 skipped
```

```bash
PYTHONPATH=src:tests python3 -m pytest -q tests/test_gui_e2e.py
```

Result:

```text
15 passed, 3 skipped
```

## Important Fixture Fix

`tests/test_gui_e2e.py` fabricated a preview window cache at the old
`1s/5s` key, while the product route canonicalizes playback windows to `0s/8s`.
The fixture now writes `preview_windows/000000000000_008000.webm` and touches it
newer than the source/CSV, matching product cache behavior.

## Current Git State

The refactor checkpoint is committed as `5d266c2` on `refactor/runtime-api-engine`
(one commit ahead of `feat/first-release` at `80ba932`).

## CLI Output Parity (2026-07-01)

Direct CLI parity was verified against the shipped baseline. The same 68s test
video (`rallyclip-prod/testing_vids/Set_1_Play_Uncut_Derek_vs._Alex_short_segmented.mp4`)
was analyzed twice on CPU with identical flags (`--write-csv --segment-video
--yolo-device cpu`): once from a clean worktree at `80ba932`
(`feat/first-release` baseline) and once from this branch at `5d266c2`.

Results:

- `parity_segments.csv` identical (3 intervals: 5.800–43.400, 46.000–56.400,
  60.000–65.000)
- `parity_segmented.mp4` byte-for-byte identical (same SHA-1, 13.5 MB, 53.0s)
- Output file naming and locations unchanged

## Completed in This Pass (2026-07-02)

1. Saved-match/library file resolution moved into
   `rallyclip_core.library.SavedMatchStore`; `gui.app` helpers are thin
   delegators (`e161b69`). Unit tests: `tests/test_saved_match_store.py`.
2. Job start/status/cancel/export wired through `RallyClipServices`; routes
   keep only HTTP glue (`1f20343`). Facade tests:
   `tests/test_api_job_lifecycle.py`.
3. JSON serialization contracts for `RunResult` and `SavedMatch` in
   `rallyclip_api.serialization` (`e64e810`). Tests:
   `tests/test_api_serialization.py`.
4. CLI is a thin client of the facade via `RallyClipServices.run_analysis`
   with a lazy engine import (`94f8077`). CLI contract/smoke tests pass
   unchanged.
5. Committed golden CLI parity test: 24s/2.4MB fixture clip +
   `golden_segments.csv` in `tests/fixtures/golden_cli/`, byte-exact CSV
   assertion, self-skips without heavy deps or model artifacts (`077931c`).
6. Runtime video decode consolidated onto PyAV: `runtime/video_frames.py`
   `VideoFrameReader` replaces the last runtime `cv2.VideoCapture` uses in
   court detection and the preprocessor duration probe (`73cceaa`). OpenCV
   remains image-ops only at runtime; the package itself cannot be dropped
   until the ONNX YOLO migration because Ultralytics requires it.

## What Is Still Not Done

- `main` has not been rebased onto this branch.
- Real `start_end_attention_voting` production preprocess/infer is not
  implemented; only the interface and decoder test exist.
- ONNX YOLO replacement (which would also allow dropping OpenCV entirely),
  storage migration, CI/CD, and v0.2 product work remain out of scope for
  this branch.

## Recommended Next Steps

1. Decide whether to rebase or fast-forward `main` now that CLI and GUI/API
   parity are proven and the facade wiring is complete.
2. Implement `start_end_attention_voting` preprocess/infer when the E2E
   start/end-head model is ready.
