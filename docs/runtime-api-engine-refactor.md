# Runtime/API/Engine Refactor

**Branch:** `refactor/runtime-api-engine`  
**Base:** `feat/first-release` at `80ba932`  
**Status:** In progress, uncommitted checkpoint  
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
- library listing
- playback manifest

Still to wire:

- start job
- job status
- cancel job
- export match
- saved-match storage resolution

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

The branch currently has uncommitted changes.

Modified tracked files:

- `src/cli/main.py`
- `src/gui/app.py`
- `src/gui/native_player.py`
- `src/runtime/assets.py`
- `src/runtime/defaults.py`
- `tests/test_gui_e2e.py`

New files:

- `ENVIRONMENT.md`
- `docs/runtime-api-engine-refactor.md`
- `src/rallyclip_api/`
- `src/rallyclip_core/`
- `src/rallyclip_engine/`
- `tests/test_runtime_api_engine.py`

## What Is Still Not Done

- `main` has not been rebased onto this branch.
- CLI has smoke/parity coverage, but full CLI parity should be checked before
  replacing `main`.
- Saved-match storage and file resolution still mostly live in `gui.app`.
- Flask still owns much of the job lifecycle glue.
- CLI calls the shared engine directly; it is not yet a thin client of the
  application service facade.
- API JSON contracts need explicit serialization tests for `RunResult` and
  saved-match models.
- Real `start_end_attention_voting` production preprocess/infer is not
  implemented; only the interface and decoder test exist.
- ONNX YOLO replacement, storage migration, CI/CD, and v0.2 product work remain
  out of scope for this branch.

## Recommended Next Steps

1. Commit the current refactor checkpoint.
2. Run a direct CLI command against a tiny/synthetic video or known fixture and
   verify CSV/video output paths still match pre-refactor behavior.
3. Move saved-match/library file resolution from `gui.app` into
   `rallyclip_api`/`rallyclip_core`.
4. Wire job start/status/cancel/export through `RallyClipServices`.
5. Add a stronger golden parity test proving the shipped pipeline produces the
   same intervals before and after the refactor.
6. Only after CLI and GUI/API parity are proven, rebase or fast-forward `main`
   to this branch.
