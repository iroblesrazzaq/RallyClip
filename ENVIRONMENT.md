# RallyClip Development Environment

This branch targets the runtime/API/engine refactor on top of the shipped Mac app baseline.
For branch-specific architecture and handoff notes, see
`docs/runtime-api-engine-refactor.md`.

## Python

- Use Python `>=3.10`.
- Recommended local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev,desktop]"
```

## Optional Extras

Browser E2E tests:

```bash
pip install -e ".[e2e-ui]"
playwright install chromium
```

Packaging tools:

```bash
pip install -e ".[pack]"
```

## Local Run Commands

CLI:

```bash
rallyclip --help
rallyclip --video path/to/video.mp4
```

Browser development UI:

```bash
rallyclip gui
```

Native desktop shell:

```bash
rallyclip-desktop
```

Frozen/headless app shape:

```bash
dist/RallyClip.app/Contents/MacOS/RallyClip --cli --help
```

## Model Assets

- RallyClip model artifacts live under `models/rallyclip_v0.3.1/`.
- Required files are `model.onnx`, `scaler.json`, and `manifest.json`.
- YOLO pose weights are resolved from `models/` or downloaded/cached by Ultralytics when needed.
- The artifact manifest declares the default analysis pipeline. The current shipped artifact resolves to `frame_probability_hysteresis`.

## Test Commands

Use `PYTHONPATH=src:tests` when running directly from a checkout without an
editable install:

```bash
PYTHONPATH=src:tests python3 -m pytest -q
```

Import/startup isolation:

```bash
PYTHONPATH=src:tests python3 -m pytest -q tests/test_gui_startup_imports.py
```

CLI contract and smoke:

```bash
PYTHONPATH=src:tests python3 -m pytest -q tests/test_cli_config_contract.py tests/test_cli_pipeline_smoke.py
```

Runtime/API/engine refactor:

```bash
PYTHONPATH=src:tests python3 -m pytest -q tests/test_runtime_api_engine.py
```

GUI/API smoke:

```bash
PYTHONPATH=src:tests python3 -m pytest -q tests/test_gui_smoke.py tests/test_gui_e2e.py
```

Optional browser E2E:

```bash
PYTHONPATH=src:tests python3 -m pytest -q tests/test_gui_playwright.py
```

Current refactor parity suite:

```bash
PYTHONPATH=src:tests python3 -m compileall -q src tests
PYTHONPATH=src:tests python3 -m pytest -q \
  tests/test_runtime_api_engine.py \
  tests/test_cli_config_contract.py \
  tests/test_cli_pipeline_smoke.py \
  tests/test_gui_smoke.py \
  tests/test_gui_startup_imports.py \
  tests/test_native_playback.py \
  tests/test_video_validation.py
PYTHONPATH=src:tests python3 -m pytest -q tests/test_gui_e2e.py
```
