# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Python package (CLI, GUI, preprocessing, inference, segmentation, training).
- `tests/` contains pytest suites; test files follow `test_*.py` naming.
- `configs/` and `config.toml` define runtime and training settings (see `configs/train/base.yaml`).
- `models/` stores trained weights and scalers; `yolov8s-pose.pt` is the default pose model.
- `docs/` includes deeper training notes (`docs/training.md`).
- Outputs land in `output_videos/` and `output_csvs/` by default.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` (or conda): create an isolated env.
- `pip install .`: install the CLI package.
- `pip install .[dev]`: install dev extras (pytest).
- `rallyvision --video path/to/video.mp4`: run point extraction on a match.
- `rallyvision gui`: launch the local GUI.
- `pytest`: run the test suite.
- `python train.py --config configs/train/base.yaml`: run the training pipeline.

## Coding Style & Naming Conventions
- Use PEP 8 conventions: 4-space indentation, `snake_case` for functions/modules, `CapWords` for classes.
- Prefer explicit, descriptive names for model and feature components (e.g., `features_builder.py`).
- Keep new modules under the appropriate `src/` subpackage; avoid dumping new logic into `cli/`.

## Testing Guidelines
- Framework: `pytest` (see `tests/` and `conftest.py`).
- Name new tests `test_<unit>.py` and individual tests `test_<behavior>()`.
- Include small fixture data in `tests/` or `data/` and avoid large binary assets in git.

## Commit & Pull Request Guidelines
- Current history shows short, single-line summaries without strict convention. Use a concise, present-tense subject (<= 72 chars) and mention the primary change (e.g., “add segment metrics tests”).
- PRs should include: a brief summary, how to test (commands + result), and screenshots/gifs for GUI changes.

## Configuration & Assets
- Update `config.toml` for local runs; don’t commit machine-specific paths.
- Ensure `models/` contains `lstm_300_v0.1.pth` and `scaler_300_v0.1.joblib` before inference.

## Agent-Specific Rules
- Do not create, edit, move, or delete any file that is not tracked by git.
- Before modifying files, verify they are tracked (for example: `git ls-files <path>`).
- Limit changes to tracked repository files required by the task.
