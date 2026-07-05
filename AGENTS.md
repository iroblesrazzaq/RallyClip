# Repository Guidelines

## Project Structure & Module Organization
- `src/rallyclip_core/` — contracts (RunRequest/RunResult), library store, playback.
- `src/rallyclip_engine/` — AnalysisModel pipeline (manifest-first selection); all analysis flows through `run_analysis`.
- `src/rallyclip_api/` — RallyClipServices facade + JSON serialization. CLI (`src/cli/`) and Flask GUI (`src/gui/`) are thin facade clients.
- `src/extraction/`, `src/preprocessing/`, `src/features/`, `src/infer/`, `src/segmentation/` — pipeline stages. Pose runs on `extraction/yolo_onnx_runner.py` (onnxruntime; no torch).
- `src/training/` — training/eval code; the only place torch/ultralytics are required (`pip install .[train]`).
- `models/rallyclip_v0.3.1/` — the shipped artifact: `manifest.json` (authoritative contract: imgsz=960, conf=0.25, fps=5, seq_len=100), `model.onnx`, `scaler.json`, `yolov8n-pose-960-dynamic.onnx`.
- `tests/` — pytest; markers `slow` and `e2e` gate the heavy suites.

## Build, Test, and Development Commands
- `pip install .` — torch-free runtime install. Extras: `[dev]` pytest, `[train]` torch/ultralytics, `[desktop]` pywebview, `[e2e-ui]` Playwright, `[pack]` PyInstaller.
- `PYTHONPATH=src python -m cli.main --video match.mp4 ...` — run analysis from source.
- `rallyclip-desktop` — system-webview shell (WKWebView/WebView2) over the localhost Flask backend; all behavior is `/api/*` HTTP.
- `pytest -m "not slow and not e2e"` — fast suite (<1 min). `pytest -m e2e` — backend + browser e2e.
- `pyinstaller --noconfirm RallyClip.spec` — build the desktop bundle (torch-free, no Qt).

## Invariants (do not break)
- The manifest is authoritative for pipeline parameters; only explicit CLI flags override (with a warning).
- Golden parity: `tests/test_cli_golden_parity.py` must stay byte-equal on this machine (0.25s tolerance cross-platform). Regenerating goldens is a deliberate, documented act.
- The runtime must never import torch/ultralytics (`tests/test_yolo_onnx_runner.py::test_analysis_run_stays_torch_free`).
- PyAV is the only runtime video decoder; cv2 is image ops only.
- CLI status prints are ASCII-only (Windows cp1252 consoles).

## Coding Style & Naming Conventions
- PEP 8: 4-space indent, `snake_case` functions/modules, `CapWords` classes.
- Comments state constraints the code can't show; match the density of surrounding code.

## Commit & Pull Request Guidelines
- Concise present-tense subject (<= 72 chars), one concern per commit.
- Never add a Claude co-author trailer.
- `main` is PR-only; CI (3 OS x test/e2e) must be green before merge; the user merges.

## Agent-Specific Rules
- Do not create, edit, move, or delete any file that is not tracked by git (verify with `git ls-files <path>`).
- Harness/progress files (feature_list.json, claude-progress.md, session-handoff.md) live in the sibling `RallyClip/` worktree on the `docs` branch, not here.
