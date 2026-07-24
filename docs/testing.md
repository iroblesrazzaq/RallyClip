# Testing gates — exact commands + last-known results

Run from repo root. `$PY` throughout:

```bash
PY=/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip/.venv-train/bin/python3
```

(That venv lives in the sibling primary clone; it has the full stack + pytest 9.0.3,
Python 3.11.14, cv2 4.13 (torch 2.12 present in this env but the runtime never imports it). Conda `tennis_env` is an
equivalent alternative; `.venv-train` is what was verified 2026-07-03.)

## Default gate (run every session; the branch gate before any commit)

```bash
PYTHONPATH=src:tests $PY -m pytest -q -m "not slow and not e2e" -p no:cacheprovider
```

Last known (2026-07-23, fix/windows-cuda-ort Greptile P1s): **231 passed, 48 deselected, ~38s** (2 sklearn warnings, benign).

## Compile gate (cheap, run with the default gate)

```bash
$PY -m compileall -q src tests
```

Last known (2026-07-03): exit 0.

## Golden CLI parity (run when touching pipeline/engine/extraction/decode)

```bash
PYTHONPATH=src:tests $PY -m pytest -q -p no:cacheprovider tests/test_cli_golden_parity.py
```

Last known (2026-07-03): **1 passed, 14.0s**. Boundaries compared at 0.25s tolerance;
byte-exact only holds on the machine+env the golden was generated on.

## Full e2e (44 tests; heavy — run before merges/releases, not per-commit)

```bash
PYTHONPATH=src:tests $PY -m pytest -q -m e2e -p no:cacheprovider
```

Not run 2026-07-03 (needs `pip install .[dev,e2e-ui]` + `playwright install chromium`;
quality/court e2e need local footage — `RALLYCLIP_COURT_VIDEO_DIR`/`RALLYCLIP_YOLO_WEIGHTS`,
self-skip if absent). CI runs unit+e2e on 3 OS; green as of 2026-07-04 snapshot.
Court e2e wants `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Lint

No linter is configured in this repo (no ruff/flake8 config; CI doesn't lint).
Repo-wide `ruff check src tests` (ruff 0.14.10, default rules) shows **23 pre-existing
errors** (16 F401 unused-import, 5 F841, 1 E402, 1 F541) — legacy debt: not ours to
fix, not a license to add new ones.

**Scoped lint for new work** — lint only the files you touched:

```bash
/Users/ismaelrobles-razzaq/anaconda3/bin/ruff check <changed files>
```

New/changed files must be clean; leave the legacy 23 alone unless you're already
editing those lines.

## Branch gate (before every commit)

1. Default gate passes at last-known counts or better (new tests raise the count — update this file).
2. Compile gate passes.
3. Scoped lint clean on touched files.
4. Golden parity if you touched `src/{rallyclip_engine,rallyclip_core,extraction,features,preprocessing,segmentation,runtime}`.
5. Never commit to `main` (protected; PR only). Work lands on topic branches / `feat/first-release`.
