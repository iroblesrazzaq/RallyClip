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

Last known (2026-10-27, native macOS on runtime v0.6.0, rebuilt on main after #64, macOS `.venv-train` py3.11):
**616 passed, 6 skipped, 49 deselected, ~79s** (#62 alone: 603 passed) (some skip without local model artifacts). `tests/test_release_packaging.py` 24 passed;
`tests/test_artifact_fetch.py` 6 passed; `tests/test_agent_review.py` 59 passed.
(2026-10-27, large-video import: 594 passed, 49 deselected.)
(2026-10-23, review round cap merged with main after #68: 542 passed, 16 skipped, 49 deselected.)
(2026-10-22 main, cancel-during-hash fix: 363 passed, 11 skipped; `tests/test_update_release.py` 15 passed.)
(2026-09-09, Linux py3.12: 321 passed, 6 skipped, 27 deselected; `tests/test_release_packaging.py`
21 passed; `tests/test_artifact_fetch.py` 6 passed.)

## Compile gate (cheap, run with the default gate)

```bash
$PY -m compileall -q src tests
```

Last known (2026-10-27): exit 0 with `.venv-train`.

## Native macOS

`swift test --package-path macos`: **5 passed** on 2026-10-27 (also 2026-10-07), including a
synthetic camera-boundary composition/export/frame-content test. Swift requires
normal host access to its caches and media services.

`$PY scripts/native/smoke.py --backend dist/native/RallyClip.app/Contents/Resources/Backend/RallyClipBackend`
checks a frozen import, edit/reset/delete, decoded-frame playback through three
cuts, exact export duration, and delayed audio across a camera boundary. Run after
`RALLYCLIP_BUILD_PYTHON="$PY" bash scripts/native/build.sh`. See docs/native-macos.md.

## Golden CLI parity (run when touching pipeline/engine/extraction/decode)

```bash
PYTHONPATH=src:tests $PY -m pytest -q -p no:cacheprovider tests/test_cli_golden_parity.py
```

Last known (2026-10-27, v0.6.0 golden): **1 passed, 16.1s**. Boundaries compared at 0.5s tolerance;
byte-exact only holds on the machine+env the golden was generated on. ubuntu/windows
ORT can split a TCN point across a 1-frame pointness dip; v0.5.0 ships
`merge_gap_sec: 0.5` so those fragments become one segment.

## Full e2e (44 tests; heavy — run before merges/releases, not per-commit)

```bash
PYTHONPATH=src:tests $PY -m pytest -q -m e2e -p no:cacheprovider
```

Not run 2026-07-03 (needs `uv sync --extra cpu --extra dev --extra e2e-ui` + `uv run playwright install chromium`;
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
6. Before every push: the review gate below.

## Review gate (before every push)

Claude Code authors; Cursor (Grok) and Greptile review locally, headless, read-only.
Run via the `/agent-review` skill (`.claude/skills/agent-review/SKILL.md`), which wraps:

```bash
python3 scripts/agent/review.py            # cursor (3 lenses) + greptile vs origin/main
python3 scripts/agent/review.py --lenses correctness,invariants,tests,perf   # pipeline IO/memory work
```

Reviews committed changes only (`merge-base..HEAD`); refuses to run with uncommitted
tracked edits, since Cursor reads the live checkout (`--allow-dirty` to override). Output (diff, raw reviewer
output, `findings.json`) goes to `$TMPDIR/rallyclip-review-<branch>-<ts>-XXXX/`, never the
repo. Cursor model defaults to `cursor-grok-4.6-high` (`--model` /
`RALLYCLIP_REVIEW_MODEL`; `agent --list-models`). Exit 0 ok, 2 a reviewer failed
(partial results kept), 3 a reviewer modified the tree. Triage every finding
(confirmed / rejected + reason / unsure) into `triage.json` next to `findings.json`,
fix confirmed high+medium, then a `## Review digest` table in the PR body.
`review.py` enforces the round limit per branch: at most 3 rounds, and the next round
runs only if the last one had a confirmed high/P1 and HEAD moved (exit 4 = stop and
write the digest). Rejected findings from earlier rounds go into the Cursor prompt and
mark repeats (`repeat_of`). Round state: `<git-common-dir>/rallyclip-review/<branch>.json`. Needs `agent` (Cursor CLI,
logged in) and `greptile` (`greptile whoami`) on PATH. No Bugbot, no PR-comment reviews.

Runtime (2026-10-21 dry run, ~34 KB diff): Greptile ~2–4 min; each Cursor lens
~7–11 min (lenses run in parallel), so run it in the background. The Cursor backend
can drop long streams (`RetriableError: Connection stalled`); the script retries once,
and `--timeout` kills the whole `agent` process group. If Cursor is down
(`Cannot use this model ... Available models:` with an empty list), run
`--reviewers greptile` and say so in the digest.

Unit tests for the script: `tests/test_agent_review.py` (45, no reviewer CLIs invoked).

