# PROGRESS — overwrite me at every session end

_Last updated: 2026-07-03 (session: harness creation)._

## Repo state as found

- Branch `refactor/runtime-api-engine` at `bb72dd6`, clean tree. Content-equal to
  `main`/`feat/first-release` per the container CLAUDE.md snapshot (verify with git).
- Default gate: 212 passed / 44 deselected (27.5s). Golden CLI parity: 1 passed (14s).
  compileall clean. 23 pre-existing ruff errors, no lint config (see docs/testing.md).
- CI green (first time in repo history) as of the 2026-07-04 snapshot; not yet a
  required status check on protected `main`.

## What this session did

Created the doc harness: AGENTS.md, features.json, docs/{REPO_MAP,PROGRESS,DECISIONS,
ENVIRONMENT,testing}.md, docs/archive/. Replaced the old generic root AGENTS.md and
moved root ENVIRONMENT.md content into docs/ENVIRONMENT.md. No source code changed.

## Next steps (in order)

1. **ONNX pose swap** (`onnx-pose-swap` in features.json — the active feature).
   Follow `docs/onnx-pose-parity-plan.md`; currently at Stage 0/1 boundary:
   contract audited, plan committed, parity numbers exist in `../YOLO-ONNX` at 960.
   Next concrete step: Stage 1 — export `yolov8n-pose.pt` → ONNX at imgsz 960
   (rect/dynamic axes, nms=False and nms=True variants) and run test 1a/1b.
2. **Frozen-app rebuild + smoke test** from current code (v0.1.0 shipped from
   pre-refactor code) — `docs/e2e_test_plan.md`, `docs/cli-in-release-binary-plan.md`.
3. Housekeeping candidates: make CI a required status check on `main`; config-object
   refactor for `gui/app.py` module globals (`docs/runtime-config-refactor-plan.md`).

## Blockers / open questions

- None hard. E2e suite not verified locally this session (needs playwright install;
  quality/court e2e need local footage).
