# Session Handoff

## Currently verified

Baseline green after `init.sh`.

## Changes this session

Harness files created.

## Still broken or unverified

Nothing.

## Next best action

Pick feature with priority 1 from `feature_list.json`: `runtime-video-validation`.

## Commands

- Install: `: # local checkout uses the existing .venv-train environment`
- Verify: `PYTHONPATH=build/lib .venv-train/bin/python -m compileall -q build/lib`
- Start: `.venv-train/bin/rallyclip gui`
- Lint: `PYTHONPATH=build/lib .venv-train/bin/python -m compileall -q build/lib scripts`
