# Clean State Checklist

- [ ] `./init.sh` exits 0
- [ ] `PYTHONPATH=build/lib .venv-train/bin/python -m compileall -q build/lib` passes
- [ ] `PYTHONPATH=build/lib .venv-train/bin/python -m compileall -q build/lib scripts` passes
- [ ] `claude-progress.md` has a session record for this session
- [ ] `feature_list.json` matches actual code state (no false `passing`)
- [ ] No half-finished work left uncommitted
- [ ] `session-handoff.md` is current
- [ ] Next session can start with only `AGENTS.md` and repo contents (no oral instructions needed)
