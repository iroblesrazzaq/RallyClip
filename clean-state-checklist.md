# Clean State Checklist

- [x] `./init.sh` exits 0
- [x] Current feature has an explicit evidence definition before being marked `passing`
- [x] `claude-progress.md` has a session record for this session
- [x] `feature_list.json` matches actual code state (no false `passing`)
- [x] No half-finished release handoff work left uncommitted
- [x] `session-handoff.md` is current
- [x] Next session can start with only `AGENTS.md` and repo contents (no oral instructions needed)

Note: the docs checkout has user-managed dirty files; do not assume a clean
root worktree. The `RallyClip-perf` worktree (branch refactor/runtime-api-engine)
is CLEAN and fully pushed as of Session 9 (33a961c); main/feat/first-release are
merged and green.
