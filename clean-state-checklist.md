# Clean State Checklist

- [x] `./init.sh` exits 0
- [x] Current feature has an explicit evidence definition before being marked `passing`
- [x] `claude-progress.md` has a session record for this session
- [x] `feature_list.json` matches actual code state (no false `passing`)
- [ ] No half-finished work left uncommitted
- [x] `session-handoff.md` is current
- [x] Next session can start with only `AGENTS.md` and repo contents (no oral instructions needed)

Note: the docs checkout already has unrelated dirty files outside this release handoff. Do not assume a clean root worktree. The release worktree `/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf` was clean after pushing `feat/first-release` and `v0.1.0`; ignored `dist/` artifacts contain the notarized DMG.
