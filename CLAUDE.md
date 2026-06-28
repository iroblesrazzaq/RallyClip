Read this file completely before taking any action.

# Workflow

## Startup

1. Run `cat claude-progress.md` to load current project state.
2. Run `cat feature_list.json | python -m json.tool` (or `jq .`) to load feature status.
3. Run `cat session-handoff.md` if it exists.
4. Run `./init.sh` to verify the baseline is green.
5. Identify the highest-priority feature with status `not_started` or `in_progress`.

## End Of Session

1. Run `PYTHONPATH=build/lib .venv-train/bin/python -m compileall -q build/lib` one final time.
2. Update `feature_list.json` with current statuses and evidence.
3. Append a session record to `claude-progress.md`.
4. Write `session-handoff.md`.
5. Run through `clean-state-checklist.md`.
6. Commit all harness files: `git add AGENTS.md claude-progress.md feature_list.json session-handoff.md && git commit -m "harness: end-of-session update"`.

# Rules

- RallyClip is a local tennis match segmentation project with a CLI, desktop app, browser GUI, and training/evaluation pipeline.
- Treat startup as read-only until `./init.sh` has completed and the current feature is identified.
- Only one feature may be `in_progress` at a time.
- Before changing code, state what you will change and why.
- Run `PYTHONPATH=build/lib .venv-train/bin/python -m compileall -q build/lib` after every meaningful code change.
- Do not modify files outside the scope of the current feature without documenting why.
- If a test fails, fix it before moving on. Never skip a red test.
- If blocked, set the feature status to `blocked`, document the blocker in `claude-progress.md`, and stop.

# Definition Of Done

All of these must be true before marking a feature `passing`:

- `PYTHONPATH=build/lib .venv-train/bin/python -m compileall -q build/lib` passes with zero failures.
- `PYTHONPATH=build/lib .venv-train/bin/python -m compileall -q build/lib scripts` passes with zero errors.
- The `user_visible_behavior` described in `feature_list.json` is demonstrably working.
- Evidence (test output, screenshot path, curl response) is recorded in the feature's `evidence` field.
- `claude-progress.md` is updated with a session record.
- `session-handoff.md` is updated.
- `clean-state-checklist.md` items are all checked.

# Local PR Review Guidance

This repo uses Greptile for automated PR review. After opening or updating a PR, run this loop until the PR is clean. Do not merge; the user merges.

1. Pull the review summary and inline comments with `gh api repos/<owner>/<repo>/issues/<pr>/comments` and `gh api repos/<owner>/<repo>/pulls/<pr>/comments`.
2. Critique each claim as legitimate or not by checking the actual code.
3. Fix legitimate claims with the smallest correct change. Skip invalid ones and note why.
4. Re-trigger review by committing, pushing, and commenting `@greptileai` on the PR.
5. Verify a `Confidence Score: 5/5`.
6. Repeat until the PR is clean.
7. Hand back to the user with the 5/5 result for them to merge.
