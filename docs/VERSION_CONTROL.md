# Version control — Greptile review before push / PR

Before pushing commits or opening/updating a PR, run a local Greptile review against
`main` and act on the feedback. Do not push until Greptile has nothing **relevant**
left.

## Command

```bash
greptile review --branch main --agent
```

- `--branch main` — diff the current topic branch against `main` (not the remote
  default alone if they diverge).
- `--agent` — plain text output for coding agents (alias of `--text`).

Requires a logged-in Greptile CLI (`greptile login` / `greptile whoami`). Prefer
`--json` only when you need machine-parseable comments.

Related: `greptile review --resume` continues an unfinished review;
`greptile review show` reopens a previous review.

## Required loop (agents)

1. Finish the local commits you intend to publish (branch gate in `docs/testing.md`
   still applies before commit).
2. Run `greptile review --branch main --agent`.
3. Read every finding. **You decide relevance** — Greptile may flag style nits,
   false positives, out-of-scope suggestions, or issues that do not apply to this
   change. Ignore or explicitly dismiss irrelevant items; do not treat the tool as
   an automatic merge blocker.
4. For each **relevant** finding: fix the code (or docs/tests), commit, and run
   Greptile again.
5. Repeat steps 2–4 until a review pass brings up **no relevant feedback**.
6. Only then push the commit(s) and/or open/update the PR.

If Greptile is unavailable (auth, outage, CLI missing), say so and get explicit
user confirmation before pushing without a review.

## What “relevant” means

Relevant = would realistically break behavior, violate repo hard constraints in
`AGENTS.md`, weaken tests/security, or leave incorrect public docs for this change.

Not automatically relevant = preference-only style, speculative refactors outside
the diff, duplicate advice already addressed, or comments contradicted by existing
project conventions / tests.
