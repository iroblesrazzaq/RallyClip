# Streaming-pipeline optimization — loop runbook

This file is the recurring instruction set for the `/loop` skill. **Each `/loop` tick =
exactly one iteration below.** State lives in `JOURNAL.md` + git, so every tick re-reads
this file and the journal and continues where the last left off.

## How to start it

From the `RallyClip-perf` worktree:

```
/loop Read docs/perf/LOOP.md and execute exactly ONE iteration, then stop.
```

(No interval → the loop self-paces. It stops on its own when the backlog is exhausted or a
hard blocker is hit — see "Stopping".)

Environment: use the project venv
`/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip/.venv-train/bin/python3`
(referred to below as `$PY`). Always run `$PY -m pytest ...` and `$PY scripts/perf/...`.

## One iteration

### 0. Orient (no edits yet)
- Read the tail of `JOURNAL.md` → current iteration N, current best metrics, what's pending.
- Read `PLAN.md` backlog. Pick the **next single** unstarted item (A1→A2→…→A6, then B1→…→B5).
  One backlog item per iteration. If mid-item from last tick, continue it.
- If iteration 0 (journal has only the baseline row): commit the scaffold first
  (`git add -A && git commit -m "perf: streaming-pipeline scaffold (harness, baselines, loop docs)"`)
  so later iterations have a clean revert point. Then proceed to item A1.

### 1. State the hypothesis
Write (in your working notes for this tick): the backlog item, **what** you'll change, **why**
it preserves behaviour, and **which metric** it should move (IO / memory / both). This goes
into the journal at step 6 regardless of outcome.

### 2. Implement the smallest correct change
- Edit only files needed for this backlog item. Match surrounding style.
- Behaviour-preserving refactors only: the in-memory core and the file wrapper must produce
  identical arrays. Keep the file-writing wrappers working (training `scripts/` use them).
- If the change alters how the **release path** wires stages, mirror that wiring in
  `scripts/perf/bench_pipeline.py::run_stub` so the harness measures the path you changed.
  (You may NOT change the golden baselines or the correctness hashing — that's the anti-cheat.)

### 3. Run the gate
```
# correctness + IO + total-time (fast)
$PY scripts/perf/bench_pipeline.py --frames 6000  --json /tmp/iter.json
# memory (long-video) + scaling pair
$PY scripts/perf/bench_pipeline.py --frames 9000  --json /tmp/iter_9k.json
$PY scripts/perf/bench_pipeline.py --frames 27000 --json /tmp/iter_27k.json
# targeted tests (no slow/e2e)
$PY -m pytest -q -m "not slow and not e2e" \
  tests/test_pose_extractor_contract.py \
  tests/test_runtime_preprocessing_contract.py \
  tests/test_runtime_feature_contract.py \
  tests/test_features_builder.py \
  tests/test_features_v1.py \
  tests/test_preprocess.py \
  tests/test_player_assigner.py \
  tests/test_cli_pipeline_smoke.py \
  tests/test_cli_config_contract.py \
  tests/test_desktop_dispatch.py \
  tests/test_segment.py
```

### 4. Evaluate against the gate (all must hold to KEEP)
Compare `/tmp/iter*.json` to `docs/perf/baseline/baseline_*.json`:
1. **Tests** all pass.
2. **Correctness**: `correctness.features_sha256` AND `correctness.segments_sha256` at 6k
   equal the baseline_6000 golden. (Exact match — no tolerance.)
3. **IO**: `serialization.total_s` and `total_mb` ≤ baseline (strictly lower for IO-removal
   items; for non-IO items, not higher).
4. **Memory**: `peak_rss_delta_mb` at 27k ≤ baseline_27000 × 1.10. For Phase B items, it
   should *drop*, and (peak@27k / peak@9k) should trend toward ~1.
5. **Total time**: `elapsed_total_s` at 6k ≤ baseline_6000 × 1.10.

RSS is noisy: if a memory check is marginal (within ±10%), re-run that bench once and use the
better of the two before deciding.

### 4b. Milestone gate (ONLY at end of Phase A and end of Phase B — skip otherwise)
The per-iteration gate is stub-only (fast). At a phase boundary, also run the real pipeline
on a long 1080p `testing_app` clip and confirm the win is real:
```
TA=/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip/raw_video/testing_app
$PY scripts/perf/bench_real.py --video "$TA/1_utr9_genM_courtH_IN_angleMED_zoomIN.mp4" --json /tmp/real_1.json   # ~15-20 min, CPU
```
Require vs `docs/perf/baseline/real_1_full.json`:
- `correctness.segments_csv_sha256` **unchanged** (real segments identical),
- `serialization.writes` strictly down (3 → fewer; → 0 after Phase A),
- `peak_rss_delta_mb` ≤ real baseline (after Phase B: materially lower).
If short on time, a bounded check (`--duration 300`) is acceptable for IO/writes, but the
memory claim needs a full-length run. Journal the real numbers in the milestone entry.

### 5. Decide
- **KEEP**: `git add -A && git commit -m "perf(<item>): <one-line what> [io -Xs, rss -YMB]"`.
  If this iteration improved on the *current best*, note the new best numbers in the journal.
  Phase milestones (end of A: IO≈0; end of B: memory flat) may refresh
  `docs/perf/baseline/` ONLY for the perf metrics — never the correctness hashes.
- **REVERT**: `git restore -SW . && git clean -fd scripts/perf docs/perf 2>/dev/null` (or
  `git checkout -- <files>`), leaving the tree at the last good commit. Record why it failed.

### 6. Journal (always, KEEP or REVERT)
Append one entry to `JOURNAL.md` (human table row) and one line to `iterations.jsonl`
(structured). Use the schema in JOURNAL.md. Must include: iteration, backlog item, hypothesis,
params (bench frames, test subset), metrics before/after, test result, correctness match,
decision + commit sha or revert reason, and the next idea. **This is the required
"track params + thoughts every iteration" record — never skip it.**

### 7. Stop this tick.
Let `/loop` schedule the next one.

## Stopping (end the loop)
Stop scheduling further ticks when any of:
- All A + B backlog items are KEPT (IO ≈ 0 and memory bounded), OR
- A backlog item fails the gate **3 ticks in a row** for the same reason (hard blocker —
  journal it and hand back to the user), OR
- A change would require touching `feat/first-release` or anything outside this worktree.

On stop: write a final `JOURNAL.md` summary (before→after table, commits, what's left,
suggested merge/rebase step back onto `feat/first-release`) and hand back to the user. The
user runs the final real-video acceptance and merges.

## Survive usage limits (4h self-resume)
The model can't see its own token budget, so instead of detecting the cutoff we keep a resume
armed in advance. At the **start of each work batch**:
1. `CronList` → if a prior survival job exists, `CronDelete` it.
2. Compute ~4h from now off the `:00`/`:30` marks: `date -v+4H "+%M %H %d %m"` → `"M H DoM Mon *"`.
3. `CronCreate { recurring: false, durable: true, cron: "<that>", prompt: "<RESUME PROMPT>" }`.

RESUME PROMPT (single arg to the cron):
> Resume the RallyClip streaming-pipeline loop. cd
> /Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf, read
> docs/perf/LOOP.md + docs/perf/JOURNAL.md, re-arm the 4h survival cron, and continue
> iterations until the Phase A+B backlog is done. Journal every iteration. Never touch
> feat/first-release. When complete, CronDelete the survival job and stop.

If the session is cut off mid-work, the pending durable cron fires ~4h later (after the usage
window resets, whenever Claude Code is next running/idle) and continues. **When the backlog is
complete, `CronDelete` the survival job** so it doesn't keep firing.

## Guardrails
- Never edit `feat/first-release` or `/private/tmp/rallyclip-first-release`.
- Never edit the golden baselines' correctness hashes or `bench_pipeline.py`'s hashing/stub
  in a way that weakens the correctness check.
- One backlog item per tick. Smallest correct change. Commit per kept iteration so revert is cheap.
- If unsure whether a change preserves behaviour, it doesn't — split it smaller.
