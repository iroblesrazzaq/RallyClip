# `/goal` prompt — streaming-pipeline optimization

Run Claude Code from the **`RallyClip-perf`** worktree, then paste the block below after
`/goal`. Detail lives in [LOOP.md](LOOP.md) / [PLAN.md](PLAN.md) / [JOURNAL.md](JOURNAL.md).

```text
GOAL: Make RallyClip's release pipeline (CLI run_pipeline + GUI _run_pipeline) stop
round-tripping intermediates through .npz on disk and keep peak memory bounded on long videos
(streaming/chunking) — without changing produced segments or breaking tests. Work autonomously
through the backlog until done.

RULES
- Work ONLY in /Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf
  (branch perf/streaming-pipeline). NEVER touch feat/first-release or any other checkout.
- $PY = /Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip/.venv-train/bin/python3.
- Read docs/perf/LOOP.md, PLAN.md, JOURNAL.md first and FOLLOW LOOP.md EXACTLY — it defines the
  per-iteration runbook, the gate commands, the backlog (A1→A6 then B1→B5), and the 4h resume.

EACH ITERATION: pick the next single backlog item; make the smallest behaviour-preserving
change (mirror any release-path wiring in scripts/perf/bench_pipeline.py::run_stub; never alter
the frozen goldens). Run the gate. KEEP only if tests pass, features+segments sha at 6k ==
baseline_6000, serialization not worse (→0 for IO items), peak_rss@27k ≤ baseline×1.10, and
total@6k ≤ baseline×1.10 — else `git restore -SW .`. On KEEP: commit + append a JOURNAL.md entry
and iterations.jsonl line (hypothesis, params, before/after metrics, tests, correctness,
decision, next idea). Run the bench_real.py milestone gate at the end of Phase A and Phase B.
ALWAYS journal.

SURVIVE LIMITS: at the start of each batch, re-arm a durable ~4h resume — CronList/CronDelete
any old one, then CronCreate {recurring:false, durable:true, cron from
`date -v+4H "+%M %H %d %m"` as "M H DoM Mon *", prompt: "Resume the RallyClip streaming-pipeline
loop: cd /Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf, read
docs/perf/LOOP.md+JOURNAL.md, re-arm the 4h cron, continue until the A+B backlog is done,
journal each iteration, never touch feat/first-release, CronDelete the job and stop when done."}.

DONE: all A+B items kept (serialization ≈ 0, peak RSS bounded) + milestone gate passes → write
a final JOURNAL.md summary, CronDelete the survival job, stop. The user merges.
```
