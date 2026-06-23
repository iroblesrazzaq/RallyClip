# `/goal` prompt — streaming-pipeline optimization

Run Claude Code from the **`RallyClip-perf`** worktree, then paste everything in the fenced
block below after `/goal`. It drives the gated optimization loop autonomously, journals every
iteration, and re-arms a durable 4h resume so it survives usage-limit cutoffs.

Detailed runbook: [LOOP.md](LOOP.md) · backlog + gates: [PLAN.md](PLAN.md) · log: [JOURNAL.md](JOURNAL.md).

```text
GOAL: Make RallyClip's release inference pipeline (CLI run_pipeline + GUI _run_pipeline) stop
round-tripping intermediates through .npz on disk, and keep peak memory bounded on long videos
via streaming/chunking — WITHOUT changing produced segments and WITHOUT breaking tests. Work
autonomously through the backlog until done, committing each kept change.

WORKSPACE (hard rules)
- Work ONLY in: /Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf
  (branch perf/streaming-pipeline). NEVER edit feat/first-release or any other checkout.
- Python = /Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip/.venv-train/bin/python3
  (called $PY below). Source videos for the real gate are in
  /Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip/raw_video/testing_app/.
- Source of truth: docs/perf/LOOP.md (runbook), docs/perf/PLAN.md (backlog + gates),
  docs/perf/JOURNAL.md (state). Read them first and FOLLOW LOOP.md EXACTLY.

LOOP (one backlog item per iteration; repeat until DONE)
1. Read JOURNAL.md tail for current state. Pick the next single unstarted item: A1→A2→…→A6,
   then B1→…→B5 (see PLAN.md).
2. Implement the smallest behaviour-preserving change. If you change how the release path wires
   stages, mirror it in scripts/perf/bench_pipeline.py::run_stub. Do NOT alter the frozen
   correctness goldens or the bench's hashing/stub (that's the anti-cheat).
3. Gate (run all):
   $PY scripts/perf/bench_pipeline.py --frames 6000  --json /tmp/iter.json
   $PY scripts/perf/bench_pipeline.py --frames 9000  --json /tmp/iter_9k.json
   $PY scripts/perf/bench_pipeline.py --frames 27000 --json /tmp/iter_27k.json
   $PY -m pytest -q -m "not slow and not e2e" tests/test_pose_extractor_contract.py
     tests/test_runtime_preprocessing_contract.py tests/test_runtime_feature_contract.py
     tests/test_features_builder.py tests/test_features_v1.py tests/test_preprocess.py
     tests/test_player_assigner.py tests/test_cli_pipeline_smoke.py
     tests/test_cli_config_contract.py tests/test_desktop_dispatch.py tests/test_segment.py
4. KEEP only if ALL hold (else REVERT with `git restore -SW .` and journal why):
   - all tests pass;
   - features_sha256 AND segments_sha256 at 6k == docs/perf/baseline/baseline_6000.json (exact);
   - serialization.total_s/total_mb not worse than baseline (→ ~0 for IO-removal items);
   - peak_rss_delta_mb at 27k ≤ baseline_27000 × 1.10 (RSS is noisy; re-run once if marginal);
   - elapsed_total_s at 6k ≤ baseline_6000 × 1.10.
5. MILESTONE gate (only at end of Phase A and end of Phase B): 
   $PY scripts/perf/bench_real.py --video "<testing_app/1_...mp4>" --json /tmp/real.json
   Require segments_csv_sha256 == docs/perf/baseline/real_1_full.json, serialization.writes
   strictly down (→ 0 after Phase A), peak_rss_delta_mb ≤ real baseline (lower after Phase B).
6. On KEEP: git commit -m "perf(<item>): <what> [io -Xs, rss -YMB]". ALWAYS append one
   JOURNAL.md entry + one iterations.jsonl line: iteration, item, hypothesis, params (bench
   frames, test subset), before/after metrics, test result, correctness match, decision (sha or
   revert reason), next idea.
7. Continue to the next item.

DONE when all A+B items are KEPT (serialization ≈ 0 and peak RSS bounded) and the milestone
gate passes. Then write a final JOURNAL.md summary (before→after table, commits, suggested
rebase onto feat/first-release), CronDelete the survival job, and stop. The user merges.

SURVIVE USAGE LIMITS (do this at the START of every batch, before working)
- CronList; if a prior survival job exists, CronDelete it.
- Compute ~4h ahead off the :00/:30 marks: run `date -v+4H "+%M %H %d %m"` → "M H DoM Mon *".
- CronCreate { recurring: false, durable: true, cron: "<that>", prompt:
  "Resume the RallyClip streaming-pipeline loop. cd
   /Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf, read
   docs/perf/LOOP.md + docs/perf/JOURNAL.md, re-arm the 4h survival cron, and continue
   iterations until the Phase A+B backlog is done. Journal every iteration. Never touch
   feat/first-release. When complete, CronDelete the survival job and stop." }
- If cut off mid-work, the pending durable cron fires ~4h later (after the usage window resets,
  when Claude Code is running/idle) and resumes. Re-arm it at the start of each batch; delete it
  when the backlog is complete.

GUARDRAILS: one backlog item per iteration; smallest correct change; commit per kept iteration
so revert is cheap; never weaken the correctness goldens; never edit outside the worktree.
```
