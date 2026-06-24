# Streaming-pipeline optimization — journal

One entry per `/loop` iteration. The loop **must** append here every tick (KEEP or REVERT).
This is the running record of params + thoughts. Newest at the bottom.

## Frozen correctness goldens (never change)
The refactor must reproduce these exactly at the matching frame count:

| frames | features_sha256  | segments_sha256  | n_seg |
|-------:|------------------|------------------|------:|
|  6,000 | 29d72f5f11c8cb19 | 980cc0ee139b33a9 |  120  |
|  9,000 | c7dce9e87e51cdf6 | 66fa566e573e54d6 |  180  |
| 27,000 | a8c09cb78e60142d | 173a354afa9665f5 |  540  |

## Perf baseline ("before", to beat)
| frames | peak RSS Δ (MB) | serialization (s / MB) | handoff (s) | total (s) |
|-------:|----------------:|-----------------------:|------------:|----------:|
|  6,000 |          71.16  |        0.4211 / 2.000  |     1.3102  |   1.3228  |
|  9,000 |          87.10  |        0.5760 / 2.975  |     2.3608  |   2.3779  |
| 27,000 |         201.80  |        1.5717 / 8.824  |     5.5492  |   5.6002  |

**Targets:** end of Phase A → serialization ≈ 0 (no intermediate NPZ). End of Phase B →
peak RSS at 27k ≈ peak RSS at 6k (memory bounded, flat vs length). Correctness hashes
unchanged throughout.

> Note: the model contract runs at **target_fps = 5**, so a ~30-min match ≈ **9,000**
> processed frames — the 9k stub row is the representative full-length point.

## Real baseline (1080p testing_app clip, full-length, CPU)
Honest "before" from the actual CLI `run_pipeline` (real YOLO/decode/court detection).
One clip is enough for the milestone gate. Source: `raw_video/testing_app/`.
File: `docs/perf/baseline/real_1_full.json`. (Frozen — the milestone gate's reference.)

| video | dur | peak RSS Δ (MB) | serialization (s / MB, w/r) | total (s) | segments | segments_csv_sha256 |
|-------|----:|----------------:|-----------------------------|----------:|---------:|---------------------|
| 1 (utr9, 1080p/60, 32min) | full | **518.9** | **2.199 / 20.783 (3w/3r)** | 1060.5 | 78 | `1bb060cc2debc42a` |

Reading: the 3 NPZ round-trips cost ~2.2s and ~20.8 MB of disk traffic; peak RSS 519 MB =
fixed floor (YOLO + onnxruntime + 1080p decode, ~360 MB) + ~155 MB of accumulated pipeline
data. **Phase A target:** writes 3 → 0, serialization → ~0, csv sha unchanged. **Phase B
target:** peak RSS drops toward the ~360 MB floor (data no longer fully materialized), csv
sha still unchanged. (Validation 60s-bound run earlier: 363 MB / 0.19s / sha be57c49f… —
consistent with the floor.)

## Entry schema (copy per iteration)

```
### Iteration N — <backlog item, e.g. A1> — <KEEP|REVERT> (<commit sha | reason>)
- when: <ISO ts>   base_commit: <sha>
- hypothesis: <what change + why behaviour-preserving + which metric it should move>
- params: bench frames {6000, 9000, 27000} @ 15fps; tests = gate subset (see LOOP.md)
- change: <files touched + one-line idea>
- metrics (after vs baseline):
  | frames | peak RSS | ser s | ser MB | handoff s | total s |
  |  6000  |  ..(..)  | ..    | ..     | ..        | ..      |
  | 27000  |  ..(..)  | ..    | ..     | ..        | ..      |
- tests: <PASS n / FAIL ...>
- correctness: features_sha match=<Y/N>  segments_sha match=<Y/N>
- decision: <KEEP commit <sha> | REVERT: <reason>>
- thoughts / next: <what this taught; next backlog item or refinement>
```

---

### Iteration 0 — baseline — KEEP (scaffold)
- when: 2026-06-22   base_commit: c0f61a2 (feat/first-release)
- hypothesis: n/a — establish the harness, golden correctness hashes, and perf baseline that
  every later iteration is gated against.
- params: stub harness, synthetic detections (2 players/frame), tiny frames; bench frames
  {6000, 9000, 27000} @ 15fps. Court mask = synthetic zeros (no YOLO). Determinism verified
  (features_sha + segments_sha stable across repeated runs).
- change: added `scripts/perf/bench_pipeline.py`, `docs/perf/baseline/baseline_*.json`,
  `docs/perf/{PLAN,LOOP,JOURNAL}.md`, `docs/perf/iterations.jsonl`. No `src/` changes.
- metrics: see "Perf baseline" table above.
- tests: gate subset PASS (53 passed).
- correctness: goldens recorded (see table). The 3 NPZ round-trips confirmed: 3 writes / 3
  reads per run; serialization + peak RSS both grow linearly with frame count.
- decision: KEEP (scaffold; commit on first loop tick).
- thoughts / next: start **A1** — give `PoseExtractor` an in-memory core so the raw pose NPZ
  write can be skipped on the release path. Watch `serialization.writes` drop 3→2 and the IO
  time fall; correctness hashes must stay identical.

### Iteration 1 — A1 (PoseExtractor in-memory core) — KEEP (88a431f)
- when: 2026-06-22T23:35   base_commit: 4a09808
- hypothesis: split `extract_pose_data` into a pure core `extract_pose_frames` (returns the
  per-frame data list — the exact object that was being serialized) + a thin save-wrapper.
  Behaviour-preserving (same list, same NPZ, same return path); enables A4 to hand the list to
  preprocess in memory. Should not move any metric this iteration (run_stub still calls the
  file wrapper); proves the core is byte-identical via the frozen shas.
- params: bench frames {6000, 9000, 27000} @ 15fps; tests = gate subset (11 files).
- change: `src/extraction/pose_extractor.py` — renamed body to `extract_pose_frames` (dropped
  `output_dir`, returns list); new `extract_pose_data` wrapper calls core then saves. No
  run_stub change (release wiring unchanged until A4).
- metrics: machine was under load (load avg ~3–4), so absolute baseline_*.json comparison is
  contaminated — everything (incl. unchanged code) ran ~50% slow. Did a same-machine A/B
  (stash parent vs mine) to isolate the change:
  | frames | PARENT rss/total | MINE rss/total | ser (both) |
  |  6000  | 79.0 / 1.21      | 71.1 / 1.22    | 0.38s w3/r3 |
  | 27000  | 308.0 / 5.99     | 258.3 / 5.53   | ~2.0s w3/r3 |
  → MINE ≤ PARENT on rss, total flat, serialization byte-identical. No regression from the change.
- tests: PASS (53 passed).
- correctness: features_sha match=Y (29d72f5f11c8cb19) segments_sha match=Y (980cc0ee139b33a9)
  at 6k; 27k shas also golden-identical.
- decision: KEEP commit 88a431f. (Absolute gate vs stale baseline fails only due to current
  machine load — A/B vs parent on the same host is the noise-controlled truth: perf-neutral.)
- thoughts / next: **A2** — `DataPreprocessor.preprocess_frames(pose_data, court_mask, src_wh)`
  pure core; make `preprocess_single_video` a load→core→save wrapper. Still no metric move
  expected (run_stub unchanged until A4); the 3→0 write drop lands at A4 when CLI + run_stub
  chain the cores in memory.
