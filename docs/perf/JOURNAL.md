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

### Iteration 2 — A2 (DataPreprocessor in-memory core) — KEEP (b916097)
- when: 2026-06-23T22:08   base_commit: 0f853c1
- hypothesis: extract `preprocess_frames(pose_data, court_mask, src_w, src_h) -> dict` (the
  pure per-frame filter→assign→rescale transform) and make `preprocess_single_video` a
  load→core→save wrapper. Behaviour-preserving (same dict, same NPZ); per-frame independent
  given a fixed mask, so streamable in Phase B. No metric move expected (run_stub still calls
  the file wrapper until A4).
- params: bench frames {6000, 9000, 27000} @ 15fps; tests = gate subset (11 files).
- change: `src/preprocessing/data_preprocessor.py` — new `preprocess_frames` core; wrapper now
  loads NPZ, resolves mask + native res, calls core, saves. No run_stub change.
- metrics: machine load even higher (load avg ~6.5), so absolute baseline contaminated again;
  used same-machine A/B (stash parent vs mine):
  | frames | PARENT rss/total | MINE rss/total | ser (mine) |
  |  6000  | 83.9 / 1.34      | 74.3 / 1.24    | 0.37s w3/r3 |
  | 27000  | 283.0 / 5.60     | 285.1 / 5.43   | 1.56s w3/r3 |
  → MINE ≈ PARENT (27k rss +0.8% = noise, total better). No regression.
- tests: PASS (53 passed).
- correctness: features_sha + segments_sha match golden at 6k/9k/27k (Y/Y all three).
- decision: KEEP commit b916097.
- thoughts / next: **A3** — `FeatureEngineer.build_features(...)` pure core; make
  `create_features_from_preprocessed` a load→core→save wrapper. Then A4 wires all three cores
  in memory (CLI + run_stub) and serialization should drop 3w/3r → 0.

### Iteration 3 — A3 (FeatureEngineer in-memory core) — KEEP (47d189d)
- when: 2026-06-23T22:11   base_commit: 254e504
- hypothesis: extract `build_features(targets, near, far) -> (features, targets)` (pure feature
  matrix builder, prev-frame state only) and make `create_features_from_preprocessed` a
  load→core→save wrapper. Behaviour-preserving (same arrays, same NPZ); prev-frame-only state
  means streamable in Phase B. No metric move expected (run_stub file-based until A4).
- params: bench frames {6000, 9000, 27000} @ 15fps; tests = gate subset (11 files).
- change: `src/features/feature_engineer.py` — new `build_features` core; wrapper loads NPZ,
  calls core, saves. No run_stub change.
- metrics: load avg ~5; same-machine A/B:
  | frames | PARENT rss/total | MINE rss/total | ser (mine) |
  |  6000  | 78.9 / 1.24      | 84.6 / 1.27    | 0.37s w3/r3 |
  | 27000  | 283.9 / 5.43     | 270.9 / 5.43   | 1.55s w3/r3 |
  → 6k rss +7% = load noise (small abs); 27k (the memory point) MINE lower, total flat. No regression.
- tests: PASS (53 passed).
- correctness: features_sha + segments_sha match golden at 6k/9k/27k (Y/Y all three).
- decision: KEEP commit 47d189d.
- thoughts / next: **A4** — all three cores now exist. Rewire CLI `run_pipeline` to chain
  cores in memory (pose_frames → preprocess_frames → build_features → inference), dropping the
  raw-pose NPZ (and the persistent `pose_data/` write bug) + the two tmp NPZs. MIRROR the same
  wiring in `run_stub`. Expect serialization 3w/3r → 0w/0r, total_s/total_mb → ~0. Correctness
  shas must stay identical. This is the first iteration where the IO metric actually moves.

### Iteration 4 — A4 (CLI in-memory hand-off) — KEEP (9b66946)
- when: 2026-06-24T00:00   base_commit: 1935ec4
- hypothesis: rewire CLI `run_pipeline` to call the three cores and chain them in memory
  (extract_pose_frames → preprocess_frames → build_features → scaler → inference), dropping all
  intermediate NPZ round-trips + the `tempfile.TemporaryDirectory` + the persistent
  `pose_data/raw/` write bug. Mirror the same wiring in `run_stub`. Serialization should drop
  3w/3r → 0; correctness shas must stay identical (NPZ round-trip is lossless for float32, so
  in-memory == reloaded). First iteration where IO actually moves.
- params: bench frames {6000, 9000, 27000} @ 15fps; tests = gate subset (11 files).
- change: `src/cli/main.py` (chain cores, remove tempfile + np.load), `scripts/perf/
  bench_pipeline.py::run_stub` (mirror in-memory chain; hashing/stub untouched),
  `tests/test_cli_pipeline_smoke.py` (test was asserting the OLD NPZ wiring — now asserts the
  in-memory wiring + that np.load of intermediates never happens; renamed
  `..._closes_feature_npz` → `..._in_memory`).
- metrics (mine vs baseline; load avg ~4):
  | frames | ser before | ser after | rss before | rss after | total before | total after |
  |  6000  | 0.42s 3w/3r | **0.0s 0w/0r** | 71.2 | 43.6 | 1.32 | 0.80 |
  |  9000  | 0.58s 3w/3r | **0.0s 0w/0r** | 87.1 | 64.0 | 2.38 | 1.19 |
  | 27000  | 1.57s 3w/3r | **0.0s 0w/0r** | 201.8 | 126.4 (best of 2; 192 worst) | 5.60 | 3.55 |
  → serialization eliminated; peak RSS *dropped* too (savez/np.load buffer spikes gone), total down.
- gate checks: correctness all golden=Y; ser→0 (0w); rss@27k 126.4 ≤ 222.0 (×1.10) Y; total@6k 0.80 ≤ 1.455 Y.
- tests: PASS (53; smoke test updated to new wiring).
- correctness: features_sha + segments_sha match golden at 6k/9k/27k (Y/Y all three).
- decision: KEEP commit 9b66946.
- note on method: dropped the stash-based A/B — it collided with a pre-existing user stash
  (`release GUI changes`) and an interrupt left edits parked in a stash, producing a spurious
  w3 reading. Now measuring the working tree directly vs frozen baselines; ser=0 and golden
  shas are load-independent and unambiguous, so no A/B needed for an IO-removal item.
- thoughts / next: **A5** — GUI `_run_pipeline` (`src/gui/app.py` or `desktop.py`): same
  in-memory chain; drop the `job_dir` intermediate NPZs but KEEP the final outputs + saved-match
  library item. Then **A6** guard test (assert run_pipeline writes no intermediate .npz), then
  the Phase-A milestone real-video gate (`bench_real.py`, writes 3→0, csv sha unchanged).

### Iteration 5 — A5 (GUI in-memory hand-off) — KEEP (468b20f)
- when: 2026-06-24T00:20   base_commit: c370ac6
- hypothesis: rewire GUI `_run_pipeline` to chain the three cores in memory, dropping the
  raw/preprocessed/features NPZs written into `job_dir`. Keep progress callbacks, step/cancel
  handling, final outputs, and the saved-match library item. Core failures now propagate to the
  existing `except Exception` handler (same job-failed outcome as the old success-bool checks),
  so behaviour is preserved. No bench/run_stub change (run_stub already mirrors the in-memory
  chain from A4; GUI uses the same cores).
- params: bench frames {6000, 27000} @ 15fps; tests = gate subset (11 files).
- change: `src/gui/app.py` — `extract_pose_frames` → `_source_frame_shape` + `preprocess_frames`
  → `build_features`; removed the two `success_*`/`Path(...).exists()` checks (cores raise),
  the `np.load(features_npz)`, and the now-dead intermediate-NPZ cleanup loop. Verified import +
  py_compile; no leftover intermediate-NPZ references (only inert `None` placeholders in the job
  paths dict, which nothing reads).
- metrics: GUI isn't exercised by the stub bench, so bench reflects the shared cores: 6k/27k
  still golden, serialization 0w/0r, rss 43.7 / 191.7. (Real GUI write-elimination is covered by
  code review + the shared-core contract; the CLI milestone gate proves the cores write nothing.)
- tests: PASS (53 passed).
- correctness: features_sha + segments_sha match golden at 6k/27k (Y/Y).
- decision: KEEP commit 468b20f.
- thoughts / next: **A6** — add a guard test asserting `run_pipeline` (CLI) writes **no**
  intermediate `.npz` (sandbox cwd + assert no `*.npz` created and `pose_data/` never appears).
  Then the **Phase-A milestone** real-video gate via `bench_real.py` (writes 3→0, csv sha
  unchanged, peak RSS ≤ real baseline).

### Iteration 6 — A6 (guard test: no intermediate NPZ) — KEEP (71a3e34)
- when: 2026-06-24T00:35   base_commit: 66b30be
- hypothesis: a guard test pins the Phase-A design so a future change can't silently
  reintroduce intermediate NPZ writes on the release path.
- params: tests = gate subset (11 files, now 54 tests); bench 6k spot-check.
- change: `tests/test_cli_pipeline_smoke.py` — new `test_run_pipeline_writes_no_intermediate_npz`:
  fakes whose file-writing wrappers raise AssertionError + `np.savez_compressed` patched to
  raise + sandboxed cwd + assert no `*.npz` / no `pose_data/` on disk. (write_csv/segment_video
  off so the test is purely about intermediates.)
- metrics: n/a (test-only). Bench 6k unchanged: golden, ser 0.0s/0w.
- tests: PASS (54 passed; +1 guard).
- correctness: features_sha + segments_sha match golden at 6k (Y/Y).
- decision: KEEP commit 71a3e34.
- thoughts / next: **Phase-A milestone gate** — `bench_real.py` on testing_app clip 1; require
  segments_csv_sha256 == 1bb060cc2debc42a (unchanged), serialization.writes 3→0, peak RSS ≤
  518.9. Then start Phase B (B1: pose generator).

### Phase-A milestone — DEFERRED (user request, 2026-06-24)
The ~18-min real-video `bench_real.py` run was deferred at the user's request to proceed
straight to Phase B. Phase A's IO-removal is already proven at the stub level (serialization
3w/3r → 0w/0r, goldens unchanged) and pinned by the A6 guard test (release path makes no NPZ
write / no `pose_data/`). The real-video acceptance (csv sha 1bb060cc2debc42a unchanged, writes
3→0, peak RSS ≤ 518.9) is owed before merge — run it at the end of Phase B alongside the Phase-B
milestone, or the user runs it directly:
`$PY scripts/perf/bench_real.py --video "$TA/1_utr9_...mp4" --device cpu --json /tmp/real_1.json`.

### Iteration 7 — B1 (pose streaming generator) — KEEP (64728bc)
- when: 2026-06-24T00:55   base_commit: b8966bb
- hypothesis: convert `extract_pose_frames` into a generator `iter_pose_frames` that yields
  per-frame dicts in stream order while buffering only one detection batch (a `pending` chunk
  filled on flush, then `yield from` + `.clear()`), making pose peak memory O(batch). Keep
  `extract_pose_frames` = `list(iter_pose_frames(...))` so all current callers are unchanged.
  Behaviour-preserving (same sequence); no metric move yet (run_stub still materializes the
  list); correctness shas must stay identical.
- params: bench frames {6000, 9000, 27000} @ 15fps; tests = gate subset (11 files).
- change: `src/extraction/pose_extractor.py` — generator core + list wrapper; `_flush_batch`
  now fills `pending[pos]` (chunk-relative) instead of `all_frames_data[idx]` (global). The
  pending list is never rebound (mutated in place + .clear()) so the closure sees one object.
- metrics (mine; load avg ~4): 6k rss 43.6 / total 1.05; 9k rss 64.7 / 1.61; 27k rss 192.1 /
  4.79; serialization 0w (unchanged). Memory unchanged this iter by design (list still built);
  drop lands when the chain consumes the generator (later B wiring).
- tests: PASS (54 passed; pose contract incl. box_conf/imgsz/coords still green).
- correctness: features_sha + segments_sha == golden at 6k/9k/27k (Y/Y all three).
- decision: KEEP commit 64728bc.
- thoughts / next: **B2** — preprocess consumes the pose stream and yields per-frame
  preprocessed records (generator core), keeping `preprocess_frames` as the list-materializing
  wrapper. Stateless per frame given a fixed court mask, so a 1:1 streaming map.

### Iteration 8 — B2 (preprocess streaming generator) — KEEP (15da4f3)
- when: 2026-06-25T00:10   base_commit: 08c31d8
- hypothesis: add `iter_preprocess_frames(pose_stream, court_mask, src_w, src_h)` yielding one
  `(frame_data, target, near_player, far_player)` record per frame (per-frame independent → O(1)
  peak); make `preprocess_frames` materialize it into the dict. The release feature path needs
  only target/near/far, so it can later consume records directly. Behaviour-preserving; no
  metric move yet (wrapper still builds the lists).
- params: bench frames {6000, 9000, 27000} @ 15fps; tests = gate subset (11 files).
- change: `src/preprocessing/data_preprocessor.py` — generator core + list-materializing
  wrapper. Debug log dropped the `len(pose_data)` total (stream has no length); cosmetic only.
- metrics (mine): 6k rss 37.6 / total 1.07; 9k 51.4 / 1.62; 27k 168.7 / 4.85; ser 0w.
- tests: PASS (54). correctness: golden Y/Y at 6k/9k/27k.
- decision: KEEP commit 15da4f3.
- thoughts / next: **B3** — features consume the preprocessed stream keeping only prev-frame
  state, emit rows. Add `iter_build_features` / a stream-consuming core; keep `build_features`
  as the array-materializing wrapper.

### Iteration 9 — B3 (feature streaming generator) — KEEP (3913812)
- when: 2026-06-25T00:30   base_commit: e0173dd
- hypothesis: add `iter_build_features(preprocessed_records)` that skips target<0 frames,
  retains only the previous kept frame as state, and yields `(feature_vector, target)`; make
  `build_features` materialize it over the parallel arrays. State advances only across kept
  frames, exactly as the old `annotated_indices` loop. Behaviour-preserving; no metric move yet.
- params: bench frames {6000, 9000, 27000} @ 15fps; tests = gate subset (11 files).
- change: `src/features/feature_engineer.py` — generator core + array-materializing wrapper
  (wrapper feeds synthetic `(None, target, near, far)` records).
- metrics (mine): 6k rss 42.8 / total 1.08; 9k 64.8 / 1.60; 27k 192.0 / 4.83; ser 0w.
- tests: PASS (54). correctness: golden Y/Y at 6k/9k/27k.
- decision: KEEP commit 3913812.
- thoughts / next: **B4** — the three streaming cores now exist (iter_pose → iter_preprocess →
  iter_build_features). Make inference consume features incrementally (fill seq_len windows,
  accumulate summed_probs/counts; retain only O(num_frames) floats) AND wire run_pipeline +
  run_stub to chain the generators end-to-end. THIS is where peak RSS should drop (27k → ~6k).
  Need to read infer/inference.py run_windowed_inference_average* to design the streaming
  windowing without changing the averaged-probs result (gate: segments_sha unchanged).
