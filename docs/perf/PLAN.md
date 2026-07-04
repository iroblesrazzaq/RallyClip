# Streaming-pipeline optimization — plan & backlog

**Branch:** `perf/streaming-pipeline` (worktree, based on `feat/first-release`).
**Goal:** make the RallyClip *release* inference pipeline (CLI `run_pipeline` + GUI
`_run_pipeline`) stop round-tripping intermediates through `.npz` on disk, and keep peak
memory bounded on long videos via streaming/chunking — **without** changing the produced
segments and **without** breaking tests.

> ⚠️ Do not touch `feat/first-release` — it is being edited elsewhere. All work happens on
> `perf/streaming-pipeline` in this worktree and is merged back later.

## The problem (today)

Both call sites run the same staged sequence and serialize intermediates between stages:

```
compute_court_mask(video)            -> mask            (in memory ✓)
extract_pose_data(video)             -> writes raw .npz (object array: per-frame dicts)
preprocess_single_video(raw.npz)     -> writes preprocessed .npz (object arrays)
create_features_from_preprocessed()  -> writes features .npz (float32 matrix)
np.load(features.npz) -> inference -> smooth -> hysteresis -> segments
CSV + segment_video                  (legitimate final output)
```

Three `np.savez_compressed` → `np.load` round-trips of data that is immediately re-consumed
in the same process. Pure overhead: pickle + zlib + disk write + disk read + decompress.
Bonus bug: in the **CLI**, `extract_pose_data` with no `output_dir` writes the raw pose NPZ
to a *persistent* `pose_data/raw/...` dir (litters the repo), not even tmp.

Measured today (stub harness, see below):

| frames | peak RSS Δ | serialization | handoff time |
|-------:|-----------:|--------------:|-------------:|
|  6,000 |    ~71 MB  | 0.42s / 2.0MB |        1.3s  |
|  9,000 |    ~87 MB  | 0.58s / 3.0MB |        2.4s  |
| 27,000 |   ~202 MB  | 1.57s / 8.8MB |        5.5s  |

Both memory and IO scale linearly with video length. The model contract runs at
**target_fps = 5**, so a 30-min match ≈ **9,000 processed frames** — the **9k row is the
representative full-length point**; 6k/27k bracket it for a scaling check.

**Real memory test:** the honest test uses the two long 1080p/60fps/~30-min clips in
`raw_video/testing_app/` (`1_*.mp4`, `2_*.mp4`). They are higher-res and longer than
`data/raw_videos/`, exercise the 1080p→720p rescale path, real (variable) detection counts,
and real decode buffers — none of which the stub captures. Run them via
`scripts/perf/bench_real.py` (real CLI `run_pipeline`, real YOLO on CPU). This is the
**milestone/acceptance** gate, not per-iteration (≈18 min/run on CPU). One clip is enough;
real "before" baseline: `docs/perf/baseline/real_1_full.json` (peak RSS 519 MB, ser
2.2s/20.8 MB, 3w/3r, 78 segments, csv sha 1bb060cc2debc42a).

## Target architecture

Classic "extract the pure transform, make I/O a thin shell". Each stage gets an **in-memory
core**; the existing file-writing methods become thin wrappers (load → core → save) kept for
the training `scripts/`. Then the release path streams.

### Phase A — in-memory hand-off (kill disk IO), behaviour-preserving
- **A1** `PoseExtractor`: pure core returning the in-memory frame list (no NPZ). Existing
  `extract_pose_data(...)` becomes `core + np.savez_compressed` wrapper.
- **A2** `DataPreprocessor`: `preprocess_frames(pose_data, court_mask, src_wh) -> dict`. Make
  `preprocess_single_video(...)` a load→core→save wrapper.
- **A3** `FeatureEngineer`: `build_features(targets, near, far) -> (features, targets)`. Make
  `create_features_from_preprocessed(...)` a load→core→save wrapper.
- **A4** CLI `run_pipeline`: call the cores; drop the `tempfile` NPZs; fixes the persistent
  `pose_data/` write bug.
- **A5** GUI `_run_pipeline`: call the cores; drop the `job_dir` intermediate NPZs (keep the
  final outputs + saved-match library item).
- **A6** Guard test: assert `run_pipeline` writes **no** intermediate `.npz`.

### Phase B — streaming / bounded memory
The feature step needs only the *previous* frame; inference windows need only a `seq_len`
buffer — so the whole chain is streamable and peak memory becomes `O(seq_len)`, independent
of video length.
- **B1** Pose extraction yields per-batch results (generator) instead of accumulating the
  full `all_frames_data` list.
- **B2** Preprocess consumes the stream, yields per-frame preprocessed records (stateless
  given a fixed court mask).
- **B3** Features consume the preprocessed stream, keep prev-frame state, emit feature rows
  into a pre-allocated array (or straight into windowed inference).
- **B4** Inference consumes features incrementally: fill `seq_len` windows, run, accumulate
  `summed_probs`/`counts`; retain only `O(num_frames)` floats. Smoothing + hysteresis stay
  global (cheap, 1 float/frame).
- **B5** Confirm peak RSS at 27k ≈ peak RSS at 6k (bounded); refresh the memory golden.

### Phase C — optional (scope guard; only if A+B land cleanly)
- **C1** Avoid decoding the video twice (court detection decodes separately from the pose
  pass). Bigger change; note, don't auto-start.

## Gates (every iteration)

A change is **KEPT** only if all hold; otherwise **REVERT**:
1. **Tests green** — targeted subset (see LOOP.md), excluding `slow`/`e2e`.
2. **Correctness** — `bench --frames 6000` `features_sha256` **and** `segments_sha256`
   equal the frozen golden (`docs/perf/baseline/baseline_6000.json`). This is the anti-cheat:
   the golden was computed from the original pipeline and never changes.
3. **IO down, not up** — `serialization.total_s`/`total_mb` ≤ baseline (→ ~0 by end of A).
4. **Memory non-regression** — `peak_rss_delta` at 9k (and 27k) ≤ baseline × 1.10 (RSS is
   noisy; use the ratio, not equality). Phase B milestone: 9k→27k growth ratio approaches ~1.
5. **No total-time regression** — `elapsed_total_s` at 6k ≤ baseline × 1.10.

**Milestone gate** (end of Phase A, end of Phase B — not every iteration): run
`bench_real.py` on a `testing_app` clip and require: segment-CSV sha == real baseline,
`serialization.writes` strictly down (3 → fewer, → 0 after A), and `peak_rss_delta` ≤ real
baseline. After Phase B, peak RSS on the full clip should drop materially (data no longer
fully materialized).

## Harness & goldens
- `scripts/perf/bench_pipeline.py` — **per-iteration gate.** Drives the real stage code on a
  synthetic, length-scalable workload (stub YOLO, tiny frames so peak RSS reflects *data*).
  Fast + deterministic. Emits JSON.
- `docs/perf/baseline/baseline_{6000,9000,27000}.json` — frozen stub "before" goldens.
- `scripts/perf/bench_real.py` — **milestone/acceptance gate.** Runs the actual CLI
  `run_pipeline` on a real video (real YOLO/decode/court detection), measuring the same
  metrics + a segment-CSV hash. Use the `raw_video/testing_app/` clips. ≈15–20 min/run (CPU).
- `docs/perf/baseline/real_1_full.json` — frozen real "before" baseline (full-length).
- Final acceptance (before hand-back): re-run `bench_real.py` on the testing_app clip and
  confirm segment-CSV sha unchanged, serialization writes → 0, and peak RSS bounded/lower.

See **LOOP.md** for the per-iteration runbook and **JOURNAL.md** for the running log.
