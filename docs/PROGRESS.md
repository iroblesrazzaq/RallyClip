# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-09 (session: assisted DMG update, Greptile cancel + checksum)._

_Last updated: 2026-07-19 (session 2: pose contract landed; **YOLO26 reversed → YOLOv8 + CoreML** after ANE benchmarks; unified training_data root; normalize running)._

_Last updated: 2026-07-19 evening (session 2: v8 pose contract, unified data root,
corpus extract, frozen split, classic retrain + benchmarks; e2e run in flight)._

_Last updated: 2026-07-20 (session 4: added the boundary-heatmap head e2e_heatmap;
it is the new best model on both new_data and legacy. Session-3 court/loader work below
is still uncommitted alongside it)._

## Session 4 — boundary-heatmap head (NEW BEST MODEL)

Added `train.head: e2e_heatmap` (twin startness/endness Gaussian heatmaps, BSN
family) as new files only — no edits to the classic or e2e_seg code paths. Built
after a critical review of the first plan changed four things: **hybrid decode**
(pointness-runs define segments, heatmaps refine edges via soft-argmax) as the
default instead of fragile peak-pair; **balanced-BCE** on the soft Gaussian as the
default loss (focal opt-in); **peak-NMS + segment-merge** in the decoder; and a
**val-swept** comparison (not a single default-param eval).

Result (val-swept, same 3 new_data + 3 legacy subsets as the other benches):

| Model | run_id | new_data test | Legacy test |
|---|---|---|---|
| classic, NO court | 20260719_165056 | 31.1% | 49.0% |
| classic, WITH court | 20260720_court_classic | 37.1% | 31.3% |
| e2e_seg, court | 20260720_court_e2e | 22.9% | 34.0% |
| **heatmap, court (hybrid)** | **20260720_court_heatmap** | **42.2%** | **46.3%** |

New best on new_data **and** recovers most of the legacy regression — only model strong on
both at once. FN low (new_data 1.4%, legacy 13.9%): the hybrid decode preserved recall.
Still overfits early (best epoch 2), bad_seg still ~29% (boundary precision is the
residual error). New files: `models/heatmap_lstm.py`, `train/heatmap_loss.py`,
`eval/heatmap_evaluator.py`, `train/heatmap_loop.py`; +tests `tests/test_heatmap_loss.py`
(11, all pass; full suite 293 pass). Bench: `benchmarks/bench_court_heatmap.py`.
Next levers: soft-argmax TIME loss term (currently shape-only), σ/focal sweep,
shorter patience, serve-convention label fix (§3b-ii) before trusting exact numbers.

---

_Session 3 notes (still current — court + loader work, all uncommitted):_

Full handoff: `../../CHAT_HANDOFF.md`. Backlog + idea dump: `../../TODO.md`.
Why-log: `docs/DECISIONS.md` (session-3 block dated 2026-07-20).

- `main` has published app **v0.5.0**. Inference artifact stays
  `artifact-rallyclip_v0.5.0`.
- Active work: assisted update (`cursor/assisted-dmg-update-5f28`) as app **0.5.1**.
- PR: https://github.com/iroblesrazzaq/RallyClip/pull/58

## Session 2 (2026-07-19, later) — what shipped (uncommitted)

## Git state

Branch `feat/desktop-auto-update`. **9 files modified, UNCOMMITTED** (session-3 work):
- `src/preprocessing/court_detector_impl.py` — manifest-aware `_load_yolo`; middle-
  anchored outward-expanding `extract_clean_frame` + helpers (`_detect_person_boxes`,
  `_boxes_to_mask`, `_quad_iou`, `_homography_to_base`); `process_video(target_time=None)`.
- `src/preprocessing/data_preprocessor.py` — middle-first `_court_sample_times`; midpoint anchor.
- `src/training/courts/cache.py` — `target_time` Optional (None=midpoint); multi-anchor retry; loud fail.
- `src/training/dataset/hdf5_dataset.py` — **in-memory load** (6 GB guard). THE speedup.
- `src/training/io/fingerprint.py` — `court_target_time` Optional → "middle" in hash.
- `src/training/pipeline.py` — Optional court_target_time; post-preprocess court-health summary.
- `src/training/preprocess/preprocessor.py` — Optional court_target_time; loud per-video no-mask warn.
- `configs/train/base.yaml` — `court.target_time: null`.
- `configs/train/holdout.yaml` — test set +`133663e7` (7 test videos now).

1. Latest app release is the newest published `v*` tag from `/releases`.
2. Frozen app downloads to a unique staging file, verifies SHA-256 (sidecar
   must name the DMG), then replaces `~/Downloads`.
3. `POST /api/update/cancel` stops the server-side transfer; the button
   becomes Cancel.
4. Source/GUI opens that release URL. App version **0.5.1**; ONNX stays
   `models/rallyclip_v0.5.0`.
5. Default gate: **321 passed, 6 skipped, 27 deselected**.

## Next steps

1. Greptile 5/5, then user merges.
2. Tag `v0.5.1` (do not retag `v0.5.0`; do not re-upload the ONNX zip).

**DONE (2026-07-19 pm):** normalize 27/27 → provenance layout
(`../training_data/sources/` + flat interface layers; RallyClip/data deleted,
dups removed) → norm= path tag in paths.py → **full corpus extract→preprocess→
features complete: 27/27/27**, no skips, under
`pose_data/norm=1280x720@5fps/yolo=yolov8n-960@4a3fe0de/conf=0p25/imgsz=960/`
(extract via CoreML EP; run log `../training_data/pipeline_run.log`).

- `a6c7e52` training: robust pipeline hardening + unified data layout
- `7196b5b` extraction: manifest-defined pose backends (YOLOv8n contract)
- `cd3f22f` docs: session decisions, progress, repo map updates
- `2e732a1` train: fix e2e_seg launch (seg_head key, data-derived pos_weight)

Prior local commits still unpushed: a6c7e52, 7196b5b, ecfe385, 2e732a1, d52084d.
**Nothing committed this session** — tests pass (see below); commit when ready.

## What happened this session

1. **Found + fixed a corpus-wide silent failure**: court detection had been failing on
   ALL videos (manifest handed to ultralytics). All prior training used NO court filter.
2. **Court detector rewrite** (user spec): midpoint anchor, outward homography expansion
   with IoU/person stop, multi-anchor retry, loud health reporting. 30/32 videos detect.
3. **HDF5 loader fix**: gzip chunks (228,7,23) made per-item reads ~470 s/epoch; in-memory
   load → ~30 s/epoch, bit-identical. Permanent speedup for all training.
4. **Re-ran the corpus with court filtering** and retrained classic + e2e; benchmarked.
5. **Added 3 new new_data videos earlier same day** → corpus 32 videos (133663e7→test, 59f28ea6
   + b7009388→train).

## Results (six-bin acceptable rate, each model val-swept; new_data subset = 3, legacy = 3)

| Model | run_id | new_data test | Legacy test |
|---|---|---|---|
| v0.3.1 shipped | (bundled) | 12.9% | — |
| classic, NO court | 20260719_165056 | 31.1% | 49.0% |
| **classic, WITH court** | **20260720_court_classic** | **37.1%** | 31.3% |
| e2e, court, patience 5 (best ep2) | 20260720_court_e2e | 22.9% | 34.0% |
| e2e, court, patience 10 (best ep9) | 20260720_court_e2e_p10 | 19.3% | 10.2% |

**Best model = classic + court (37.1% new_data).** Court filtering helped the new_data target domain
(+6) but regressed legacy (−18, FN-driven) — confounded with the train-set change; user
parked it (masks were legacy-tuned; new_data is the target). e2e underperforms and overfits
early; longer patience didn't help (run variance > patience effect). Benchmark scripts +
outputs in `../training_data/benchmarks/` (bench_court_classic.py, sweep_court_classic.py,
bench_court_e2e.py, bench_court_e2e_p10.py).

## Next (user's call — nothing running)

1. Decide the working model: classic+court is the current best; commit the session's code.
2. **Serve-convention label fix** (TODO §3b-ii) — poisons boundary supervision for any head.
3. **Gaussian startness/endness heatmap head** (TODO §3c, user idea 2026-07-20) — soft
   Gaussian boundary targets + soft-argmax decode; targets the persistent bad_seg failure.
4. (Parked) legacy court-mask regression — cap far-court masking / isolate court-vs-train.
5. Push branch / merge when a shippable artifact exists.

## Ops notes

- **Always wrap training in `caffeinate -ims`** — laptop sleep suspends (not kills) the
  process; stacked restarts clobber the run dir.
- Run: `cd RallyClip-perf && caffeinate -ims ~/anaconda3/bin/python train.py --config
  base.yaml,data_container.yaml,holdout.yaml[,<overlay>] --steps <steps>`.
- Force court recompute after a court change: `court.force: true` (cached npz is keyed by
  video stem and reused otherwise). Tests: `~/anaconda3/bin/python -m pytest tests/ -q`
  (282 pass minus heavy GUI/e2e).
