# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-09 (session: assisted DMG update, Greptile cancel + checksum)._

_Last updated: 2026-07-19 (session 2: pose contract landed; **YOLO26 reversed → YOLOv8 + CoreML** after ANE benchmarks; unified training_data root; normalize running)._

_Last updated: 2026-07-19 evening (session 2: v8 pose contract, unified data root,
corpus extract, frozen split, classic retrain + benchmarks; e2e run in flight)._

Full handoff: `../../CHAT_HANDOFF.md`. Backlog + idea log: `../../TODO.md`.
Why-log: `docs/DECISIONS.md` (3 entries dated 2026-07-19).

- `main` has published app **v0.5.0**. Inference artifact stays
  `artifact-rallyclip_v0.5.0`.
- Active work: assisted update (`cursor/assisted-dmg-update-5f28`) as app **0.5.1**.
- PR: https://github.com/iroblesrazzaq/RallyClip/pull/58

## Session 2 (2026-07-19, later) — what shipped (uncommitted)

## Git state

Branch `feat/desktop-auto-update`, clean tree, 4 local commits (NOT pushed):

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

## What shipped today

1. **Pose backend**: YOLO26 evaluated then DITCHED (CoreML fp16 breaks its e2e head,
   30–180 px; v8-static-CoreML = 90 fps sub-pixel). Manifest backend system in
   `models/pose/yolov8n/` + `src/extraction/pose_backend.py`; provider cpu|coreml;
   identity tag `yolov8n-960@4a3fe0de`; extractor torch-free.
2. **Data**: container-level `../training_data/` with provenance `sources/` tree,
   flat symlink/annotation interface layers, `norm=1280x720@5fps` contract tag in
   paths.py. `RallyClip/data` deleted (archived); dup videos removed (md5-checked).
3. **Corpus**: 29 videos / 25.5 h / 2571 segs fully normalized→extracted (CoreML,
   ~55–65 fps)→preprocessed→featured. New sessions: a9051e (reclassified match,
   service practice), 3e5f (unscored split-serve, 121 segs, 45% in-point).
4. **Frozen split** in `configs/train/holdout.yaml`: test 6 (3 legacy + 3 new_data),
   val 3 by-video, train 20. new_data test subset = clean v0.3.1 benchmark.
5. **Dataset** `datasets/20260719_165056` (6615/970/1375 seqs of 100×362).
6. **Classic retrain** run `20260719_165056` (best ep4, val bal_acc .881, early stop ep9).
7. **Benchmarks (six-bin, new_data test)**: v0.3.1 12.9% acceptable → classic 28.6% →
   +swept hysteresis (.6/.45/σ2/2s) 30.0% → +offsets(−.25/−.25) 31.1%. Legacy test
   37.4→49.0% with offsets. Diagnosis: new_data residual = boundary VARIANCE (bias-correction
   doesn't move it) → e2e head is the lever. Scripts/outputs in `../training_data/benchmarks/`.

## In flight

- **e2e_seg training** `runs/20260719_e2e` (dataset symlinked to 20260719_165056),
  log `../training_data/train_e2e_20260719.log`, selection_metric=acceptable.
  When done: benchmark with `../training_data/benchmarks/bench_new_sv.py` pattern
  (swap RUN dir + seg decode) and compare vs classic 31.1% / v0.3.1 12.9%.

## Next (in rough order)

1. e2e benchmark + three-way comparison.
2. Postprocess extras: gap-merge knob in sweep; recall-leaning operating point.
3. Serve-convention fix experiment (ignore(-1) between-serve gaps in legacy) →
   rebuild dataset → retrain both heads.
4. FN>FP asymmetry: pos_weight multiplier / recall-weighted selection_metric.
5. If e2e boundary variance persists: outside-offset heads (TODO §3c).
6. Push branch / merge when a shippable artifact exists (bundle manifest v0.4).
