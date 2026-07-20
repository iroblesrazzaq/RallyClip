# PROGRESS — overwrite me at every session end

_Last updated: 2026-09-09 (session: assisted DMG update, Greptile cancel + checksum)._

_Last updated: 2026-07-19 (session 2: pose contract landed; **YOLO26 reversed → YOLOv8 + CoreML** after ANE benchmarks; unified training_data root; normalize running)._

> **Supersedes below:** the YOLO26 items in this file describe infrastructure that
> survives, but the active bundle is now `models/pose/yolov8n/` (v26 bundle
> deleted, exports parked in ../YOLO-ONNX/exports/). Training extract runs
> `provider: coreml` (~90 fps, sub-pixel parity). See DECISIONS.md 2026-07-19
> session-2 entry and ../TODO.md §2b for the measurements and rationale.

- `main` has published app **v0.5.0**. Inference artifact stays
  `artifact-rallyclip_v0.5.0`.
- Active work: assisted update (`cursor/assisted-dmg-update-5f28`) as app **0.5.1**.
- PR: https://github.com/iroblesrazzaq/RallyClip/pull/58

## Session 2 (2026-07-19, later) — what shipped (uncommitted)

1. **Unified corpus root `../training_data/`** — source_videos/ (27 symlinks: 16 new_data
   matches + 11 old originals; `c31e6888` excluded — video but zero new_data JSONs),
   annotations/ (27, converted new_data labels + old JSONs), raw_videos/ (normalize output).
   new_data merge audit clean (eca4 176→146 = pid dedup). Flips deferred.
2. **YOLO26n pose contract** — `models/pose/yolo26n/` bundle: ONNX @960 dynamic
   (exported via `../YOLO-ONNX/.venv-export`, ultralytics 8.4.102, opset 20) +
   `manifest.json` (head family, letterbox, COCO-17 kpt names, shas). Parity vs
   `.pt` ≤1e-4 px.
3. **Runner** — `decode_yolo26_e2e` ([1,300,57], NMS in graph) + `decode_pose`
   shape dispatch (56→v8 raw, 57→e2e) in `src/extraction/yolo_onnx_runner.py`.
4. **Backend loader** — `src/extraction/pose_backend.py`: manifest → (model, meta),
   sha-verified; cache tag = `name@sha8` (`yolo26n-960@116bf9a9`).
5. **Extractor** — `YoloHdf5Extractor` takes manifest path; ONNX path is torch/
   ultralytics-free; imgsz from manifest; HDF5 attrs get model sha + contract
   version + head family. `base.yaml` → `yolo.model: models/pose/yolo26n/manifest.json`.
6. **Decision: CPU ORT only for YOLO26** (no CoreML for now — e2e NMS-in-graph
   partitioning risk; revisit post-extract).
7. Tests: `tests/test_pose_backend.py` (12) + full suite green (264 passed; one
   GUI latency benchmark fails only under ffmpeg load).

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

**Next:** build dataset (holdout overlay) → commit the working tree →
train classic + e2e heads on the new corpus → benchmark vs bundled v0.3.1.

## Repo / worktree state

| Checkout | Branch | Notes |
|---|---|---|
| **This tree (`RallyClip-perf/`)** | `feat/desktop-auto-update` @ `e1c32a7` (tracks `origin/main`) | **Active.** Large uncommitted training-pipeline land (normalize, new_data ingest, e2e head, holdout, tests). |
| Sibling `../RallyClip/` | `docs` @ `3729f84` | Primary clone; parked. Local data/models only. |
| Sibling `../rallyclip-prod/` | separate repo | Modal cloud experiments — out of scope. |
| Sibling `../YOLO-ONNX/` | separate repo | Pose ONNX lab (YOLO26 + v8@960 parity). |

Shipping desktop still on **YOLOv8n ONNX@960** + optional CoreML EP (static 544×960). Training extract still Ultralytics `.pt` until YOLO26 contract lands.

Handoff for other chats: `../CHAT_HANDOFF.md`. Data/training backlog: `../TODO.md`.

## What shipped this session (uncommitted in this worktree)

1. **new_data labels (container `new_data/`)** — deterministic point segments for singles_matches; `ignore_before_s`; check videos; docs in `sv/SINGLES_MATCHES_LABELING.md`.
2. **Normalize stage** — 1280×720 @ 5 fps (`src/training/normalize/`, `scripts/normalize_videos.py`); `preprocess.target_fps: 5`.
3. **new_data label ingest** — `scripts/convert_new_data_labels.py` → annotations + ignore prefix.
4. **Ignore targets / flip guard / cache fingerprints / loud stage failures / eval fixes / determinism / data-derived `pos_weight`.**
5. **E2E segment head** — `train.head: classic | e2e_seg`; seg_loop/loss/lstm/evaluator/segments6; fps from preprocess.
6. **Holdout** — `configs/train/holdout.yaml` + sweep overlap hard-error.
7. **Tests** — unit/smoke/characterization (~43); parity script `scripts/parity_check_normalize_yolo.py`.

## Decisions locked this session

- Training pose backend target: **YOLO26n-pose ONNX** (not Ultralytics `.pt` forever; not keep-v8-for-train). Desktop stays v8 until retrain.
- Contract ownership: export/parity in `YOLO-ONNX/`; bundled manifest + runner + training extract in this repo (`models/pose/yolo26n/`, `src/extraction/`, `src/training/pose/`).
- Existing `yolo26n-pose.onnx` at 640×640 is **not** the corpus contract — re-export at training imgsz required.

## Next steps (in order)

1. **YOLO26n ONNX contract** — manifest, e2e decode in runner, wire `YoloHdf5Extractor`, keypoint mapping check, fingerprint, parity gate (`../TODO.md` §2b).
2. Export YOLO26 @ 960 rect (+ static CoreML sibling when ready); re-extract corpus under new cache tag.
3. Commit robust-training + YOLO26 extract wiring; keep desktop on v8 until new model artifact.
4. Retrain classic + e2e on YOLO26 features with holdout frozen.
5. Optional later: split `pipeline.py` / ArtifactLayout (deferred Phase 6).

## Open mess (user callout)

Training pipeline / worktrees feel “all over the place.” Source of truth for code is **this worktree**; source of truth for new_data labels is **`../new_data/`**; pose lab is **`../YOLO-ONNX/`**. Consolidate extract defaults so train and (future) desktop share one YOLO26 contract after retrain.
