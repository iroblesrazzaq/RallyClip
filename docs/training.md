# Training Pipeline

This repo includes a modular, step-based training pipeline driven by a YAML config (default: `configs/train/base.yaml`).

## Data layout (default `data_root: data`)
```
data/
  source_videos/         # input videos (.mp4, .mov, ...); older roots: raw_videos/
  videos/norm=<WxH@fps>/ # normalized copies the pipeline works on
  annotations/           # per-video JSON labels
  pose_data/norm=<WxH@fps>/
    raw/                 # YOLO outputs (HDF5)
    courts/              # court masks (.npz)
    preprocessed/        # downsampled + filtered detections (HDF5)
    features/            # feature vectors (HDF5)
  datasets/<run_id>/     # train/val/test splits + scaler + manifest
  runs/<run_id>/         # checkpoints + metrics
  visualizations/<run_id>/ # overlay videos (debug)
```

Annotation files are expected at `data/annotations/<video_filename>.json` where `<video_filename>` includes the extension (e.g., `match1.mp4.json`).
Raw/preprocessed/features are nested by YOLO model, confidence, and `imgsz` (plus target FPS where relevant).

## Annotation format
```json
{
  "video_path": "data/raw_videos/match1.mp4",
  "segments": [
    {"start_time": 12.34, "end_time": 23.45, "label": "in_play"}
  ],
  "metadata": {
    "surface": "hard",
    "indoor": false
  }
}
```

Convert existing CSVs:
```bash
python scripts/convert_annotations.py --data-root data
```

## Pipeline steps
The pipeline is broken into explicit steps so you can re-run only what you need:

1. `extract`: YOLO pose inference to raw HDF5.
2. `preprocess`: downsample to target FPS, filter with court mask, assign near/far players.
3. `features`: feature engineering (registry-based, default `v1`).
4. `dataset`: build train/val/test sequences, fit scaler, write manifest.
5. `train`: train model, log metrics, checkpoint best/last.
6. `eval`: evaluate on test split.

## Quickstart
Install the training extra first (`uv sync --extra cpu --extra train`), then:
```bash
uv run python train.py --config configs/train/base.yaml
```
Override steps:
```bash
uv run python train.py --config configs/train/base.yaml --steps extract,preprocess,features
```

## Useful scripts
- `scripts/extract_yolo.py`: run YOLO extraction only.
- `scripts/cache_court_masks.py`: cache court masks only.
- `scripts/preprocess_yolo.py`: preprocess only.
- `scripts/build_features.py`: feature engineering only.
- `scripts/build_dataset.py`: dataset builder only.
- `scripts/convert_annotations.py`: CSV -> JSON conversion.
- `scripts/visualize_overlays.py`: debugging overlays.

## Visualization overlays
```bash
uv run python visualize.py --config configs/train/base.yaml --stage yolo --video match1.mp4
uv run python visualize.py --config configs/train/base.yaml --stage court --video match1.mp4
uv run python visualize.py --config configs/train/base.yaml --stage preproc --video match1.mp4
```

Outputs land in `data/visualizations/<run_id>/` and use the source video FPS.

## Output artifacts
- `data/pose_data/raw/yolo=<model>/conf=<conf>/imgsz=<imgsz>/*.h5`: raw YOLO detections per frame.
- `data/pose_data/preprocessed/yolo=<model>/conf=<conf>/imgsz=<imgsz>/fps=<fps>/*.h5`: downsampled per-frame detections + targets.
- `data/pose_data/features/yolo=<model>/conf=<conf>/imgsz=<imgsz>/fps=<fps>/*.h5`: feature vectors + targets.
- `data/datasets/<run_id>/train.h5`, `val.h5`, `test.h5`.
- `data/datasets/<run_id>/scaler.joblib`.
- `data/datasets/<run_id>/dataset_manifest.json`.
- `data/runs/<run_id>/checkpoints/` and `data/runs/<run_id>/metrics.jsonl`.

## Config highlights
- `data_root`: base data directory.
- `videos`: optional explicit list (overrides per-step `videos`).
- `overwrite_all`: force regenerate outputs for all steps.
- `preprocess.target_fps`: downsample rate for training (default 15).
- `features.feature_set`: feature registry key.
- `dataset.seq_len_seconds` + `overlap_seconds`: sequence windowing in seconds.
- `dataset.split.strategy`: `by_video`, `within_video`, or `hybrid`.
- `train.device`: `cuda`, `mps`, or `cpu` (auto-picks if null).
- `yolo.imgsz`: input resize for YOLO (higher = more compute).
- `train.head`: `classic` (BiLSTM per-frame logit + hysteresis) or `e2e_heatmap`
  (TCN pointness + start/end heatmaps, v0.6). The heatmap head's val/test
  decode is the runtime's (`infer.heatmap_decode`), plus a training-only
  `pairdp` mode.
- `preprocess.slot_mode`: `classic` (full-frame near/far) or `netline`
  (2-slot singles gated on the DL court detector's net line; needs
  `preprocess.court_geometry_dir`).

## Reproducing v0.6.0
`configs/train/v060_netline.yaml` holds the shipped recipe (TCN c64 L5 k3,
cosine lr 1e-4, net-line slots, `v2_nearfar`). Overlay it on `base.yaml` and
set `dataset.split` for the run.

Net-line slots need two inputs the pipeline steps do not create:
- far-crop poses from `scripts/extract_crop_poses.py`, run after `extract`
  has written the raw poses. It works on a 1920x1080 canvas, which is why the recipe normalizes at 1080p.
  Preprocess refuses a crop side-car whose frame grid or frame size differs
  from the raw poses.
- court geometry: one DL-court-detector JSON per video in
  `preprocess.court_geometry_dir`. The detector run that wrote
  `training_data_1080p/courts_dl` is not in this repo.

So run it in two passes:
```bash
uv run python train.py --config configs/train/base.yaml,configs/train/v060_netline.yaml --steps normalize,extract
uv run python scripts/extract_crop_poses.py --data-root ../training_data_netline
uv run python train.py --config configs/train/base.yaml,configs/train/v060_netline.yaml --steps preprocess,features,dataset,train,eval
```
The CV and final-model drivers (`cv_netline.py`, `final_train.py`,
`decode_sweep.py`) live next to the data in `training_data_netline/`, not in
this repo.
