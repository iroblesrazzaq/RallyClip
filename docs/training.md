# Training Pipeline

This repo includes a modular, step-based training pipeline driven by a YAML config (default: `configs/train/base.yaml`).

## Data layout (default `data_root: data`)
```
data/
  raw_videos/            # input videos (.mp4, .mov, ...)
  annotations/           # per-video JSON labels
  pose_data/
    raw/                 # YOLO outputs (HDF5)
    courts/              # court masks (.npz)
    preprocessed/        # downsampled + filtered detections (HDF5)
    features/            # feature vectors (HDF5)
  datasets/<run_id>/     # train/val/test splits + scaler + manifest
  runs/<run_id>/         # checkpoints + metrics
  visualizations/<run_id>/ # overlay videos (debug)
```

Annotation files are expected at `data/annotations/<video_filename>.json` where `<video_filename>` includes the extension (e.g., `match1.mp4.json`).

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
```bash
python train.py --config configs/train/base.yaml
```
Override steps:
```bash
python train.py --config configs/train/base.yaml --steps extract,preprocess,features
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
python visualize.py --config configs/train/base.yaml --stage yolo --video match1.mp4
python visualize.py --config configs/train/base.yaml --stage court --video match1.mp4
python visualize.py --config configs/train/base.yaml --stage preproc --video match1.mp4
```

Outputs land in `data/visualizations/<run_id>/` and use the source video FPS.

## Output artifacts
- `data/pose_data/raw/.../*.h5`: raw YOLO detections per frame.
- `data/pose_data/preprocessed/.../*.h5`: downsampled per-frame detections + targets.
- `data/pose_data/features/.../*.h5`: feature vectors + targets.
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
