# Training Pipeline TODO

This file defines the training pipeline plan for RallyClip. The goal is to move
all training code into this repo (on a new branch) with a clean, modular, and
repeatable workflow that supports full-fps YOLO extraction, flexible frame-rate
downsampling, multiple split strategies, robust metrics, and W&B tracking.

## Goals
- Keep a strict separation between YOLO extraction, preprocessing, feature
  engineering, dataset creation, training, and evaluation.
- Cache heavy steps (YOLO outputs and court masks) and allow fast re-runs.
- Support flexible train/val/test split strategies (by video, within video,
  hybrid).
- Track metrics (frame + segment) and experiments (W&B).
- Save all artifacts needed to fully reproduce a run.
- Provide manual visualization outputs for QA.

## Proposed data layout (repo-local)
data/
  raw_videos/
  annotations/
    <video>.csv              # manual labels (start/end)
    <video>.json             # canonical manifest (generated from CSV)
  metadata/
    videos.jsonl             # per-video metadata and provenance
  pose_data/
    raw/                     # YOLO outputs at native fps (HDF5)
      <video>.h5
    courts/
      <video>.npz            # cached court mask + metadata
      court_log.jsonl        # per-video success/failure log
    preprocessed/
      <video>__fps15.h5
    features/
      <video>__fps15__v1.h5
  datasets/
    <run_id>/
      train.h5
      val.h5
      test.h5
      scaler.joblib
      dataset_manifest.json
  runs/
    <run_id>/
      config.yaml
      metrics.jsonl
      checkpoints/
      artifacts/
  visualizations/
    <run_id>/
      <video>__yolo.mp4
      <video>__court.mp4
      <video>__preproc.mp4

## Annotation format (canonical JSON)
- Keep CSV as source input, but normalize to JSON for training.
- JSON fields (example):
  {
    "video_path": "data/raw_videos/foo.mp4",
    "segments": [
      {"start_time": 12.3, "end_time": 18.9, "label": "in_play"}
    ],
    "metadata": {
      "surface": "hard",
      "indoor_outdoor": "outdoor",
      "gender": "mens",
      "level": "college",
      "camera_angle": "baseline",
      "court_color": "blue"
    }
  }

## Split strategy support
- by_video: full videos held out for val/test.
- within_video: temporal splits within each video.
- hybrid: hold out some videos for test, split remaining videos into train/val.
- All split strategies are configured per run with a fixed random seed.

## Metrics
Frame-level:
- accuracy
- balanced_accuracy
- f1
- auroc

Segment-level:
- precision, recall, f1 (IoU-matched segments)
- mean_iou
- coverage (fraction of true time covered by predicted segments)
- specificity (true negative rate, as "leave out" quality)

## Experiment tracking
- Primary: W&B (enabled only when config says so).
- Local fallback: JSONL logs saved in data/runs/<run_id>/metrics.jsonl.

## Checkpointing
- Save "best" checkpoint by val balanced accuracy (configurable).
- Save "last" checkpoint and optional periodic checkpoints.
- Persist model weights, scaler, config, and run metadata (git commit hash,
  dataset manifest, feature set, split strategy).

## New modules and files
- train.py (repo root): config-driven CLI entrypoint for the pipeline.
- visualize.py (repo root): config-driven visualization entrypoint.
- scripts/extract_yolo.py: standalone YOLO-to-HDF5 extraction.
- scripts/cache_court_masks.py: standalone court mask cache runner.
- scripts/convert_annotations.py: CSV -> JSON annotation converter.
- scripts/preprocess_yolo.py: HDF5 preprocessing runner.
- scripts/build_features.py: feature builder runner.
- scripts/build_dataset.py: dataset builder runner.
- src/training/
  - io/annotations.py            # CSV -> JSON, JSON loader
  - io/metadata.py               # per-video metadata registry
  - pose/pose_extractor.py       # full-fps YOLO extraction -> HDF5
  - courts/cache.py              # court mask cache + logging
  - preprocess/preprocessor.py   # downsample, filter, player assignment
  - features/registry.py         # feature set registry
  - features/v1.py               # current feature engineering
  - dataset/builder.py           # offline HDF5 dataset creation
  - dataset/splits.py            # split strategies
  - dataset/hdf5_dataset.py      # in-memory dataset loader
  - models/lstm.py               # LSTM model definition
  - train/loop.py                # training loop + metrics + W&B
  - eval/evaluator.py            # evaluation on val/test
  - eval/checkpoint.py           # evaluate saved checkpoints
  - metrics/frame.py             # frame metrics
  - metrics/segment.py           # segment metrics
  - viz/renderer.py              # video overlay helpers
  - viz/stages.py                # yolo/court/preproc visualizers

## TODO checklist (implementation plan)
1) Branch and scaffolding
- [ ] Create branch: training-pipeline
- [ ] Add src/training/ package skeleton and __init__.py files
- [ ] Add train.py CLI stub with config parsing

2) Annotation and metadata
- [ ] Add CSV -> JSON converter (preserve CSV input)
- [ ] Add JSON schema validation
- [ ] Add metadata registry in data/metadata/videos.jsonl
- [ ] Create a small script to update metadata per video

3) YOLO extraction (native fps)
- [ ] Add HDF5 writer for per-frame YOLO outputs
- [ ] Store boxes, keypoints, confidences, timestamps, fps
- [ ] Ensure extraction is independent from downstream steps
- [ ] Add resumable mode (skip if already extracted)

4) Court detection cache
- [ ] Store mask + metadata in data/pose_data/courts/<video>.npz
- [ ] Log per-video success/failure to court_log.jsonl
- [ ] Expose cache load/save helpers

5) Preprocessing (downsampling + filtering)
- [ ] Load native-fps YOLO HDF5
- [ ] Downsample to target fps without re-running YOLO (timestamp-based schedule)
- [ ] Apply court mask filter + player assignment
- [ ] Save preprocessed HDF5 with per-frame labels
- [ ] Persist sampled frame indices + timestamps for visualization mapping

6) Feature engineering registry
- [ ] Implement FeatureSetRegistry (name -> class)
- [ ] Add FeatureSetV1 to match current feature vector
- [ ] Allow feature flags per run in config
- [ ] Save feature metadata with output

7) Dataset creation (offline default)
- [ ] Build sequences with overlap from feature HDF5
- [ ] Implement split strategies (by_video, within_video, hybrid)
- [ ] Fit scaler on train only, save scaler.joblib
- [ ] Save datasets to data/datasets/<run_id>/train|val|test.h5
- [ ] Emit dataset_manifest.json (videos, splits, metadata)
- [ ] Add optional on-the-fly Dataset class for research

8) Training loop
- [ ] Implement LSTM training loop with metrics
- [ ] Add W&B logging (optional)
- [ ] Add early stopping + checkpointing
- [ ] Save run config + git commit hash

9) Evaluation
- [ ] Frame-level eval on val/test
- [ ] Segment-level eval (hysteresis + IoU matching)
- [ ] Produce eval summary JSON and per-video breakdown

10) CLI and configs
- [ ] Add config templates in configs/train/*.yaml
- [ ] Support step selection (extract, preprocess, features, dataset, train, eval)
- [ ] Add dry-run mode to validate config and paths

11) Documentation
- [ ] Write docs/training.md with end-to-end usage
- [ ] Add examples for common workflows
- [ ] Document how to add new feature sets and models

12) Visualization (manual QA)
- [ ] Implement visualize.py with stage selection: yolo|court|preproc
- [ ] YOLO stage: draw boxes + keypoints + COCO skeleton lines
- [ ] Court stage: overlay mask (alpha) + optional detected lines
- [ ] Preproc stage: draw near/far player overlays with centroid, vx/vy, ax/ay, speed, accel
- [ ] Render at native fps; blank overlays on non-sampled frames
- [ ] Use stored sampled frame indices for deterministic mapping
