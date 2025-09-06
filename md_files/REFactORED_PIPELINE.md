# Tennis Point Detection - Refactored Pipeline

This document describes the refactored data processing pipeline for tennis point detection.

## Pipeline Overview

The refactored pipeline consists of three main stages, each with its own dedicated class:

### 1. Original YOLO Inference
- **Purpose**: Extract raw pose data from videos using YOLOv8-pose
- **Output**: NPZ files with raw YOLO inference results
- **Key Features**: 
  - Preserves all frames from original video
  - Tracks annotation status for each frame (-100, 0, 1)
  - Debuggable - can inspect raw YOLO output

### 2. Data Preprocessing (`tennis_preprocessor.py`)
- **Class**: `TennisDataPreprocessor`
- **Purpose**: Apply court filtering and player assignment
- **Input**: Raw YOLO NPZ files + corresponding video files
- **Output**: Preprocessed NPZ files with:
  - All frame data (every frame from original video)
  - Target labels (-100, 0, 1)
  - Near/far player assignments
  - Separate arrays for clarity and modularity
- **Key Features**:
  - Court mask filtering to remove detections outside playable area
  - Player assignment (near/far) using data_processor heuristics
  - Preserves frame alignment with original video
  - Enables visualization for debugging

### 3. Feature Engineering (`tennis_feature_engineer.py`)
- **Class**: `TennisFeatureEngineer`
- **Purpose**: Create final feature vectors for training
- **Input**: Preprocessed NPZ files
- **Output**: 
  - Feature arrays: (n_annotated_frames, 288)
  - Target arrays: (n_annotated_frames,)
  - Only for annotated frames (status >= 0)
- **Key Features**:
  - Comprehensive feature engineering (bounding boxes, keypoints, velocities, accelerations, limb lengths)
  - Handles missing data appropriately (-1 values)
  - Ready for LSTM training

## Usage

### 1. Original YOLO Inference
```bash
# Using existing pose_extractor.py
python pose_extractor.py 0 60 15 0.05 /path/to/video.mp4 s /path/to/annotations.csv
```

### 2. Data Preprocessing
```bash
python preprocess_data_pipeline.py \
    --input-dir pose_data/unfiltered/yolos_0.05conf_15fps_0s_to_60s \
    --video-dir raw_videos \
    --output-dir preprocessed_data
```

### 3. Feature Engineering
```bash
python create_features_pipeline.py \
    --input-dir preprocessed_data \
    --output-dir training_features
```

## File Formats

### Raw YOLO NPZ Files
Contains 'frames' array with dictionaries for each frame:
- `boxes`: Bounding box coordinates
- `keypoints`: Keypoint coordinates
- `conf`: Keypoint confidence values
- `annotation_status`: Frame annotation status (-100, 0, or 1)

### Preprocessed NPZ Files
Separate arrays for clarity:
- `frames`: All frame data (filtered)
- `targets`: Annotation status for each frame
- `near_players`: Near player data for each frame
- `far_players`: Far player data for each frame

### Feature NPZ Files
- `features`: Array of shape (n_annotated_frames, 288)
- `targets`: Array of shape (n_annotated_frames,)

## Key Improvements

1. **True Modularity**: Each stage has its own dedicated class
2. **Single Responsibility**: Each class handles one specific task
3. **Separation of Concerns**: Preprocessing and feature engineering are completely separate
4. **Debuggability**: Each stage saves intermediate results for inspection
5. **Visualization Ready**: Preprocessed data format enables easy visualization
6. **Efficiency**: Feature engineering only processes annotated frames
7. **Clarity**: Separate arrays in NPZ files for better organization

## Class Interfaces

### TennisDataPreprocessor
```python
class TennisDataPreprocessor:
    def __init__(self, screen_width=1280, screen_height=720)
    def generate_court_mask(self, video_path)
    def filter_frame_by_court(self, frame_data, mask)
    def assign_players_to_frame(self, frame_data)
    def preprocess_single_video(self, input_npz_path, video_path, output_npz_path, overwrite=False)
```

### TennisFeatureEngineer
```python
class TennisFeatureEngineer:
    def __init__(self, screen_width=1280, screen_height=720, feature_vector_size=288)
    def create_feature_vector(self, assigned_players, previous_assigned_players=None)
    def create_features_from_preprocessed(self, input_npz_path, output_file, overwrite=False)
```