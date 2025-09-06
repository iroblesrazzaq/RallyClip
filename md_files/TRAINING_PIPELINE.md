# Tennis Point Detection - Training Pipeline

This document describes the updated training pipeline for the tennis point detection LSTM model.

## Updated Pipeline Overview

1. **Video Processing**: Extract pose data using `pose_extractor.py`
2. **Data Filtering**: Filter pose data using `filter_pose_data.py`
3. **Feature Engineering**: Convert filtered data to features using `prepare_training_data.py`
4. **Model Training**: Train the LSTM model using `train_tennis_lstm.py`

## Key Changes

### 1. Annotation Status Tracking

The updated pipeline now tracks annotation status for each frame:
- `-100`: Frame was skipped during pose extraction (should be ignored)
- `0`: Frame was processed but not in a point/play segment
- `1`: Frame was processed and is in a point/play segment

### 2. Data Flow

```
Video Files
    ↓
pose_extractor.py (with annotations CSV)
    ↓
NPZ files with annotation_status field
    ↓
filter_pose_data.py (preserves annotation_status)
    ↓
Filtered NPZ files with annotation_status
    ↓
prepare_training_data.py (uses annotation_status)
    ↓
Feature vectors (.npy) and status files (_status.npy)
    ↓
tennis_dataset.py (loads features and status)
    ↓
Training batches for LSTM model
```

## Usage

### 1. Extract Pose Data

```bash
python pose_extractor.py 0 60 15 0.05 /path/to/video.mp4 s /path/to/annotations.csv
```

### 2. Filter Pose Data

```bash
python filter_pose_data.py --input-dir pose_data/unfiltered/yolos_0.05conf_15fps_0s_to_60s --video-path /path/to/video.mp4
```

### 3. Prepare Training Data

```bash
python prepare_training_data.py --input pose_data/filtered/court_filtered_yolos_0.05conf_15fps_0s_to_60s --output training_data
```

### 4. Train Model

```bash
python test_files/train_tennis_lstm.py --data_dir training_data --model_save_path tennis_point_lstm.pth
```

## File Formats

### NPZ Files

Each NPZ file contains a 'frames' array with dictionaries for each frame:
- `boxes`: Bounding box coordinates
- `keypoints`: Keypoint coordinates
- `conf`: Keypoint confidence values
- `annotation_status`: Frame annotation status (-100, 0, or 1)

### Feature Files

- `_features.npy`: Feature vectors for training (only frames with status >= 0)
- `_status.npy`: Annotation status for each frame

### Dataset Loading

The `TennisDataset` class loads both feature vectors and their corresponding annotation status,
using the status values directly as training labels.