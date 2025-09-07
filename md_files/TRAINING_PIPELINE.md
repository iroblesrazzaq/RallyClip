# Tennis Point Detection - Training Pipeline

This document describes the updated training pipeline for the tennis point detection LSTM model.

## Updated Pipeline Overview

1. **Video Processing**: Extract pose data using `pose_extractor.py`
2. **Data Preprocessing**: Filter pose data and assign players using `preprocess_data_pipeline.py`
3. **Feature Engineering**: Convert preprocessed data to features using `create_features_pipeline.py`
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
preprocess_data_pipeline.py (preserves annotation_status)
    ↓
Preprocessed NPZ files with separate arrays:
- frames: All frame data (every frame in original video)
- targets: Annotation status for each frame (-100, 0, 1)
- near_players: Near player data for each frame
- far_players: Far player data for each frame
    ↓
create_features_pipeline.py (only processes annotated frames)
    ↓
Feature arrays ready for LSTM training
```

## Detailed Pipeline Steps

### 1. Pose Extraction
```bash
python pose_extractor.py start_time duration target_fps confidence_threshold video_path model_size annotations_path
```

### 2. Data Preprocessing
```bash
python preprocess_data_pipeline.py \\
    --input-dir pose_data/unfiltered/yolos_0.05conf_15fps_0s_to_60s \\
    --video-dir raw_videos \\
    --output-dir preprocessed_data
```

### 3. Feature Engineering
```bash
python create_features_pipeline.py \\
    --input-dir preprocessed_data \\
    --output-dir training_features
```

### 4. Model Training
```bash
python train_tennis_lstm.py \\
    --input-dir training_features \\
    --output-dir models
```

## Benefits of Updated Pipeline

1. **True Modularity**: Each stage has its own dedicated class with single responsibility
2. **Improved Data Organization**: Separate arrays in NPZ files for clarity
3. **Efficiency**: Feature engineering only processes annotated frames
4. **Visualization Ready**: Preprocessed data format enables easy visualization
5. **Debuggability**: Intermediate results saved at each stage for inspection