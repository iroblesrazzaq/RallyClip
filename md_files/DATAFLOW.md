# FILE TO DOCUMENT HOW DATA IS TRACKED THROUGH PIPELINE

## Refactored Pipeline

### 1. extract_all.py
- Saves pose data in npz format
- Saves bounding boxes, keypoints, and keypoint confidence values to numpy arrays for EACH FRAME
- If a frame is not selected for annotation, SAVES EMPTY FRAME DATA TO NPZ FILE
- OUTPUT NPZ FILE HAS VIDEO FPS * LEN VIDEO SECONDS # ENTRIES
- Tracks annotation status: -100 (not annotated), 0 (annotated, not in play), 1 (annotated, in play)

### 2. tennis_preprocessor.py (TennisDataPreprocessor class)
- Takes raw YOLO NPZ files and applies court mask filtering
- Combines court filtering with data_processor methods for merging BBs and assigning players
- Saves to NPZ file with separate arrays:
  - All frame data (every frame in original video)
  - Target labels (-100, 0, 1)
  - Near/far player assignments
- Enables visualization to verify data correctness

### 3. tennis_feature_engineer.py (TennisFeatureEngineer class)
- Takes preprocessed NPZ files and creates feature vectors
- Only processes annotated frames (status >= 0)
- Creates individual feature vectors using data_processor methods
- Output: 
  - Feature array of shape (n_annotated_frames, 288)
  - Target array of shape (n_annotated_frames,)
  - Saved to organized directory structure

## Data Flow Summary

1. **Raw Inference** → 2. **Preprocessing (TennisDataPreprocessor)** → 3. **Feature Engineering (TennisFeatureEngineer)** → 4. **Training Ready**

Each stage is handled by a dedicated class with single responsibility, providing true modularity.