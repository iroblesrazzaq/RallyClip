# Pose Data Directory Structure Fix Summary

## Problem Identified
The tennis tracker project had inconsistent directory naming with both "raw" and "unfiltered" directories being created, leading to redundancy and confusion in the data pipeline.

## Root Cause Analysis
1. The `run_pipeline.py` script was creating directories with "unfiltered" naming convention
2. The `pose_extractor.py` script was hard-coded to save files to "unfiltered" directory
3. This mismatch caused files to be saved in different locations, breaking the pipeline flow

## Fixes Applied

### 1. Updated `data_scripts/run_pipeline.py`
- Modified `create_directory_structure()` function to consistently use "raw" instead of "unfiltered"
- Updated directory path construction to ensure all pipeline stages use the same base directory

### 2. Updated `data_scripts/pose_extractor.py`  
- Changed hard-coded directory creation from `"pose_data/unfiltered"` to `"pose_data/raw"`
- Maintained the same subdirectory naming convention for consistency

### 3. Cleaned Up Redundant Directories
- Removed the redundant "unfiltered" directory structure
- Preserved all existing data in the standardized "raw" directory

## Results
- ✅ Consistent directory structure using "raw" naming throughout the pipeline
- ✅ Eliminated redundant "unfiltered" directories
- ✅ Proper file organization with standardized naming conventions
- ✅ Seamless integration with downstream pipeline components (preprocessing, feature engineering)
- ✅ All 7 test videos processed successfully with 0 failures

## Directory Structure Now
```
pose_data/
├── raw/
│   └── yolos_0.05conf_15fps_0s_to_10s/
│       ├── Video1_posedata_0s_to_10s_yolos.npz
│       ├── Video2_posedata_0s_to_10s_yolos.npz
│       └── ... (7 total files)
├── preprocessed/
└── features/
```

This fix ensures that all pose extraction data flows consistently through the pipeline using the standardized "raw" directory structure.