# Tennis Data Pipeline Runner

This script orchestrates the complete tennis data processing pipeline using a single entry point.

## Configuration Files

Configuration files are stored in the `data_configs/` directory. Each config file is a JSON with the following structure:

```json
{
    "start_time": 0,
    "duration": 60,
    "fps": 15,
    "conf": 0.05,
    "model_size": "s",
    "videos_to_process": "annotated",
    "steps_to_run": ["extractor", "preprocessor", "feature_extractor"],
    "overwrite": false,
    "save_court_masks": true
}
```

### Configuration Parameters

- `start_time`: Start time in seconds for video processing
- `duration`: Duration in seconds to process
- `fps`: Target frame rate for processing
- `conf`: Confidence threshold for YOLO detection
- `model_size`: YOLO model size (n, s, m, l)
- `videos_to_process`: Which videos to process
  - `"annotated"`: Only videos with corresponding CSV files in `annotations/`
  - `"all"`: All videos in `raw_videos/`
  - `[video_list]`: Specific video filenames as an array
- `steps_to_run`: Which pipeline steps to execute
  - `"extractor"`: Run pose extraction
  - `"preprocessor"`: Run data preprocessing
  - `"feature_extractor"`: Run feature engineering
- `overwrite`: Whether to overwrite existing files
- `save_court_masks`: Whether to save court masks in preprocessed NPZ files (optional)

### Example Configurations

1. `config1.json`: Process annotated videos with all steps and save court masks
2. `config_all_videos.json`: Process all videos with all steps, no court masks
3. `config_selected_videos.json`: Process specific videos with only extraction
4. `config_stepwise.json`: Process annotated videos with stepwise execution and save court masks

## Usage

```bash
python run_pipeline.py --config data_configs/config1.json
```

## Directory Structure

The pipeline creates a consistent directory structure based on the configuration parameters:

```
pose_data/
├── raw/
│   └── {model_size}_{conf}_{start_time}_{duration}_{fps}/
├── preprocessed/
│   └── {model_size}_{conf}_{start_time}_{duration}_{fps}/
└── features/
    └── {model_size}_{conf}_{start_time}_{duration}_{fps}/
```

This structure enables stepwise execution - you can run the extractor, then later run the preprocessor on the already extracted data.

## Court Mask Handling

When `save_court_masks` is set to `true` in the config:
- Successfully generated court masks are embedded in the preprocessed NPZ files
- Failed court detection does not stop processing (falls back to processing without filtering)
- Court masks are saved as a `court_mask` field in the NPZ file
- When `save_court_masks` is `false` or omitted, no court masks are saved