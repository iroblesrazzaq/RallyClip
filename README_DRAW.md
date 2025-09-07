# Draw Script

This script visualizes pose data from NPZ files on corresponding videos.

## Usage

### Process a single NPZ file:
```bash
python draw.py --npz-path <path_to_npz_file> [--start-time <seconds>] [--duration <seconds>]
```

### Process all NPZ files in a directory:
```bash
python draw.py --npz-dir <path_to_npz_directory> --draw-all [--start-time <seconds>] [--duration <seconds>]
```

## Features

### Raw NPZ Files
- Draws bounding boxes
- Draws keypoints
- Draws keypoint connections (limbs)
- Shows confidence values
- Processes all frames in the video

### Preprocessed NPZ Files
- Draws Player 1 (near player) in red
- Draws Player 2 (far player) in green
- Labels players clearly
- Draws keypoints and limb connections
- Overlays court mask (transparent red) if available
- Processes all frames in the video

## Output Structure

The output videos are saved in the `sanity_check_clips/` directory with the following structure:

```
sanity_check_clips/
├── raw/
│   └── {model_size}_{conf}_{start_time}_{duration}_{fps}/
│       └── draw_{npz_filename}.mp4
└── preprocessed/
    └── {model_size}_{conf}_{start_time}_{duration}_{fps}/
        └── draw_{npz_filename}.mp4
```

## Examples

Process a raw pose file for the first 30 seconds:
```bash
python draw.py --npz-path pose_data/raw/s_0.05_0_60_15/video_posedata_0s_to_60s_s15fps_0.05conf.npz --duration 30
```

Process a preprocessed file from 10s to 40s:
```bash
python draw.py --npz-path pose_data/preprocessed/s_0.05_0_60_15/video_preprocessed.npz --start-time 10 --duration 30
```

Process all raw files in a directory:
```bash
python draw.py --npz-dir pose_data/raw/s_0.05_0_60_15 --draw-all
```

## Requirements

- OpenCV (cv2)
- NumPy
- The corresponding video files must be in the `raw_videos/` directory