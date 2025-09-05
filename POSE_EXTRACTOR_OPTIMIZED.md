# Optimized Pose Extractor

This is an optimized version of the pose extractor that leverages Apple MPS parallel inference capabilities by processing frames in batches.

## Key Improvements

1. **Batch Processing**: Instead of processing one frame at a time, this version processes multiple frames in batches to take advantage of the parallel processing capabilities of Apple MPS.

2. **Memory Efficient**: Uses a chunked processing approach to balance memory usage with batch efficiency.

3. **Better Progress Reporting**: More accurate progress updates during processing.

## Usage

The optimized pose extractor can be used in the same way as the original:

```bash
python pose_extractor_optimized.py [start_time] [duration] [target_fps] [confidence_threshold] [video_path] [model_size] [batch_size]
```

### Parameters

- `start_time`: Start time in seconds (default: 0)
- `duration`: Duration in seconds (default: 10)
- `target_fps`: Target frame rate for consistent temporal sampling (default: 15)
- `confidence_threshold`: Confidence threshold for pose detection (default: 0.05)
- `video_path`: Path to the input video file (default: "raw_videos/Monica Greene unedited tennis match play.mp4")
- `model_size`: YOLO model size (n, s, m, l) (default: s)
- `batch_size`: Number of frames to process in parallel (default: 8)

### Examples

```bash
# Default settings
python pose_extractor_optimized.py

# Process 30 seconds at 30 FPS with a medium model and batch size of 16
python pose_extractor_optimized.py 0 30 30 0.05 "path/to/video.mp4" m 16

# Process 60 seconds at 15 FPS with a large model and batch size of 4
python pose_extractor_optimized.py 0 60 15 0.05 "path/to/video.mp4" l 4
```

## Batch Processing

For processing all videos in the `raw_videos` directory, use the batch script:

```bash
python extract_all_optimized.py [start_time] [duration] [target_fps] [confidence_threshold] [model_size] [batch_size]
```

## Performance Benefits

On Apple M-series chips with MPS support, this optimized version should provide significant performance improvements over the original single-frame processing approach, especially when using larger batch sizes.

The optimal batch size depends on your specific hardware:
- For M2 MacBook Pro: Batch sizes between 8-16 typically provide good performance
- Larger batch sizes may cause memory issues
- Smaller batch sizes may not fully utilize the parallel processing capabilities

## Output

The output format is identical to the original pose extractor, ensuring compatibility with the rest of the pipeline. Files are saved with `_optimized` in the filename to distinguish them from the original version.