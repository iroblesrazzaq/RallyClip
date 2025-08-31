#!/usr/bin/env python3
"""
draw_all.py - Batch annotate all saved pose data files

This script finds all .npz pose data files in the pose_data directory and creates
annotated videos for each one using the video_annotator.py script.

The script automatically extracts fps and confidence threshold from the directory names
and passes them to the video_annotator.py script.

Usage:
    python draw_all.py [start_time] [duration] [model_size] [overwrite]
    
    start_time: Optional start time in seconds to filter files
    duration: Optional duration in seconds to filter files  
    model_size: Optional YOLO model size (n, s, m, l) to filter files
    overwrite: Optional boolean to overwrite existing annotated videos (default: False)
               Accepts: true, 1, yes, y (case insensitive)
    
    If only one argument is provided, it's treated as model_size.
    If three arguments are provided, they're treated as [start_time] [duration] [model_size].
    If four arguments are provided, they're treated as [start_time] [duration] [model_size] [overwrite].
    If no arguments are provided, processes all .npz files.

Examples:
    python draw_all.py                                    # Process all .npz files
    python draw_all.py s                                  # Process only small model files
    python draw_all.py m                                  # Process only medium model files
    python draw_all.py 0 60 s                             # Process small model files from 0s to 60s
    python draw_all.py 30 60 m                            # Process medium model files from 30s to 90s
    python draw_all.py 0 120 l                            # Process large model files from 0s to 120s
    python draw_all.py 0 60 s true                        # Process small model files from 0s to 60s, overwrite existing
    python draw_all.py 30 60 m yes                        # Process medium model files from 30s to 90s, overwrite existing
"""

import os
import glob
import subprocess
import time
import sys
import re


def get_npz_files(model_size=None, start_time=None, duration=None):
    """
    Get all .npz files from pose_data subdirectories, optionally filtered by model size and time range.
    
    Args:
        model_size (str, optional): Model size to filter by (n, s, m, l)
        start_time (int, optional): Start time in seconds to filter by
        duration (int, optional): Duration in seconds to filter by
        
    Returns:
        list: List of .npz file paths
    """
    pose_data_dir = "pose_data"
    
    if not os.path.exists(pose_data_dir):
        print(f"❌ Pose data directory '{pose_data_dir}' not found!")
        return []
    
    # Get all .npz files from subdirectories
    all_files = []
    for subdir in os.listdir(pose_data_dir):
        subdir_path = os.path.join(pose_data_dir, subdir)
        if os.path.isdir(subdir_path) and subdir.endswith('s'):  # New format ends with time range
            npz_pattern = os.path.join(subdir_path, "*.npz")
            all_files.extend(glob.glob(npz_pattern))
    
    filtered_files = all_files
    
    # Filter by model size
    if model_size:
        model_filtered = []
        for file_path in filtered_files:
            filename = os.path.basename(file_path)
            if f"_yolo{model_size}.npz" in filename:
                model_filtered.append(file_path)
        filtered_files = model_filtered
    
    # Filter by time range
    if start_time is not None and duration is not None:
        end_time = start_time + duration
        time_filtered = []
        for file_path in filtered_files:
            # Extract time range from subdirectory name
            subdir_name = os.path.basename(os.path.dirname(file_path))
            # Parse time range from subdirectory like "yolom_0.05conf_30s_to_90s"
            time_match = re.search(r'_(\d+)s_to_(\d+)s$', subdir_name)
            if time_match:
                file_start = int(time_match.group(1))
                file_end = int(time_match.group(2))
                if file_start == start_time and file_end == end_time:
                    time_filtered.append(file_path)
        filtered_files = time_filtered
    
    return filtered_files


def extract_video_info_from_filename(npz_path):
    """
    Extract video path, start time, duration, model size, fps, and confidence threshold from .npz filename and subdirectory.
    
    Args:
        npz_path (str): Path to .npz file
        
    Returns:
        dict: Dictionary with video_path, start_time, duration, model_size, target_fps, confidence_threshold
    """
    filename = os.path.basename(npz_path)
    subdir_name = os.path.basename(os.path.dirname(npz_path))
    
    # Extract model size from filename
    model_size_match = re.search(r'_yolo([nslm])\.npz$', filename)
    model_size = model_size_match.group(1) if model_size_match else 's'
    
    # Extract time range, fps, and confidence threshold from subdirectory name (new format)
    # Format: yolo{model_size}_{confidence_threshold}conf_{target_fps}fps_{start_time}s_to_{end_time}s
    time_match = re.search(r'_(\d+\.\d+)conf_(\d+)fps_(\d+)s_to_(\d+)s$', subdir_name)
    if time_match:
        confidence_threshold = float(time_match.group(1))
        target_fps = int(time_match.group(2))
        start_time = int(time_match.group(3))
        end_time = int(time_match.group(4))
    else:
        # Fallback to old format without fps
        time_match = re.search(r'_(\d+\.\d+)conf_(\d+)s_to_(\d+)s$', subdir_name)
        if time_match:
            confidence_threshold = float(time_match.group(1))
            target_fps = 15  # Default fps
            start_time = int(time_match.group(2))
            end_time = int(time_match.group(3))
        else:
            # Fallback to filename parsing (oldest format)
            time_match = re.search(r'_posedata_(\d+)s_to_(\d+)s_', filename)
            if not time_match:
                print(f"⚠️  Could not parse time range from filename or subdirectory: {filename}")
                return None
            confidence_threshold = 0.05  # Default confidence threshold
            target_fps = 15  # Default fps
            start_time = int(time_match.group(1))
            end_time = int(time_match.group(2))
    
    duration = end_time - start_time
    
    # Extract base video name (remove _posedata_... suffix)
    base_name = filename.split('_posedata_')[0]
    video_path = os.path.join("raw_videos", f"{base_name}.mp4")
    
    return {
        'video_path': video_path,
        'start_time': start_time,
        'duration': duration,
        'model_size': model_size,
        'target_fps': target_fps,
        'confidence_threshold': confidence_threshold,
        'npz_path': npz_path
    }


def run_annotation_command(video_path, start_time, duration, target_fps, confidence_threshold, model_size):
    """
    Run the video annotation command.
    
    Args:
        video_path (str): Path to video file
        start_time (int): Start time in seconds
        duration (int): Duration in seconds
        target_fps (int): Target frame rate
        confidence_threshold (float): Confidence threshold
        model_size (str): Model size
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = ["python", "video_annotator.py", str(start_time), str(duration), str(target_fps), str(confidence_threshold), video_path, model_size]
    
    print(f"🔄 Running: {' '.join(cmd)}")
    
    start_time_cmd = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time_cmd = time.time()
    
    if result.returncode == 0:
        print(f"✅ Success ({end_time_cmd - start_time_cmd:.2f}s)")
        return True
    else:
        print(f"❌ Failed: {result.stderr.strip()}")
        return False


def main():
    # Parse command line arguments
    if len(sys.argv) >= 5:
        # Format: python draw_all.py [start_time] [duration] [model_size] [overwrite]
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        model_size = sys.argv[3]
        overwrite = sys.argv[4].lower() in ['true', '1', 'yes', 'y']
    elif len(sys.argv) >= 4:
        # Format: python draw_all.py [start_time] [duration] [model_size]
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        model_size = sys.argv[3]
        overwrite = False
    elif len(sys.argv) >= 2:
        # Format: python draw_all.py [model_size]
        start_time = None
        duration = None
        model_size = sys.argv[1]
        overwrite = False
    else:
        # No arguments - process all files
        start_time = None
        duration = None
        model_size = None
        overwrite = False
    
    print("🎨 Batch Annotating All Pose Data Files")
    print("=" * 50)
    
    if model_size and start_time is not None and duration is not None:
        print(f"Model size filter: {model_size}")
        print(f"Time range filter: {start_time}s to {start_time + duration}s")
    elif model_size:
        print(f"Model size filter: {model_size}")
    elif start_time is not None and duration is not None:
        print(f"Time range filter: {start_time}s to {start_time + duration}s")
    else:
        print("Processing all files")
    
    print(f"Overwrite existing files: {overwrite}")
    
    # Get .npz files
    npz_files = get_npz_files(model_size, start_time, duration)
    
    if not npz_files:
        print("❌ No .npz files found!")
        if model_size and start_time is not None and duration is not None:
            print(f"   No files found for model size '{model_size}' and time range {start_time}s to {start_time + duration}s")
        elif model_size:
            print(f"   No files found for model size '{model_size}'")
        elif start_time is not None and duration is not None:
            print(f"   No files found for time range {start_time}s to {start_time + duration}s")
        return False
    
    print(f"\n📁 Found {len(npz_files)} .npz files:")
    for i, npz_file in enumerate(npz_files, 1):
        print(f"   {i}. {os.path.basename(npz_file)}")
    
    # Process each file
    print(f"\n🎬 Processing {len(npz_files)} files...")
    print("=" * 50)
    
    successful = 0
    failed = 0
    skipped = 0
    start_time_total = time.time()
    
    for i, npz_file in enumerate(npz_files, 1):
        print(f"\n📹 Processing {i}/{len(npz_files)}: {os.path.basename(npz_file)}")
        
        # Extract info from filename
        info = extract_video_info_from_filename(npz_file)
        if not info:
            print(f"❌ Skipping {os.path.basename(npz_file)} - could not parse filename")
            failed += 1
            continue
        
        # Check if video file exists
        if not os.path.exists(info['video_path']):
            print(f"❌ Video file not found: {info['video_path']}")
            failed += 1
            continue
        
        # Check if annotated video already exists
        base_name = os.path.splitext(os.path.basename(info['video_path']))[0]
        # Construct the expected annotated video path using the new directory structure
        subdir_name = f"yolo{info['model_size']}_{info['confidence_threshold']}conf_{info['target_fps']}fps_{info['start_time']}s_to_{info['start_time'] + info['duration']}s"
        annotated_path = f"sanity_check_clips/{subdir_name}/{base_name}_annotated_{info['start_time']}s_to_{info['start_time'] + info['duration']}s_yolo{info['model_size']}.mp4"
        
        if os.path.exists(annotated_path) and not overwrite:
            print(f"⏭️  Skipping - annotated video already exists: {os.path.basename(annotated_path)} (use --overwrite to force)")
            skipped += 1
            continue
        
        # Run annotation
        if run_annotation_command(
            info['video_path'], 
            info['start_time'], 
            info['duration'], 
            info['target_fps'],
            info['confidence_threshold'],
            info['model_size']
        ):
            successful += 1
        else:
            failed += 1
    
    # Summary
    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    
    print(f"\n🎉 BATCH ANNOTATION COMPLETED")
    print("=" * 50)
    print(f"Total files found: {len(npz_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(npz_files)*100):.1f}%")
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if successful > 0:
        print(f"Average time per processed file: {total_time/successful:.2f} seconds")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
