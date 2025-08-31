#!/usr/bin/env python3
"""
draw_filtered_all.py - Batch annotate all successfully court-filtered pose data files

This script finds all .npz pose data files in court_filtered_* directories (successfully filtered data only)
and creates annotated videos for each one using the video_annotator.py script.
It specifically excludes unfiltered_* directories where court detection failed.

Usage:
    python draw_filtered_all.py [start_time] [duration] [target_fps] [confidence_threshold] [model_size] [overwrite]
    
    start_time: Optional start time in seconds to filter files
    duration: Optional duration in seconds to filter files  
    target_fps: Optional target FPS to filter files
    confidence_threshold: Optional confidence threshold to filter files
    model_size: Optional YOLO model size (n, s, m, l) to filter files
    overwrite: Optional boolean to overwrite existing annotated videos (default: False)
               Accepts: true, 1, yes, y (case insensitive)
    
    If only one argument is provided, it's treated as model_size.
    If three arguments are provided, they're treated as [start_time] [duration] [model_size].
    If four arguments are provided, they're treated as [start_time] [duration] [target_fps] [confidence_threshold].
    If five arguments are provided, they're treated as [start_time] [duration] [target_fps] [confidence_threshold] [model_size].
    If six arguments are provided, they're treated as [start_time] [duration] [target_fps] [confidence_threshold] [model_size] [overwrite].
    If no arguments are provided, processes all court_filtered_* .npz files.

Examples:
    python draw_filtered_all.py                                    # Process all court_filtered_* .npz files
    python draw_filtered_all.py s                                  # Process only small model court_filtered_* files
    python draw_filtered_all.py m                                  # Process only medium model court_filtered_* files
    python draw_filtered_all.py 0 60 s                             # Process small model court_filtered_* files from 0s to 60s
    python draw_filtered_all.py 30 60 m                            # Process medium model court_filtered_* files from 30s to 90s
    python draw_filtered_all.py 0 120 l                            # Process large model court_filtered_* files from 0s to 120s
    python draw_filtered_all.py 0 60 15 0.05 s                    # Process small model court_filtered_* files from 0s to 60s, 15 FPS, 0.05 conf
    python draw_filtered_all.py 30 60 10 0.03 m                   # Process medium model court_filtered_* files from 30s to 90s, 10 FPS, 0.03 conf
    python draw_filtered_all.py 0 60 15 0.05 s true               # Process small model court_filtered_* files from 0s to 60s, 15 FPS, 0.05 conf, overwrite existing
    python draw_filtered_all.py 30 60 10 0.03 m yes               # Process medium model court_filtered_* files from 30s to 90s, 10 FPS, 0.03 conf, overwrite existing
"""

import os
import glob
import subprocess
import time
import sys
import re


def get_filtered_npz_files(model_size=None, start_time=None, duration=None, target_fps=None, confidence_threshold=None):
    """
    Get all .npz files from court_filtered_* subdirectories (successfully filtered data only),
    optionally filtered by model size, time range, fps, and confidence threshold.
    
    This function specifically excludes unfiltered_* directories where court detection failed.
    
    Args:
        model_size (str, optional): Model size to filter by (n, s, m, l)
        start_time (int, optional): Start time in seconds to filter by
        duration (int, optional): Duration in seconds to filter by
        target_fps (int, optional): Target FPS to filter by
        confidence_threshold (float, optional): Confidence threshold to filter by
        
    Returns:
        list: List of .npz file paths from successfully filtered data only
    """
    pose_data_dir = "pose_data"
    
    if not os.path.exists(pose_data_dir):
        print(f"❌ Pose data directory '{pose_data_dir}' not found!")
        return []
    
    # Get all .npz files from court_filtered_* subdirectories (successfully filtered data only)
    all_files = []
    for subdir in os.listdir(pose_data_dir):
        subdir_path = os.path.join(pose_data_dir, subdir)
        # Only look at successfully filtered data (court_filtered_*), exclude unfiltered data
        if (os.path.isdir(subdir_path) and 
            subdir.startswith('court_filtered_') and 
            subdir.endswith('s')):  # New format ends with time range
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
            # Parse time range from subdirectory like "court_filtered_yolom_0.05conf_30s_to_90s"
            time_match = re.search(r'_(\d+)s_to_(\d+)s$', subdir_name)
            if time_match:
                file_start = int(time_match.group(1))
                file_end = int(time_match.group(2))
                if file_start == start_time and file_end == end_time:
                    time_filtered.append(file_path)
        filtered_files = time_filtered
    
    # Filter by target FPS
    if target_fps is not None:
        fps_filtered = []
        for file_path in filtered_files:
            # Extract FPS from subdirectory name
            subdir_name = os.path.basename(os.path.dirname(file_path))
            # Parse FPS from subdirectory like "court_filtered_yolom_0.05conf_15fps_30s_to_90s"
            fps_match = re.search(r'_(\d+)fps_', subdir_name)
            if fps_match:
                file_fps = int(fps_match.group(1))
                if file_fps == target_fps:
                    fps_filtered.append(file_path)
        filtered_files = fps_filtered
    
    # Filter by confidence threshold
    if confidence_threshold is not None:
        conf_filtered = []
        for file_path in filtered_files:
            # Extract confidence threshold from subdirectory name
            subdir_name = os.path.basename(os.path.dirname(file_path))
            # Parse confidence from subdirectory like "court_filtered_yolom_0.05conf_15fps_30s_to_90s"
            conf_match = re.search(r'_(\d+\.\d+)conf_', subdir_name)
            if conf_match:
                file_conf = float(conf_match.group(1))
                if abs(file_conf - confidence_threshold) < 0.001:  # Float comparison
                    conf_filtered.append(file_path)
        filtered_files = conf_filtered
    
    return filtered_files


def extract_video_info_from_filename(npz_path):
    """
    Extract video information from .npz filename.
    
    Args:
        npz_path (str): Path to .npz file
        
    Returns:
        dict: Dictionary with video info or None if parsing fails
    """
    filename = os.path.basename(npz_path)
    
    # Parse filename like "Monica Greene unedited tennis match play_posedata_30s_to_90s_yolos.npz"
    # Extract base video name (everything before _posedata)
    if '_posedata_' not in filename:
        return None
    
    base_name = filename.split('_posedata_')[0]
    video_path = f"raw_videos/{base_name}.mp4"
    
    # Extract time range from filename
    time_match = re.search(r'_posedata_(\d+)s_to_(\d+)s_', filename)
    if not time_match:
        return None
    
    start_time = int(time_match.group(1))
    end_time = int(time_match.group(2))
    duration = end_time - start_time
    
    # Extract model size from filename
    model_match = re.search(r'_yolo([nslm])\.npz$', filename)
    if not model_match:
        return None
    
    model_size = model_match.group(1)
    
    # Extract other parameters from subdirectory name
    subdir_name = os.path.basename(os.path.dirname(npz_path))
    
    # Parse confidence threshold
    conf_match = re.search(r'_(\d+\.\d+)conf_', subdir_name)
    confidence_threshold = float(conf_match.group(1)) if conf_match else 0.05
    
    # Parse target FPS
    fps_match = re.search(r'_(\d+)fps_', subdir_name)
    target_fps = int(fps_match.group(1)) if fps_match else 15
    
    return {
        'video_path': video_path,
        'start_time': start_time,
        'duration': duration,
        'target_fps': target_fps,
        'confidence_threshold': confidence_threshold,
        'model_size': model_size,
        'npz_path': npz_path
    }


def run_annotation_command(video_path, start_time, duration, target_fps, confidence_threshold, model_size, overwrite=False):
    """
    Run the video annotation command.
    
    Args:
        video_path (str): Path to video file
        start_time (int): Start time in seconds
        duration (int): Duration in seconds
        target_fps (int): Target frame rate
        confidence_threshold (float): Confidence threshold
        model_size (str): Model size
        overwrite (bool): Overwrite existing files
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = ["python", "video_annotator.py", str(start_time), str(duration), str(target_fps), str(confidence_threshold), video_path, model_size]
    if overwrite:
        cmd.append("true")
    
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
    # Parse command line arguments (same logic as draw_all.py)
    if len(sys.argv) >= 7:
        # Format: python draw_filtered_all.py [start_time] [duration] [target_fps] [confidence_threshold] [model_size] [overwrite]
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        target_fps = int(sys.argv[3])
        confidence_threshold = float(sys.argv[4])
        model_size = sys.argv[5]
        overwrite = sys.argv[6].lower() in ['true', '1', 'yes', 'y']
    elif len(sys.argv) >= 6:
        # Format: python draw_filtered_all.py [start_time] [duration] [target_fps] [confidence_threshold] [model_size]
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        target_fps = int(sys.argv[3])
        confidence_threshold = float(sys.argv[4])
        model_size = sys.argv[5]
        overwrite = False
    elif len(sys.argv) >= 4:
        # Format: python draw_filtered_all.py [start_time] [duration] [model_size]
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        model_size = sys.argv[3]
        target_fps = None
        confidence_threshold = None
        overwrite = False
    elif len(sys.argv) >= 2:
        # Format: python draw_filtered_all.py [model_size]
        start_time = None
        duration = None
        model_size = sys.argv[1]
        target_fps = None
        confidence_threshold = None
        overwrite = False
    else:
        # No arguments - process all files
        start_time = None
        duration = None
        model_size = None
        target_fps = None
        confidence_threshold = None
        overwrite = False
    
    print("🎨 Batch Annotating All Court-Filtered Pose Data Files")
    print("=" * 60)
    
    if model_size and start_time is not None and duration is not None:
        print(f"Model size filter: {model_size}")
        print(f"Time range filter: {start_time}s to {start_time + duration}s")
        if target_fps is not None:
            print(f"Target FPS filter: {target_fps}")
        if confidence_threshold is not None:
            print(f"Confidence threshold filter: {confidence_threshold}")
    elif model_size:
        print(f"Model size filter: {model_size}")
    elif start_time is not None and duration is not None:
        print(f"Time range filter: {start_time}s to {start_time + duration}s")
    else:
        print("Processing all court_filtered_* files")
    
    print(f"Overwrite existing files: {overwrite}")
    
    # Get .npz files from court_filtered_* directories
    npz_files = get_filtered_npz_files(model_size, start_time, duration, target_fps, confidence_threshold)
    
    if not npz_files:
        print("❌ No successfully court-filtered .npz files found!")
        print("   (Only looking at court_filtered_* directories, not unfiltered_* directories)")
        if model_size and start_time is not None and duration is not None:
            print(f"   No files found for model size '{model_size}' and time range {start_time}s to {start_time + duration}s")
        elif model_size:
            print(f"   No files found for model size '{model_size}'")
        elif start_time is not None and duration is not None:
            print(f"   No files found for time range {start_time}s to {start_time + duration}s")
        return False
    
    print(f"\n📁 Found {len(npz_files)} successfully court-filtered .npz files:")
    for i, npz_file in enumerate(npz_files, 1):
        print(f"   {i}. {os.path.basename(npz_file)}")
    
    # Process each file
    print(f"\n🎬 Processing {len(npz_files)} files...")
    print("=" * 60)
    
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
        
        # Check if annotated video already exists (in court_filtered subdirectory)
        base_name = os.path.splitext(os.path.basename(info['video_path']))[0]
        # Construct the expected annotated video path using the court_filtered directory structure
        subdir_name = f"yolo{info['model_size']}_{info['confidence_threshold']}conf_{info['target_fps']}fps_{info['start_time']}s_to_{info['start_time'] + info['duration']}s"
        annotated_path = f"sanity_check_clips/court_filtered/{subdir_name}/{base_name}_annotated_{info['start_time']}s_to_{info['start_time'] + info['duration']}s_yolo{info['model_size']}.mp4"
        
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
            info['model_size'],
            overwrite
        ):
            successful += 1
        else:
            failed += 1
    
    # Summary
    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    
    print(f"\n🎉 BATCH COURT-FILTERED ANNOTATION COMPLETED")
    print("=" * 60)
    print(f"Total successfully court-filtered files found: {len(npz_files)}")
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
