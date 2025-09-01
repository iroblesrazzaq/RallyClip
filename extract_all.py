#!/usr/bin/env python3
"""
extract_all.py - Batch extract pose data from all videos

This script finds all video files in the raw_videos directory and runs pose extraction
on each one using the pose_extractor.py script.

Usage:
    python extract_all.py [start_time] [duration] [target_fps] [confidence_threshold] [model_size]
    
    start_time: Start time in seconds (default: 0)
    duration: Duration in seconds (default: 60)
    target_fps: Target frame rate for consistent temporal sampling (default: 15)
    confidence_threshold: Confidence threshold for pose detection (default: 0.05)
    model_size: YOLO model size (n, s, m, l) (default: s)

Examples:
    python extract_all.py                                    # Default: 0s to 60s, 15 FPS, 0.05 conf, small model
    python extract_all.py 0 30 10 0.03 m                    # 0s to 30s, 10 FPS, 0.03 conf, medium model
    python extract_all.py 30 60 15 0.05 l                   # 30s to 90s, 15 FPS, 0.05 conf, large model
"""

import os
import glob
import subprocess
import time
import sys


def get_video_files():
    """
    Get all video files from raw_videos directory.
    
    Returns:
        list: List of video file paths
    """
    raw_videos_dir = "raw_videos"
    
    if not os.path.exists(raw_videos_dir):
        print(f"❌ Raw videos directory '{raw_videos_dir}' not found!")
        return []
    
    # Get all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    
    for ext in video_extensions:
        pattern = os.path.join(raw_videos_dir, ext)
        video_files.extend(glob.glob(pattern))
    
    return sorted(video_files)


def run_extraction_command(video_path, start_time, duration, target_fps, confidence_threshold, model_size):
    """
    Run the pose extraction command.
    
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
    cmd = ["python", "pose_extractor.py", str(start_time), str(duration), str(target_fps), str(confidence_threshold), video_path, model_size]
    
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
    if len(sys.argv) >= 2:
        start_time = int(sys.argv[1])
    else:
        start_time = 0
    
    if len(sys.argv) >= 3:
        duration = int(sys.argv[2])
    else:
        duration = 60
    
    if len(sys.argv) >= 4:
        target_fps = int(sys.argv[3])
    else:
        target_fps = 15
    
    if len(sys.argv) >= 5:
        confidence_threshold = float(sys.argv[4])
    else:
        confidence_threshold = 0.05
    
    if len(sys.argv) >= 6:
        model_size = sys.argv[5]
    else:
        model_size = "s"
    
    print("🎯 Batch Extracting Pose Data from All Videos")
    print("=" * 60)
    print(f"Start time: {start_time}s")
    print(f"Duration: {duration}s")
    print(f"Target FPS: {target_fps}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"YOLO model: {model_size} ({get_model_name(model_size)})")
    
    # Get video files
    video_files = get_video_files()
    
    if not video_files:
        print("❌ No video files found in raw_videos directory!")
        return False
    
    print(f"\n📁 Found {len(video_files)} video files:")
    for i, video_file in enumerate(video_files, 1):
        print(f"   {i}. {os.path.basename(video_file)}")
    
    # Process each video
    print(f"\n🎬 Processing {len(video_files)} videos...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    start_time_total = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n📹 Processing {i}/{len(video_files)}")
        print("=" * 60)
        print(f"🎬 Processing: {os.path.basename(video_path)}")
        print("=" * 60)
        
        # Check if pose data already exists
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        subdir_name = f"yolo{model_size}_{confidence_threshold}conf_{target_fps}fps_{start_time}s_to_{start_time + duration}s"
        pose_data_path = f"pose_data/unfiltered/{subdir_name}/{base_name}_posedata_{start_time}s_to_{start_time + duration}s_yolo{model_size}.npz"
        
        if os.path.exists(pose_data_path):
            print(f"⏭️  Skipping - pose data already exists: {os.path.basename(pose_data_path)}")
            successful += 1
            continue
        
        # Run extraction
        if run_extraction_command(video_path, start_time, duration, target_fps, confidence_threshold, model_size):
            successful += 1
        else:
            failed += 1
    
    # Summary
    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    
    print(f"\n🎉 BATCH EXTRACTION COMPLETED")
    print("=" * 60)
    print(f"Total videos processed: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(video_files)*100):.1f}%")
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if successful > 0:
        print(f"Average time per video: {total_time/successful:.2f} seconds")
    
    return failed == 0


def get_model_name(model_size):
    """Get full model name from size abbreviation."""
    model_names = {
        'n': 'nano',
        's': 'small', 
        'm': 'medium',
        'l': 'large'
    }
    return model_names.get(model_size, model_size)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
