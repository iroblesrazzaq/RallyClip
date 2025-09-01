#!/usr/bin/env python3
"""
Batch processing script for the two-part pose estimation pipeline:
Runs the pipeline on all videos in the raw_videos directory.

Usage: python test_all.py [start_time] [duration] [target_fps] [model_size]
  - start_time: Start time in seconds (default: 0)
  - duration: Duration in seconds (default: 60)
  - target_fps: Target frame rate for consistent temporal sampling (default: 15)
  - model_size: YOLO model size - n, s, m, l (default: s)

Examples:
  python test_all.py 0 60 15 s
  python test_all.py 0 30 10 n
  python test_all.py 0 120 15 l
"""

import subprocess
import sys
import time
import os
import glob


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        print(f"   Runtime: {end_time - start_time:.2f} seconds")
        return True
    else:
        print(f"❌ {description} failed")
        print(f"   Error: {result.stderr.strip()}")
        return False


def get_model_name(model_size):
    """Get the full model name from size abbreviation."""
    model_names = {
        "n": "nano",
        "s": "small", 
        "m": "medium",
        "l": "large"
    }
    return model_names.get(model_size, "unknown")


def get_video_files():
    """Get all video files from raw_videos directory."""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        pattern = os.path.join("raw_videos", ext)
        video_files.extend(glob.glob(pattern))
    
    # Filter out hidden files and sort
    video_files = [f for f in video_files if not os.path.basename(f).startswith('.')]
    video_files.sort()
    
    return video_files


def process_single_video(video_path, start_time, duration, target_fps, model_size="s"):
    """Process a single video through the pipeline."""
    print(f"\n{'='*80}")
    print(f"🎬 Processing: {os.path.basename(video_path)}")
    print(f"{'='*80}")
    
    # Step 1: Extract pose data
    extract_cmd = ["python", "pose_extractor.py", str(start_time), str(duration), str(target_fps), video_path, model_size]
    if not run_command(extract_cmd, f"Step 1: Pose Data Extraction - {os.path.basename(video_path)}"):
        print(f"❌ Failed to extract pose data for {os.path.basename(video_path)}")
        return False
    
    # Step 2: Create annotated video
    annotate_cmd = ["python", "video_annotator.py", str(start_time), str(duration), video_path, model_size]
    if not run_command(annotate_cmd, f"Step 2: Video Annotation - {os.path.basename(video_path)}"):
        print(f"❌ Failed to create annotated video for {os.path.basename(video_path)}")
        return False
    
    # Check output files with model size
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    pose_data_file = f"pose_data/unfiltered/{base_name}_posedata_{start_time}s_to_{start_time + duration}s_yolo{model_size}.npz"
    video_file = f"sanity_check_clips/{base_name}_annotated_{start_time}s_to_{start_time + duration}s_yolo{model_size}.mp4"
    
    success = True
    if os.path.exists(pose_data_file):
        size = os.path.getsize(pose_data_file) / 1024  # KB
        print(f"✅ Pose data: {pose_data_file} ({size:.1f} KB)")
    else:
        print(f"❌ Pose data file not found: {pose_data_file}")
        success = False
    
    if os.path.exists(video_file):
        size = os.path.getsize(video_file) / (1024 * 1024)  # MB
        print(f"✅ Annotated video: {video_file} ({size:.1f} MB)")
    else:
        print(f"❌ Annotated video not found: {video_file}")
        success = False
    
    return success


def main():
    # Parse command line arguments
    if len(sys.argv) >= 4:
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        target_fps = int(sys.argv[3])
        model_size = sys.argv[4] if len(sys.argv) > 4 else "s"
    else:
        start_time = 0
        duration = 60  # Default to 60 seconds
        target_fps = 15  # Default to 15 FPS
        model_size = "s"  # Default to small model
    
    print("🎯 Batch Processing All Videos in raw_videos Directory")
    print("=" * 80)
    print(f"Start time: {start_time}s")
    print(f"Duration: {duration}s")
    print(f"Target FPS: {target_fps}")
    print(f"YOLO model: {model_size} ({get_model_name(model_size)})")
    
    # Get all video files
    video_files = get_video_files()
    
    if not video_files:
        print("❌ No video files found in raw_videos directory")
        return False
    
    print(f"\n📁 Found {len(video_files)} video files:")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(video)}")
    
    # Process each video
    total_start_time = time.time()
    successful_videos = 0
    failed_videos = 0
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*80}")
        print(f"📹 Processing video {i}/{len(video_files)}")
        print(f"{'='*80}")
        
        if process_single_video(video_path, start_time, duration, target_fps, model_size):
            successful_videos += 1
        else:
            failed_videos += 1
    
    # Final summary
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    
    print(f"\n{'='*80}")
    print("🎉 BATCH PROCESSING COMPLETED")
    print(f"{'='*80}")
    print(f"Total videos processed: {len(video_files)}")
    print(f"Successful: {successful_videos}")
    print(f"Failed: {failed_videos}")
    print(f"Success rate: {(successful_videos/len(video_files)*100):.1f}%")
    print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    print(f"Average time per video: {total_runtime/len(video_files):.2f} seconds")
    
    if successful_videos == len(video_files):
        print("\n🎯 All videos processed successfully!")
        return True
    else:
        print(f"\n⚠️  {failed_videos} videos failed to process")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
