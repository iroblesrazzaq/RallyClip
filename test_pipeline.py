#!/usr/bin/env python3
"""
Test script for the two-part pose estimation pipeline:
1. pose_extractor.py - Extracts pose data and saves to .npz
2. video_annotator.py - Creates annotated videos from the data

Usage: python test_pipeline.py [start_time] [duration]
"""

import subprocess
import sys
import time
import os


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
        if result.stdout:
            print("   Output:", result.stdout.strip())
        return True
    else:
        print(f"❌ {description} failed")
        print(f"   Error: {result.stderr.strip()}")
        return False


def main():
    # Parse command line arguments
    if len(sys.argv) >= 3:
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        every_k_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    else:
        start_time = 0
        duration = 5  # Default to 5 seconds for quick testing
        every_k_frames = 1
    
    print("🎯 Testing Two-Part Pose Estimation Pipeline")
    print("=" * 50)
    print(f"Start time: {start_time}s")
    print(f"Duration: {duration}s")
    print(f"Every k frames: {every_k_frames}")
    
    # Step 1: Extract pose data
    extract_cmd = ["python", "pose_extractor.py", str(start_time), str(duration), str(every_k_frames)]
    if not run_command(extract_cmd, "Step 1: Pose Data Extraction"):
        print("❌ Pipeline failed at step 1")
        return False
    
    # Step 2: Create annotated video
    annotate_cmd = ["python", "video_annotator.py", str(start_time), str(duration)]
    if not run_command(annotate_cmd, "Step 2: Video Annotation"):
        print("❌ Pipeline failed at step 2")
        return False
    
    # Check output files
    print("\n📁 Checking output files...")
    
    # Check pose data file
    base_name = "Monica Greene unedited tennis match play"
    pose_data_file = f"pose_data/{base_name}_posedata_{start_time}s_to_{start_time + duration}s.npz"
    if os.path.exists(pose_data_file):
        size = os.path.getsize(pose_data_file) / 1024  # KB
        print(f"✅ Pose data file: {pose_data_file} ({size:.1f} KB)")
    else:
        print(f"❌ Pose data file not found: {pose_data_file}")
    
    # Check annotated video file
    video_file = f"sanity_check_clips/{base_name}_annotated_{start_time}s_to_{start_time + duration}s.mp4"
    if os.path.exists(video_file):
        size = os.path.getsize(video_file) / (1024 * 1024)  # MB
        print(f"✅ Annotated video: {video_file} ({size:.1f} MB)")
    else:
        print(f"❌ Annotated video not found: {video_file}")
    
    print("\n🎉 Pipeline completed successfully!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
