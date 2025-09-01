#!/usr/bin/env python3
"""
Comprehensive Pipeline Testing Script

This script automates the testing of the full data preparation workflow:
1. Mask generation (manual_court.py)
2. Pose extraction (pose_extractor.py) 
3. Pose filtering (filter_pose_data.py)

It executes each pipeline script in sequence and verifies that the expected
output files are created at each stage.

Usage: python test_filtering_pipeline.py [start_time] [duration] [target_fps] [model_size] [video_path]

Arguments:
    start_time: Start time in seconds (default: 0)
    duration: Duration in seconds (default: 10)
    target_fps: Target frame rate (default: 15)
    model_size: YOLO model size - n, s, m, l (default: s)
    video_path: Path to video file (default: "raw_videos/Monica Greene unedited tennis match play.mp4")

Examples:
    python test_filtering_pipeline.py 0 10 15 s
    python test_filtering_pipeline.py 0 30 10 n "raw_videos/my_video.mp4"
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


def verify_file_exists(file_path, description):
    """Verify that a file exists and is not empty."""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        if size > 0:
            size_kb = size / 1024
            print(f"✅ {description}: {file_path} ({size_kb:.1f} KB)")
            return True
        else:
            print(f"❌ {description}: {file_path} (empty file)")
            return False
    else:
        print(f"❌ {description}: {file_path} (not found)")
        return False


def verify_directory_exists(dir_path, description):
    """Verify that a directory exists and contains files."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        files = glob.glob(os.path.join(dir_path, "*.npz"))
        if files:
            print(f"✅ {description}: {dir_path} ({len(files)} files)")
            return True
        else:
            print(f"❌ {description}: {dir_path} (no .npz files)")
            return False
    else:
        print(f"❌ {description}: {dir_path} (not found)")
        return False


def main():
    """Main function to test the complete pipeline."""
    # Parse command line arguments
    if len(sys.argv) >= 2:
        start_time = int(sys.argv[1])
    else:
        start_time = 0
    
    if len(sys.argv) >= 3:
        duration = int(sys.argv[2])
    else:
        duration = 10
    
    if len(sys.argv) >= 4:
        target_fps = int(sys.argv[3])
    else:
        target_fps = 15
    
    if len(sys.argv) >= 5:
        model_size = sys.argv[4]
    else:
        model_size = "s"
    
    if len(sys.argv) >= 6:
        video_path = sys.argv[5]
    else:
        video_path = "raw_videos/Monica Greene unedited tennis match play.mp4"
    
    # Validate inputs
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at '{video_path}'")
        sys.exit(1)
    
    if model_size not in ['n', 's', 'm', 'l']:
        print(f"❌ Error: Invalid model size '{model_size}'. Must be n, s, m, or l.")
        sys.exit(1)
    
    print("=" * 80)
    print("🎯 COMPREHENSIVE PIPELINE TESTING")
    print("=" * 80)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Start time: {start_time}s, Duration: {duration}s")
    print(f"Target FPS: {target_fps}, Model size: {model_size}")
    print()
    
    # Define file paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Step 1: Mask generation paths (no longer needed - generated on-the-fly)
    # mask_path = f"court_masks/{base_name}_mask.png"
    
    # Step 2: Pose extraction paths
    npz_dir = f"pose_data/unfiltered/yolo{model_size}_{0.03}conf_{target_fps}fps_{start_time}s_to_{start_time + duration}s"
    input_npz_path = f"{npz_dir}/{base_name}_posedata_{start_time}s_to_{start_time + duration}s_yolo{model_size}.npz"
    
    # Step 3: Pose filtering paths (new court_filter_ structure)
    filtered_npz_dir = f"pose_data/filtered/court_filter_yolo{model_size}_{0.03}conf_{target_fps}fps_{start_time}s_to_{start_time + duration}s"
    
    # List of expected files/directories for verification
    expected_items = [
        (input_npz_path, "Input pose data file"),
        (filtered_npz_dir, "Filtered pose data directory")
    ]
    
    # Start timing
    script_start_time = time.time()
    
    # Step 1: Generate mask (no longer needed - generated on-the-fly in filter script)
    print("STEP 1: Skipping mask generation (will be generated on-the-fly)")
    print("   Mask will be generated automatically during filtering")
    
    # Step 2: Extract pose data (only if it doesn't exist)
    print("\nSTEP 2: Checking pose data...")
    if os.path.exists(input_npz_path) and os.path.getsize(input_npz_path) > 0:
        print(f"✅ Pose data already exists: {input_npz_path}")
        print("   Skipping pose extraction step")
    else:
        print("🔄 Pose data not found, extracting...")
        extract_cmd = ["python", "pose_extractor.py", str(start_time), str(duration), str(target_fps), "0.03", video_path, model_size]
        if not run_command(extract_cmd, "Pose data extraction"):
            print("❌ Pipeline failed at Step 2: Pose extraction")
            sys.exit(1)
    
    # Step 3: Filter pose data
    print("\nSTEP 3: Filtering pose data...")
    filter_cmd = [
        "python", "filter_pose_data.py",
        "--input-dir", npz_dir,
        "--video-path", video_path
    ]
    
    # Add overwrite flag if needed (you can add this as a parameter later)
    # if overwrite:
    #     filter_cmd.append("--overwrite")
    if not run_command(filter_cmd, "Pose data filtering"):
        print("❌ Pipeline failed at Step 3: Pose filtering")
        sys.exit(1)
    
    # Step 4: Verify all outputs
    print("\nSTEP 4: Verifying outputs...")
    all_items_exist = True
    
    for item_path, description in expected_items:
        if description.endswith("directory"):
            if not verify_directory_exists(item_path, description):
                all_items_exist = False
        else:
            if not verify_file_exists(item_path, description):
                all_items_exist = False
    
    # Calculate total runtime
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    
    # Final results
    print("\n" + "=" * 80)
    if all_items_exist:
        print("🎉 PIPELINE TEST PASSED")
        print("✅ All expected files and directories were created successfully")
    else:
        print("❌ PIPELINE TEST FAILED")
        print("❌ Some expected files or directories were missing")
    
    print(f"🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    print("=" * 80)
    
    return all_items_exist


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
