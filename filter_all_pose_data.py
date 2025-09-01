#!/usr/bin/env python3
"""
Batch Pose Data Filtering Script

This script runs filter_pose_data.py for all videos in the raw_videos directory.
It processes each video file and applies court filtering to the corresponding pose data.

Usage:
    python filter_all_pose_data.py --input-dir <pose_data_subdir> [--overwrite]

Arguments:
    --input-dir: Required. Path to the pose_data subdirectory containing .npz files
    --overwrite: Optional. Flag to overwrite existing filtered files

Example:
    python filter_all_pose_data.py --input-dir "pose_data/unfiltered/yolos_0.03conf_10fps_30s_to_90s"
"""

import os
import sys
import argparse
import subprocess
import glob
import time


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run court filtering on all videos in raw_videos directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python filter_all_pose_data.py --input-dir "pose_data/unfiltered/yolos_0.03conf_10fps_30s_to_90s"
    
    python filter_all_pose_data.py --input-dir "pose_data/unfiltered/yolos_0.03conf_10fps_30s_to_90s" --overwrite
        """
    )
    
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Path to the pose_data subdirectory containing .npz files'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing filtered files (default: False)'
    )
    
    return parser.parse_args()


def get_video_files():
    """Get all video files from raw_videos directory."""
    raw_videos_dir = "raw_videos"
    
    if not os.path.exists(raw_videos_dir):
        print(f"❌ Raw videos directory '{raw_videos_dir}' not found!")
        return []
    
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    
    for ext in video_extensions:
        pattern = os.path.join(raw_videos_dir, ext)
        video_files.extend(glob.glob(pattern))
    
    # Sort for consistent ordering
    video_files.sort()
    return video_files


def run_filter_for_video(input_dir, video_path, overwrite=False):
    """Run filter_pose_data.py for a single video."""
    cmd = [
        "python", "filter_pose_data.py",
        "--input-dir", input_dir,
        "--video-path", video_path
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            # Check if court filtering was successful or if it failed but continued without filtering
            output = result.stdout.strip()
            if "Court filtering: Enabled" in output:
                print(f"✅ Success (with court filtering): {os.path.basename(video_path)}")
            elif "Court filtering: Disabled" in output:
                print(f"⚠️  Success (without court filtering): {os.path.basename(video_path)}")
            else:
                print(f"✅ Success: {os.path.basename(video_path)}")
            return True
        else:
            print(f"❌ Failed: {os.path.basename(video_path)}")
            print(f"   Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout: {os.path.basename(video_path)} (took longer than 5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error: {os.path.basename(video_path)} - {e}")
        return False


def main():
    """Main function to run filtering on all videos."""
    args = parse_args()
    
    print("🎯 BATCH POSE DATA FILTERING")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Overwrite: {args.overwrite}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Get all video files
    video_files = get_video_files()
    
    if not video_files:
        print("❌ No video files found in raw_videos directory!")
        sys.exit(1)
    
    print(f"Found {len(video_files)} video files to process")
    print()
    
    # Start timing
    script_start_time = time.time()
    
    # Process each video
    successful = 0
    successful_with_filtering = 0
    successful_without_filtering = 0
    failed = 0
    
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"🔄 Processing {i}/{len(video_files)}: {video_name}")
        
        if run_filter_for_video(args.input_dir, video_path, args.overwrite):
            successful += 1
            # Note: We can't easily track filtering status here since it's in the subprocess output
            # The individual video processing will show the status
        else:
            failed += 1
        
        print()  # Empty line for readability
    
    # Calculate total runtime
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    
    # Summary
    print("=" * 50)
    print("🎯 BATCH FILTERING SUMMARY")
    print("=" * 50)
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful * 100 / len(video_files)):.1f}%")
    print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    print(f"\nNote: Check individual video outputs above for court filtering status")
    
    if failed == 0:
        print("\n🎉 All videos processed successfully!")
    else:
        print(f"\n❌ {failed} videos failed to process")
    
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
