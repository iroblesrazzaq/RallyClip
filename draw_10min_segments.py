#!/usr/bin/env python3
"""
draw_10min_segments.py - Draw 10-minute samples from every video with annotations (FIXED)

This script finds all video files in raw_videos and creates annotated videos for 10-minute samples
with off-court masking (transparent gray), bounding boxes, pose keypoints, and centroids.
Fixed: Annotations now sync properly with video timing (no more slow-motion bounding boxes).

Usage:
    python draw_10min_segments.py [pose_data_dir] [sample_duration] [overwrite]
    
    pose_data_dir: Path to the pose data directory (e.g., "pose_data/filtered/yolos_0.2conf_10fps_0s_to_999999s")
    sample_duration: Duration of each sample in minutes (default: 10)
    overwrite: Boolean to overwrite existing videos (default: False)
               Accepts: true, 1, yes, y (case insensitive)

Examples:
    python draw_10min_segments.py "pose_data/filtered/yolos_0.2conf_10fps_0s_to_999999s"                    # 10-minute samples, no overwrite
    python draw_10min_segments.py "pose_data/filtered/yolos_0.2conf_10fps_0s_to_999999s" 15                 # 15-minute samples, no overwrite
    python draw_10min_segments.py "pose_data/filtered/yolos_0.2conf_10fps_0s_to_999999s" 10 true            # 10-minute samples, overwrite existing
    python draw_10min_segments.py "pose_data/filtered/yolon_0.2conf_15fps_0s_to_999999s" 20 yes             # 20-minute samples, overwrite existing
"""

import os
import glob
import subprocess
import time
import sys
import math


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


def get_video_duration(video_path):
    """
    Get video duration using ffprobe.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        float: Duration in seconds, or None if failed
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
        else:
            print(f"⚠️  Could not get duration for {os.path.basename(video_path)}")
            return None
            
    except Exception as e:
        print(f"⚠️  Error getting duration for {os.path.basename(video_path)}: {e}")
        return None


def find_matching_pose_data(video_path, segment_start, segment_duration, pose_data_dir):
    """
    Find matching pose data file for a video segment.
    
    Args:
        video_path (str): Path to video file
        segment_start (int): Start time in seconds
        segment_duration (int): Duration in seconds
        pose_data_dir (str): Directory containing pose data files
        
    Returns:
        str: Path to pose data file, or None if not found
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if not os.path.exists(pose_data_dir):
        print(f"⚠️  Pose data directory not found: {pose_data_dir}")
        return None
    
    # Look for .npz files that match our video
    npz_pattern = os.path.join(pose_data_dir, f"{base_name}*.npz")
    npz_files = glob.glob(npz_pattern)
    
    # First try to find exact segment match
    for npz_file in npz_files:
        if f"{segment_start}s_to_{segment_start + segment_duration}s" in npz_file:
            return npz_file
    
    # If no exact match, look for full video coverage (e.g., 0s_to_999999s)
    for npz_file in npz_files:
        if "0s_to_999999s" in npz_file or "0s_to_" in npz_file:
            print(f"📁 Found full video pose data: {os.path.basename(npz_file)}")
            return npz_file
    
    return None


def run_draw_command(video_path, pose_data_path, segment_start, segment_duration, overwrite=False):
    """
    Run the video annotation command.
    
    Args:
        video_path (str): Path to video file
        pose_data_path (str): Path to pose data file
        segment_start (int): Start time in seconds
        segment_duration (int): Duration in seconds
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [
        "python", "video_annotator.py",
        str(segment_start), str(segment_duration),
        "15", "0.2", video_path, "s", str(overwrite).lower(),
        "--data-path", pose_data_path
    ]
    
    print(f"🔄 Running: {' '.join(cmd)}")
    
    start_time_cmd = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
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
        pose_data_dir = sys.argv[1]
    else:
        print("❌ Error: pose_data_dir is required!")
        print("Usage: python draw_10min_segments.py [pose_data_dir] [segment_duration] [overwrite]")
        return False
    
    if len(sys.argv) >= 3:
        segment_duration_minutes = int(sys.argv[2])
    else:
        segment_duration_minutes = 10
    
    if len(sys.argv) >= 4:
        overwrite = sys.argv[3].lower() in ['true', '1', 'yes', 'y']
    else:
        overwrite = False
    
    sample_duration_seconds = segment_duration_minutes * 60
    
    print("🎯 Drawing 10-Minute Samples from All Videos")
    print("=" * 60)
    print(f"Pose data directory: {pose_data_dir}")
    print(f"Sample duration: {segment_duration_minutes} minutes ({sample_duration_seconds} seconds)")
    print(f"Overwrite: {overwrite}")
    print(f"Features: Off-court masking (transparent gray), bounding boxes, pose keypoints, centroids")
    
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
        
        # Get video duration
        duration = get_video_duration(video_path)
        if duration is None:
            print(f"⚠️  Skipping {os.path.basename(video_path)} - could not determine duration")
            continue
        
        # Take a single 10-minute sample from the beginning of the video
        if duration <= sample_duration_seconds:
            # Video is shorter than requested sample, use the whole video
            segment_start = 0
            actual_segment_duration = duration
            print(f"📊 Video duration: {duration:.1f}s ({duration/60:.1f} minutes)")
            print(f"📊 Using entire video (shorter than {segment_duration_minutes} minutes)")
        else:
            # Always start from the beginning (t=0) for consistent frame alignment
            segment_start = 0
            actual_segment_duration = min(sample_duration_seconds, duration)
            print(f"📊 Video duration: {duration:.1f}s ({duration/60:.1f} minutes)")
            print(f"📊 Taking {actual_segment_duration/60:.1f} minute sample from 0s to {actual_segment_duration/60:.1f}s")
        
        print(f"\n🎬 Processing single sample: {segment_start}s to {segment_start + actual_segment_duration}s ({actual_segment_duration}s)")
        
        # Find matching pose data
        pose_data_path = find_matching_pose_data(video_path, segment_start, actual_segment_duration, pose_data_dir)
        
        if pose_data_path is None:
            print(f"⚠️  No pose data found for sample {segment_start}s to {segment_start + actual_segment_duration}s")
            continue
        
        print(f"📁 Found pose data: {os.path.basename(pose_data_path)}")
        
        # DEBUG: Print time dimension lengths
        print(f"🔍 DEBUG: Analyzing time dimensions...")
        
        # Get video FPS and calculate total video frames vs segment frames
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_video_frames = int(duration * video_fps)
            segment_frames = int(actual_segment_duration * video_fps)
            cap.release()
            print(f"   📹 Video: {segment_frames} frames at {video_fps:.1f} FPS for {actual_segment_duration}s segment")
            print(f"   📹 Video: {total_video_frames} total frames for entire {duration:.1f}s video")
        except Exception as e:
            print(f"   ⚠️  Could not get video FPS: {e}")
            total_video_frames = "unknown"
            segment_frames = "unknown"
        
        # Get pose data array length
        try:
            import numpy as np
            pose_data = np.load(pose_data_path, allow_pickle=True)
            pose_frames = len(pose_data['frames'])
            print(f"   📊 Pose data: {pose_frames} frames in .npz file")
            
            if total_video_frames != "unknown":
                if pose_frames == total_video_frames:
                    print(f"   ✅ Total frame counts match! Video and pose data are aligned.")
                else:
                    print(f"   ❌ TOTAL FRAME MISMATCH! Video has {total_video_frames} frames, pose data has {pose_frames} frames")
                    print(f"   💡 This explains the timing sync issue!")
                    
                # Check segment alignment
                if segment_start == 0:
                    print(f"   📍 Segment starts at 0s, so first {segment_frames} frames should align perfectly")
                else:
                    print(f"   📍 Segment starts at {segment_start}s, so frames {int(segment_start * video_fps)}-{int((segment_start + actual_segment_duration) * video_fps)} should align")
            else:
                print(f"   ⚠️  Cannot verify alignment (video FPS unknown)")
                
        except Exception as e:
            print(f"   ⚠️  Could not read pose data: {e}")
        
        # Check if output video already exists
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_annotated_{segment_start}s_to_{segment_start + actual_segment_duration}s.mp4"
        output_path = f"sanity_check_clips/{output_filename}"
        
        if os.path.exists(output_path) and not overwrite:
            print(f"⏭️  Skipping - annotated video already exists: {output_filename}")
            continue
        
        # Run annotation
        if run_draw_command(video_path, pose_data_path, segment_start, actual_segment_duration, overwrite):
            successful += 1
        else:
            failed += 1
        
        print()  # Empty line for readability
    
    # Summary
    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    
    print(f"\n🎉 10-MINUTE SAMPLE DRAWING COMPLETED")
    print("=" * 60)
    print(f"Total videos processed: {len(video_files)}")
    print(f"Successful samples: {successful}")
    print(f"Failed samples: {failed}")
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if successful > 0:
        print(f"Average time per sample: {total_time/successful:.2f} seconds")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
