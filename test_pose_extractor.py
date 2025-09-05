#!/usr/bin/env python3
"""
Test script to compare the performance of the original and optimized pose extractors.
"""

import time
import os
import sys

def test_performance():
    """Test the performance of both extractors."""
    # Check if video file exists
    video_path = "raw_videos/Monica Greene unedited tennis match play.mp4"
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please make sure you have a test video in the raw_videos directory.")
        return
    
    # Test parameters
    start_time = 0
    duration = 30  # 30 seconds for testing
    target_fps = 15
    confidence_threshold = 0.05
    model_size = "s"
    batch_size = 8
    
    print("Testing Pose Extractor Performance")
    print("=" * 50)
    print(f"Video: {video_path}")
    print(f"Duration: {duration}s")
    print(f"Target FPS: {target_fps}")
    print(f"Model: yolov8{model_size}-pose.pt")
    print(f"Batch size: {batch_size}")
    print()
    
    # Test optimized version
    print("Testing Optimized Pose Extractor...")
    start_time_opt = time.time()
    
    # Import and run optimized version
    try:
        from pose_extractor_optimized import OptimizedPoseExtractor
        extractor_opt = OptimizedPoseExtractor(
            model_path=f"yolov8{model_size}-pose.pt",
            batch_size=batch_size
        )
        
        output_path_opt = extractor_opt.extract_pose_data(
            video_path=video_path,
            start_time_seconds=start_time,
            duration_seconds=duration,
            target_fps=target_fps,
            confidence_threshold=confidence_threshold
        )
        
        end_time_opt = time.time()
        elapsed_opt = end_time_opt - start_time_opt
        print(f"Optimized version completed in: {elapsed_opt:.2f} seconds")
        
    except Exception as e:
        print(f"Error testing optimized version: {e}")
        return
    
    print("\nPerformance test completed!")

if __name__ == "__main__":
    test_performance()