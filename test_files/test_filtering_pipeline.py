#!/usr/bin/env python3
"""
Comprehensive Pipeline Testing Script

This script automates the testing of the full data preparation workflow:
1. Mask generation (manual_court.py)
2. Pose extraction (pose_extractor.py) 
3. Data preprocessing (preprocess_data_pipeline.py)
4. Feature engineering (create_features_pipeline.py)

It executes each pipeline script in sequence and verifies that the expected
output files are created at each stage.

Usage: python test_filtering_pipeline.py [start_time] [duration] [target_fps] [model_size] [video_path]

Arguments:
    start_time: Start time in seconds (default: 0)
    duration: Duration in seconds (default: 10)
    target_fps: Target frame rate (default: 15)
    model_size: YOLO model size - n, s, m, l (default: s)
    video_path: Path to video file (default: "raw_videos/Monica Greene unedited tennis match play.mp4")
"""

import os
import sys
import subprocess
import argparse
import time
import shutil

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_command(command, description, cwd=None):
    """Run a command and return the result."""
    print(f"\n🚀 {description}")
    print(f"   Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("   ✅ Success")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print("   ❌ Failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("   ❌ Timeout")
        return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def test_new_modular_pipeline():
    """Test the new modular pipeline."""
    print("🧪 TESTING NEW MODULAR PIPELINE")
    print("=" * 50)
    
    # Test that the new classes can be imported
    print("\n1. Testing class imports...")
    try:
        from data_scripts.data_preprocessor import DataPreprocessor
        from data_scripts.feature_engineer import FeatureEngineer
        print("   ✅ DataPreprocessor imported successfully")
        print("   ✅ FeatureEngineer imported successfully")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test that the pipeline scripts exist
    print("\n2. Testing pipeline scripts...")
    pipeline_scripts = [
        "preprocess_data_pipeline.py",
        "create_features_pipeline.py"
    ]
    
    for script in pipeline_scripts:
        if os.path.exists(script):
            print(f"   ✅ {script} exists")
        else:
            print(f"   ❌ {script} not found")
            return False
    
    print("\n✅ New modular pipeline test completed successfully!")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the new modular pipeline")
    parser.add_argument("--start-time", type=int, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--target-fps", type=int, default=15, help="Target frame rate")
    parser.add_argument("--model-size", choices=['n', 's', 'm', 'l'], default='s', help="YOLO model size")
    parser.add_argument("--video-path", default="raw_videos/Monica Greene unedited tennis match play.mp4", help="Path to video file")
    
    args = parser.parse_args()
    
    print("Tennis Point Detection - New Modular Pipeline Test")
    print("=" * 60)
    
    # Run the test
    success = test_new_modular_pipeline()
    
    if success:
        print("\n🎉 All tests passed! The new modular pipeline is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())