#!/usr/bin/env python3
"""
Tennis Data Pipeline Runner

This script orchestrates the complete tennis data processing pipeline:
1. Pose extraction (pose_extractor.py)
2. Data preprocessing (preprocess_data_pipeline.py)
3. Feature engineering (create_features_pipeline.py)

Usage:
    python run_data_pipeline.py --config data_configs/config1.json
"""

import os
import sys
import json
import argparse
import subprocess
import glob
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def get_annotated_videos(raw_videos_dir, annotations_dir):
    """Get list of videos that have corresponding annotation CSV files."""
    annotated_videos = []
    annotation_files = glob.glob(os.path.join(annotations_dir, "*.csv"))
    
    for annotation_file in annotation_files:
        # Get the video filename from the annotation filename
        basename = os.path.basename(annotation_file)
        if basename.endswith(".csv"):
            video_filename = basename[:-4]  # Remove .csv extension
        else:
            video_filename = basename
            
        # Check if the corresponding video exists with various extensions
        video_found = False
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            video_path = os.path.join(raw_videos_dir, f"{video_filename}")
            if video_path.endswith(ext) and os.path.exists(video_path):
                annotated_videos.append(os.path.basename(video_path))
                video_found = True
                break
            elif not video_path.endswith(ext):
                video_path_with_ext = video_path + ext
                if os.path.exists(video_path_with_ext):
                    annotated_videos.append(os.path.basename(video_path_with_ext))
                    video_found = True
                    break
        
        if not video_found:
            print(f"Warning: No video file found for annotation {basename}")
    
    return annotated_videos

def get_all_videos(raw_videos_dir):
    """Get list of all videos in the raw videos directory."""
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv']
    all_videos = []
    
    for ext in video_extensions:
        videos = glob.glob(os.path.join(raw_videos_dir, ext))
        for video in videos:
            all_videos.append(os.path.basename(video))
    
    return all_videos

def get_selected_videos(raw_videos_dir, video_names):
    """Get list of specific videos to process."""
    selected_videos = []
    
    for video_name in video_names:
        video_path = os.path.join(raw_videos_dir, video_name)
        if os.path.exists(video_path):
            selected_videos.append(video_name)
        else:
            print(f"Warning: Video {video_name} not found in {raw_videos_dir}")
    
    return selected_videos

def get_videos_to_process(config):
    """Get the list of videos to process based on config."""
    raw_videos_dir = "raw_videos"
    annotations_dir = "annotations"
    
    videos_to_process = config.get("videos_to_process", "annotated")
    
    if videos_to_process == "annotated":
        return get_annotated_videos(raw_videos_dir, annotations_dir)
    elif videos_to_process == "all":
        return get_all_videos(raw_videos_dir)
    elif isinstance(videos_to_process, list):
        return get_selected_videos(raw_videos_dir, videos_to_process)
    else:
        raise ValueError(f"Invalid videos_to_process value: {videos_to_process}")

def create_directory_structure(config):
    """Create the directory structure for the pipeline."""
    # Extract parameters from config
    start_time = config["start_time"]
    duration = config["duration"]
    fps = config["fps"]
    conf = config["conf"]
    model_size = config["model_size"]
    
    # Create directory name based on parameters
    dir_name = f"{model_size}_{conf}_{start_time}_{duration}_{fps}"
    
    # Define directory paths
    unfiltered_dir = os.path.join("pose_data", "raw", dir_name)
    preprocessed_dir = os.path.join("pose_data", "preprocessed", dir_name)
    features_dir = os.path.join("pose_data", "features", dir_name)
    
    # Create directories if they don't exist
    os.makedirs(unfiltered_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    return unfiltered_dir, preprocessed_dir, features_dir

def run_pose_extractor(video_name, output_dir, config, overwrite=False):
    """Run pose extraction for a single video."""
    # Extract parameters from config
    start_time = config["start_time"]
    duration = config["duration"]
    fps = config["fps"]
    conf = config["conf"]
    model_size = config["model_size"]
    
    # Define paths
    video_path = os.path.join("raw_videos", video_name)
    annotations_path = os.path.join("annotations", f"{os.path.splitext(video_name)[0]}.csv")
    
    # Check if annotation file exists
    if not os.path.exists(annotations_path):
        print(f"Warning: No annotation file found for {video_name}, running without annotations")
        annotations_path = "None"  # Or handle as needed
    
    # Define output file path
    output_file = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_posedata_{start_time}s_to_{start_time + duration}s_{model_size}{fps}fps_{conf}conf.npz")
    
    # Check if output file already exists and overwrite is False
    if os.path.exists(output_file) and not overwrite:
        print(f"  ✓ Already exists, skipping: {os.path.basename(output_file)}")
        return True
    
    # Build command
    cmd = [
        "python", "pose_extractor.py",
        str(start_time),
        str(duration),
        str(fps),
        str(conf),
        video_path,
        model_size,
        annotations_path
    ]
    
    # Run the command
    print(f"  Running pose extraction for: {video_name}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"  ✅ Pose extraction completed for: {video_name}")
            if result.stdout:
                print(f"    Output: {result.stdout.strip()}")
            return True
        else:
            print(f"  ❌ Pose extraction failed for: {video_name}")
            if result.stderr:
                print(f"    Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ Pose extraction timeout for: {video_name}")
        return False
    except Exception as e:
        print(f"  ❌ Pose extraction exception for: {video_name}: {e}")
        return False

def run_preprocessor(input_dir, video_names, output_dir, overwrite=False):
    """Run data preprocessing."""
    # Build command
    cmd = [
        "python", "preprocess_data_pipeline.py",
        "--input-dir", input_dir,
        "--video-dir", "raw_videos",
        "--output-dir", output_dir
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    # Run the command
    print(f"  Running data preprocessing...")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"  ✅ Data preprocessing completed")
            if result.stdout:
                print(f"    Output: {result.stdout.strip()}")
            return True
        else:
            print(f"  ❌ Data preprocessing failed")
            if result.stderr:
                print(f"    Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ Data preprocessing timeout")
        return False
    except Exception as e:
        print(f"  ❌ Data preprocessing exception: {e}")
        return False

def run_feature_engineer(input_dir, output_dir, overwrite=False):
    """Run feature engineering."""
    # Build command
    cmd = [
        "python", "create_features_pipeline.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    # Run the command
    print(f"  Running feature engineering...")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"  ✅ Feature engineering completed")
            if result.stdout:
                print(f"    Output: {result.stdout.strip()}")
            return True
        else:
            print(f"  ❌ Feature engineering failed")
            if result.stderr:
                print(f"    Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ Feature engineering timeout")
        return False
    except Exception as e:
        print(f"  ❌ Feature engineering exception: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run the tennis data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_data_pipeline.py --config data_configs/config1.json
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    print("=== Tennis Data Pipeline Runner ===")
    print(f"Loading config from: {args.config}")
    
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1
    
    # Print config summary
    print(f"Configuration:")
    print(f"  Start time: {config['start_time']}")
    print(f"  Duration: {config['duration']}")
    print(f"  FPS: {config['fps']}")
    print(f"  Confidence: {config['conf']}")
    print(f"  Model size: {config['model_size']}")
    print(f"  Videos to process: {config['videos_to_process']}")
    print(f"  Steps to run: {config['steps_to_run']}")
    print(f"  Overwrite: {config['overwrite']}")
    
    # Get videos to process
    try:
        videos = get_videos_to_process(config)
        print(f"  Videos to process: {len(videos)}")
        for video in videos[:5]:  # Show first 5 videos
            print(f"    - {video}")
        if len(videos) > 5:
            print(f"    ... and {len(videos) - 5} more")
    except Exception as e:
        print(f"❌ Error getting videos to process: {e}")
        return 1
    
    if not videos:
        print("❌ No videos found to process")
        return 1
    
    # Create directory structure
    try:
        unfiltered_dir, preprocessed_dir, features_dir = create_directory_structure(config)
        print(f"  Directory structure created:")
        print(f"    Unfiltered: {unfiltered_dir}")
        print(f"    Preprocessed: {preprocessed_dir}")
        print(f"    Features: {features_dir}")
    except Exception as e:
        print(f"❌ Error creating directory structure: {e}")
        return 1
    
    # Get steps to run
    steps_to_run = config.get("steps_to_run", ["extractor", "preprocessor", "feature_extractor"])
    overwrite = config.get("overwrite", False)
    
    # Run pose extractor if requested
    if "extractor" in steps_to_run:
        print("\n1. Running Pose Extraction...")
        successful_extractions = 0
        failed_extractions = 0
        
        for video_name in videos:
            if run_pose_extractor(video_name, unfiltered_dir, config, overwrite):
                successful_extractions += 1
            else:
                failed_extractions += 1
        
        print(f"  Pose extraction summary:")
        print(f"    Successful: {successful_extractions}")
        print(f"    Failed: {failed_extractions}")
        
        if failed_extractions > 0 and "preprocessor" in steps_to_run:
            print("  ⚠️  Some extractions failed, preprocessing may be incomplete")
    
    # Run preprocessor if requested
    if "preprocessor" in steps_to_run:
        print("\n2. Running Data Preprocessing...")
        if run_preprocessor(unfiltered_dir, videos, preprocessed_dir, overwrite):
            print("  Data preprocessing completed successfully")
        else:
            print("  ❌ Data preprocessing failed")
            if "feature_extractor" in steps_to_run:
                print("  ⚠️  Feature engineering will be skipped")
    
    # Run feature engineer if requested
    if "feature_extractor" in steps_to_run:
        print("\n3. Running Feature Engineering...")
        if run_feature_engineer(preprocessed_dir, features_dir, overwrite):
            print("  Feature engineering completed successfully")
        else:
            print("  ❌ Feature engineering failed")
    
    print("\n🎯 Pipeline execution completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())