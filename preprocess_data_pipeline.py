#!/usr/bin/env python3
"""
Tennis Data Preprocessing Pipeline

This script uses the TennisDataPreprocessor class to preprocess tennis pose data.
"""

import os
import sys
import argparse
import glob
import time
from data_scripts.data_preprocessor import DataPreprocessor

def find_video_file(base_name, video_dir):
    """
    Find the video file corresponding to a base name.
    
    Args:
        base_name (str): Base name of the video
        video_dir (str): Directory to search for video files
        
    Returns:
        str or None: Path to video file or None if not found
    """
    # Try to find the video file
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    
    for ext in video_extensions:
        # Try exact match first
        pattern = os.path.join(video_dir, f"{base_name}.{ext.replace('*', '')}")
        if os.path.exists(pattern):
            return pattern
            
        # Try with extension matching
        pattern = os.path.join(video_dir, f"{base_name}{ext[1:]}")
        if os.path.exists(pattern):
            return pattern
    
    # Try a more general pattern
    for ext in video_extensions:
        pattern = os.path.join(video_dir, ext)
        matching_files = glob.glob(pattern)
        for match in matching_files:
            if base_name in os.path.basename(match):
                return match
    
    return None

def preprocess_all_videos(input_dir, video_dir, output_dir, overwrite=False):
    """
    Preprocess all videos in a directory using the TennisDataPreprocessor.
    
    Args:
        input_dir (str): Directory containing .npz files
        video_dir (str): Directory containing video files
        output_dir (str): Directory to save preprocessed .npz files
        overwrite (bool): Whether to overwrite existing files
    """
    # Find all .npz files in the input directory
    npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
    
    if not npz_files:
        print(f"❌ No .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} .npz files to process")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Process each file
    successful = 0
    failed = 0
    
    for npz_file in npz_files:
        try:
            # Derive video name from npz file name
            base_name = os.path.splitext(os.path.basename(npz_file))[0]
            # Remove the suffix to get the video name
            # e.g., "video_name_posedata_0s_to_60s_yolos" -> "video_name"
            if "_posedata" in base_name:
                video_name = "_".join(base_name.split("_posedata")[0].split("_")[:-1])
            else:
                video_name = base_name.split("_posedata")[0]
            
            # Look for corresponding video file
            video_file = find_video_file(video_name, video_dir)
            
            if video_file is None:
                print(f"  ⚠️  No matching video file found for {base_name}, skipping...")
                failed += 1
                continue
            
            # Create output file path
            output_file = os.path.join(output_dir, f"{base_name}_preprocessed.npz")
            
            print(f"Processing: {os.path.basename(npz_file)}")
            if preprocessor.preprocess_single_video(npz_file, video_file, output_file, overwrite):
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"Failed to process {npz_file}: {e}")
            failed += 1
    
    print(f"\nPreprocessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Preprocess tennis pose data using TennisDataPreprocessor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python preprocess_data_pipeline.py --input-dir "pose_data/unfiltered/yolos_0.05conf_15fps_0s_to_60s" --video-dir "raw_videos" --output-dir "preprocessed_data"
    
    python preprocess_data_pipeline.py --input-dir "pose_data/unfiltered/yolos_0.05conf_15fps_0s_to_60s" --video-dir "raw_videos" --output-dir "preprocessed_data" --overwrite
        """
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing raw pose .npz files')
    parser.add_argument('--video-dir', type=str, default='raw_videos',
                        help='Directory containing video files (default: raw_videos)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save preprocessed .npz files')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing preprocessed files')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory {args.input_dir} does not exist")
        return
    
    # Check if video directory exists
    if not os.path.exists(args.video_dir):
        print(f"❌ Error: Video directory {args.video_dir} does not exist")
        return
    
    print("=== Tennis Data Preprocessing Pipeline ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Overwrite: {'Yes' if args.overwrite else 'No'}")
    print()
    
    # Start timing
    start_time = time.time()
    
    # Preprocess all videos
    preprocess_all_videos(
        input_dir=args.input_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite
    )
    
    # Calculate and print total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")

if __name__ == "__main__":
    main()