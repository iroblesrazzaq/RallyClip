#!/usr/bin/env python3
"""
Pose Data Filtering Script

This script filters pose data from existing .npz files based on a playable area mask.
It ONLY works when NPZ files exist in the specified subdirectory and outputs to court_filter_ prefixed directories.
The mask is created on-the-fly from the first video file and then discarded.

Usage:
    python filter_pose_data.py --input-dir <pose_data_subdir> --video-path <video_file> [--visualize]

Arguments:
    --input-dir: Required. Path to the pose_data subdirectory containing .npz files
    --video-path: Required. Path to the video file for mask generation
    --visualize: Optional. Flag to create verification videos for the first few files

Example:
    python filter_pose_data.py --input-dir "pose_data/yolos_0.03conf_10fps_30s_to_90s" --video-path "raw_videos/video.mp4"
"""

import os
import sys
import argparse
import numpy as np
import cv2
import time
import glob
from data_scripts.court_detector import CourtDetector


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Filter pose data from existing .npz files based on playable area mask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python filter_pose_data.py --input-dir "pose_data/unfiltered/yolos_0.03conf_10fps_30s_to_90s" --video-path "raw_videos/video.mp4"
    
    python filter_pose_data.py --input-dir "pose_data/unfiltered/yolos_0.03conf_10fps_30s_to_90s" --video-path "raw_videos/video.mp4" --visualize
        """
    )
    
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Path to the pose_data subdirectory containing .npz files'
    )
    
    parser.add_argument(
        '--video-path',
        required=True,
        help='Path to the video file for mask generation'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create verification videos for the first few files (optional)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing filtered files (default: False)'
    )
    
    return parser.parse_args()


def filter_single_npz_file(input_npz_path, output_npz_path, mask=None):
    """
    Filter a single .npz file based on the playable area mask.
    
    Args:
        input_npz_path (str): Path to input .npz file
        output_npz_path (str): Path to output .npz file
        mask (np.ndarray, optional): The playable area mask. If None, no filtering is applied.
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load pose data
        pose_data = np.load(input_npz_path, allow_pickle=True)['frames']
        
        # If no mask is provided (court detection failed), copy data without filtering
        if mask is None:
            print(f"  ⚠️  No court mask available - copying data without filtering")
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_npz_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the original data without filtering
            np.savez_compressed(output_npz_path, frames=pose_data)
            return True
        
        # Initialize data structures for filtered results
        filtered_frames_data = []
        
        # Main filtering loop
        for frame_idx, frame_data in enumerate(pose_data):
            # Extract arrays from frame data
            boxes = frame_data['boxes']
            keypoints = frame_data['keypoints']
            conf = frame_data['conf']
            
            # Initialize empty lists for the current frame's surviving data
            kept_boxes = []
            kept_keypoints = []
            kept_conf = []
            
            # Inner loop: filter each person in the frame
            for i, box in enumerate(boxes):
                # Calculate bounding box centroid
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                # Check if centroid is inside the playable area
                # First, perform boundary check to prevent IndexError
                if (0 <= center_y < mask.shape[0] and 
                    0 <= center_x < mask.shape[1] and 
                    mask[int(center_y), int(center_x)] == 0):  
                    
                    # Keep this person's data
                    kept_boxes.append(box)
                    kept_keypoints.append(keypoints[i])
                    kept_conf.append(conf[i])
            
            # Assemble new frame data
            filtered_frame = {
                'boxes': np.array(kept_boxes),
                'keypoints': np.array(kept_keypoints),
                'conf': np.array(kept_conf)
            }
            
            # Preserve annotation status if it exists in the original frame data
            if 'annotation_status' in frame_data:
                filtered_frame['annotation_status'] = frame_data['annotation_status']
            
            filtered_frames_data.append(filtered_frame)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_npz_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the filtered data
        np.savez_compressed(output_npz_path, frames=filtered_frames_data)
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_npz_path}: {e}")
        return False


def main():
    """
    Main function to filter pose data from a single .npz file corresponding to a video.
    """
    # Parse command-line arguments
    args = parse_args()
    
    print("=== Pose Data Filtering ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Video path: {args.video_path}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"⚠️  Video file not found: {args.video_path}")
        print(f"  Continuing without court filtering...")
        mask = None
        court_filtering_enabled = False
    else:
        # Generate mask on-the-fly
        print("🔄 Generating court mask...")
        try:
            detector = CourtDetector()
            mask, clean_frame, metadata = detector.process_video(args.video_path, target_time=60)
            
            # Check if court detection was successful
            if mask is None:
                print(f"⚠️  Court detection failed: {metadata.get('error', 'Unknown error')}")
                print(f"  Metadata: {metadata}")
                print(f"  Continuing without court filtering...")
                court_filtering_enabled = False
            else:
                print(f"✓ Generated mask: {mask.shape}")
                print(f"  Metadata: {metadata}")
                court_filtering_enabled = True
                
                # Save the court mask for later use in visualization
                try:
                    os.makedirs("court_masks", exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
                    mask_path = f"court_masks/{base_name}_court_mask.npz"
                    np.savez_compressed(mask_path, mask=mask, metadata=metadata)
                    print(f"✓ Court mask saved to: {mask_path}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not save court mask: {e}")
                
        except Exception as e:
            print(f"❌ Error during court detection: {e}")
            print(f"  Continuing without court filtering...")
            mask = None
            court_filtering_enabled = False
    
    # --- START OF FIX ---
    # The original code searched for all .npz files. The corrected code finds
    # the ONE specific .npz file that corresponds to the given --video-path.

    # Derive the base name of the video to find its matching data file.
    base_video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    # Use a glob pattern because the full npz filename contains extra details (model, time, etc.)
    npz_file_pattern = os.path.join(args.input_dir, f"{base_video_name}*.npz")
    matching_files = glob.glob(npz_file_pattern)

    if not matching_files:
        print(f"❌ Error: No matching .npz file found for video '{base_video_name}' in {args.input_dir}")
        return # Exit this specific run, but don't kill the whole batch process.
    
    # Assume the first match is the correct one.
    npz_file_to_process = matching_files[0]
    print(f"✓ Found corresponding .npz file to process: {os.path.basename(npz_file_to_process)}")
    print()
    # --- END OF FIX ---
    
    # Create output directory name with appropriate prefix
    input_dir_name = os.path.basename(args.input_dir)
    if court_filtering_enabled:
        output_dir_name = f"court_filtered_{input_dir_name}"
        output_dir = os.path.join("pose_data", "filtered", output_dir_name)
    else:
        # If filtering is disabled, data is considered "unfiltered" by court status
        output_dir_name = f"unfiltered_{input_dir_name}"
        output_dir = os.path.join("pose_data", "filtered", output_dir_name) # Still place in 'filtered' parent dir for consistency
    
    print(f"Output directory: {output_dir}")
    print(f"Court filtering: {'Enabled' if court_filtering_enabled else 'Disabled'}")
    print()
    
    # Process the single identified .npz file
    filename = os.path.basename(npz_file_to_process)
    output_npz_path = os.path.join(output_dir, filename)
    
    print(f"Processing: {filename}")
    
    # Check if output file already exists
    if os.path.exists(output_npz_path) and os.path.getsize(output_npz_path) > 0:
        if args.overwrite:
            print(f"  🔄 Already exists, overwriting")
        else:
            print(f"  ✓ Already exists, skipping")
            return
    
    # Filter the file
    if filter_single_npz_file(npz_file_to_process, output_npz_path, mask):
        print(f"  ✓ Filtered successfully")
    else:
        print(f"  ❌ Failed to filter")


if __name__ == "__main__":
    # Start timing
    script_start_time = time.time()
    
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
    
    # Calculate and print total runtime
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
