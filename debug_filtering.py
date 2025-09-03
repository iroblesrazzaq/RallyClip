#!/usr/bin/env python3
"""
Debug script to investigate filtering inconsistency
"""

import numpy as np
import os
import glob

def debug_filtering():
    # Paths
    unfiltered_dir = "pose_data/unfiltered/yolos_0.2conf_10fps_0s_to_999999s"
    filtered_dir = "pose_data/filtered/court_filtered_yolos_0.2conf_10fps_0s_to_999999s"
    court_masks_dir = "court_masks"
    
    print("🔍 DEBUGGING FILTERING INCONSISTENCY")
    print("=" * 50)
    
    # Check if directories exist
    print(f"Unfiltered dir exists: {os.path.exists(unfiltered_dir)}")
    print(f"Filtered dir exists: {os.path.exists(filtered_dir)}")
    print(f"Court masks dir exists: {os.path.exists(court_masks_dir)}")
    
    # Get sample files
    unfiltered_files = glob.glob(os.path.join(unfiltered_dir, "*.npz"))
    filtered_files = glob.glob(os.path.join(filtered_dir, "*.npz"))
    mask_files = glob.glob(os.path.join(court_masks_dir, "*_court_mask.npz"))
    
    print(f"\nFound {len(unfiltered_files)} unfiltered files")
    print(f"Found {len(filtered_files)} filtered files")
    print(f"Found {len(mask_files)} court mask files")
    
    if not unfiltered_files or not filtered_files:
        print("❌ Missing files for comparison")
        return
    
    # Compare first file
    unfiltered_file = unfiltered_files[0]
    filtered_file = filtered_files[0]
    
    print(f"\n📊 COMPARING FIRST FILE:")
    print(f"Unfiltered: {os.path.basename(unfiltered_file)}")
    print(f"Filtered: {os.path.basename(filtered_file)}")
    
    # Extract video name from NPZ filename
    base_name = os.path.basename(unfiltered_file)
    video_name = base_name.replace("_posedata_0s_to_999999s_yolos.npz", "")
    expected_mask = f"{video_name}_court_mask.npz"
    expected_mask_path = os.path.join(court_masks_dir, expected_mask)
    
    print(f"Video name extracted: {video_name}")
    print(f"Expected mask: {expected_mask}")
    print(f"Expected mask exists: {os.path.exists(expected_mask_path)}")
    
    # Show all available masks
    print(f"\nAvailable court masks:")
    for mask_file in mask_files:
        mask_name = os.path.basename(mask_file)
        print(f"  - {mask_name}")
    
    # Load data
    try:
        unfiltered_data = np.load(unfiltered_file, allow_pickle=True)['frames']
        filtered_data = np.load(filtered_file, allow_pickle=True)['frames']
        
        print(f"Unfiltered frames: {len(unfiltered_data)}")
        print(f"Filtered frames: {len(filtered_data)}")
        
        # Compare first few frames
        for frame_idx in range(min(3, len(unfiltered_data), len(filtered_data))):
            print(f"\nFrame {frame_idx}:")
            
            if frame_idx < len(unfiltered_data):
                unfiltered_frame = unfiltered_data[frame_idx]
                unfiltered_boxes = unfiltered_frame['boxes']
                print(f"  Unfiltered boxes: {len(unfiltered_boxes)}")
                
                # Show first few boxes
                for i, box in enumerate(unfiltered_boxes[:3]):
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    print(f"    Box {i}: center=({center_x:.1f}, {center_y:.1f})")
            
            if frame_idx < len(filtered_data):
                filtered_frame = filtered_data[frame_idx]
                filtered_boxes = filtered_frame['boxes']
                print(f"  Filtered boxes: {len(filtered_boxes)}")
                
                # Show first few boxes
                for i, box in enumerate(filtered_boxes[:3]):
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    print(f"    Box {i}: center=({center_x:.1f}, {center_y:.1f})")
        
        # Check if filtering actually reduced data
        total_unfiltered = sum(len(frame['boxes']) for frame in unfiltered_data)
        total_filtered = sum(len(frame['boxes']) for frame in filtered_data)
        
        print(f"\n📈 TOTAL BOXES:")
        print(f"Unfiltered: {total_unfiltered}")
        print(f"Filtered: {total_filtered}")
        print(f"Reduction: {total_unfiltered - total_filtered} boxes removed")
        
        if total_unfiltered == total_filtered:
            print("⚠️  WARNING: No filtering occurred - all boxes were kept!")
        else:
            print(f"✓ Filtering removed {total_unfiltered - total_filtered} boxes")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
    
    # Check court mask
    if mask_files:
        # Use the expected mask if it exists, otherwise use the first available
        if os.path.exists(expected_mask_path):
            mask_file = expected_mask_path
            print(f"\n🎾 COURT MASK ANALYSIS (EXPECTED MASK):")
        else:
            mask_file = mask_files[0]
            print(f"\n🎾 COURT MASK ANALYSIS (FALLBACK MASK - NOT THE RIGHT ONE!):")
        
        print(f"Mask file: {os.path.basename(mask_file)}")
        
        try:
            mask_data = np.load(mask_file, allow_pickle=True)
            mask = mask_data['mask']
            
            print(f"Mask shape: {mask.shape}")
            print(f"Mask dtype: {mask.dtype}")
            print(f"Mask unique values: {np.unique(mask)}")
            print(f"Mask min/max: {mask.min()}/{mask.max()}")
            
            # Count pixels
            total_pixels = mask.size
            inside_pixels = np.sum(mask == 0)
            outside_pixels = np.sum(mask != 0)
            
            print(f"Total pixels: {total_pixels}")
            print(f"Inside pixels (0): {inside_pixels} ({inside_pixels/total_pixels*100:.1f}%)")
            print(f"Outside pixels (non-0): {outside_pixels} ({outside_pixels/total_pixels*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error loading mask: {e}")

if __name__ == "__main__":
    debug_filtering()
