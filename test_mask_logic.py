#!/usr/bin/env python3
"""
Test script to verify mask interpretation logic
"""

import numpy as np

def test_mask_logic():
    print("🧪 TESTING MASK INTERPRETATION LOGIC")
    print("=" * 50)
    
    # Simulate the mask values we see
    mask_values = [0, 255]  # Inside = 0, Outside = 255
    
    print("Mask values: 0 (inside), 255 (outside)")
    print()
    
    print("FILTERING LOGIC (filter_pose_data.py):")
    print("  Uses: mask[y, x] == 0")
    for val in mask_values:
        result = val == 0
        status = "KEEP" if result else "REMOVE"
        print(f"    {val} == 0 = {result} -> {status}")
    
    print()
    print("VISUALIZATION LOGIC (video_annotator.py):")
    print("  Uses: court_mask[y, x] == 0")
    for val in mask_values:
        result = val == 0
        status = "GREEN (inside)" if result else "BLUE (outside)"
        print(f"    {val} == 0 = {result} -> {status}")
    
    print()
    print("COMPARISON:")
    print("  Both should give the same result:")
    print("  - Inside (0): KEEP + GREEN")
    print("  - Outside (255): REMOVE + BLUE")
    
    # Test with actual mask data
    print("\n" + "=" * 50)
    print("TESTING WITH ACTUAL MASK DATA")
    
    try:
        # Load the actual mask
        mask_data = np.load("court_masks/Aditi Narayan ｜ Matchplay_court_mask.npz", allow_pickle=True)
        mask = mask_data['mask']
        
        print(f"Mask loaded: {mask.shape}, dtype: {mask.dtype}")
        print(f"Unique values: {np.unique(mask)}")
        
        # Test a few sample coordinates
        test_coords = [
            (360, 640),  # Center of 720x1280 image
            (100, 100),  # Top-left area
            (600, 1000), # Bottom-right area
        ]
        
        print("\nTesting sample coordinates:")
        for y, x in test_coords:
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                mask_val = mask[y, x]
                filtering_result = ~mask_val
                visualization_result = mask_val == 0
                
                print(f"  Position ({y}, {x}): mask[{y}, {x}] = {mask_val}")
                print(f"    Filtering (~{mask_val}): {'KEEP' if filtering_result else 'REMOVE'}")
                print(f"    Visualization ({mask_val} == 0): {'GREEN' if visualization_result else 'BLUE'}")
                print(f"    Consistent: {'✓' if filtering_result == visualization_result else '❌'}")
            else:
                print(f"  Position ({y}, {x}): OUT OF BOUNDS")
                
    except Exception as e:
        print(f"❌ Error loading mask: {e}")

if __name__ == "__main__":
    test_mask_logic()
