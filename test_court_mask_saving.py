#!/usr/bin/env python3
"""
Test script to verify court mask saving functionality.
"""

import numpy as np
import tempfile
import os
from data_scripts.data_preprocessor import DataPreprocessor

def test_court_mask_saving():
    """Test that court masks are saved when requested."""
    print("Testing court mask saving functionality...")
    
    # Create mock data in the expected format
    mock_frames = [
        {
            'boxes': np.array([[100, 200, 300, 400]]),
            'keypoints': np.array([[[150, 250], [160, 260], [170, 270]]]),
            'conf': np.array([[0.9, 0.8, 0.7]])
        },
        {
            'boxes': np.array([]),
            'keypoints': np.array([]),
            'conf': np.array([])
        }
    ]
    
    mock_targets = np.array([1, -100])
    mock_near_players = [
        {
            'box': np.array([100, 200, 300, 400]),
            'keypoints': np.array([[150, 250], [160, 260], [170, 270]]),
            'conf': np.array([0.9, 0.8, 0.7])
        },
        None
    ]
    mock_far_players = [None, None]
    
    # Create a mock court mask
    mock_mask = np.zeros((720, 1280), dtype=np.uint8)
    
    # Test with save_court_masks=True
    print("Testing with save_court_masks=True...")
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_file = f.name
    
    try:
        # Save with court mask
        np.savez_compressed(
            temp_file,
            frames=mock_frames,
            targets=mock_targets,
            near_players=mock_near_players,
            far_players=mock_far_players,
            court_mask=mock_mask
        )
        
        # Load and verify
        data = np.load(temp_file, allow_pickle=True)
        
        assert 'frames' in data
        assert 'targets' in data
        assert 'near_players' in data
        assert 'far_players' in data
        assert 'court_mask' in data
        
        print("✓ Court mask saved successfully when requested")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    # Test without court mask (default behavior)
    print("Testing without court mask...")
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_file = f.name
    
    try:
        # Save without court mask
        np.savez_compressed(
            temp_file,
            frames=mock_frames,
            targets=mock_targets,
            near_players=mock_near_players,
            far_players=mock_far_players
        )
        
        # Load and verify
        data = np.load(temp_file, allow_pickle=True)
        
        assert 'frames' in data
        assert 'targets' in data
        assert 'near_players' in data
        assert 'far_players' in data
        assert 'court_mask' not in data
        
        print("✓ No court mask saved when not requested")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    print("\n✅ Court mask saving functionality test passed!")

def main():
    """Run all tests."""
    print("=== Testing Court Mask Saving Functionality ===\n")
    
    try:
        test_court_mask_saving()
        print("\n🎉 All tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()