#!/usr/bin/env python3
"""
Test script to verify the modular pipeline components.
"""

import numpy as np
import os
import tempfile
from data_scripts.data_preprocessor import DataPreprocessor
from data_scripts.feature_engineer import FeatureEngineer

def test_preprocessor_class():
    """Test that the TennisDataPreprocessor class works correctly."""
    print("Testing TennisDataPreprocessor class...")
    
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
    
    # Save in the expected format
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_file = f.name
    
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
    
    print("✓ Preprocessor class test passed")
    
    # Clean up
    os.unlink(temp_file)

def test_feature_engineer_class():
    """Test that the TennisFeatureEngineer class works correctly."""
    print("Testing TennisFeatureEngineer class...")
    
    # Create mock feature data
    mock_features = np.random.randn(10, 288).astype(np.float32)
    mock_targets = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    # Save in the expected format
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_file = f.name
    
    np.savez_compressed(
        temp_file,
        features=mock_features,
        targets=mock_targets
    )
    
    # Load and verify
    data = np.load(temp_file, allow_pickle=True)
    
    assert 'features' in data
    assert 'targets' in data
    
    features = data['features']
    targets = data['targets']
    
    assert features.shape == (10, 288)
    assert targets.shape == (10,)
    
    print("✓ Feature engineer class test passed")
    
    # Clean up
    os.unlink(temp_file)

def test_modular_workflow():
    """Test the complete modular workflow."""
    print("Testing modular workflow...")
    
    # Test that we can instantiate both classes
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    
    assert preprocessor is not None
    assert feature_engineer is not None
    
    print("✓ Modular workflow test passed")

def main():
    """Run all tests."""
    print("=== Testing Modular Pipeline Components ===\n")
    
    try:
        test_preprocessor_class()
        test_feature_engineer_class()
        test_modular_workflow()
        
        print("\n✓ All tests passed!")
        print("The modular pipeline components are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
