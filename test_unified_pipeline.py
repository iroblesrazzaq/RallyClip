#!/usr/bin/env python3
"""
Comprehensive test script to verify the unified pipeline components.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pipeline_components():
    """Test that all pipeline components can be imported and used."""
    print("Testing unified pipeline components...")
    
    # Test DataPreprocessor import
    try:
        from data_scripts.data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor(save_court_masks=True)
        print("✓ DataPreprocessor imported and instantiated successfully")
    except Exception as e:
        print(f"❌ DataPreprocessor import failed: {e}")
        return False
    
    # Test FeatureEngineer import
    try:
        from data_scripts.feature_engineer import FeatureEngineer
        feature_engineer = FeatureEngineer()
        print("✓ FeatureEngineer imported and instantiated successfully")
    except Exception as e:
        print(f"❌ FeatureEngineer import failed: {e}")
        return False
    
    # Test main pipeline script exists
    if os.path.exists("run_pipeline.py"):
        print("✓ run_pipeline.py exists")
    else:
        print("❌ run_pipeline.py not found")
        return False
    
    # Test config files exist
    config_files = [
        "data_configs/config1.json",
        "data_configs/config_all_videos.json",
        "data_configs/config_selected_videos.json",
        "data_configs/config_stepwise.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file} exists")
        else:
            print(f"❌ {config_file} not found")
            return False
    
    print("\n✅ All pipeline components verified successfully!")
    return True

def main():
    """Run all tests."""
    print("=== Testing Unified Pipeline Components ===\n")
    
    try:
        if test_pipeline_components():
            print("\n🎉 All tests passed!")
            return True
        else:
            print("\n💥 Some tests failed!")
            return False
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()