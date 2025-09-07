#!/usr/bin/env python3
"""
Test script to verify the draw functionality.
"""

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_draw_script():
    """Test that the draw script can be imported and has the required functions."""
    print("Testing draw script functionality...")
    
    try:
        # Import the functions we need to test
        from draw import get_video_path_from_npz, is_preprocessed_npz
        
        print("✓ Draw script imported successfully")
        print("✓ Required functions available")
        
        # Test the functions exist and are callable
        assert callable(get_video_path_from_npz)
        assert callable(is_preprocessed_npz)
        
        print("✓ Functions are callable")
        
        return True
        
    except Exception as e:
        print(f"❌ Draw script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Testing Draw Script ===\n")
    
    try:
        if test_draw_script():
            print("\n✅ All tests passed!")
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