#!/usr/bin/env python3
"""
Test to verify centroid values for missing players.
"""

import numpy as np
from data_scripts.data_processor import DataProcessor

def test_centroid_for_missing_players():
    # Initialize the data processor
    processor = DataProcessor(screen_width=1280, screen_height=720)
    
    # Test case: Only near player present
    print("=== Testing centroid for missing far player ===")
    player1_box = np.array([100, 500, 200, 700])  # [x1, y1, x2, y2]
    player1_keypoints = np.random.rand(17, 2) * 100  # Random keypoints
    player1_conf = np.random.rand(17)  # Random confidence scores
    
    near_only_frame_data = {
        'boxes': np.array([player1_box]),
        'keypoints': np.array([player1_keypoints]),
        'conf': np.array([player1_conf])
    }
    
    near_only_assigned = processor.assign_players(near_only_frame_data)
    near_only_feature_vector = processor.create_feature_vector(near_only_assigned)
    
    print(f"Near player assigned: {near_only_assigned['near_player'] is not None}")
    print(f"Far player assigned: {near_only_assigned['far_player'] is not None}")
    print(f"Feature vector shape: {near_only_feature_vector.shape}")
    
    # Check near player features (first 58 elements)
    near_player_features = near_only_feature_vector[:58]
    print(f"Near player features (first 10): {near_player_features[:10]}")
    print(f"Near player has -1 values: {np.any(near_player_features == -1.0)}")
    
    # Check far player features (last 58 elements)
    far_player_features = near_only_feature_vector[58:]
    print(f"Far player features (first 10): {far_player_features[:10]}")
    print(f"Far player all -1 values: {np.all(far_player_features == -1.0)}")
    
    # Specifically check centroid values for missing far player
    # Centroid should be at indices 5 and 6 within the far player section (5+58 and 6+58 in the full vector)
    far_centroid_x = near_only_feature_vector[58 + 5]  # presence(1) + bbox(4) + centroid_x(1) = index 5 in player section
    far_centroid_y = near_only_feature_vector[58 + 6]  # presence(1) + bbox(4) + centroid_x(1) + centroid_y(1) = index 6 in player section
    print(f"Far player centroid X: {far_centroid_x} (should be -1)")
    print(f"Far player centroid Y: {far_centroid_y} (should be -1)")
    
    # Verify near player centroid is calculated correctly
    near_centroid_x = near_only_feature_vector[5]  # presence(1) + bbox(4) + centroid_x(1) = index 5 in vector
    near_centroid_y = near_only_feature_vector[6]  # presence(1) + bbox(4) + centroid_x(1) + centroid_y(1) = index 6 in vector
    expected_centroid_x = (100 + 200) / 2  # (x1 + x2) / 2
    expected_centroid_y = (500 + 700) / 2  # (y1 + y2) / 2
    print(f"Near player centroid X: {near_centroid_x} (expected: {expected_centroid_x})")
    print(f"Near player centroid Y: {near_centroid_y} (expected: {expected_centroid_y})")

if __name__ == "__main__":
    test_centroid_for_missing_players()