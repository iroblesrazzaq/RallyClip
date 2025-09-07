#!/usr/bin/env python3
"""
Test script to verify the data processor is working correctly.
"""

import numpy as np
from data_scripts.data_processor import DataProcessor

def test_data_processor():
    """Test the data processor with a simple example."""
    # Initialize the data processor
    processor = DataProcessor(screen_width=1280, screen_height=720)
    
    # Create some test data with two players
    # Player 1 (near player) - bottom half of screen
    player1_box = np.array([100, 500, 200, 700])  # [x1, y1, x2, y2]
    player1_keypoints = np.random.rand(17, 2) * 100  # Random keypoints
    player1_conf = np.random.rand(17)  # Random confidence scores
    
    # Player 2 (far player) - top half of screen
    player2_box = np.array([800, 200, 900, 400])  # [x1, y1, x2, y2]
    player2_keypoints = np.random.rand(17, 2) * 100  # Random keypoints
    player2_conf = np.random.rand(17)  # Random confidence scores
    
    # Create frame data
    frame_data = {
        'boxes': np.array([player1_box, player2_box]),
        'keypoints': np.array([player1_keypoints, player2_keypoints]),
        'conf': np.array([player1_conf, player2_conf])
    }
    
    # Apply the data processor
    assigned_players = processor.assign_players(frame_data)
    
    # Create feature vector (no previous frame)
    feature_vector = processor.create_feature_vector(assigned_players)
    
    # Print results
    print("Test Results:")
    print(f"Near player assigned: {assigned_players['near_player'] is not None}")
    print(f"Far player assigned: {assigned_players['far_player'] is not None}")
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Feature vector (first 30 elements): {feature_vector[:30]}")
    print(f"Number of -1 values in feature vector: {np.sum(feature_vector == -1.0)}")
    
    # Test with missing players
    print("\nTesting with missing players...")
    
    # Case 1: No players detected
    empty_frame_data = {
        'boxes': np.array([]),
        'keypoints': np.array([]),
        'conf': np.array([])
    }
    
    empty_assigned = processor.assign_players(empty_frame_data)
    empty_feature_vector = processor.create_feature_vector(empty_assigned)
    
    print(f"Empty frame - Near player assigned: {empty_assigned['near_player'] is not None}")
    print(f"Empty frame - Far player assigned: {empty_assigned['far_player'] is not None}")
    print(f"Empty frame - Feature vector shape: {empty_feature_vector.shape}")
    print(f"Empty frame - Number of -1 values: {np.sum(empty_feature_vector == -1.0)}")
    print(f"Empty frame - Number of 0 values (velocity/accel): {np.sum(empty_feature_vector == 0.0)}")
    
    # Test velocity calculation with consecutive detections
    print("\nTesting velocity calculation...")
    
    # Previous frame with specific keypoints for testing
    prev_keypoints_near = np.array([[50, 100], [55, 105], [60, 110], [65, 115], [70, 120], 
                                   [75, 125], [80, 130], [85, 135], [90, 140], [95, 145],
                                   [100, 150], [105, 155], [110, 160], [115, 165], 
                                   [120, 170], [125, 175], [130, 180]])
    
    prev_keypoints_far = np.array([[800, 50], [805, 55], [810, 60], [815, 65], [820, 70],
                                  [825, 75], [830, 80], [835, 85], [840, 90], [845, 95],
                                  [850, 100], [855, 105], [860, 110], [865, 115],
                                  [870, 120], [875, 125], [880, 130]])
    
    prev_frame_data = {
        'boxes': np.array([[50, 400, 150, 600], [750, 100, 850, 300]]),  # Previous positions
        'keypoints': np.array([prev_keypoints_near, prev_keypoints_far]),
        'conf': np.random.rand(2, 17)
    }
    
    prev_assigned = processor.assign_players(prev_frame_data)
    
    # Current frame (with movement)
    current_keypoints_near = np.array([[100, 150], [105, 155], [110, 160], [115, 165], [120, 170],
                                      [125, 175], [130, 180], [135, 185], [140, 190], [145, 195],
                                      [150, 200], [155, 205], [160, 210], [165, 215],
                                      [170, 220], [175, 225], [180, 230]])
    
    current_keypoints_far = np.array([[850, 100], [855, 105], [860, 110], [865, 115], [870, 120],
                                     [875, 125], [880, 130], [885, 135], [890, 140], [895, 145],
                                     [900, 150], [905, 155], [910, 160], [915, 165],
                                     [920, 170], [925, 175], [930, 180]])
    
    current_frame_data = {
        'boxes': np.array([[100, 500, 200, 700], [800, 200, 900, 400]]),  # Moved positions
        'keypoints': np.array([current_keypoints_near, current_keypoints_far]),
        'conf': np.random.rand(2, 17)
    }
    
    current_assigned = processor.assign_players(current_frame_data)
    velocity_feature_vector = processor.create_feature_vector(current_assigned, prev_assigned)
    
    print(f"Velocity test - Feature vector shape: {velocity_feature_vector.shape}")
    print(f"Velocity test - First 35 elements: {velocity_feature_vector[:35]}")
    
    # Check that velocity values are not all zero (they should have actual values)
    # Player velocity starts at index 7 (presence=1 + bbox=4 + centroid=2)
    near_vel_x = velocity_feature_vector[7]
    near_vel_y = velocity_feature_vector[8]
    print(f"Near player velocity: ({near_vel_x}, {near_vel_y})")
    print(f"Velocity non-zero: {near_vel_x != 0.0 or near_vel_y != 0.0}")
    
    # Check keypoint velocity (starts after player features + keypoint positions/confidences)
    # Player features: 11 (presence + bbox + centroid + velocity + acceleration)
    # Keypoint positions: 34, Keypoint confidences: 17
    # Keypoint velocity starts at index: 1 + 11 + 34 + 17 = 63
    kp_vel_start = 1 + 11 + 34 + 17
    first_kp_vel_x = velocity_feature_vector[kp_vel_start]
    first_kp_vel_y = velocity_feature_vector[kp_vel_start + 1]
    print(f"First keypoint velocity: ({first_kp_vel_x}, {first_kp_vel_y})")
    print(f"Keypoint velocity non-zero: {first_kp_vel_x != 0.0 or first_kp_vel_y != 0.0}")
    
    # Check limb lengths (starts after all previous features)
    # Total features before limb lengths: 1 + 11 + 34 + 17 + 34 + 34 = 131
    limb_start = 1 + 11 + 34 + 17 + 34 + 34
    first_limb_length = velocity_feature_vector[limb_start]
    print(f"First limb length: {first_limb_length}")
    print(f"Limb length not -1: {first_limb_length != -1.0}")

if __name__ == "__main__":
    test_data_processor()
