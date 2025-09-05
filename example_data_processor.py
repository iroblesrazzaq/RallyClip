#!/usr/bin/env python3
"""
Example script demonstrating the DataProcessor class usage.
"""

import numpy as np
from data_processor import DataProcessor

def main():
    # Initialize the data processor
    processor = DataProcessor(screen_width=1280, screen_height=720)
    
    # Example 1: Frame with two players
    print("=== Example 1: Frame with two players ===")
    frame_data = {
        'boxes': np.array([
            [100, 500, 200, 700],  # Player 1 (near, bottom of screen)
            [800, 200, 900, 400]   # Player 2 (far, top of screen)
        ]),
        'keypoints': np.random.rand(2, 17, 2) * 100,  # Random keypoints
        'conf': np.random.rand(2, 17)  # Random confidence scores
    }
    
    # Assign players
    assigned_players = processor.assign_players(frame_data)
    print(f"Near player assigned: {assigned_players['near_player'] is not None}")
    print(f"Far player assigned: {assigned_players['far_player'] is not None}")
    
    # Create feature vector
    feature_vector = processor.create_feature_vector(assigned_players)
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Number of -1 values: {np.sum(feature_vector == -1.0)}")
    print(f"First 35 elements: {feature_vector[:35]}")
    
    # Example 2: Frame with no players
    print("\n=== Example 2: Frame with no players ===")
    empty_frame_data = {
        'boxes': np.array([]),
        'keypoints': np.array([]),
        'conf': np.array([])
    }
    
    empty_assigned = processor.assign_players(empty_frame_data)
    empty_feature_vector = processor.create_feature_vector(empty_assigned)
    print(f"Near player assigned: {empty_assigned['near_player'] is not None}")
    print(f"Far player assigned: {empty_assigned['far_player'] is not None}")
    print(f"Feature vector shape: {empty_feature_vector.shape}")
    print(f"Number of -1 values: {np.sum(empty_feature_vector == -1.0)}")
    print(f"Number of 0 values (velocity/accel): {np.sum(empty_feature_vector == 0.0)}")
    
    # Example 3: Velocity calculation with consecutive frames
    print("\n=== Example 3: Velocity calculation ===")
    
    # Previous frame with specific keypoints
    prev_keypoints_near = np.array([[50, 100], [55, 105], [60, 110], [65, 115], [70, 120], 
                                   [75, 125], [80, 130], [85, 135], [90, 140], [95, 145],
                                   [100, 150], [105, 155], [110, 160], [115, 165], 
                                   [120, 170], [125, 175], [130, 180]])
    
    prev_keypoints_far = np.array([[800, 50], [805, 55], [810, 60], [815, 65], [820, 70],
                                  [825, 75], [830, 80], [835, 85], [840, 90], [845, 95],
                                  [850, 100], [855, 105], [860, 110], [865, 115],
                                  [870, 120], [875, 125], [880, 130]])
    
    prev_frame_data = {
        'boxes': np.array([[50, 400, 150, 600], [750, 100, 850, 300]]),
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
                                     [900, 150], [905, 105], [910, 160], [915, 165],
                                     [920, 170], [925, 175], [930, 180]])
    
    current_frame_data = {
        'boxes': np.array([[100, 500, 200, 700], [800, 200, 900, 400]]),
        'keypoints': np.array([current_keypoints_near, current_keypoints_far]),
        'conf': np.random.rand(2, 17)
    }
    
    current_assigned = processor.assign_players(current_frame_data)
    velocity_vector = processor.create_feature_vector(current_assigned, prev_assigned)
    
    print(f"Velocity vector shape: {velocity_vector.shape}")
    # Show player velocity components (indices 7,8 for near player; 7+144,8+144 for far player)
    near_vel_x = velocity_vector[7]
    near_vel_y = velocity_vector[8]
    far_vel_x = velocity_vector[7 + 144]  # 144 = features per player
    far_vel_y = velocity_vector[8 + 144]
    print(f"Near player velocity: ({near_vel_x}, {near_vel_y})")
    print(f"Far player velocity: ({far_vel_x}, {far_vel_y})")
    
    # Show keypoint velocity components
    # Keypoint velocity starts after player features + keypoint positions/confidences
    kp_vel_start = 1 + 11 + 34 + 17  # presence(1) + player_features(11) + keypoints(34) + conf(17)
    first_kp_vel_x = velocity_vector[kp_vel_start]
    first_kp_vel_y = velocity_vector[kp_vel_start + 1]
    print(f"Near player first keypoint velocity: ({first_kp_vel_x}, {first_kp_vel_y})")
    
    # Show limb length components
    # Limb lengths start after all previous features
    limb_start = 1 + 11 + 34 + 17 + 34 + 34  # presence + player_features + keypoints + conf + kp_vel + kp_accel
    first_limb_length = velocity_vector[limb_start]
    print(f"Near player first limb length: {first_limb_length}")
    
    # Example 4: Show centroid calculation
    print("\n=== Example 4: Centroid calculation ===")
    box = np.array([100, 200, 300, 400])  # [x1, y1, x2, y2]
    centroid = processor._calculate_centroid(box)
    print(f"Box: {box}")
    print(f"Centroid: ({centroid[0]}, {centroid[1]})")

if __name__ == "__main__":
    main()
