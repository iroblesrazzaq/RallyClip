#!/usr/bin/env python3
"""
Comprehensive test demonstrating the DataProcessor with velocity/acceleration features.
"""

import numpy as np
from data_scripts.data_processor import DataProcessor

def test_comprehensive_scenario():
    """Test a comprehensive scenario with various player detection patterns."""
    processor = DataProcessor(screen_width=1280, screen_height=720)
    
    print("=== Comprehensive DataProcessor Test ===\n")
    
    # Simulate a sequence of frames with different player detection patterns
    frame_sequence = [
        # Frame 1: Both players detected
        {
            'boxes': np.array([[100, 500, 200, 700], [800, 200, 900, 400]]),
            'description': 'Both players present'
        },
        # Frame 2: Only near player detected
        {
            'boxes': np.array([[150, 550, 250, 750]]),
            'description': 'Only near player present'
        },
        # Frame 3: No players detected
        {
            'boxes': np.array([]),
            'description': 'No players detected'
        },
        # Frame 4: Both players detected again
        {
            'boxes': np.array([[200, 600, 300, 800], [850, 250, 950, 450]]),
            'description': 'Both players present (moved)'
        }
    ]
    
    # Process each frame
    previous_assigned = None
    for i, frame_info in enumerate(frame_sequence):
        print(f"--- Frame {i+1}: {frame_info['description']} ---")
        
        # Create frame data with specific keypoints for testing
        keypoints_data = []
        for j in range(len(frame_info['boxes'])):
            # Create specific keypoints for each player
            base_x = frame_info['boxes'][j][0]
            base_y = frame_info['boxes'][j][1]
            keypoints = np.array([[base_x + k*5, base_y + k*5] for k in range(17)])
            keypoints_data.append(keypoints)
        
        frame_data = {
            'boxes': frame_info['boxes'],
            'keypoints': np.array(keypoints_data) if keypoints_data else np.array([]),
            'conf': np.random.rand(len(frame_info['boxes']), 17) if frame_info['boxes'].size > 0 else np.array([])
        }
        
        # Assign players
        assigned_players = processor.assign_players(frame_data)
        
        # Create feature vector
        feature_vector = processor.create_feature_vector(assigned_players, previous_assigned)
        
        # Analyze results
        near_present = assigned_players['near_player'] is not None
        far_present = assigned_players['far_player'] is not None
        
        print(f"  Players detected - Near: {near_present}, Far: {far_present}")
        print(f"  Feature vector shape: {feature_vector.shape}")
        print(f"  -1 values (missing data): {np.sum(feature_vector == -1.0)}")
        print(f"  0 values (velocity/accel): {np.sum(feature_vector == 0.0)}")
        
        # Check specific features for present players
        if near_present:
            # Near player features start at index 0
            near_centroid_x = feature_vector[5]   # presence(1) + bbox(4) + centroid_x(1)
            near_centroid_y = feature_vector[6]   # presence(1) + bbox(4) + centroid_x(1) + centroid_y(1)
            near_vel_x = feature_vector[7]        # + velocity_x(1)
            near_vel_y = feature_vector[8]        # + velocity_y(1)
            print(f"  Near player - Centroid: ({near_centroid_x}, {near_centroid_y}), Velocity: ({near_vel_x}, {near_vel_y})")
            
            # Check keypoint velocity (after player features + keypoint positions/confidences)
            kp_vel_start = 1 + 11 + 34 + 17  # presence(1) + player_features(11) + keypoints(34) + conf(17)
            if len(feature_vector) > kp_vel_start:
                first_kp_vel_x = feature_vector[kp_vel_start]
                first_kp_vel_y = feature_vector[kp_vel_start + 1]
                print(f"  Near player - First keypoint velocity: ({first_kp_vel_x}, {first_kp_vel_y})")
            
            # Check limb lengths (after all previous features)
            limb_start = 1 + 11 + 34 + 17 + 34 + 34  # presence + player_features + keypoints + conf + kp_vel + kp_accel
            if len(feature_vector) > limb_start:
                first_limb_length = feature_vector[limb_start]
                print(f"  Near player - First limb length: {first_limb_length}")
        
        if far_present:
            # Far player features start at index 144 (features per player)
            far_centroid_x = feature_vector[144 + 5]  # offset + presence(1) + bbox(4) + centroid_x(1)
            far_centroid_y = feature_vector[144 + 6]  # offset + presence(1) + bbox(4) + centroid_x(1) + centroid_y(1)
            far_vel_x = feature_vector[144 + 7]       # + velocity_x(1)
            far_vel_y = feature_vector[144 + 8]       # + velocity_y(1)
            print(f"  Far player - Centroid: ({far_centroid_x}, {far_centroid_y}), Velocity: ({far_vel_x}, {far_vel_y})")
        
        print()
        previous_assigned = assigned_players

    print("=== Test Complete ===")

def test_velocity_acceleration_accuracy():
    """Test the accuracy of velocity and acceleration calculations."""
    processor = DataProcessor(screen_width=1280, screen_height=720)
    
    print("=== Velocity/Acceleration Accuracy Test ===\n")
    
    # Create a sequence with known movements
    positions = [
        (100, 200),  # Frame 1
        (150, 250),  # Frame 2 - moved (+50, +50)
        (200, 300),  # Frame 3 - moved (+50, +50) again
        (180, 280),  # Frame 4 - moved (-20, -20)
    ]
    
    # Create corresponding keypoints for each position
    keypoints_sequence = []
    for pos in positions:
        x, y = pos
        keypoints = np.array([[x + k, y + k] for k in range(17)])
        keypoints_sequence.append(keypoints)
    
    previous_assigned = None
    for i in range(len(positions)):
        x, y = positions[i]
        box = np.array([x-50, y-50, x+50, y+50])  # Create box around position
        
        frame_data = {
            'boxes': np.array([box]),
            'keypoints': np.array([keypoints_sequence[i]]),
            'conf': np.random.rand(1, 17)
        }
        
        assigned_players = processor.assign_players(frame_data)
        feature_vector = processor.create_feature_vector(assigned_players, previous_assigned)
        
        # Extract velocity (assuming only near player)
        vel_x = feature_vector[7]
        vel_y = feature_vector[8]
        
        # Extract keypoint velocity
        kp_vel_start = 1 + 11 + 34 + 17  # presence(1) + player_features(11) + keypoints(34) + conf(17)
        first_kp_vel_x = feature_vector[kp_vel_start]
        first_kp_vel_y = feature_vector[kp_vel_start + 1]
        
        # Extract limb length
        limb_start = 1 + 11 + 34 + 17 + 34 + 34  # All previous features
        first_limb_length = feature_vector[limb_start]
        
        print(f"Frame {i+1}: Position ({x}, {y})")
        if i > 0:
            expected_vel_x = x - positions[i-1][0]
            expected_vel_y = y - positions[i-1][1]
            print(f"  Player Velocity - Calculated: ({vel_x}, {vel_y}), Expected: ({expected_vel_x}, {expected_vel_y})")
            print(f"  Player Accuracy: vel_x={'✓' if vel_x == expected_vel_x else '✗'}, vel_y={'✓' if vel_y == expected_vel_y else '✗'}")
            
            # Keypoint velocity should match player velocity for consistent movement
            print(f"  Keypoint Velocity - First keypoint: ({first_kp_vel_x}, {first_kp_vel_y})")
            print(f"  Keypoint Accuracy: vel_x={'✓' if first_kp_vel_x == expected_vel_x else '✗'}, vel_y={'✓' if first_kp_vel_y == expected_vel_y else '✗'}")
            
            print(f"  First Limb Length: {first_limb_length}")
        else:
            print(f"  Player Velocity - Calculated: ({vel_x}, {vel_y}) (no previous frame for comparison)")
            print(f"  Keypoint Velocity - First keypoint: ({first_kp_vel_x}, {first_kp_vel_y})")
            print(f"  First Limb Length: {first_limb_length}")
        print()
        
        previous_assigned = assigned_players

if __name__ == "__main__":
    test_comprehensive_scenario()
    print("\n" + "="*50 + "\n")
    test_velocity_acceleration_accuracy()