#!/usr/bin/env python3
"""
Tennis Feature Engineer

This module contains the FeatureEngineer class that handles
feature vector creation from preprocessed tennis pose data.
"""

import numpy as np
import os

class FeatureEngineer:
    """
    Creates feature vectors from preprocessed tennis pose data.
    
    This class takes preprocessed data with player assignments and creates
    feature vectors for training the tennis point detection model.
    """
    
    def __init__(self, screen_width=1280, screen_height=720, feature_vector_size=288):
        """
        Initialize the feature engineer.
        
        Args:
            screen_width (int): Width of the video frames
            screen_height (int): Height of the video frames
            feature_vector_size (int): Size of feature vectors (default: 288)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.feature_vector_size = feature_vector_size
        self.screen_center_x = screen_width / 2
    
    def _calculate_centroid(self, box):
        """
        Calculate the centroid (center point) of a bounding box.
        
        Args:
            box (np.array): Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            tuple: (center_x, center_y) coordinates of the centroid
        """
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return (center_x, center_y)

    def _calculate_velocity(self, current_pos, previous_pos, dt=1.0):
        """
        Calculate velocity vector between two positions.
        
        Args:
            current_pos (tuple): Current (x, y) position
            previous_pos (tuple): Previous (x, y) position
            dt (float): Time difference between frames (default 1.0 for normalized units)
            
        Returns:
            tuple: (vx, vy) velocity components
        """
        if current_pos is None or previous_pos is None:
            return (0.0, 0.0)  # No measurable movement if position is missing
        vx = (current_pos[0] - previous_pos[0]) / dt
        vy = (current_pos[1] - previous_pos[1]) / dt
        return (vx, vy)

    def _calculate_acceleration(self, current_vel, previous_vel, dt=1.0):
        """
        Calculate acceleration vector between two velocities.
        
        Args:
            current_vel (tuple): Current (vx, vy) velocity
            previous_vel (tuple): Previous (vx, vy) velocity
            dt (float): Time difference between frames (default 1.0 for normalized units)
            
        Returns:
            tuple: (ax, ay) acceleration components
        """
        if current_vel is None or previous_vel is None:
            return (0.0, 0.0)  # No measurable acceleration if velocity is missing
        ax = (current_vel[0] - previous_vel[0]) / dt
        ay = (current_vel[1] - previous_vel[1]) / dt
        return (ax, ay)

    def _calculate_keypoint_velocity(self, current_keypoints, previous_keypoints, dt=1.0):
        """
        Calculate velocity for each keypoint.
        
        Args:
            current_keypoints (np.array): Current keypoints array of shape (17, 2)
            previous_keypoints (np.array): Previous keypoints array of shape (17, 2)
            dt (float): Time difference between frames
            
        Returns:
            np.array: Velocity array of shape (17, 2) with (vx, vy) for each keypoint
        """
        if current_keypoints is None or previous_keypoints is None:
            return np.zeros((17, 2))  # Return zero velocities if keypoints are missing
        
        # Calculate velocity for each keypoint
        velocities = (current_keypoints - previous_keypoints) / dt
        return velocities

    def _calculate_keypoint_acceleration(self, current_velocities, previous_velocities, dt=1.0):
        """
        Calculate acceleration for each keypoint.
        
        Args:
            current_velocities (np.array): Current velocities array of shape (17, 2)
            previous_velocities (np.array): Previous velocities array of shape (17, 2)
            dt (float): Time difference between frames
            
        Returns:
            np.array: Acceleration array of shape (17, 2) with (ax, ay) for each keypoint
        """
        if current_velocities is None or previous_velocities is None:
            return np.zeros((17, 2))  # Return zero accelerations if velocities are missing
        
        # Calculate acceleration for each keypoint
        accelerations = (current_velocities - previous_velocities) / dt
        return accelerations

    def _calculate_limb_lengths(self, keypoints):
        """
        Calculate lengths of anatomically connected limbs/joints.
        Uses standard COCO keypoint connectivity.
        
        Args:
            keypoints (np.array): Keypoints array of shape (num_keypoints, 2)
            
        Returns:
            np.array: Array of limb lengths
        """
        if keypoints is None:
            return np.full(14, -1.0)  # Return -1 for missing data
            
        # Standard COCO keypoint connections (14 main limbs)
        # Format: (keypoint_index_1, keypoint_index_2, name)
        connections = [
            (5, 7, "shoulder_left->elbow_left"),      # 0
            (7, 9, "elbow_left->wrist_left"),         # 1
            (6, 8, "shoulder_right->elbow_right"),    # 2
            (8, 10, "elbow_right->wrist_right"),      # 3
            (11, 13, "hip_left->knee_left"),          # 4
            (13, 15, "knee_left->ankle_left"),        # 5
            (12, 14, "hip_right->knee_right"),        # 6
            (14, 16, "knee_right->ankle_right"),      # 7
            (5, 6, "shoulder_left->shoulder_right"),  # 8
            (11, 12, "hip_left->hip_right"),          # 9
            (5, 11, "shoulder_left->hip_left"),       # 10
            (6, 12, "shoulder_right->hip_right"),     # 11
            (6, 5, "shoulder_right->shoulder_left"),  # 12 (redundant but useful for symmetry)
            (12, 11, "hip_right->hip_left")           # 13 (redundant but useful for symmetry)
        ]
        
        limb_lengths = []
        for i, j, name in connections:
            # Calculate Euclidean distance between connected keypoints
            if i < len(keypoints) and j < len(keypoints):
                dist = np.sqrt(np.sum((keypoints[i] - keypoints[j]) ** 2))
                limb_lengths.append(dist)
            else:
                limb_lengths.append(-1.0)  # Missing keypoint
                
        return np.array(limb_lengths)

    def create_feature_vector(self, assigned_players, previous_assigned_players=None, num_keypoints=17):
        """
        Creates a fixed-size 1D NumPy vector from the assigned player data.
        This is the designated place for feature engineering.
        
        For missing players, -1 values are used to represent absent data,
        which is outside the valid coordinate range and clearly identifiable.
        
        Velocity and acceleration are calculated only when we have consecutive detections.
        For missing frames, velocity/acceleration are set to 0 (no measurable movement).
        
        Includes velocity and acceleration for each keypoint, and limb lengths in addition to overall player features.
        
        Args:
            assigned_players (dict): The output from the `assign_players` method for current frame.
            previous_assigned_players (dict): The output from the `assign_players` method for previous frame.
            num_keypoints (int): The number of keypoints per player.

        Returns:
            np.ndarray: A flat vector ready for the LSTM.
        """
        # Define the structure: 
        # 1 (exists) + 4 (bbox) + 2 (centroid) + 2 (player velocity) + 2 (player acceleration) + 
        # 17*2 (kp_xy) + 17 (kp_conf) + 17*2 (kp_velocity) + 17*2 (kp_acceleration) + 14 (limb_lengths) = 144 features per player
        features_per_player = 1 + 4 + 2 + 2 + 2 + (num_keypoints * 3) + (num_keypoints * 2) + (num_keypoints * 2) + 14
        vector = np.full(features_per_player * 2, -1.0)  # Use -1 for missing values

        # --- Near Player ---
        if assigned_players['near_player']:
            player_data = assigned_players['near_player']
            # Mark as present
            vector[0] = 1.0
            # Calculate centroid
            centroid = self._calculate_centroid(player_data['box'])
            
            # Initialize velocity and acceleration
            velocity = (0.0, 0.0)
            acceleration = (0.0, 0.0)
            kp_velocities = np.zeros((num_keypoints, 2))
            kp_accelerations = np.zeros((num_keypoints, 2))
            limb_lengths = np.full(14, -1.0)  # Default to -1 for missing data
            
            # Calculate velocity, acceleration, and keypoint features if we have previous frame data
            if (previous_assigned_players and 
                previous_assigned_players['near_player']):
                # We have consecutive detections, calculate actual velocity
                prev_centroid = self._calculate_centroid(previous_assigned_players['near_player']['box'])
                velocity = self._calculate_velocity(centroid, prev_centroid)
                
                # Calculate keypoint velocities
                current_kps = player_data['keypoints']
                prev_kps = previous_assigned_players['near_player']['keypoints']
                kp_velocities = self._calculate_keypoint_velocity(current_kps, prev_kps)
                
                # Calculate limb lengths
                limb_lengths = self._calculate_limb_lengths(player_data['keypoints'])
                
                # For acceleration, we would need previous velocities (which would require 3 consecutive frames)
                # For now, we'll set acceleration to 0 unless we have a more sophisticated approach
                
            # Basic Features + Centroid + Velocity + Acceleration + Keypoint data + Keypoint velocity + Keypoint acceleration + Limb lengths
            flat_features = np.concatenate([
                player_data['box'],
                centroid,
                velocity,
                acceleration,
                player_data['keypoints'].flatten(),
                player_data['conf'],
                kp_velocities.flatten(),
                kp_accelerations.flatten(),
                limb_lengths
            ])
            vector[1:features_per_player] = flat_features
            
        else:
            # For missing players, set velocity and acceleration to 0 (no movement)
            # but keep the rest as -1 to indicate missing player
            offset = 1 + 4 + 2  # Skip presence, bbox, centroid
            # Set player velocity and acceleration to 0
            vector[offset:offset+4] = [0.0, 0.0, 0.0, 0.0]  # Velocity (0,0) + Acceleration (0,0)
            # Set keypoint velocity and acceleration to 0 (after skipping keypoint positions and confidences)
            kp_offset = offset + 4 + (num_keypoints * 3)  # Skip player features + keypoint positions/confidences
            vector[kp_offset:kp_offset+(num_keypoints * 4)] = 0.0  # Keypoint velocity + acceleration = 0
            # Limb lengths remain -1 (already set by initial fill)
            
        # --- Far Player ---
        offset = features_per_player
        if assigned_players['far_player']:
            player_data = assigned_players['far_player']
            # Mark as present
            vector[offset] = 1.0
            # Calculate centroid
            centroid = self._calculate_centroid(player_data['box'])
            
            # Initialize velocity and acceleration
            velocity = (0.0, 0.0)
            acceleration = (0.0, 0.0)
            kp_velocities = np.zeros((num_keypoints, 2))
            kp_accelerations = np.zeros((num_keypoints, 2))
            limb_lengths = np.full(14, -1.0)  # Default to -1 for missing data
            
            # Calculate velocity, acceleration, and keypoint features if we have previous frame data
            if (previous_assigned_players and 
                previous_assigned_players['far_player']):
                # We have consecutive detections, calculate actual velocity
                prev_centroid = self._calculate_centroid(previous_assigned_players['far_player']['box'])
                velocity = self._calculate_velocity(centroid, prev_centroid)
                
                # Calculate keypoint velocities
                current_kps = player_data['keypoints']
                prev_kps = previous_assigned_players['far_player']['keypoints']
                kp_velocities = self._calculate_keypoint_velocity(current_kps, prev_kps)
                
                # Calculate limb lengths
                limb_lengths = self._calculate_limb_lengths(player_data['keypoints'])
                
            # Basic Features + Centroid + Velocity + Acceleration + Keypoint data + Keypoint velocity + Keypoint acceleration + Limb lengths
            flat_features = np.concatenate([
                player_data['box'],
                centroid,
                velocity,
                acceleration,
                player_data['keypoints'].flatten(),
                player_data['conf'],
                kp_velocities.flatten(),
                kp_accelerations.flatten(),
                limb_lengths
            ])
            vector[offset+1 : offset+features_per_player] = flat_features
            
        else:
            # For missing players, set velocity and acceleration to 0 (no movement)
            # but keep the rest as -1 to indicate missing player
            pos_offset = offset + 1 + 4 + 2  # Skip presence, bbox, centroid
            # Set player velocity and acceleration to 0
            vector[pos_offset:pos_offset+4] = [0.0, 0.0, 0.0, 0.0]  # Velocity (0,0) + Acceleration (0,0)
            # Set keypoint velocity and acceleration to 0 (after skipping keypoint positions and confidences)
            kp_offset = pos_offset + 4 + (num_keypoints * 3)  # Skip player features + keypoint positions/confidences
            vector[kp_offset:kp_offset+(num_keypoints * 4)] = 0.0  # Keypoint velocity + acceleration = 0
            # Limb lengths remain -1 (already set by initial fill)
            
        return vector
    
    def create_features_from_preprocessed(self, input_npz_path, output_file, overwrite=False):
        """
        Create feature vectors from preprocessed pose data.
        
        Args:
            input_npz_path (str): Path to preprocessed .npz file
            output_file (str): Path to output .npz file
            overwrite (bool): Whether to overwrite existing files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if output file already exists
            if os.path.exists(output_file) and not overwrite:
                print(f"  ✓ Already exists, skipping: {os.path.basename(output_file)}")
                return True
            
            # Load preprocessed data
            print(f"  Loading preprocessed data from: {input_npz_path}")
            data = np.load(input_npz_path, allow_pickle=True)
            
            # Extract arrays
            frames = data['frames']
            targets = data['targets']
            near_players = data['near_players']
            far_players = data['far_players']
            
            print(f"  Loaded {len(frames)} frames")
            print(f"  Annotation status distribution:")
            print(f"    -100 (skipped): {np.sum(targets == -100)}")
            print(f"    0 (not in play): {np.sum(targets == 0)}")
            print(f"    1 (in play): {np.sum(targets == 1)}")
            
            # Create feature vectors only for annotated frames (status >= 0)
            annotated_indices = np.where(targets >= 0)[0]
            print(f"  Creating features for {len(annotated_indices)} annotated frames")
            
            feature_vectors = []
            feature_targets = []
            previous_players = None
            
            for idx in annotated_indices:
                # Create assigned players dictionary for this frame
                assigned_players = {
                    'near_player': near_players[idx],
                    'far_player': far_players[idx]
                }
                
                # Create feature vector
                feature_vector = self.create_feature_vector(assigned_players, previous_players)
                
                feature_vectors.append(feature_vector)
                feature_targets.append(targets[idx])
                
                # Update previous players for velocity/acceleration calculations
                previous_players = assigned_players
                
                # Progress indicator
                if len(feature_vectors) % 100 == 0:
                    print(f"    Created features for {len(feature_vectors)}/{len(annotated_indices)} annotated frames")
            
            # Convert to numpy arrays
            if feature_vectors:
                feature_array = np.array(feature_vectors)
                target_array = np.array(feature_targets)
                
                print(f"  Feature array shape: {feature_array.shape}")
                print(f"  Target array shape: {target_array.shape}")
            else:
                # Create empty arrays with correct shapes
                feature_array = np.empty((0, self.feature_vector_size))
                target_array = np.empty((0,))
                print(f"  No annotated frames found, creating empty arrays")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save feature vectors and targets
            np.savez_compressed(
                output_file,
                features=feature_array,
                targets=target_array
            )
            
            print(f"  ✓ Features saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing {input_npz_path}: {e}")
            import traceback
            traceback.print_exc()
            return False