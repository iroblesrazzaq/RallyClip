# data_processor.py
import numpy as np

class DataProcessor:
    """
    Processes pose data and extracts features for tennis player tracking.
    Separates data processing from feature engineering for flexibility.
    """
    def __init__(self, screen_width=1280, screen_height=720, merge_iou_thresh=0.6):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_width / 2
        self.merge_iou_thresh = merge_iou_thresh
        
        # Define the edge zones for conditional merging
        self.left_zone_x = screen_width * 0.10
        self.right_zone_x = screen_width * 0.90
        self.bottom_zone_y = screen_height * 0.80

    def _calculate_iou(self, box1, box2):
        """Calculate the Intersection over Union of two bounding boxes."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _conditional_merge_boxes(self, boxes, keypoints, confs):
        """Implement the conditional merge logic for edge-zone merging."""
        if len(boxes) <= 1:
            return boxes, keypoints, confs

        # 1. Group boxes into clumps based on IoU
        detections = [{'box': boxes[i], 'keypoints': keypoints[i], 'conf': confs[i], 'clump_id': -1} for i in range(len(boxes))]
        clump_count = 0
        for i in range(len(detections)):
            if detections[i]['clump_id'] == -1:
                detections[i]['clump_id'] = clump_count
                for j in range(i + 1, len(detections)):
                    if self._calculate_iou(detections[i]['box'], detections[j]['box']) > self.merge_iou_thresh:
                        detections[j]['clump_id'] = clump_count
                clump_count += 1
        
        if clump_count == len(detections): 
            return boxes, keypoints, confs

        # 2. Process each clump
        final_boxes, final_keypoints, final_confs = [], [], []
        for clump_id in range(clump_count):
            clump = [d for d in detections if d['clump_id'] == clump_id]
            if len(clump) == 1:
                final_boxes.append(clump[0]['box'])
                final_keypoints.append(clump[0]['keypoints'])
                final_confs.append(clump[0]['conf'])
                continue

            # 3. Check if the clump is in an edge zone
            min_x1 = min(d['box'][0] for d in clump)
            min_y1 = min(d['box'][1] for d in clump)
            max_x2 = max(d['box'][2] for d in clump)
            max_y2 = max(d['box'][3] for d in clump)
            clump_center_x = (min_x1 + max_x2) / 2
            
            is_in_edge_zone = (
                clump_center_x < self.left_zone_x or
                clump_center_x > self.right_zone_x or
                max_y2 > self.bottom_zone_y # Check the bottom of the merged box
            )

            # 4. Merge if in zone, otherwise keep separate
            if is_in_edge_zone:
                merged_box = [min_x1, min_y1, max_x2, max_y2]
                # Use data from the largest original box in the clump
                best_detection = max(clump, key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]))
                final_boxes.append(merged_box)
                final_keypoints.append(best_detection['keypoints'])
                final_confs.append(best_detection['conf'])
            else: # Not in an edge zone, so do not merge
                for d in clump:
                    final_boxes.append(d['box'])
                    final_keypoints.append(d['keypoints'])
                    final_confs.append(d['conf'])

        return np.array(final_boxes), np.array(final_keypoints), np.array(final_confs)

    def assign_players(self, frame_data):
        """
        Applies the v3 heuristic to identify and assign near and far players.
        
        Returns:
            dict: A dictionary with 'near_player' and 'far_player' keys.
                  Each value is a detection dictionary or None if not found.
        """
        boxes = frame_data.get('boxes', np.array([]))
        keypoints = frame_data.get('keypoints', np.array([]))
        confs = frame_data.get('conf', np.array([]))

        # 1. Conditional Merge
        clean_boxes, clean_keypoints, clean_confs = self._conditional_merge_boxes(boxes, keypoints, confs)

        assigned_players = {'near_player': None, 'far_player': None}
        
        if len(clean_boxes) == 0:
            return assigned_players

        # Create a list of candidate detections
        candidates = [{
            'box': clean_boxes[i], 
            'keypoints': clean_keypoints[i], 
            'conf': clean_confs[i]
        } for i in range(len(clean_boxes))]

        # 2. Find Near Player (lowest bottom edge)
        near_player_idx = max(range(len(candidates)), key=lambda i: candidates[i]['box'][3])
        near_player = candidates[near_player_idx]
        assigned_players['near_player'] = near_player
        del candidates[near_player_idx]

        # 3. Find Far Player (closest to center line)
        if len(candidates) > 0:
            far_player_idx = min(range(len(candidates)), key=lambda i: abs(((candidates[i]['box'][0] + candidates[i]['box'][2]) / 2) - self.screen_center_x))
            far_player = candidates[far_player_idx]
            assigned_players['far_player'] = far_player
            
        return assigned_players

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
            current_keypoints (np.array): Current keypoints array of shape (num_keypoints, 2)
            previous_keypoints (np.array): Previous keypoints array of shape (num_keypoints, 2)
            dt (float): Time difference between frames
            
        Returns:
            np.array: Velocity array of shape (num_keypoints, 2) with (vx, vy) for each keypoint
        """
        if current_keypoints is None or previous_keypoints is None:
            # Return zeros for all keypoints if data is missing
            return np.zeros((current_keypoints.shape[0] if current_keypoints is not None else 17, 2))
        
        # Calculate velocity for each keypoint
        velocities = (current_keypoints - previous_keypoints) / dt
        return velocities

    def _calculate_keypoint_acceleration(self, current_velocities, previous_velocities, dt=1.0):
        """
        Calculate acceleration for each keypoint.
        
        Args:
            current_velocities (np.array): Current velocities array of shape (num_keypoints, 2)
            previous_velocities (np.array): Previous velocities array of shape (num_keypoints, 2)
            dt (float): Time difference between frames
            
        Returns:
            np.array: Acceleration array of shape (num_keypoints, 2) with (ax, ay) for each keypoint
        """
        if current_velocities is None or previous_velocities is None:
            # Return zeros for all keypoints if data is missing
            return np.zeros((current_velocities.shape[0] if current_velocities is not None else 17, 2))
        
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

# --- Example Usage (can be placed in a separate main script) ---
#
# from data_processor import DataProcessor
#
# # 1. Initialize the processor
# processor = DataProcessor(screen_width=1280, screen_height=720)
#
# # 2. Load the court-filtered data
# npz_path = 'path/to/your/court_filtered_data.npz'
# all_frames_data = np.load(npz_path, allow_pickle=True)['frames']
#
# # 3. Process the entire video sequence
# lstm_input_sequence = []
# previous_players = None
# for frame_data in all_frames_data:
#     # Step A: Assign players using the core heuristic
#     assigned_players = processor.assign_players(frame_data)
#     
#     # Step B: Convert the assignment into a feature vector
#     feature_vector = processor.create_feature_vector(assigned_players, previous_players)
#     
#     lstm_input_sequence.append(feature_vector)
#     previous_players = assigned_players  # Store for next iteration
#
# # 4. Final result is a NumPy array ready for the model
# lstm_ready_data = np.array(lstm_input_sequence)
# print(f"Successfully created LSTM input data with shape: {lstm_ready_data.shape}")