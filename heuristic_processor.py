# heuristic_processor.py
import numpy as np

class HeuristicProcessor:
    """
    Implements the refined heuristic (v3) to process pose data.
    It separates player assignment from feature engineering for flexibility.
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
        # [Implement the standard IoU calculation logic here]
        # This calculates the intersection over union of two bounding boxes.
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
        # [Implement the conditional merge logic here]
        # This method should only merge boxes if their clump is in an edge zone.
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

    def create_feature_vector(self, assigned_players, num_keypoints=17):
        """
        Creates a fixed-size 1D NumPy vector from the assigned player data.
        This is the designated place for future feature engineering.
        
        For missing players, -1 values are used to represent absent data,
        which is outside the valid coordinate range and clearly identifiable.
        
        Args:
            assigned_players (dict): The output from the `assign_players` method.
            num_keypoints (int): The number of keypoints per player.

        Returns:
            np.ndarray: A flat vector ready for the LSTM.
        """
        # Define the structure: 1 (exists) + 4 (bbox) + 17*2 (kp_xy) + 17 (kp_conf) = 56 features per player
        features_per_player = 1 + 4 + (num_keypoints * 3)
        vector = np.full(features_per_player * 2, -1.0)  # Use -1 for missing values

        # --- Near Player ---
        if assigned_players['near_player']:
            player_data = assigned_players['near_player']
            # Mark as present
            vector[0] = 1.0
            # Basic Features
            flat_features = np.concatenate([
                player_data['box'],
                player_data['keypoints'].flatten(),
                player_data['conf']
            ])
            vector[1:features_per_player] = flat_features
            
            # **FUTURE:** Add engineered features here (e.g., velocity, acceleration)
            # by appending them to `flat_features` and adjusting `features_per_player`.
            
        # --- Far Player ---
        offset = features_per_player
        if assigned_players['far_player']:
            player_data = assigned_players['far_player']
            # Mark as present
            vector[offset] = 1.0
            # Basic Features
            flat_features = np.concatenate([
                player_data['box'],
                player_data['keypoints'].flatten(),
                player_data['conf']
            ])
            vector[offset+1 : offset+features_per_player] = flat_features
            
        return vector

# --- Example Usage (can be placed in a separate main script) ---
#
# from heuristic_processor import HeuristicProcessor
#
# # 1. Initialize the processor
# processor = HeuristicProcessor(screen_width=1280, screen_height=720)
#
# # 2. Load the court-filtered data
# npz_path = 'path/to/your/court_filtered_data.npz'
# all_frames_data = np.load(npz_path, allow_pickle=True)['frames']
#
# # 3. Process the entire video sequence
# lstm_input_sequence = []
# for frame_data in all_frames_data:
#     # Step A: Assign players using the core heuristic
#     assigned_players = processor.assign_players(frame_data)
#     
#     # Step B: Convert the assignment into a feature vector
#     feature_vector = processor.create_feature_vector(assigned_players)
#     
#     lstm_input_sequence.append(feature_vector)
#
# # 4. Final result is a NumPy array ready for the model
# lstm_ready_data = np.array(lstm_input_sequence)
# print(f"Successfully created LSTM input data with shape: {lstm_ready_data.shape}")