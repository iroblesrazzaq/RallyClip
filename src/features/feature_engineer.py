import os
import numpy as np


class FeatureEngineer:
    def __init__(self, screen_width: int = 1280, screen_height: int = 720, feature_vector_size: int = 288) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.feature_vector_size = feature_vector_size
        self.screen_center_x = screen_width / 2

    def _calculate_centroid(self, box):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return (center_x, center_y)

    def _calculate_velocity(self, current_pos, previous_pos, dt: float = 1.0):
        if current_pos is None or previous_pos is None:
            return (0.0, 0.0)
        vx = (current_pos[0] - previous_pos[0]) / dt
        vy = (current_pos[1] - previous_pos[1]) / dt
        return (vx, vy)

    def _calculate_acceleration(self, current_vel, previous_vel, dt: float = 1.0):
        if current_vel is None or previous_vel is None:
            return (0.0, 0.0)
        ax = (current_vel[0] - previous_vel[0]) / dt
        ay = (current_vel[1] - previous_vel[1]) / dt
        return (ax, ay)

    def _calculate_keypoint_velocity(self, current_keypoints, previous_keypoints, dt: float = 1.0):
        if current_keypoints is None or previous_keypoints is None:
            return np.zeros((17, 2))
        return (current_keypoints - previous_keypoints) / dt

    def _calculate_limb_lengths(self, keypoints):
        if keypoints is None:
            return np.full(14, -1.0)
        connections = [
            (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12), (5, 11), (6, 12), (6, 5), (12, 11)
        ]
        limb_lengths = []
        for i, j in connections:
            if i < len(keypoints) and j < len(keypoints):
                dist = np.sqrt(np.sum((keypoints[i] - keypoints[j]) ** 2))
                limb_lengths.append(dist)
            else:
                limb_lengths.append(-1.0)
        return np.array(limb_lengths)

    def create_feature_vector(self, assigned_players, previous_assigned_players=None, previous_velocities=None, num_keypoints: int = 17):
        features_per_player = 1 + 4 + 2 + 2 + 2 + 1 + 1 + (num_keypoints * 3) + (num_keypoints * 2) + (num_keypoints * 2) + num_keypoints + num_keypoints + 14
        vector = np.full(features_per_player * 2, -1.0)

        if assigned_players['near_player']:
            player_data = assigned_players['near_player']
            vector[0] = 1.0
            centroid = self._calculate_centroid(player_data['box'])
            velocity = (0.0, 0.0)
            acceleration = (0.0, 0.0)
            speed = -1.0
            acceleration_magnitude = -1.0
            kp_velocities = np.zeros((num_keypoints, 2))
            kp_accelerations = np.zeros((num_keypoints, 2))
            kp_speeds = np.full(num_keypoints, -1.0)
            kp_acceleration_magnitudes = np.full(num_keypoints, -1.0)
            limb_lengths = np.full(14, -1.0)
            if (previous_assigned_players and previous_assigned_players['near_player']):
                prev_centroid = self._calculate_centroid(previous_assigned_players['near_player']['box'])
                velocity = self._calculate_velocity(centroid, prev_centroid)
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                if previous_velocities and previous_velocities['near_player']:
                    acceleration = self._calculate_acceleration(velocity, previous_velocities['near_player'])
                    acceleration_magnitude = np.sqrt(acceleration[0]**2 + acceleration[1]**2)
                current_kps = player_data['keypoints']
                prev_kps = previous_assigned_players['near_player']['keypoints']
                kp_velocities = self._calculate_keypoint_velocity(current_kps, prev_kps)
                kp_speeds = np.sqrt(kp_velocities[:, 0]**2 + kp_velocities[:, 1]**2)
                kp_acceleration_magnitudes = np.sqrt(kp_accelerations[:, 0]**2 + kp_accelerations[:, 1]**2)
                limb_lengths = self._calculate_limb_lengths(player_data['keypoints'])
            flat_features = np.concatenate([
                player_data['box'], centroid, velocity, acceleration, [speed, acceleration_magnitude],
                player_data['keypoints'].flatten(), player_data['conf'], kp_velocities.flatten(), kp_accelerations.flatten(),
                kp_speeds, kp_acceleration_magnitudes, limb_lengths
            ])
            vector[1:features_per_player] = flat_features
        else:
            offset = 1 + 4 + 2
            vector[offset:offset+4] = [0.0, 0.0, 0.0, 0.0]
            vector[offset+4:offset+6] = [-1.0, -1.0]
            kp_offset = offset + 6 + (num_keypoints * 3)
            vector[kp_offset:kp_offset+(num_keypoints * 4)] = 0.0
            mag_offset = kp_offset + (num_keypoints * 4)
            vector[mag_offset:mag_offset+(num_keypoints * 2)] = -1.0

        offset = features_per_player
        if assigned_players['far_player']:
            player_data = assigned_players['far_player']
            vector[offset] = 1.0
            centroid = self._calculate_centroid(player_data['box'])
            velocity = (0.0, 0.0)
            acceleration = (0.0, 0.0)
            speed = -1.0
            acceleration_magnitude = -1.0
            kp_velocities = np.zeros((num_keypoints, 2))
            kp_accelerations = np.zeros((num_keypoints, 2))
            kp_speeds = np.full(num_keypoints, -1.0)
            kp_acceleration_magnitudes = np.full(num_keypoints, -1.0)
            limb_lengths = np.full(14, -1.0)
            if (previous_assigned_players and previous_assigned_players['far_player']):
                prev_centroid = self._calculate_centroid(previous_assigned_players['far_player']['box'])
                velocity = self._calculate_velocity(centroid, prev_centroid)
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                if previous_velocities and previous_velocities['far_player']:
                    acceleration = self._calculate_acceleration(velocity, previous_velocities['far_player'])
                    acceleration_magnitude = np.sqrt(acceleration[0]**2 + acceleration[1]**2)
                current_kps = player_data['keypoints']
                prev_kps = previous_assigned_players['far_player']['keypoints']
                kp_velocities = self._calculate_keypoint_velocity(current_kps, prev_kps)
                kp_speeds = np.sqrt(kp_velocities[:, 0]**2 + kp_velocities[:, 1]**2)
                kp_acceleration_magnitudes = np.sqrt(kp_accelerations[:, 0]**2 + kp_accelerations[:, 1]**2)
                limb_lengths = self._calculate_limb_lengths(player_data['keypoints'])
            flat_features = np.concatenate([
                player_data['box'], centroid, velocity, acceleration, [speed, acceleration_magnitude],
                player_data['keypoints'].flatten(), player_data['conf'], kp_velocities.flatten(), kp_accelerations.flatten(),
                kp_speeds, kp_acceleration_magnitudes, limb_lengths
            ])
            vector[offset+1 : offset+self.feature_vector_size] = flat_features
        else:
            pos_offset = offset + 1 + 4 + 2
            vector[pos_offset:pos_offset+4] = [0.0, 0.0, 0.0, 0.0]
            vector[pos_offset+4:pos_offset+6] = [-1.0, -1.0]
            kp_offset = pos_offset + 6 + (num_keypoints * 3)
            vector[kp_offset:kp_offset+(num_keypoints * 4)] = 0.0
            mag_offset = kp_offset + (num_keypoints * 4)
            vector[mag_offset:mag_offset+(num_keypoints * 2)] = -1.0
        return vector

    def create_features_from_preprocessed(self, input_npz_path: str, output_file: str, overwrite: bool = False) -> bool:
        try:
            if os.path.exists(output_file) and not overwrite:
                print(f"  ✓ Already exists, skipping: {os.path.basename(output_file)}")
                return True
            print(f"  Loading preprocessed data from: {input_npz_path}")
            data = np.load(input_npz_path, allow_pickle=True)
            frames = data['frames']
            targets = data['targets']
            near_players = data['near_players']
            far_players = data['far_players']
            annotated_indices = np.where(targets >= 0)[0]
            feature_vectors, feature_targets = [], []
            previous_players = None
            previous_velocities = {'near_player': None, 'far_player': None}
            for idx in annotated_indices:
                assigned_players = {'near_player': near_players[idx], 'far_player': far_players[idx]}
                feature_vector = self.create_feature_vector(assigned_players, previous_players, previous_velocities)
                feature_vectors.append(feature_vector)
                feature_targets.append(targets[idx])
                current_velocities = {'near_player': None, 'far_player': None}
                if (assigned_players['near_player'] and previous_players and previous_players['near_player']):
                    current_centroid = self._calculate_centroid(assigned_players['near_player']['box'])
                    prev_centroid = self._calculate_centroid(previous_players['near_player']['box'])
                    current_velocities['near_player'] = self._calculate_velocity(current_centroid, prev_centroid)
                if (assigned_players['far_player'] and previous_players and previous_players['far_player']):
                    current_centroid = self._calculate_centroid(assigned_players['far_player']['box'])
                    prev_centroid = self._calculate_centroid(previous_players['far_player']['box'])
                    current_velocities['far_player'] = self._calculate_velocity(current_centroid, prev_centroid)
                previous_players = assigned_players
                previous_velocities = current_velocities
            if feature_vectors:
                feature_array = np.array(feature_vectors)
                target_array = np.array(feature_targets)
            else:
                feature_array = np.empty((0, self.feature_vector_size))
                target_array = np.empty((0,))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.savez_compressed(output_file, features=feature_array, targets=target_array)
            print(f"  ✓ Features saved to: {output_file}")
            return True
        except Exception as e:
            print(f"  ❌ Error processing {input_npz_path}: {e}")
            import traceback
            traceback.print_exc()
            return False


