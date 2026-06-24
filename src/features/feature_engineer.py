import os
import logging
import numpy as np

from training.features.v1 import FeatureSetV1


class FeatureEngineer:
    def __init__(self, screen_width: int = 1280, screen_height: int = 720, feature_vector_size: int | None = None, target_fps: float = 5.0) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.feature_set = FeatureSetV1(screen_width=screen_width, screen_height=screen_height)
        expected_dim = self.feature_set.feature_dim()
        if feature_vector_size is not None and feature_vector_size != expected_dim:
            logging.warning("Ignoring legacy runtime feature size %s; v1 requires %s", feature_vector_size, expected_dim)
        self.feature_vector_size = expected_dim
        self.target_fps = float(target_fps)
        self.dt = 1.0 / self.target_fps if self.target_fps > 0 else 1.0
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

    def _pack_player(self, player_data, num_keypoints: int = 17):
        if player_data is None:
            return {
                "exists": False,
                "box": np.full((4,), -1.0, dtype=np.float32),
                "keypoints": np.full((num_keypoints, 2), -1.0, dtype=np.float32),
                "conf": np.full((num_keypoints,), -1.0, dtype=np.float32),
                "box_conf": -1.0,
            }
        return {
            "exists": True,
            "box": np.asarray(player_data["box"], dtype=np.float32),
            "keypoints": np.asarray(player_data["keypoints"], dtype=np.float32),
            "conf": np.asarray(player_data["conf"], dtype=np.float32),
            "box_conf": float(player_data.get("box_conf", -1.0)),
        }

    def _normalize_prev_motion(self, previous_velocities):
        if not previous_velocities:
            return None
        if "near" in previous_velocities or "far" in previous_velocities:
            return previous_velocities
        return {
            "near": previous_velocities.get("near_player"),
            "far": previous_velocities.get("far_player"),
        }

    def _player_velocity(self, player_data, previous_player_data):
        if player_data is None or previous_player_data is None:
            return None
        centroid = self._calculate_centroid(player_data["box"])
        prev_centroid = self._calculate_centroid(previous_player_data["box"])
        return self._calculate_velocity(centroid, prev_centroid, self.dt)

    def _player_keypoint_velocity(self, player_data, previous_player_data):
        if player_data is None or previous_player_data is None:
            return None
        return self._calculate_keypoint_velocity(
            np.asarray(player_data["keypoints"], dtype=np.float32),
            np.asarray(previous_player_data["keypoints"], dtype=np.float32),
            self.dt,
        )

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
        previous_assigned_players = previous_assigned_players or {}
        near = self._pack_player(assigned_players.get('near_player'), num_keypoints)
        far = self._pack_player(assigned_players.get('far_player'), num_keypoints)
        prev_near = self._pack_player(previous_assigned_players.get('near_player'), num_keypoints) if previous_assigned_players else None
        prev_far = self._pack_player(previous_assigned_players.get('far_player'), num_keypoints) if previous_assigned_players else None
        return self.feature_set.build_feature_vector(
            near,
            far,
            prev_near,
            prev_far,
            self._normalize_prev_motion(previous_velocities),
            self.dt,
            num_keypoints=num_keypoints,
        )

    def build_features(self, targets, near_players, far_players):
        """In-memory feature core.

        Builds the v1 feature matrix from the preprocessed per-frame targets +
        near/far player records, carrying prev-frame state for velocity/acceleration.
        Returns ``(feature_array, target_array)`` -- the arrays the release path feeds
        straight into inference and that :meth:`create_features_from_preprocessed`
        serializes. Only the previous frame is needed as state, so this is streamable
        (Phase B).
        """
        annotated_indices = np.where(targets >= 0)[0]
        feature_vectors, feature_targets = [], []
        previous_players = None
        previous_velocities = {
            'near': {'centroid': None, 'keypoints': None},
            'far': {'centroid': None, 'keypoints': None},
        }
        for idx in annotated_indices:
            assigned_players = {'near_player': near_players[idx], 'far_player': far_players[idx]}
            feature_vector = self.create_feature_vector(assigned_players, previous_players, previous_velocities)
            feature_vectors.append(feature_vector)
            feature_targets.append(targets[idx])
            current_velocities = {
                'near': {
                    'centroid': self._player_velocity(
                        assigned_players['near_player'],
                        previous_players['near_player'] if previous_players else None,
                    ),
                    'keypoints': self._player_keypoint_velocity(
                        assigned_players['near_player'],
                        previous_players['near_player'] if previous_players else None,
                    ),
                },
                'far': {
                    'centroid': self._player_velocity(
                        assigned_players['far_player'],
                        previous_players['far_player'] if previous_players else None,
                    ),
                    'keypoints': self._player_keypoint_velocity(
                        assigned_players['far_player'],
                        previous_players['far_player'] if previous_players else None,
                    ),
                },
            }
            previous_players = assigned_players
            previous_velocities = current_velocities
        if feature_vectors:
            feature_array = np.array(feature_vectors, dtype=np.float32)
            target_array = np.array(feature_targets)
        else:
            feature_array = np.empty((0, self.feature_vector_size), dtype=np.float32)
            target_array = np.empty((0,))
        return feature_array, target_array

    def create_features_from_preprocessed(self, input_npz_path: str, output_file: str, overwrite: bool = False) -> bool:
        """File-writing wrapper around :meth:`build_features` (training scripts).

        Behaviour-preserving load->core->save: loads the preprocessed arrays, runs the
        core, and serializes the same ``features`` / ``targets`` matrices.
        """
        try:
            if os.path.exists(output_file) and not overwrite:
                logging.info("Features skip (exists): %s", os.path.basename(output_file))
                return True
            logging.info("Loading preprocessed data from: %s", input_npz_path)
            with np.load(input_npz_path, allow_pickle=True) as data:
                targets = data['targets'].copy()
                near_players = data['near_players'].copy()
                far_players = data['far_players'].copy()
            feature_array, target_array = self.build_features(targets, near_players, far_players)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.savez_compressed(output_file, features=feature_array, targets=target_array)
            logging.info("Features saved to: %s", output_file)
            return True
        except Exception as e:
            logging.error("Error processing %s: %s", input_npz_path, e, exc_info=True)
            return False

