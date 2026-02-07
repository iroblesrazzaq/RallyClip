from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class FeatureSetV1:
    screen_width: int = 1280
    screen_height: int = 720

    def __post_init__(self) -> None:
        self.screen_center_x = self.screen_width / 2

    @staticmethod
    def feature_dim(num_keypoints: int = 17) -> int:
        per_player = 1 + 4 + 2 + 2 + 2 + 1 + 1 + (num_keypoints * 3) + (num_keypoints * 2) + (num_keypoints * 2) + num_keypoints + num_keypoints + 14 + 1
        return per_player * 2

    def build_feature_vector(
        self,
        near: Dict[str, np.ndarray],
        far: Dict[str, np.ndarray],
        prev_near: Dict[str, np.ndarray] | None,
        prev_far: Dict[str, np.ndarray] | None,
        prev_velocities: Dict | None,
        dt: float,
        num_keypoints: int = 17,
    ) -> np.ndarray:
        per_player = 1 + 4 + 2 + 2 + 2 + 1 + 1 + (num_keypoints * 3) + (num_keypoints * 2) + (num_keypoints * 2) + num_keypoints + num_keypoints + 14 + 1
        vector = np.full(per_player * 2, -1.0, dtype=np.float32)

        vector[:per_player] = self._player_features(near, prev_near, prev_velocities, dt, num_keypoints)
        vector[per_player:] = self._player_features(far, prev_far, prev_velocities, dt, num_keypoints, prefix="far")
        return vector

    def _player_features(
        self,
        player: Dict[str, np.ndarray],
        prev_player: Dict[str, np.ndarray] | None,
        prev_velocities: Dict | None,
        dt: float,
        num_keypoints: int,
        prefix: str = "near",
    ) -> np.ndarray:
        exists = player.get("exists", False)
        feature_len = 1 + 4 + 2 + 2 + 2 + 1 + 1 + (num_keypoints * 3) + (num_keypoints * 2) + (num_keypoints * 2) + num_keypoints + num_keypoints + 14 + 1
        feature = np.full(feature_len, -1.0, dtype=np.float32)
        if not exists:
            feature[0] = 0.0
            return feature

        box = player["box"]
        keypoints = player["keypoints"]
        conf = player["conf"]
        box_conf = player.get("box_conf", -1.0)

        centroid = self._centroid(box)
        velocity = (0.0, 0.0)
        acceleration = (0.0, 0.0)
        speed = -1.0
        accel_mag = -1.0

        kp_vel = np.zeros((num_keypoints, 2), dtype=np.float32)
        kp_accel = np.zeros((num_keypoints, 2), dtype=np.float32)
        kp_speed = np.full(num_keypoints, -1.0, dtype=np.float32)
        kp_accel_mag = np.full(num_keypoints, -1.0, dtype=np.float32)
        limb_lengths = self._limb_lengths(keypoints)

        prev_centroid_vel, prev_kp_vel = self._motion_parts(prev_velocities, prefix)

        if prev_player and prev_player.get("exists", False):
            prev_centroid = self._centroid(prev_player["box"])
            velocity = self._velocity(centroid, prev_centroid, dt)
            speed = float(np.sqrt(velocity[0] ** 2 + velocity[1] ** 2))
            if prev_centroid_vel is not None:
                acceleration = self._acceleration(velocity, prev_centroid_vel, dt)
                accel_mag = float(np.sqrt(acceleration[0] ** 2 + acceleration[1] ** 2))

            prev_kps = prev_player["keypoints"]
            kp_vel = self._keypoint_velocity(keypoints, prev_kps, dt)
            kp_speed = np.sqrt(kp_vel[:, 0] ** 2 + kp_vel[:, 1] ** 2)
            if prev_kp_vel is not None and prev_kp_vel.shape == kp_vel.shape:
                kp_accel = self._keypoint_acceleration(kp_vel, prev_kp_vel, dt)
            kp_accel_mag = np.sqrt(kp_accel[:, 0] ** 2 + kp_accel[:, 1] ** 2)

        feature[0] = 1.0
        idx = 1
        feature[idx:idx + 4] = box
        idx += 4
        feature[idx:idx + 2] = centroid
        idx += 2
        feature[idx:idx + 2] = velocity
        idx += 2
        feature[idx:idx + 2] = acceleration
        idx += 2
        feature[idx] = speed
        idx += 1
        feature[idx] = accel_mag
        idx += 1
        feature[idx:idx + num_keypoints * 2] = keypoints.flatten()
        idx += num_keypoints * 2
        feature[idx:idx + num_keypoints] = conf
        idx += num_keypoints
        feature[idx:idx + num_keypoints * 2] = kp_vel.flatten()
        idx += num_keypoints * 2
        feature[idx:idx + num_keypoints * 2] = kp_accel.flatten()
        idx += num_keypoints * 2
        feature[idx:idx + num_keypoints] = kp_speed
        idx += num_keypoints
        feature[idx:idx + num_keypoints] = kp_accel_mag
        idx += num_keypoints
        feature[idx:idx + 14] = limb_lengths
        idx += 14
        feature[idx] = box_conf
        return feature

    @staticmethod
    def _centroid(box: np.ndarray) -> Tuple[float, float]:
        return (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)

    @staticmethod
    def _velocity(curr: Tuple[float, float], prev: Tuple[float, float], dt: float) -> Tuple[float, float]:
        if dt <= 0:
            return (0.0, 0.0)
        return ((curr[0] - prev[0]) / dt, (curr[1] - prev[1]) / dt)

    @staticmethod
    def _acceleration(curr: Tuple[float, float], prev: Tuple[float, float], dt: float) -> Tuple[float, float]:
        if dt <= 0:
            return (0.0, 0.0)
        return ((curr[0] - prev[0]) / dt, (curr[1] - prev[1]) / dt)

    @staticmethod
    def _keypoint_velocity(curr: np.ndarray, prev: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0:
            return np.zeros((curr.shape[0], 2), dtype=np.float32)
        return (curr - prev) / dt

    @staticmethod
    def _keypoint_acceleration(curr_vel: np.ndarray, prev_vel: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0:
            return np.zeros_like(curr_vel, dtype=np.float32)
        return (curr_vel - prev_vel) / dt

    @staticmethod
    def _motion_parts(prev_velocities: Dict | None, prefix: str):
        if not prev_velocities:
            return None, None
        value = prev_velocities.get(prefix)
        if value is None:
            return None, None
        if isinstance(value, dict):
            return value.get("centroid"), value.get("keypoints")
        if isinstance(value, tuple) and len(value) == 2:
            return value, None
        return None, None

    @staticmethod
    def _limb_lengths(keypoints: np.ndarray) -> np.ndarray:
        connections = [
            (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12), (5, 11), (6, 12), (6, 5), (12, 11),
        ]
        limb_lengths = []
        for i, j in connections:
            if i < len(keypoints) and j < len(keypoints):
                dist = np.sqrt(np.sum((keypoints[i] - keypoints[j]) ** 2))
                limb_lengths.append(dist)
            else:
                limb_lengths.append(-1.0)
        return np.array(limb_lengths, dtype=np.float32)
