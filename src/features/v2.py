"""Feature set v2: v1 minus exactly-redundant magnitudes, over N player slots.

Two changes from v1:

1. **Drops the four derived-magnitude groups** (36 dims/slot). Each is exactly
   the norm of a vector already in the block -- verified algebraically on raw
   features with max absolute error 0.0, not approximately:
       speed              == |velocity|
       accel_mag          == |acceleration|
       keypoint_speed     == |keypoint_vel|     (17)
       keypoint_accel_mag == |keypoint_accel|   (17)
   A network computes a norm trivially, so these cost parameters and add noise
   without adding information. Ablation on the 720p corpus: removing them
   *improved* test F1 (41.5% -> 42.1%).
   Per-slot block: 181 -> 145.

2. **Generalises to N slots.** v1 hardcodes (near, far); v2 takes an ordered
   list, so the far-player crop pass can contribute two more slots
   (near, far, crop0, crop1) -> 4 x 145 = 580 dims.

Slot layout is otherwise identical to v1, so the same interpretation and
normalisation apply. An absent slot is filled with the -1 "not observed"
sentinel, exactly as v1 fills an undetected far player.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_SLOTS = ("near", "far", "crop0", "crop1")
NEARFAR_SLOTS = ("near", "far")


# Per-slot column groups, in block order (see per_slot_dim). Used by the
# dataset step to ablate groups without rebuilding cached features.
SLOT_GROUPS = (
    ("exists", 1), ("box", 4), ("centroid", 2), ("velocity", 2), ("acceleration", 2),
    ("keypoints_xy", 34), ("keypoint_conf", 17), ("keypoint_vel", 34), ("keypoint_accel", 34),
    ("limb_lengths", 14), ("box_conf", 1),
)


def kept_columns(slots: Tuple[str, ...], drop_groups=(), drop_slots=()) -> np.ndarray:
    """Column indices that survive dropping the named groups (in every slot)
    and the named slots entirely."""
    names = {name for name, _ in SLOT_GROUPS}
    unknown = (set(drop_groups) - names) | (set(drop_slots) - set(slots))
    if unknown:
        raise ValueError(f"Unknown feature groups/slots: {sorted(unknown)}")
    keep, col = [], 0
    for slot in slots:
        for name, width in SLOT_GROUPS:
            if slot not in drop_slots and name not in drop_groups:
                keep.extend(range(col, col + width))
            col += width
    return np.asarray(keep, dtype=np.int64)


@dataclass
class FeatureSetV2:
    screen_width: int = 1280
    screen_height: int = 720
    slots: Tuple[str, ...] = DEFAULT_SLOTS

    def __post_init__(self) -> None:
        self.screen_center_x = self.screen_width / 2

    # --- layout ---------------------------------------------------------
    @staticmethod
    def per_slot_dim(num_keypoints: int = 17) -> int:
        # exists(1) box(4) centroid(2) velocity(2) acceleration(2)
        # keypoints_xy(2k) keypoint_conf(k) keypoint_vel(2k) keypoint_accel(2k)
        # limb_lengths(14) box_conf(1)
        return 1 + 4 + 2 + 2 + 2 + (num_keypoints * 2) + num_keypoints + \
            (num_keypoints * 2) + (num_keypoints * 2) + 14 + 1

    def feature_dim(self, num_keypoints: int = 17) -> int:
        return self.per_slot_dim(num_keypoints) * len(self.slots)

    # --- construction ---------------------------------------------------
    def build_feature_vector(
        self,
        players: Dict[str, Optional[Dict[str, np.ndarray]]],
        prev_players: Dict[str, Optional[Dict[str, np.ndarray]]] | None,
        prev_velocities: Dict | None,
        dt: float,
        num_keypoints: int = 17,
    ) -> np.ndarray:
        """players/prev_players are keyed by slot name; a missing or non-existent
        slot yields the sentinel block."""
        per = self.per_slot_dim(num_keypoints)
        vector = np.full(per * len(self.slots), -1.0, dtype=np.float32)
        prev_players = prev_players or {}
        for i, slot in enumerate(self.slots):
            vector[i * per:(i + 1) * per] = self._slot_features(
                players.get(slot), prev_players.get(slot), prev_velocities, dt,
                num_keypoints, prefix=slot,
            )
        return vector

    def _slot_features(
        self,
        player: Optional[Dict[str, np.ndarray]],
        prev_player: Optional[Dict[str, np.ndarray]],
        prev_velocities: Dict | None,
        dt: float,
        num_keypoints: int,
        prefix: str = "near",
    ) -> np.ndarray:
        per = self.per_slot_dim(num_keypoints)
        feature = np.full(per, -1.0, dtype=np.float32)
        if not player or not player.get("exists", False):
            feature[0] = 0.0
            return feature

        box = player["box"]
        keypoints = player["keypoints"]
        conf = player["conf"]
        box_conf = player.get("box_conf", -1.0)

        centroid = self._centroid(box)
        velocity = (0.0, 0.0)
        acceleration = (0.0, 0.0)
        kp_vel = np.zeros((num_keypoints, 2), dtype=np.float32)
        kp_accel = np.zeros((num_keypoints, 2), dtype=np.float32)
        limb_lengths = self._limb_lengths(keypoints)

        prev_centroid_vel, prev_kp_vel = self._motion_parts(prev_velocities, prefix)

        if prev_player and prev_player.get("exists", False):
            prev_centroid = self._centroid(prev_player["box"])
            velocity = self._velocity(centroid, prev_centroid, dt)
            if prev_centroid_vel is not None:
                acceleration = self._acceleration(velocity, prev_centroid_vel, dt)
            prev_kps = prev_player["keypoints"]
            kp_vel = self._keypoint_velocity(keypoints, prev_kps, dt)
            if prev_kp_vel is not None and prev_kp_vel.shape == kp_vel.shape:
                kp_accel = self._keypoint_acceleration(kp_vel, prev_kp_vel, dt)

        feature[0] = 1.0
        idx = 1
        feature[idx:idx + 4] = box; idx += 4
        feature[idx:idx + 2] = centroid; idx += 2
        feature[idx:idx + 2] = velocity; idx += 2
        feature[idx:idx + 2] = acceleration; idx += 2
        feature[idx:idx + num_keypoints * 2] = keypoints.flatten(); idx += num_keypoints * 2
        feature[idx:idx + num_keypoints] = conf; idx += num_keypoints
        feature[idx:idx + num_keypoints * 2] = kp_vel.flatten(); idx += num_keypoints * 2
        feature[idx:idx + num_keypoints * 2] = kp_accel.flatten(); idx += num_keypoints * 2
        feature[idx:idx + 14] = limb_lengths; idx += 14
        feature[idx] = box_conf
        return feature

    # --- helpers (identical semantics to v1) ----------------------------
    @staticmethod
    def _centroid(box: np.ndarray) -> Tuple[float, float]:
        return (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)

    @staticmethod
    def _velocity(curr, prev, dt: float):
        if dt <= 0:
            return (0.0, 0.0)
        return ((curr[0] - prev[0]) / dt, (curr[1] - prev[1]) / dt)

    @staticmethod
    def _acceleration(curr, prev, dt: float):
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
        out: List[float] = []
        for i, j in connections:
            if i < len(keypoints) and j < len(keypoints):
                out.append(float(np.sqrt(np.sum((keypoints[i] - keypoints[j]) ** 2))))
            else:
                out.append(-1.0)
        return np.array(out, dtype=np.float32)


@dataclass
class FeatureSetV2NearFar(FeatureSetV2):
    """v2 layout restricted to the full-frame slots (290 dims).

    This is the baseline for the far-crop experiment: identical feature
    definitions and identical corpus, differing only in whether the crop slots
    are present. Comparing the crop run against a v1/720p champion instead would
    confound the feature change, the corpus change, and the crop.
    """

    slots: Tuple[str, ...] = NEARFAR_SLOTS


# --- streaming (shared by the training feature builder and the runtime) --------
def pack_player(kps: np.ndarray, conf: np.ndarray, box: np.ndarray, box_conf) -> Dict[str, np.ndarray]:
    """One slot's arrays -> the player dict build_feature_vector takes. A slot
    whose keypoints are all the -1 sentinel does not exist."""
    exists = bool(np.any(kps >= 0))
    return {
        "exists": exists,
        "keypoints": kps,
        "conf": conf,
        "box": box,
        "box_conf": float(box_conf) if np.ndim(box_conf) == 0 else float(box_conf[0]),
    }


def player_velocity(player: Optional[Dict[str, np.ndarray]], prev_player: Optional[Dict[str, np.ndarray]], dt: float):
    if not player or not player.get("exists") or not prev_player or not prev_player.get("exists"):
        return None
    box = player["box"]
    prev_box = prev_player["box"]
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    pcx = (prev_box[0] + prev_box[2]) / 2
    pcy = (prev_box[1] + prev_box[3]) / 2
    if dt <= 0:
        return (0.0, 0.0)
    return ((cx - pcx) / dt, (cy - pcy) / dt)


def keypoint_velocity(player: Optional[Dict[str, np.ndarray]], prev_player: Optional[Dict[str, np.ndarray]], dt: float):
    if not player or not player.get("exists") or not prev_player or not prev_player.get("exists"):
        return None
    if dt <= 0:
        return np.zeros((player["keypoints"].shape[0], 2), dtype=np.float32)
    return (player["keypoints"] - prev_player["keypoints"]) / dt


class V2FeatureStream:
    """Row-by-row v2 features; motion state carries across consecutive rows."""

    def __init__(self, builder: FeatureSetV2, dt: float, keep: Optional[np.ndarray] = None) -> None:
        self.builder = builder
        self.dt = float(dt)
        self.keep = keep
        self.prev_slots = {slot: None for slot in builder.slots}
        self.prev_motion = {slot: {"centroid": None, "keypoints": None} for slot in builder.slots}

    def step(self, current: Dict[str, Optional[Dict[str, np.ndarray]]]) -> np.ndarray:
        """current: slot name -> player dict (or None for an absent slot)."""
        vec = self.builder.build_feature_vector(current, self.prev_slots, self.prev_motion, self.dt)
        self.prev_motion = {
            slot: {
                "centroid": player_velocity(current.get(slot), self.prev_slots[slot], self.dt),
                "keypoints": keypoint_velocity(current.get(slot), self.prev_slots[slot], self.dt),
            }
            for slot in self.builder.slots
        }
        self.prev_slots = {slot: current.get(slot) for slot in self.builder.slots}
        return vec if self.keep is None else vec[self.keep]


FEATURE_SETS = {"v2": FeatureSetV2, "v2_nearfar": FeatureSetV2NearFar}
