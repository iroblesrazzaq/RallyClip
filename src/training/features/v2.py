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
from typing import Dict, Optional, Tuple

import numpy as np

from training.features.v1 import FeatureSetV1

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

        # Block order matches SLOT_GROUPS.
        return np.concatenate([
            [1.0], box, centroid, velocity, acceleration,
            keypoints.flatten(), conf, kp_vel.flatten(), kp_accel.flatten(),
            limb_lengths, [box_conf],
        ]).astype(np.float32)

    # --- helpers: shared with v1 so the two layouts cannot drift ---------
    _centroid = staticmethod(FeatureSetV1._centroid)
    _velocity = staticmethod(FeatureSetV1._velocity)
    _acceleration = staticmethod(FeatureSetV1._acceleration)
    _keypoint_velocity = staticmethod(FeatureSetV1._keypoint_velocity)
    _keypoint_acceleration = staticmethod(FeatureSetV1._keypoint_acceleration)
    _motion_parts = staticmethod(FeatureSetV1._motion_parts)
    # v1's 14 limb pairs include (6, 5) and (12, 11), which repeat (5, 6) and
    # (11, 12). The shipped v0.6 model was trained on this layout, so the
    # duplicates stay.
    _limb_lengths = staticmethod(FeatureSetV1._limb_lengths)


@dataclass
class FeatureSetV2NearFar(FeatureSetV2):
    """v2 layout restricted to the full-frame slots (290 dims).

    This is the baseline for the far-crop experiment: identical feature
    definitions and identical corpus, differing only in whether the crop slots
    are present. Comparing the crop run against a v1/720p champion instead would
    confound the feature change, the corpus change, and the crop.
    """

    slots: Tuple[str, ...] = NEARFAR_SLOTS
