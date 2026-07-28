from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

from training.features.registry import FeatureRegistry
from training.io.fingerprint import (
    build_features_fingerprint,
    fingerprint_dict,
    read_h5_fingerprint,
    tmp_path_for,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureBuildConfig:
    feature_set: str
    target_fps: float
    overwrite: bool = False
    screen_width: int = 1280
    screen_height: int = 720


class FeatureBuilder:
    def __init__(self, cfg: FeatureBuildConfig) -> None:
        self.cfg = cfg
        self.registry = FeatureRegistry()

    def build(
        self,
        preproc_h5: Path,
        output_path: Path,
        overwrite: bool = False,
    ) -> Optional[Path]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_payload = build_features_fingerprint(
            preproc_h5_path=preproc_h5,
            feature_set=self.cfg.feature_set,
            target_fps=self.cfg.target_fps,
        )
        fingerprint = fingerprint_dict(fingerprint_payload)

        if output_path.exists() and not (overwrite or self.cfg.overwrite):
            if _is_valid_features_h5(output_path) and read_h5_fingerprint(output_path) == fingerprint:
                logger.info("Skipping existing features: %s", output_path)
                return output_path
            logger.warning("Existing features file is stale/invalid; regenerating: %s", output_path)

        builder_cls = self.registry.get(self.cfg.feature_set)
        builder = builder_cls(screen_width=self.cfg.screen_width, screen_height=self.cfg.screen_height)

        with h5py.File(preproc_h5, "r") as h5f:
            targets = h5f["targets"][:]
            labeled_idx = np.where(targets >= 0)[0]
            if labeled_idx.size == 0:
                logger.warning("No labeled frames in %s", preproc_h5)
                return None

            timestamps = h5f["frames"]["timestamps"][:]
            frame_index = h5f["frames"]["frame_index"][:]

            players = h5f["players"]
            # v1 is fixed at (near, far); v2 declares its own ordered slots so the
            # far-crop pass can contribute extra ones. Slots absent from the
            # preprocessed file (e.g. crop* on a video with no side-car) resolve
            # to None and are written as the -1 "not observed" sentinel block.
            slots = tuple(getattr(builder, "slots", ("near", "far")))
            slot_arrays = {}
            for slot in slots:
                if slot not in players:
                    slot_arrays[slot] = None
                    continue
                slot_arrays[slot] = (
                    players[slot][:], players[f"{slot}_conf"][:],
                    players[f"{slot}_box"][:], players[f"{slot}_box_conf"][:],
                )
            missing = [s for s, v in slot_arrays.items() if v is None]
            if missing:
                logger.warning("Feature slots absent from %s (filled with sentinel): %s",
                               preproc_h5.name, ", ".join(missing))
            near_kps, near_conf, near_box, near_box_conf = slot_arrays["near"]
            far_kps, far_conf, far_box, far_box_conf = slot_arrays["far"]

            dt = 1.0 / float(self.cfg.target_fps)
            feature_vectors = []
            feature_targets = []
            feature_frames = []
            feature_times = []

            multi_slot = hasattr(builder, "slots")
            prev_slots = {slot: None for slot in slots}
            prev_motion = {slot: {"centroid": None, "keypoints": None} for slot in slots}

            for idx in labeled_idx:
                current = {}
                for slot in slots:
                    arr = slot_arrays[slot]
                    current[slot] = (
                        None if arr is None
                        else _pack_player(arr[0][idx], arr[1][idx], arr[2][idx], arr[3][idx])
                    )

                if multi_slot:
                    vec = builder.build_feature_vector(current, prev_slots, prev_motion, dt)
                else:
                    # v1 signature is positional (near, far, ...); keep it exactly
                    # so existing v1 artifacts stay reproducible.
                    vec = builder.build_feature_vector(
                        current["near"], current["far"],
                        prev_slots["near"], prev_slots["far"], prev_motion, dt,
                    )
                feature_vectors.append(vec)
                feature_targets.append(int(targets[idx]))
                feature_frames.append(int(frame_index[idx]))
                feature_times.append(float(timestamps[idx]))

                prev_motion = {
                    slot: {
                        "centroid": _player_velocity(current[slot], prev_slots[slot], dt),
                        "keypoints": _keypoint_velocity(current[slot], prev_slots[slot], dt),
                    }
                    for slot in slots
                }
                prev_slots = current

        features = np.asarray(feature_vectors, dtype=np.float32)
        targets_arr = np.asarray(feature_targets, dtype=np.int8)
        frames_arr = np.asarray(feature_frames, dtype=np.int64)
        times_arr = np.asarray(feature_times, dtype=np.float64)

        if np.any(targets_arr < 0):
            raise RuntimeError(f"Feature builder produced unlabeled targets from {preproc_h5}")

        tmp_path = tmp_path_for(output_path)
        try:
            with h5py.File(tmp_path, "w") as out:
                out.create_dataset("features", data=features, compression="gzip")
                out.create_dataset("targets", data=targets_arr)
                out.create_dataset("frame_index", data=frames_arr)
                out.create_dataset("timestamps", data=times_arr)
                out.attrs["feature_set"] = self.cfg.feature_set
                out.attrs["feature_dim"] = features.shape[1]
                out.attrs["target_fps"] = float(self.cfg.target_fps)
                out.attrs["source"] = str(preproc_h5)
                out.attrs["fingerprint"] = fingerprint
            tmp_path.replace(output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        logger.info("Saved features to %s", output_path)
        return output_path


def _is_valid_features_h5(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as h5f:
            return all(key in h5f for key in ("features", "targets", "frame_index", "timestamps"))
    except Exception:
        return False


def _pack_player(kps: np.ndarray, conf: np.ndarray, box: np.ndarray, box_conf: np.ndarray) -> Dict[str, np.ndarray]:
    exists = bool(np.any(kps >= 0))
    return {
        "exists": exists,
        "keypoints": kps,
        "conf": conf,
        "box": box,
        "box_conf": float(box_conf) if np.ndim(box_conf) == 0 else float(box_conf[0]),
    }


def _player_velocity(player: Optional[Dict[str, np.ndarray]], prev_player: Optional[Dict[str, np.ndarray]], dt: float):
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


def _keypoint_velocity(player: Optional[Dict[str, np.ndarray]], prev_player: Optional[Dict[str, np.ndarray]], dt: float):
    if not player or not player.get("exists") or not prev_player or not prev_player.get("exists"):
        return None
    if dt <= 0:
        return np.zeros((player["keypoints"].shape[0], 2), dtype=np.float32)
    return (player["keypoints"] - prev_player["keypoints"]) / dt
