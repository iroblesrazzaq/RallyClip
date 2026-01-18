from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

from training.features.registry import FeatureRegistry

logger = logging.getLogger(__name__)


@dataclass
class FeatureBuildConfig:
    feature_set: str
    target_fps: float
    overwrite: bool = False


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
        if output_path.exists() and not (overwrite or self.cfg.overwrite):
            logger.info("Skipping existing features: %s", output_path)
            return output_path

        builder_cls = self.registry.get(self.cfg.feature_set)
        builder = builder_cls()

        with h5py.File(preproc_h5, "r") as h5f:
            targets = h5f["targets"][:]
            labeled_idx = np.where(targets >= 0)[0]
            if labeled_idx.size == 0:
                logger.warning("No labeled frames in %s", preproc_h5)
                return None

            timestamps = h5f["frames"]["timestamps"][:]
            frame_index = h5f["frames"]["frame_index"][:]

            players = h5f["players"]
            near_kps = players["near"][:]
            far_kps = players["far"][:]
            near_conf = players["near_conf"][:]
            far_conf = players["far_conf"][:]
            near_box = players["near_box"][:]
            far_box = players["far_box"][:]
            near_box_conf = players["near_box_conf"][:]
            far_box_conf = players["far_box_conf"][:]

            dt = 1.0 / float(self.cfg.target_fps)
            feature_vectors = []
            feature_targets = []
            feature_frames = []
            feature_times = []

            prev_near = None
            prev_far = None
            prev_vel = {"near": None, "far": None}

            for idx in labeled_idx:
                near = _pack_player(near_kps[idx], near_conf[idx], near_box[idx], near_box_conf[idx])
                far = _pack_player(far_kps[idx], far_conf[idx], far_box[idx], far_box_conf[idx])

                vec = builder.build_feature_vector(near, far, prev_near, prev_far, prev_vel, dt)
                feature_vectors.append(vec)
                feature_targets.append(int(targets[idx]))
                feature_frames.append(int(frame_index[idx]))
                feature_times.append(float(timestamps[idx]))

                prev_vel = {
                    "near": _player_velocity(near, prev_near, dt),
                    "far": _player_velocity(far, prev_far, dt),
                }
                prev_near = near
                prev_far = far

        features = np.asarray(feature_vectors, dtype=np.float32)
        targets_arr = np.asarray(feature_targets, dtype=np.int8)
        frames_arr = np.asarray(feature_frames, dtype=np.int64)
        times_arr = np.asarray(feature_times, dtype=np.float64)

        with h5py.File(output_path, "w") as out:
            out.create_dataset("features", data=features, compression="gzip")
            out.create_dataset("targets", data=targets_arr)
            out.create_dataset("frame_index", data=frames_arr)
            out.create_dataset("timestamps", data=times_arr)
            out.attrs["feature_set"] = self.cfg.feature_set
            out.attrs["feature_dim"] = features.shape[1]
            out.attrs["target_fps"] = float(self.cfg.target_fps)
            out.attrs["source"] = str(preproc_h5)

        logger.info("Saved features to %s", output_path)
        return output_path


def _pack_player(kps: np.ndarray, conf: np.ndarray, box: np.ndarray, box_conf: np.ndarray) -> Dict[str, np.ndarray]:
    exists = bool(np.any(kps >= 0))
    return {
        "exists": exists,
        "keypoints": kps,
        "conf": conf,
        "box": box,
        "box_conf": float(box_conf) if np.ndim(box_conf) == 0 else float(box_conf[0]),
    }


def _player_velocity(player: Dict[str, np.ndarray], prev_player: Optional[Dict[str, np.ndarray]], dt: float):
    if not player.get("exists") or not prev_player or not prev_player.get("exists"):
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
