from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from training.courts.cache import CourtMaskCache
from training.io.annotations import load_annotations_json
from training.preprocess.player_assigner import PlayerAssigner

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    target_fps: float
    save_court_masks: bool
    court_model_path: str
    court_target_time: int
    court_force: bool = False


class Hdf5Preprocessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        self.assigner = PlayerAssigner()
        self.court_cache = CourtMaskCache(
            model_path=cfg.court_model_path,
            target_time=cfg.court_target_time,
        )

    def preprocess(
        self,
        data_root: Path,
        raw_h5_path: Path,
        video_path: Path,
        annotations_path: Path,
        output_path: Path,
        overwrite: bool = False,
    ) -> Optional[Path]:
        if not annotations_path.exists():
            logger.warning("No annotations for %s; skipping", video_path.name)
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            logger.info("Skipping existing preprocessed file: %s", output_path)
            return output_path

        annotations = load_annotations_json(annotations_path)
        if not annotations.get("segments"):
            logger.warning("No labeled segments for %s; skipping", video_path.name)
            return None

        cache = self.court_cache.get_or_create(data_root, video_path, force=self.cfg.court_force)
        court_mask = cache.mask

        with h5py.File(raw_h5_path, "r") as raw_h5:
            frame_indices = raw_h5["frames"]["frame_index"][:]
            timestamps = raw_h5["frames"]["timestamps"][:]
            offsets = raw_h5["frames"]["frame_offsets"][:]
            boxes = raw_h5["detections"]["boxes"]
            box_conf = raw_h5["detections"]["box_conf"]
            keypoints = raw_h5["detections"]["keypoints"]
            keypoint_conf = raw_h5["detections"]["keypoint_conf"]

            sample_idx = _sample_indices(timestamps, self.cfg.target_fps)
            if sample_idx.size == 0:
                logger.warning("No frames sampled for %s", video_path.name)
                return None

            preproc_h5 = h5py.File(output_path, "w")
            try:
                frames_group = preproc_h5.create_group("frames")
                det_group = preproc_h5.create_group("detections")
                players_group = preproc_h5.create_group("players")

                frames_group.create_dataset("frame_index", data=frame_indices[sample_idx], dtype="i8")
                frames_group.create_dataset("timestamps", data=timestamps[sample_idx], dtype="f8")
                frames_group.create_dataset("frame_offsets", data=np.array([0], dtype=np.int64), maxshape=(None,), chunks=True)

                det_group.create_dataset("boxes", shape=(0, 4), maxshape=(None, 4), dtype="f4", chunks=True, compression="gzip")
                det_group.create_dataset("box_conf", shape=(0,), maxshape=(None,), dtype="f4", chunks=True, compression="gzip")
                det_group.create_dataset("keypoints", shape=(0, 17, 2), maxshape=(None, 17, 2), dtype="f4", chunks=True, compression="gzip")
                det_group.create_dataset("keypoint_conf", shape=(0, 17), maxshape=(None, 17), dtype="f4", chunks=True, compression="gzip")

                players_group.create_dataset("near", shape=(0, 17, 2), maxshape=(None, 17, 2), dtype="f4", chunks=True, compression="gzip")
                players_group.create_dataset("far", shape=(0, 17, 2), maxshape=(None, 17, 2), dtype="f4", chunks=True, compression="gzip")
                players_group.create_dataset("near_conf", shape=(0, 17), maxshape=(None, 17), dtype="f4", chunks=True, compression="gzip")
                players_group.create_dataset("far_conf", shape=(0, 17), maxshape=(None, 17), dtype="f4", chunks=True, compression="gzip")
                players_group.create_dataset("near_box", shape=(0, 4), maxshape=(None, 4), dtype="f4", chunks=True, compression="gzip")
                players_group.create_dataset("far_box", shape=(0, 4), maxshape=(None, 4), dtype="f4", chunks=True, compression="gzip")
                players_group.create_dataset("near_box_conf", shape=(0,), maxshape=(None,), dtype="f4", chunks=True, compression="gzip")
                players_group.create_dataset("far_box_conf", shape=(0,), maxshape=(None,), dtype="f4", chunks=True, compression="gzip")

                targets = _build_targets(timestamps[sample_idx], annotations)
                preproc_h5.create_dataset("targets", data=targets.astype(np.int8), dtype="i1")

                if self.cfg.save_court_masks and court_mask is not None:
                    preproc_h5.create_dataset("court_mask", data=court_mask, dtype="u1")

                for idx in sample_idx:
                    start = int(offsets[idx])
                    end = int(offsets[idx + 1])
                    frame_boxes = np.array(boxes[start:end], dtype=np.float32)
                    frame_box_conf = np.array(box_conf[start:end], dtype=np.float32)
                    frame_kps = np.array(keypoints[start:end], dtype=np.float32)
                    frame_kp_conf = np.array(keypoint_conf[start:end], dtype=np.float32)

                    filtered = _filter_by_court(frame_boxes, frame_box_conf, frame_kps, frame_kp_conf, court_mask)
                    _append_detections(frames_group, det_group, filtered)
                    _append_players(players_group, self.assigner.assign(filtered))

                preproc_h5.attrs["target_fps"] = float(self.cfg.target_fps)
                preproc_h5.attrs["raw_h5"] = str(raw_h5_path)
                preproc_h5.attrs["video"] = str(video_path)
                preproc_h5.attrs["annotations"] = str(annotations_path)
            finally:
                preproc_h5.close()

        logger.info("Preprocessed %s", output_path)
        return output_path


def _sample_indices(timestamps: np.ndarray, target_fps: float) -> np.ndarray:
    if timestamps.size == 0 or target_fps <= 0:
        return np.array([], dtype=np.int64)
    step = 1.0 / target_fps
    target_times = []
    t = timestamps[0]
    while t <= timestamps[-1] + 1e-6:
        target_times.append(t)
        t += step
    target_times = np.array(target_times, dtype=np.float64)

    sampled_indices = []
    cursor = 0
    for tt in target_times:
        while cursor < len(timestamps) and timestamps[cursor] < tt:
            cursor += 1
        if cursor >= len(timestamps):
            break
        sampled_indices.append(cursor)
    return np.array(sampled_indices, dtype=np.int64)


def _build_targets(timestamps: np.ndarray, annotations: Dict) -> np.ndarray:
    segments = annotations.get("segments", [])
    if not segments:
        return np.full(timestamps.shape[0], -100, dtype=np.int8)
    starts = np.array([seg["start_time"] for seg in segments], dtype=np.float64)
    ends = np.array([seg["end_time"] for seg in segments], dtype=np.float64)
    targets = np.zeros(timestamps.shape[0], dtype=np.int8)

    idx = 0
    for i, ts in enumerate(timestamps):
        while idx < len(ends) - 1 and ts > ends[idx]:
            idx += 1
        if starts[idx] <= ts <= ends[idx]:
            targets[i] = 1
    return targets


def _filter_by_court(
    boxes: np.ndarray,
    box_conf: np.ndarray,
    keypoints: np.ndarray,
    keypoint_conf: np.ndarray,
    court_mask: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    if court_mask is None or boxes.size == 0:
        return {
            "boxes": boxes,
            "box_conf": box_conf,
            "keypoints": keypoints,
            "keypoint_conf": keypoint_conf,
        }

    keep = []
    for i, box in enumerate(boxes):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        if 0 <= cy < court_mask.shape[0] and 0 <= cx < court_mask.shape[1]:
            if court_mask[int(cy), int(cx)] == 0:
                keep.append(i)

    if not keep:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "box_conf": np.empty((0,), dtype=np.float32),
            "keypoints": np.empty((0, 17, 2), dtype=np.float32),
            "keypoint_conf": np.empty((0, 17), dtype=np.float32),
        }

    keep_idx = np.array(keep, dtype=np.int64)
    return {
        "boxes": boxes[keep_idx],
        "box_conf": box_conf[keep_idx],
        "keypoints": keypoints[keep_idx],
        "keypoint_conf": keypoint_conf[keep_idx],
    }


def _append_detections(frames_group: h5py.Group, det_group: h5py.Group, data: Dict[str, np.ndarray]) -> None:
    boxes = data["boxes"]
    box_conf = data["box_conf"]
    keypoints = data["keypoints"]
    keypoint_conf = data["keypoint_conf"]

    offsets = frames_group["frame_offsets"]
    det_count = int(offsets[-1])
    num_dets = int(boxes.shape[0])

    if num_dets:
        _append_rows(det_group["boxes"], boxes)
        _append_rows(det_group["box_conf"], box_conf)
        _append_rows(det_group["keypoints"], keypoints)
        _append_rows(det_group["keypoint_conf"], keypoint_conf)

    offsets.resize((offsets.shape[0] + 1,))
    offsets[-1] = det_count + num_dets


def _append_players(players_group: h5py.Group, players: Dict[str, np.ndarray]) -> None:
    for key, dataset_name in (
        ("near_kps", "near"),
        ("far_kps", "far"),
        ("near_conf", "near_conf"),
        ("far_conf", "far_conf"),
        ("near_box", "near_box"),
        ("far_box", "far_box"),
        ("near_box_conf", "near_box_conf"),
        ("far_box_conf", "far_box_conf"),
    ):
        _append_rows(players_group[dataset_name], players[key])


def _append_rows(dataset: h5py.Dataset, data: np.ndarray) -> None:
    new_size = dataset.shape[0] + data.shape[0]
    dataset.resize((new_size,) + dataset.shape[1:])
    dataset[-data.shape[0]:] = data
