from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import cv2
import h5py
import numpy as np

from training.courts.cache import CourtMaskCache
from training.io.annotations import load_annotations_json
from training.io.fingerprint import (
    build_preprocess_fingerprint,
    fingerprint_dict,
    read_h5_fingerprint,
    tmp_path_for,
)
from training.normalize.normalize import CANONICAL_HEIGHT, CANONICAL_WIDTH
from training.preprocess.player_assigner import PlayerAssigner

logger = logging.getLogger(__name__)
UNLABELED_TARGET = -1


@dataclass
class PreprocessConfig:
    target_fps: float
    save_court_masks: bool
    court_model_path: str
    court_target_time: Optional[int]  # None => midpoint anchor (see CourtMaskCache)
    court_force: bool = False
    court_enabled: bool = True  # False => no court filtering at all (deliberate, no warning)
    expect_width: int = CANONICAL_WIDTH
    expect_height: int = CANONICAL_HEIGHT


class Hdf5Preprocessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        self.assigner = PlayerAssigner(
            screen_width=cfg.expect_width,
            screen_height=cfg.expect_height,
        )
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
        crop_h5_path: Optional[Path] = None,
    ) -> Optional[Path]:
        if not annotations_path.exists():
            logger.warning("No annotations for %s; skipping", video_path.name)
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_payload = build_preprocess_fingerprint(
            annotations_path=annotations_path,
            raw_h5_path=raw_h5_path,
            target_fps=self.cfg.target_fps,
            court_model_path=self.cfg.court_model_path,
            court_target_time=self.cfg.court_target_time,
            crop_h5_path=crop_h5_path,
        )
        fingerprint = fingerprint_dict(fingerprint_payload)

        if output_path.exists() and not overwrite:
            if _is_valid_preprocessed_h5(output_path) and read_h5_fingerprint(output_path) == fingerprint:
                logger.info("Skipping existing preprocessed file: %s", output_path)
                return output_path
            logger.warning("Existing preprocessed file is stale/invalid; regenerating: %s", output_path)

        annotations = load_annotations_json(annotations_path)
        if not annotations.get("segments"):
            logger.warning("No labeled segments for %s; skipping", video_path.name)
            return None

        if self.cfg.court_enabled:
            cache = self.court_cache.get_or_create(data_root, video_path, force=self.cfg.court_force)
            court_mask = cache.mask
            if not cache.success or court_mask is None:
                # Fall back to the default mask, matching runtime behaviour
                # (data_preprocessor.compute_court_mask). Passing frames through
                # unfiltered was the previous training behaviour and is worse:
                # spectators and adjacent-court players leak straight into the
                # player slots, and for the far-crop slots that is pure noise in
                # the exact block the crop exists to populate.
                court_mask = _default_court_mask(self.cfg.expect_width, self.cfg.expect_height)
                logger.warning(
                    "No court mask for %s (detection failed) -> using DEFAULT court mask",
                    video_path.name,
                )
        else:
            court_mask = None

        tmp_path = tmp_path_for(output_path)
        try:
            with h5py.File(raw_h5_path, "r") as raw_h5:
                _assert_canonical_dims(raw_h5, video_path, self.cfg.expect_width, self.cfg.expect_height)

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

                # Optional far-crop side-car (scripts/extract_crop_poses.py). It is
                # written on the same frame grid as the raw pose h5, so it indexes
                # with the same sample_idx. Verify that rather than assume it: a
                # silent misalignment would attach one frame's crop players to
                # another frame's targets.
                crop_src = crop_boxes = crop_bconf = crop_kps = crop_kconf = crop_off = None
                if crop_h5_path is not None and Path(crop_h5_path).exists():
                    crop_src = h5py.File(crop_h5_path, "r")
                    if not np.array_equal(crop_src["frames"]["frame_index"][:], frame_indices):
                        crop_src.close()
                        raise ValueError(
                            f"Crop side-car frame grid does not match raw poses for "
                            f"{video_path.name}; regenerate it (extract_crop_poses.py)."
                        )
                    crop_off = crop_src["frames"]["frame_offsets"][:]
                    crop_boxes = crop_src["detections"]["boxes"]
                    crop_bconf = crop_src["detections"]["box_conf"]
                    crop_kps = crop_src["detections"]["keypoints"]
                    crop_kconf = crop_src["detections"]["keypoint_conf"]

                with h5py.File(tmp_path, "w") as preproc_h5:
                    frames_group = preproc_h5.create_group("frames")
                    det_group = preproc_h5.create_group("detections")
                    players_group = preproc_h5.create_group("players")

                    frames_group.create_dataset("frame_index", data=frame_indices[sample_idx], dtype="i8")
                    frames_group.create_dataset("timestamps", data=timestamps[sample_idx], dtype="f8")
                    frames_group.create_dataset(
                        "frame_offsets", data=np.array([0], dtype=np.int64), maxshape=(None,), chunks=True
                    )

                    det_group.create_dataset(
                        "boxes", shape=(0, 4), maxshape=(None, 4), dtype="f4", chunks=True, compression="gzip"
                    )
                    det_group.create_dataset(
                        "box_conf", shape=(0,), maxshape=(None,), dtype="f4", chunks=True, compression="gzip"
                    )
                    det_group.create_dataset(
                        "keypoints",
                        shape=(0, 17, 2),
                        maxshape=(None, 17, 2),
                        dtype="f4",
                        chunks=True,
                        compression="gzip",
                    )
                    det_group.create_dataset(
                        "keypoint_conf",
                        shape=(0, 17),
                        maxshape=(None, 17),
                        dtype="f4",
                        chunks=True,
                        compression="gzip",
                    )

                    players_group.create_dataset(
                        "near", shape=(0, 17, 2), maxshape=(None, 17, 2), dtype="f4", chunks=True, compression="gzip"
                    )
                    players_group.create_dataset(
                        "far", shape=(0, 17, 2), maxshape=(None, 17, 2), dtype="f4", chunks=True, compression="gzip"
                    )
                    players_group.create_dataset(
                        "near_conf", shape=(0, 17), maxshape=(None, 17), dtype="f4", chunks=True, compression="gzip"
                    )
                    players_group.create_dataset(
                        "far_conf", shape=(0, 17), maxshape=(None, 17), dtype="f4", chunks=True, compression="gzip"
                    )
                    players_group.create_dataset(
                        "near_box", shape=(0, 4), maxshape=(None, 4), dtype="f4", chunks=True, compression="gzip"
                    )
                    players_group.create_dataset(
                        "far_box", shape=(0, 4), maxshape=(None, 4), dtype="f4", chunks=True, compression="gzip"
                    )
                    players_group.create_dataset(
                        "near_box_conf", shape=(0,), maxshape=(None,), dtype="f4", chunks=True, compression="gzip"
                    )
                    players_group.create_dataset(
                        "far_box_conf", shape=(0,), maxshape=(None,), dtype="f4", chunks=True, compression="gzip"
                    )
                    if crop_src is not None:
                        _create_crop_datasets(players_group)

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

                        filtered = _filter_by_court(
                            frame_boxes, frame_box_conf, frame_kps, frame_kp_conf, court_mask
                        )
                        _append_detections(frames_group, det_group, filtered)
                        _append_players(players_group, self.assigner.assign(filtered))

                        if crop_src is not None:
                            cs, ce = int(crop_off[idx]), int(crop_off[idx + 1])
                            _append_crop_slots(players_group, _crop_slots(
                                np.asarray(crop_boxes[cs:ce], dtype=np.float32),
                                np.asarray(crop_bconf[cs:ce], dtype=np.float32),
                                np.asarray(crop_kps[cs:ce], dtype=np.float32),
                                np.asarray(crop_kconf[cs:ce], dtype=np.float32),
                                court_mask,
                            ))

                    preproc_h5.attrs["target_fps"] = float(self.cfg.target_fps)
                    preproc_h5.attrs["raw_h5"] = str(raw_h5_path)
                    preproc_h5.attrs["video"] = str(video_path)
                    preproc_h5.attrs["annotations"] = str(annotations_path)
                    preproc_h5.attrs["fingerprint"] = fingerprint
                    ignore_before = annotations.get("metadata", {}).get("ignore_before_s")
                    if ignore_before is not None:
                        preproc_h5.attrs["ignore_before_s"] = float(ignore_before)

            tmp_path.replace(output_path)
        finally:
            if crop_src is not None:
                crop_src.close()
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        logger.info("Preprocessed %s", output_path)
        return output_path


def _assert_canonical_dims(raw_h5: h5py.File, video_path: Path, expect_w: int, expect_h: int) -> None:
    width = int(raw_h5.attrs.get("width", 0) or 0)
    height = int(raw_h5.attrs.get("height", 0) or 0)
    if width == 0 or height == 0:
        # Older extracts may omit dims; warn but do not hard-fail.
        logger.warning(
            "Raw HDF5 for %s missing width/height attrs; expected %dx%d after normalize",
            video_path.name,
            expect_w,
            expect_h,
        )
        return
    if width != expect_w or height != expect_h:
        raise ValueError(
            f"Video {video_path.name} has dims {width}x{height}; "
            f"expected {expect_w}x{expect_h}. Run the normalize step first."
        )


def _is_valid_preprocessed_h5(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as h5f:
            required_paths = (
                ("frames", "frame_index"),
                ("frames", "timestamps"),
                ("players", "near"),
                ("players", "far"),
            )
            for group_name, dataset_name in required_paths:
                if group_name not in h5f or dataset_name not in h5f[group_name]:
                    return False
            if "targets" not in h5f:
                return False
        return True
    except Exception:
        return False


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
    segments = list(annotations.get("segments") or [])
    if not segments:
        return np.full(timestamps.shape[0], UNLABELED_TARGET, dtype=np.int8)

    # Sort + overlap validation (also done in normalize_annotations; defensive here).
    segments = sorted(segments, key=lambda s: (float(s["start_time"]), float(s["end_time"])))
    for i in range(1, len(segments)):
        if float(segments[i]["start_time"]) < float(segments[i - 1]["end_time"]):
            raise ValueError(f"Overlapping segments: {segments[i - 1]} overlaps {segments[i]}")

    starts = np.array([float(seg["start_time"]) for seg in segments], dtype=np.float64)
    ends = np.array([float(seg["end_time"]) for seg in segments], dtype=np.float64)
    targets = np.zeros(timestamps.shape[0], dtype=np.int8)

    ignore_before = annotations.get("metadata", {}).get("ignore_before_s")
    if ignore_before is not None:
        ignore_before = float(ignore_before)
        targets[timestamps < ignore_before] = UNLABELED_TARGET

    idx = 0
    for i, ts in enumerate(timestamps):
        if targets[i] == UNLABELED_TARGET:
            continue
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
    dataset[-data.shape[0] :] = data


# --- far-player crop slots ---------------------------------------------------
CROP_SLOTS = ("crop0", "crop1")


DEFAULT_COURT_MASK_PATH = (
    Path(__file__).resolve().parents[2] / "preprocessing" / "default_court_mask.png"
)


@lru_cache(maxsize=4)
def _default_court_mask(width: int, height: int) -> Optional[np.ndarray]:
    """The runtime's fallback mask, resized to the training frame size.

    Same asset and semantics the desktop app uses when live detection fails
    (0 == on court, 255 == out). It is a majority vote over the hand-annotated
    golden courts, so it encodes the typical framing rather than this video's --
    a coarse filter, but far better than none.

    Nearest-neighbour on resize to keep it strictly binary.
    """
    if not DEFAULT_COURT_MASK_PATH.exists():
        logger.warning("Default court mask missing at %s; falling back to no filtering",
                       DEFAULT_COURT_MASK_PATH)
        return None
    mask = cv2.imread(str(DEFAULT_COURT_MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning("Default court mask unreadable at %s", DEFAULT_COURT_MASK_PATH)
        return None
    if (mask.shape[1], mask.shape[0]) != (width, height):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8) * 255


def _feet_on_court(box: np.ndarray, court_mask: Optional[np.ndarray]) -> bool:
    """Feet (bottom-centre) inside the playable court.

    Deliberately feet, not centroid: the court trapezoid narrows with height, so
    a far player near a sideline can have their feet cleanly on court while their
    centroid falls outside the narrower band above them. That error is worst for
    small players high in the frame -- exactly who the crop exists to recover.
    (_filter_by_court still uses centroid for the full-frame near/far slots;
    changing that alters every existing feature and is its own experiment.)
    """
    if court_mask is None:
        return True
    fx = int(np.clip((box[0] + box[2]) / 2, 0, court_mask.shape[1] - 1))
    fy = int(np.clip(box[3], 0, court_mask.shape[0] - 1))
    return bool(court_mask[int(fy), int(fx)] == 0)


def _crop_slots(
    boxes: np.ndarray,
    box_conf: np.ndarray,
    kps: np.ndarray,
    kp_conf: np.ndarray,
    court_mask: Optional[np.ndarray],
):
    """Court-filter -> top-2 by confidence -> order by feet-y ascending.

    Select by confidence so a low-confidence blob cannot evict a real player;
    order by position so slot identity is stable frame to frame (confidence
    ranking flips under small perturbations, which would corrupt the velocity
    and acceleration features derived from slot continuity).

    Slot 0 is the higher (farther) of the two -- usually the far-court player,
    with an intruding near player landing in slot 1.
    """
    keep = [i for i in range(len(boxes)) if _feet_on_court(boxes[i], court_mask)]
    top2 = sorted(keep, key=lambda i: -float(box_conf[i]))[:2]
    ordered = sorted(top2, key=lambda i: float(boxes[i][3]))
    out = []
    for i in ordered:
        out.append({
            "kps": kps[i], "conf": kp_conf[i],
            "box": boxes[i], "box_conf": float(box_conf[i]),
        })
    while len(out) < 2:
        out.append(None)
    return out[0], out[1]


def _append_crop_slots(players_group: h5py.Group, slots) -> None:
    """Append one frame's crop slots, using the -1 'not observed' sentinel for
    an absent slot -- the same convention _pack_players uses for a missing far
    player, so downstream code needs no special case."""
    for name, slot in zip(CROP_SLOTS, slots):
        if slot is None:
            kps = np.full((1, 17, 2), -1.0, dtype=np.float32)
            conf = np.full((1, 17), -1.0, dtype=np.float32)
            box = np.full((1, 4), -1.0, dtype=np.float32)
            bconf = np.full((1,), -1.0, dtype=np.float32)
        else:
            kps = np.asarray(slot["kps"], dtype=np.float32)[None, ...]
            conf = np.asarray(slot["conf"], dtype=np.float32)[None, ...]
            box = np.asarray(slot["box"], dtype=np.float32)[None, ...]
            bconf = np.asarray([slot["box_conf"]], dtype=np.float32)
        _append_rows(players_group[name], kps)
        _append_rows(players_group[f"{name}_conf"], conf)
        _append_rows(players_group[f"{name}_box"], box)
        _append_rows(players_group[f"{name}_box_conf"], bconf)


def _create_crop_datasets(players_group: h5py.Group) -> None:
    for name in CROP_SLOTS:
        players_group.create_dataset(
            name, shape=(0, 17, 2), maxshape=(None, 17, 2), dtype="f4", chunks=True, compression="gzip")
        players_group.create_dataset(
            f"{name}_conf", shape=(0, 17), maxshape=(None, 17), dtype="f4", chunks=True, compression="gzip")
        players_group.create_dataset(
            f"{name}_box", shape=(0, 4), maxshape=(None, 4), dtype="f4", chunks=True, compression="gzip")
        players_group.create_dataset(
            f"{name}_box_conf", shape=(0,), maxshape=(None,), dtype="f4", chunks=True, compression="gzip")
