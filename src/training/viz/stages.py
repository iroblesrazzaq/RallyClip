from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import h5py
import numpy as np

from training.paths import (
    datasets_dir,
    pose_courts_dir,
    pose_preprocessed_dir,
    pose_raw_dir,
    raw_videos_dir,
    resolve_data_root,
    visualizations_dir,
)
from training.viz.overlays import (
    draw_boxes,
    draw_keypoints,
    draw_lines,
    draw_mask,
    draw_skeleton,
    draw_text_block,
)

logger = logging.getLogger(__name__)


def render_stage(
    stage: str,
    config: Dict[str, Any],
    videos: Iterable[str],
    start_time: float = 0.0,
    duration: Optional[float] = None,
) -> None:
    stage_map = {
        "yolo": _render_yolo,
        "court": _render_court,
        "court_image": _render_court_image,
        "preproc": _render_preproc,
    }
    if stage not in stage_map:
        raise ValueError(f"Unknown stage: {stage}")

    for video in videos:
        stage_map[stage](video, config, start_time, duration)


def render_dataset_sequence(
    config: Dict[str, Any],
    dataset_run_id: str,
    split: str,
    sequence_index: int,
    output_run_id: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Path:
    data_root = resolve_data_root(config)
    dataset_h5 = datasets_dir(data_root) / dataset_run_id / f"{split}.h5"
    if not dataset_h5.exists():
        raise FileNotFoundError(f"Dataset split not found: {dataset_h5}")

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if sequence_index < 0:
        raise ValueError("sequence_index must be >= 0")

    with h5py.File(dataset_h5, "r") as ds:
        if "sequence_frame_index" not in ds or "sequence_video_index" not in ds:
            raise ValueError(
                "Dataset file missing sequence mapping metadata. Rebuild dataset with current pipeline."
            )
        total_sequences = int(ds["sequence_frame_index"].shape[0])
        if sequence_index >= total_sequences:
            raise IndexError(f"sequence_index {sequence_index} out of range (n={total_sequences})")

        seq_frame_idx = np.asarray(ds["sequence_frame_index"][sequence_index], dtype=np.int64)
        if "sequence_timestamps" in ds:
            seq_timestamps = np.asarray(ds["sequence_timestamps"][sequence_index], dtype=np.float64)
        else:
            seq_timestamps = np.full(seq_frame_idx.shape[0], -1.0, dtype=np.float64)
        seq_targets = np.asarray(ds["targets"][sequence_index], dtype=np.int8)
        seq_video_idx = int(ds["sequence_video_index"][sequence_index])

        if "video_index_to_name" not in ds:
            raise ValueError(
                "Dataset file missing video_index_to_name metadata. Rebuild dataset with current pipeline."
            )
        video_names = _decode_h5_strings(ds["video_index_to_name"][:])
        if seq_video_idx < 0 or seq_video_idx >= len(video_names):
            raise IndexError(f"sequence_video_index {seq_video_idx} out of bounds for video mapping")
        video_name = video_names[seq_video_idx]

    if max_frames is not None:
        keep = max(int(max_frames), 0)
        seq_frame_idx = seq_frame_idx[:keep]
        seq_timestamps = seq_timestamps[:keep]
        seq_targets = seq_targets[:keep]

    if seq_frame_idx.size == 0:
        raise ValueError("Sequence has no frames to render")

    yolo_cfg = config.get("yolo", {})
    preprocess_cfg = config.get("preprocess", {})
    video_path = _resolve_video(data_root, video_name)
    preproc_h5 = _preproc_h5_path(data_root, video_path, yolo_cfg, preprocess_cfg)
    if not preproc_h5.exists():
        raise FileNotFoundError(f"Preprocessed HDF5 not found: {preproc_h5}")

    with h5py.File(preproc_h5, "r") as pre:
        pre_frame_idx = np.asarray(pre["frames"]["frame_index"][:], dtype=np.int64)
        players = pre["players"]
        near_kps = np.asarray(players["near"][:], dtype=np.float32)
        far_kps = np.asarray(players["far"][:], dtype=np.float32)
        near_conf = np.asarray(players["near_conf"][:], dtype=np.float32)
        far_conf = np.asarray(players["far_conf"][:], dtype=np.float32)
        near_box = np.asarray(players["near_box"][:], dtype=np.float32)
        far_box = np.asarray(players["far_box"][:], dtype=np.float32)
        near_box_conf = np.asarray(players["near_box_conf"][:], dtype=np.float32)
        far_box_conf = np.asarray(players["far_box_conf"][:], dtype=np.float32)

    frame_to_pre_idx = {int(fi): i for i, fi in enumerate(pre_frame_idx.tolist())}
    cache = _load_court_cache(pose_courts_dir(data_root) / f"{video_path.stem}.npz")
    status_lines = _court_status_lines(cache)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps_out = float(cap.get(cv2.CAP_PROP_FPS) or preprocess_cfg.get("target_fps", 15))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    run_id = output_run_id or config.get("run_id") or "default"
    output_path = (
        visualizations_dir(data_root)
        / run_id
        / f"{video_path.stem}__dataset_{split}_seq{sequence_index}.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (width, height),
    )

    seq_frame_list = [int(v) for v in seq_frame_idx.tolist()]
    seq_step_by_frame = {frame: step for step, frame in enumerate(seq_frame_list)}
    clip_start = min(seq_frame_list)
    clip_end = max(seq_frame_list)

    prev = {"near": None, "far": None, "near_v": None, "far_v": None}
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        for frame_idx in range(clip_start, clip_end + 1):
            ret, frame = cap.read()
            if not ret:
                continue

            _draw_court_overlay(frame, cache, status_lines)
            step = seq_step_by_frame.get(int(frame_idx))
            if step is None:
                draw_text_block(frame, ["dataset_sample=no"], (10, 220), (180, 180, 180))
                ts = -1.0
            else:
                pre_idx = frame_to_pre_idx.get(int(frame_idx))
                if pre_idx is None:
                    draw_text_block(frame, ["MISSING preproc frame mapping"], (10, 220), (0, 0, 255))
                else:
                    draw_boxes(
                        frame,
                        near_box[pre_idx][None, ...],
                        np.asarray([near_box_conf[pre_idx]], dtype=np.float32),
                        (0, 255, 0),
                    )
                    draw_boxes(
                        frame,
                        far_box[pre_idx][None, ...],
                        np.asarray([far_box_conf[pre_idx]], dtype=np.float32),
                        (255, 0, 0),
                    )
                    draw_keypoints(frame, near_kps[pre_idx], near_conf[pre_idx], (0, 255, 0))
                    draw_keypoints(frame, far_kps[pre_idx], far_conf[pre_idx], (255, 0, 0))
                    draw_skeleton(frame, near_kps[pre_idx], near_conf[pre_idx], (0, 255, 0))
                    draw_skeleton(frame, far_kps[pre_idx], far_conf[pre_idx], (255, 0, 0))

                    label = int(seq_targets[step]) if step < len(seq_targets) else -1
                    info = _player_feature_text(near_box[pre_idx], far_box[pre_idx], label, prev)
                    draw_text_block(frame, info["near"], (10, 20), (0, 255, 0))
                    draw_text_block(frame, info["far"], (10, 100), (255, 0, 0))
                    draw_text_block(frame, info["meta"], (10, 180), (255, 255, 255))
                ts = float(seq_timestamps[step]) if step < len(seq_timestamps) else -1.0

            meta_lines = [
                f"split={split} seq={sequence_index} sample_step={(step + 1) if step is not None else '-'} / {len(seq_frame_idx)}",
                f"video={video_name}",
                f"orig_frame={int(frame_idx)} ts={ts:.3f}s" if ts >= 0 else f"orig_frame={int(frame_idx)}",
            ]
            draw_text_block(frame, meta_lines, (10, height - 50), (255, 255, 255))
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return output_path


def _render_yolo(video: str, config: Dict[str, Any], start_time: float, duration: Optional[float]) -> None:
    data_root = resolve_data_root(config)
    yolo_cfg = config.get("yolo", {})
    extract_cfg = config.get("extract", {})

    video_path = _resolve_video(data_root, video)
    raw_h5 = _raw_h5_path(data_root, video_path, yolo_cfg, extract_cfg)

    run_id = config.get("run_id") or "default"
    output_path = _output_path(data_root, run_id, video_path.stem, "yolo")
    _render_raw_overlay(video_path, raw_h5, output_path, start_time=start_time, duration=duration)


def _render_court(video: str, config: Dict[str, Any], start_time: float, duration: Optional[float]) -> None:
    data_root = resolve_data_root(config)
    video_path = _resolve_video(data_root, video)
    cache_path = pose_courts_dir(data_root) / f"{video_path.stem}.npz"

    run_id = config.get("run_id") or "default"
    output_path = _output_path(data_root, run_id, video_path.stem, "court")
    _render_court_overlay(video_path, cache_path, output_path, start_time=start_time, duration=duration)


def _render_court_image(video: str, config: Dict[str, Any], start_time: float, duration: Optional[float]) -> None:
    data_root = resolve_data_root(config)
    video_path = _resolve_video(data_root, video)
    cache_path = pose_courts_dir(data_root) / f"{video_path.stem}.npz"

    run_id = config.get("run_id") or "default"
    target_time = float(start_time)
    if target_time <= 0:
        target_time = float(config.get("court", {}).get("target_time", 60))
    output_path = _output_image_path(data_root, run_id, video_path.stem, "court", target_time)
    _render_court_snapshot(video_path, cache_path, output_path, target_time=target_time)


def _render_preproc(video: str, config: Dict[str, Any], start_time: float, duration: Optional[float]) -> None:
    data_root = resolve_data_root(config)
    yolo_cfg = config.get("yolo", {})
    preprocess_cfg = config.get("preprocess", {})

    video_path = _resolve_video(data_root, video)
    preproc_h5 = _preproc_h5_path(data_root, video_path, yolo_cfg, preprocess_cfg)
    run_id = config.get("run_id") or "default"
    output_path = _output_path(data_root, run_id, video_path.stem, "preproc")
    _render_preproc_overlay(video_path, preproc_h5, output_path, start_time=start_time, duration=duration)


def _render_raw_overlay(
    video_path: Path,
    h5_path: Path,
    output_path: Path,
    start_time: float = 0.0,
    duration: Optional[float] = None,
) -> None:
    if not h5_path.exists():
        raise FileNotFoundError(f"Raw HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        # Some local h5py builds fail on direct integer dataset reads.
        frame_index = np.asarray(h5f["frames"]["frame_index"].astype("float64")[:], dtype=np.int64)
        offsets = np.asarray(h5f["frames"]["frame_offsets"].astype("float64")[:], dtype=np.int64)
        boxes = h5f["detections"]["boxes"]
        box_conf = h5f["detections"]["box_conf"]
        keypoints = h5f["detections"]["keypoints"]
        kp_conf = h5f["detections"]["keypoint_conf"]
        data_w = 0
        data_h = 0

        cap, writer, scale, start_frame, end_frame = _open_video_writer(
            video_path,
            output_path,
            data_w,
            data_h,
            start_time=start_time,
            duration=duration,
        )
        try:
            pointer = 0
            idx = start_frame
            while True:
                if end_frame is not None and idx >= end_frame:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                while pointer < len(frame_index) and frame_index[pointer] < idx:
                    pointer += 1
                if pointer < len(frame_index) and frame_index[pointer] == idx:
                    start = int(offsets[pointer])
                    end = int(offsets[pointer + 1])
                    frame_boxes = _scale_boxes(np.array(boxes[start:end]), scale)
                    frame_box_conf = np.array(box_conf[start:end])
                    frame_kps = _scale_keypoints(np.array(keypoints[start:end]), scale)
                    frame_kp_conf = np.array(kp_conf[start:end])
                    draw_boxes(frame, frame_boxes, frame_box_conf, (0, 255, 0))
                    for kp, conf in zip(frame_kps, frame_kp_conf):
                        draw_keypoints(frame, kp, conf, (0, 255, 255))
                        draw_skeleton(frame, kp, conf, (0, 255, 255))
                writer.write(frame)
                idx += 1
        finally:
            cap.release()
            writer.release()


def _render_court_overlay(
    video_path: Path,
    cache_path: Path,
    output_path: Path,
    start_time: float = 0.0,
    duration: Optional[float] = None,
) -> None:
    cache = _load_court_cache(cache_path)
    status_lines = _court_status_lines(cache)

    cap, writer, _, start_frame, end_frame = _open_video_writer(
        video_path,
        output_path,
        None,
        None,
        start_time=start_time,
        duration=duration,
    )
    try:
        idx = start_frame
        while True:
            if end_frame is not None and idx >= end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break
            _draw_court_overlay(frame, cache, status_lines)
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()


def _render_court_snapshot(
    video_path: Path,
    cache_path: Path,
    output_path: Path,
    target_time: float,
) -> None:
    cache = _load_court_cache(cache_path)
    status_lines = _court_status_lines(cache)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_frame = max(int(round(max(target_time, 0.0) * fps)), 0)
        if target_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame from: {video_path}")
        _draw_court_overlay(frame, cache, status_lines)
        draw_text_block(frame, [f"frame_time={target_time:.1f}s"], (10, 20), (255, 255, 255))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write image: {output_path}")
    finally:
        cap.release()


def _render_preproc_overlay(
    video_path: Path,
    h5_path: Path,
    output_path: Path,
    start_time: float = 0.0,
    duration: Optional[float] = None,
) -> None:
    if not h5_path.exists():
        raise FileNotFoundError(f"Preprocessed HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        frame_index = np.asarray(h5f["frames"]["frame_index"].astype("float64")[:], dtype=np.int64)
        targets = h5f["targets"][:]
        players = h5f["players"]
        near_kps = players["near"][:]
        far_kps = players["far"][:]
        near_conf = players["near_conf"][:]
        far_conf = players["far_conf"][:]
        near_box = players["near_box"][:]
        far_box = players["far_box"][:]

        data_w = 0
        data_h = 0

        cap, writer, scale, start_frame, end_frame = _open_video_writer(
            video_path,
            output_path,
            data_w,
            data_h,
            start_time=start_time,
            duration=duration,
        )
        try:
            pointer = 0
            idx = start_frame
            prev = {"near": None, "far": None, "near_v": None, "far_v": None}

            while True:
                if end_frame is not None and idx >= end_frame:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                while pointer < len(frame_index) and frame_index[pointer] < idx:
                    pointer += 1
                if pointer < len(frame_index) and frame_index[pointer] == idx:
                    near_box_scaled = _scale_boxes(near_box[pointer][None, ...], scale)
                    far_box_scaled = _scale_boxes(far_box[pointer][None, ...], scale)
                    near_kps_scaled = _scale_keypoints(near_kps[pointer][None, ...], scale)
                    far_kps_scaled = _scale_keypoints(far_kps[pointer][None, ...], scale)

                    draw_boxes(frame, near_box_scaled, np.array([1.0]), (0, 255, 0))
                    draw_boxes(frame, far_box_scaled, np.array([1.0]), (255, 0, 0))
                    draw_keypoints(frame, near_kps_scaled[0], near_conf[pointer], (0, 255, 0))
                    draw_keypoints(frame, far_kps_scaled[0], far_conf[pointer], (255, 0, 0))
                    draw_skeleton(frame, near_kps_scaled[0], near_conf[pointer], (0, 255, 0))
                    draw_skeleton(frame, far_kps_scaled[0], far_conf[pointer], (255, 0, 0))

                    info = _player_feature_text(
                        near_box[pointer],
                        far_box[pointer],
                        targets[pointer],
                        prev,
                    )
                    draw_text_block(frame, info["near"], (10, 20), (0, 255, 0))
                    draw_text_block(frame, info["far"], (10, 100), (255, 0, 0))
                    draw_text_block(frame, info["meta"], (10, 180), (255, 255, 255))
                writer.write(frame)
                idx += 1
        finally:
            cap.release()
            writer.release()


def _player_feature_text(near_box: np.ndarray, far_box: np.ndarray, target: int, prev: Dict[str, Any]) -> Dict[str, Any]:
    near = _centroid(near_box)
    far = _centroid(far_box)

    near_v = _velocity(near, prev.get("near")) if prev.get("near") else (0.0, 0.0)
    far_v = _velocity(far, prev.get("far")) if prev.get("far") else (0.0, 0.0)
    near_a = _velocity(near_v, prev.get("near_v")) if prev.get("near_v") else (0.0, 0.0)
    far_a = _velocity(far_v, prev.get("far_v")) if prev.get("far_v") else (0.0, 0.0)

    prev["near"] = near
    prev["far"] = far
    prev["near_v"] = near_v
    prev["far_v"] = far_v

    near_speed = float(np.sqrt(near_v[0] ** 2 + near_v[1] ** 2))
    far_speed = float(np.sqrt(far_v[0] ** 2 + far_v[1] ** 2))
    near_acc = float(np.sqrt(near_a[0] ** 2 + near_a[1] ** 2))
    far_acc = float(np.sqrt(far_a[0] ** 2 + far_a[1] ** 2))

    return {
        "near": [
            f"near: cx={near[0]:.1f} cy={near[1]:.1f}",
            f"v=({near_v[0]:.2f},{near_v[1]:.2f}) a=({near_a[0]:.2f},{near_a[1]:.2f})",
            f"speed={near_speed:.2f} acc={near_acc:.2f}",
        ],
        "far": [
            f"far: cx={far[0]:.1f} cy={far[1]:.1f}",
            f"v=({far_v[0]:.2f},{far_v[1]:.2f}) a=({far_a[0]:.2f},{far_a[1]:.2f})",
            f"speed={far_speed:.2f} acc={far_acc:.2f}",
        ],
        "meta": [f"annotation={int(target)}"],
    }


def _centroid(box: np.ndarray) -> Tuple[float, float]:
    if np.all(box < 0):
        return (0.0, 0.0)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _velocity(curr: Tuple[float, float], prev: Tuple[float, float]) -> Tuple[float, float]:
    return (curr[0] - prev[0], curr[1] - prev[1])


def _open_video_writer(
    video_path: Path,
    output_path: Path,
    data_w: int | None,
    data_h: int | None,
    start_time: float = 0.0,
    duration: Optional[float] = None,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    scale = (1.0, 1.0)
    if data_w and data_h and data_w > 0 and data_h > 0:
        scale = (width / data_w, height / data_h)

    start_frame = max(int(round(max(start_time, 0.0) * fps)), 0)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    end_frame: Optional[int] = None
    if duration is not None:
        end_frame = start_frame + max(int(round(max(duration, 0.0) * fps)), 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    return cap, writer, scale, start_frame, end_frame


def _scale_boxes(boxes: np.ndarray, scale: Tuple[float, float]) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    sx, sy = scale
    out = boxes.copy()
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


def _scale_keypoints(kps: np.ndarray, scale: Tuple[float, float]) -> np.ndarray:
    if kps.size == 0:
        return kps
    sx, sy = scale
    out = kps.copy()
    out[..., 0] *= sx
    out[..., 1] *= sy
    return out


def _resolve_video(data_root: Path, video: str) -> Path:
    video_path = Path(video)
    if not video_path.is_absolute():
        video_path = raw_videos_dir(data_root) / video
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path


def _format_time(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _raw_h5_path(data_root: Path, video_path: Path, yolo_cfg: Dict[str, Any], extract_cfg: Dict[str, Any]) -> Path:
    conf = float(yolo_cfg.get("conf", 0.25))
    conf_tag = f"{conf:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    model_tag = Path(str(yolo_cfg.get("model", "yolov8s-pose.pt"))).name
    imgsz = int(yolo_cfg.get("imgsz", 1920))
    start_time = extract_cfg.get("start_time", 0)
    duration = extract_cfg.get("duration")
    sampling_mode = str(extract_cfg.get("sampling_mode", "full_then_downsample"))
    sample_fps = extract_cfg.get("sample_fps")
    start_tag = _format_time(float(start_time))
    dur_tag = "full" if duration in (None, "", "null") else _format_time(float(duration))
    raw_root = pose_raw_dir(data_root) / f"yolo={model_tag}" / f"conf={conf_tag}" / f"imgsz={imgsz}"
    base = f"{video_path.stem}__start{start_tag}__dur{dur_tag}"
    if sampling_mode == "downsample_then_extract":
        if sample_fps in (None, "", "null"):
            raise ValueError("sample_fps must be set when sampling_mode='downsample_then_extract'")
        sample_tag = _format_time(float(sample_fps))
        return raw_root / f"{base}__samplefps{sample_tag}.h5"
    return raw_root / f"{base}.h5"


def _preproc_h5_path(data_root: Path, video_path: Path, yolo_cfg: Dict[str, Any], preprocess_cfg: Dict[str, Any]) -> Path:
    conf = float(yolo_cfg.get("conf", 0.25))
    conf_tag = f"{conf:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    model_tag = Path(str(yolo_cfg.get("model", "yolov8s-pose.pt"))).name
    imgsz = int(yolo_cfg.get("imgsz", 1920))
    fps = preprocess_cfg.get("target_fps", 15)
    preproc_root = (
        pose_preprocessed_dir(data_root)
        / f"yolo={model_tag}"
        / f"conf={conf_tag}"
        / f"imgsz={imgsz}"
        / f"fps={fps}"
    )
    return preproc_root / f"{video_path.stem}__fps{fps}.h5"


def _output_path(data_root: Path, run_id: str, stem: str, stage: str) -> Path:
    return visualizations_dir(data_root) / run_id / f"{stem}__{stage}.mp4"


def _output_image_path(data_root: Path, run_id: str, stem: str, stage: str, target_time: float) -> Path:
    time_tag = f"{target_time:.1f}".replace(".", "p")
    return visualizations_dir(data_root) / run_id / f"{stem}__{stage}_t{time_tag}.png"


def _load_court_cache(cache_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "exists": cache_path.exists(),
        "success": False,
        "mask": None,
        "metadata": {},
        "error": "cache_missing",
        "baseline": None,
        "left_line": None,
        "right_line": None,
        "extended": None,
    }
    if not cache_path.exists():
        return info

    cache = np.load(cache_path, allow_pickle=True)
    success_raw = cache.get("success")
    success = bool(success_raw.item()) if hasattr(success_raw, "item") else bool(success_raw)
    metadata_json = cache.get("metadata_json")
    metadata = json.loads(metadata_json.item()) if metadata_json is not None else {}

    nested = metadata.get("metadata", {})
    info["success"] = success
    info["metadata"] = metadata
    info["mask"] = cache.get("mask")
    info["baseline"] = metadata.get("baseline") or nested.get("baseline")
    info["left_line"] = metadata.get("left_doubles_sideline") or nested.get("left_doubles_sideline")
    info["right_line"] = metadata.get("right_doubles_sideline") or nested.get("right_doubles_sideline")
    info["extended"] = metadata.get("extended_sidelines") or nested.get("extended_sidelines")
    info["error"] = nested.get("error") or metadata.get("error")
    if info["success"] and info["mask"] is not None:
        info["error"] = None
    elif not info["error"]:
        info["error"] = "mask_missing_or_detection_failed"
    return info


def _court_status_lines(cache: Dict[str, Any]) -> list[str]:
    if cache["success"] and cache["mask"] is not None:
        return ["COURT MASK OK"]
    reason = str(cache.get("error") or "unknown_failure")
    if len(reason) > 80:
        reason = reason[:77] + "..."
    return ["COURT MASK FAILED", f"reason={reason}"]


def _draw_court_overlay(frame: np.ndarray, cache: Dict[str, Any], status_lines: list[str]) -> None:
    draw_mask(frame, cache.get("mask"), alpha=0.3, color=(0, 0, 255))
    if cache.get("baseline"):
        draw_lines(frame, cache["baseline"], (0, 255, 0), 2)
    if cache.get("left_line"):
        draw_lines(frame, cache["left_line"], (255, 0, 0), 2)
    if cache.get("right_line"):
        draw_lines(frame, cache["right_line"], (255, 0, 0), 2)
    if cache.get("extended"):
        draw_lines(frame, cache["extended"], (255, 0, 255), 1)
    color = (0, 255, 0) if cache.get("success") and cache.get("mask") is not None else (0, 0, 255)
    draw_text_block(frame, status_lines, (10, 40), color)


def _decode_h5_strings(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out
