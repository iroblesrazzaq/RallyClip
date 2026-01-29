from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import cv2
import h5py
import numpy as np

from training.paths import (
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


def render_stage(stage: str, config: Dict[str, Any], videos: Iterable[str]) -> None:
    stage_map = {
        "yolo": _render_yolo,
        "court": _render_court,
        "preproc": _render_preproc,
    }
    if stage not in stage_map:
        raise ValueError(f"Unknown stage: {stage}")

    for video in videos:
        stage_map[stage](video, config)


def _render_yolo(video: str, config: Dict[str, Any]) -> None:
    data_root = resolve_data_root(config)
    yolo_cfg = config.get("yolo", {})
    extract_cfg = config.get("extract", {})

    video_path = _resolve_video(data_root, video)
    raw_h5 = _raw_h5_path(data_root, video_path, yolo_cfg, extract_cfg)

    run_id = config.get("run_id") or "default"
    output_path = _output_path(data_root, run_id, video_path.stem, "yolo")
    _render_raw_overlay(video_path, raw_h5, output_path)


def _render_court(video: str, config: Dict[str, Any]) -> None:
    data_root = resolve_data_root(config)
    video_path = _resolve_video(data_root, video)
    cache_path = pose_courts_dir(data_root) / f"{video_path.stem}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Court cache not found: {cache_path}")

    run_id = config.get("run_id") or "default"
    output_path = _output_path(data_root, run_id, video_path.stem, "court")
    _render_court_overlay(video_path, cache_path, output_path)


def _render_preproc(video: str, config: Dict[str, Any]) -> None:
    data_root = resolve_data_root(config)
    yolo_cfg = config.get("yolo", {})
    preprocess_cfg = config.get("preprocess", {})

    video_path = _resolve_video(data_root, video)
    preproc_h5 = _preproc_h5_path(data_root, video_path, yolo_cfg, preprocess_cfg)
    run_id = config.get("run_id") or "default"
    output_path = _output_path(data_root, run_id, video_path.stem, "preproc")
    _render_preproc_overlay(video_path, preproc_h5, output_path)


def _render_raw_overlay(video_path: Path, h5_path: Path, output_path: Path) -> None:
    if not h5_path.exists():
        raise FileNotFoundError(f"Raw HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        frame_index = h5f["frames"]["frame_index"][:]
        offsets = h5f["frames"]["frame_offsets"][:]
        boxes = h5f["detections"]["boxes"]
        box_conf = h5f["detections"]["box_conf"]
        keypoints = h5f["detections"]["keypoints"]
        kp_conf = h5f["detections"]["keypoint_conf"]
        data_w = int(h5f.attrs.get("width", 0))
        data_h = int(h5f.attrs.get("height", 0))

        cap, writer, scale = _open_video_writer(video_path, output_path, data_w, data_h)
        pointer = 0
        idx = 0
        while True:
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
        cap.release()
        writer.release()


def _render_court_overlay(video_path: Path, cache_path: Path, output_path: Path) -> None:
    cache = np.load(cache_path, allow_pickle=True)
    mask = cache.get("mask")
    metadata_json = cache.get("metadata_json")
    metadata = json.loads(metadata_json.item()) if metadata_json is not None else {}

    baseline = metadata.get("baseline") or metadata.get("metadata", {}).get("baseline")
    left_line = metadata.get("left_doubles_sideline") or metadata.get("metadata", {}).get("left_doubles_sideline")
    right_line = metadata.get("right_doubles_sideline") or metadata.get("metadata", {}).get("right_doubles_sideline")
    extended = metadata.get("extended_sidelines") or metadata.get("metadata", {}).get("extended_sidelines")

    cap, writer, _ = _open_video_writer(video_path, output_path, None, None)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        draw_mask(frame, mask, alpha=0.3, color=(0, 0, 255))
        if baseline:
            draw_lines(frame, baseline, (0, 255, 0), 2)
        if left_line:
            draw_lines(frame, left_line, (255, 0, 0), 2)
        if right_line:
            draw_lines(frame, right_line, (255, 0, 0), 2)
        if extended:
            draw_lines(frame, extended, (255, 0, 255), 1)
        writer.write(frame)
    cap.release()
    writer.release()


def _render_preproc_overlay(video_path: Path, h5_path: Path, output_path: Path) -> None:
    if not h5_path.exists():
        raise FileNotFoundError(f"Preprocessed HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        frame_index = h5f["frames"]["frame_index"][:]
        targets = h5f["targets"][:]
        players = h5f["players"]
        near_kps = players["near"][:]
        far_kps = players["far"][:]
        near_conf = players["near_conf"][:]
        far_conf = players["far_conf"][:]
        near_box = players["near_box"][:]
        far_box = players["far_box"][:]

        data_w = int(h5f.attrs.get("width", 0)) if "width" in h5f.attrs else 0
        data_h = int(h5f.attrs.get("height", 0)) if "height" in h5f.attrs else 0

        cap, writer, scale = _open_video_writer(video_path, output_path, data_w, data_h)
        pointer = 0
        idx = 0
        prev = {"near": None, "far": None, "near_v": None, "far_v": None}

        while True:
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


def _open_video_writer(video_path: Path, output_path: Path, data_w: int | None, data_h: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    scale = (1.0, 1.0)
    if data_w and data_h and data_w > 0 and data_h > 0:
        scale = (width / data_w, height / data_h)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    return cap, writer, scale


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


def _raw_h5_path(data_root: Path, video_path: Path, yolo_cfg: Dict[str, Any], extract_cfg: Dict[str, Any]) -> Path:
    conf = float(yolo_cfg.get("conf", 0.25))
    conf_tag = f"{conf:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    model_tag = Path(str(yolo_cfg.get("model", "yolov8s-pose.pt"))).name
    imgsz = int(yolo_cfg.get("imgsz", 1920))
    start_time = extract_cfg.get("start_time", 0)
    duration = extract_cfg.get("duration")
    dur_tag = "full" if duration in (None, "", "null") else str(duration)
    raw_root = pose_raw_dir(data_root) / f"yolo={model_tag}" / f"conf={conf_tag}" / f"imgsz={imgsz}"
    return raw_root / f"{video_path.stem}__start{start_time}__dur{dur_tag}.h5"


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
