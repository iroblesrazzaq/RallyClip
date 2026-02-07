#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.io.config import load_config, resolve_court_model_path  # noqa: E402
from training.io.videos import resolve_videos  # noqa: E402
from training.paths import (
    annotations_dir,
    pose_preprocessed_dir,
    pose_raw_dir,
    raw_videos_dir,
    resolve_data_root,
)  # noqa: E402
from training.preprocess.preprocessor import Hdf5Preprocessor, PreprocessConfig  # noqa: E402


def _format_time(value):
    if value in (None, "", "null"):
        return "full"
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _raw_h5_filename(stem: str, start_time, duration, sampling_mode: str, sample_fps) -> str:
    base = f"{stem}__start{_format_time(start_time)}__dur{_format_time(duration)}"
    if sampling_mode == "downsample_then_extract":
        if sample_fps in (None, "", "null"):
            raise ValueError("sample_fps must be set when sampling_mode='downsample_then_extract'")
        return f"{base}__samplefps{_format_time(sample_fps)}.h5"
    return f"{base}.h5"


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess YOLO HDF5 outputs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["annotated", "all", "list"], help="Video selection mode")
    parser.add_argument("--video", action="append", help="Explicit video list (repeatable)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    data_root = resolve_data_root(config)
    court_model_path = resolve_court_model_path(config)

    preprocess_cfg = config.get("preprocess", {})
    court_cfg = config.get("court", {})
    extract_cfg = config.get("extract", {})
    yolo_cfg = config.get("yolo", {})

    mode = args.mode or preprocess_cfg.get("mode", "annotated")
    explicit = args.video or preprocess_cfg.get("videos")

    raw_dir = raw_videos_dir(data_root)
    ann_dir = annotations_dir(data_root)

    videos = resolve_videos(mode, raw_dir, ann_dir, explicit)
    if not videos:
        raise SystemExit("No videos to process")

    yolo_model = Path(yolo_cfg.get("model", "yolov8s-pose.pt")).name
    conf = float(yolo_cfg.get("conf", 0.25))
    conf_tag = f"{conf:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    imgsz = int(yolo_cfg.get("imgsz", 1920))

    output_root = (
        pose_preprocessed_dir(data_root)
        / f"yolo={yolo_model}"
        / f"conf={conf_tag}"
        / f"imgsz={imgsz}"
        / f"fps={preprocess_cfg.get('target_fps', 15)}"
    )

    start_time = extract_cfg.get("start_time", 0)
    duration = extract_cfg.get("duration")
    sampling_mode = str(extract_cfg.get("sampling_mode", "full_then_downsample"))
    sample_fps = extract_cfg.get("sample_fps")

    preprocessor = Hdf5Preprocessor(
        PreprocessConfig(
            target_fps=float(preprocess_cfg.get("target_fps", 15)),
            save_court_masks=bool(preprocess_cfg.get("save_court_masks", False)),
            court_model_path=court_model_path,
            court_target_time=int(court_cfg.get("target_time", 60)),
            court_force=bool(court_cfg.get("force", False)),
        )
    )

    overwrite = bool(args.overwrite or preprocess_cfg.get("overwrite", False))

    for video in videos:
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = raw_dir / video
        if not video_path.exists():
            logging.warning("Video not found: %s", video_path)
            continue

        raw_h5 = (
            pose_raw_dir(data_root)
            / f"yolo={yolo_model}"
            / f"conf={conf_tag}"
            / f"imgsz={imgsz}"
            / _raw_h5_filename(video_path.stem, start_time, duration, sampling_mode, sample_fps)
        )
        if not raw_h5.exists():
            logging.warning("Raw HDF5 not found: %s", raw_h5)
            continue

        annotations_path = ann_dir / f"{video_path.name}.json"
        output_path = output_root / f"{video_path.stem}__fps{preprocess_cfg.get('target_fps', 15)}.h5"
        preprocessor.preprocess(
            data_root,
            raw_h5,
            video_path,
            annotations_path,
            output_path,
            overwrite=overwrite,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
