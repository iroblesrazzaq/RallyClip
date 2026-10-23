#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.features.builder import FeatureBuildConfig, FeatureBuilder  # noqa: E402
from training.artifact_paths import yolo_conf_imgsz_tags  # noqa: E402
from training.io.config import load_config  # noqa: E402
from training.io.videos import resolve_videos  # noqa: E402
from training.paths import (
    annotations_dir,
    pose_features_dir,
    pose_preprocessed_dir,
    raw_videos_dir,
    resolve_data_root,
)  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build feature HDF5 files from preprocessed data")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["annotated", "all", "list"], help="Video selection mode")
    parser.add_argument("--video", action="append", help="Explicit video list (repeatable)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    data_root = resolve_data_root(config)

    features_cfg = config.get("features", {})
    preprocess_cfg = config.get("preprocess", {})

    mode = args.mode or features_cfg.get("mode", "annotated")
    explicit = args.video or features_cfg.get("videos")

    raw_dir = raw_videos_dir(data_root)
    ann_dir = annotations_dir(data_root)

    videos = resolve_videos(mode, raw_dir, ann_dir, explicit)
    if not videos:
        raise SystemExit("No videos to process")

    # Same resolution as the pipeline: a manifest model supplies its own tag and imgsz.
    yolo_model, conf_tag, imgsz = yolo_conf_imgsz_tags(config)
    fps = float(preprocess_cfg.get("target_fps", 15))

    preproc_root = (
        pose_preprocessed_dir(data_root)
        / f"yolo={yolo_model}"
        / f"conf={conf_tag}"
        / f"imgsz={imgsz}"
        / f"fps={fps}"
    )
    output_root = (
        pose_features_dir(data_root)
        / f"yolo={yolo_model}"
        / f"conf={conf_tag}"
        / f"imgsz={imgsz}"
        / f"fps={fps}"
    )

    feature_set = features_cfg.get("feature_set", "v1")
    builder = FeatureBuilder(FeatureBuildConfig(feature_set=feature_set, target_fps=fps))

    overwrite = bool(args.overwrite or features_cfg.get("overwrite", False))

    for video in videos:
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = raw_dir / video
        if not video_path.exists():
            logging.warning("Video not found: %s", video_path)
            continue

        preproc_path = preproc_root / f"{video_path.stem}__fps{fps}.h5"
        if not preproc_path.exists():
            logging.warning("Preprocessed HDF5 not found: %s", preproc_path)
            continue

        output_path = output_root / f"{video_path.stem}__features__{feature_set}.h5"
        builder.build(preproc_path, output_path, overwrite=overwrite)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
