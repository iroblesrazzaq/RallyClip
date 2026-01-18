#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.features.builder import FeatureBuildConfig, FeatureBuilder  # noqa: E402
from training.io.config import load_config  # noqa: E402
from training.io.videos import resolve_videos  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build feature HDF5 files from preprocessed data")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["annotated", "all", "list"], help="Video selection mode")
    parser.add_argument("--video", action="append", help="Explicit video list (repeatable)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    data_root = Path(config.get("data_root", "data")).expanduser().resolve()

    features_cfg = config.get("features", {})
    preprocess_cfg = config.get("preprocess", {})
    yolo_cfg = config.get("yolo", {})

    mode = args.mode or features_cfg.get("mode", "annotated")
    explicit = args.video or features_cfg.get("videos")

    raw_dir = data_root / "raw_videos"
    ann_dir = data_root / "annotations"

    videos = resolve_videos(mode, raw_dir, ann_dir, explicit)
    if not videos:
        raise SystemExit("No videos to process")

    yolo_model = Path(yolo_cfg.get("model", "yolov8s-pose.pt")).name
    conf = float(yolo_cfg.get("conf", 0.25))
    conf_tag = f"{conf:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    fps = float(preprocess_cfg.get("target_fps", 15))

    preproc_root = data_root / "pose_data" / "preprocessed" / f"yolo={yolo_model}" / f"conf={conf_tag}" / f"fps={fps}"
    output_root = data_root / "pose_data" / "features" / f"yolo={yolo_model}" / f"conf={conf_tag}" / f"fps={fps}"

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
