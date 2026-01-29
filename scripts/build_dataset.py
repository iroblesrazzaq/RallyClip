#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.dataset.builder import DatasetBuilder, DatasetConfig  # noqa: E402
from training.dataset.splits import SplitConfig  # noqa: E402
from training.io.config import load_config  # noqa: E402
from training.io.videos import resolve_videos  # noqa: E402
from training.paths import (
    annotations_dir,
    datasets_dir,
    pose_features_dir,
    raw_videos_dir,
    resolve_data_root,
)  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset HDF5 files")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["annotated", "all", "list"], help="Video selection mode")
    parser.add_argument("--video", action="append", help="Explicit video list (repeatable)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    data_root = resolve_data_root(config)

    dataset_cfg = config.get("dataset", {})
    features_cfg = config.get("features", {})
    preprocess_cfg = config.get("preprocess", {})
    yolo_cfg = config.get("yolo", {})

    mode = args.mode or dataset_cfg.get("mode", "annotated")
    explicit = args.video or dataset_cfg.get("videos")

    raw_dir = raw_videos_dir(data_root)
    ann_dir = annotations_dir(data_root)

    videos = resolve_videos(mode, raw_dir, ann_dir, explicit)
    if not videos:
        raise SystemExit("No videos to process")

    yolo_model = Path(yolo_cfg.get("model", "yolov8s-pose.pt")).name
    conf = float(yolo_cfg.get("conf", 0.25))
    conf_tag = f"{conf:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    imgsz = int(yolo_cfg.get("imgsz", 1920))
    fps = float(preprocess_cfg.get("target_fps", 15))

    feature_set = features_cfg.get("feature_set", "v1")
    feature_root = (
        pose_features_dir(data_root)
        / f"yolo={yolo_model}"
        / f"conf={conf_tag}"
        / f"imgsz={imgsz}"
        / f"fps={fps}"
    )

    split_cfg = SplitConfig(
        strategy=dataset_cfg.get("split", {}).get("strategy", "hybrid"),
        seed=int(dataset_cfg.get("split", {}).get("seed", 1337)),
        val_ratio=float(dataset_cfg.get("split", {}).get("val_ratio", 0.1)),
        test_ratio=float(dataset_cfg.get("split", {}).get("test_ratio", 0.1)),
        test_videos=dataset_cfg.get("split", {}).get("test_videos", []),
        val_videos=dataset_cfg.get("split", {}).get("val_videos", []),
    )

    builder = DatasetBuilder(
        DatasetConfig(
            seq_len_seconds=float(dataset_cfg.get("seq_len_seconds", 20)),
            overlap_seconds=float(dataset_cfg.get("overlap_seconds", 10)),
            target_fps=fps,
            split=split_cfg,
        )
    )

    run_id = config.get("run_id") or "default"
    output_dir = datasets_dir(data_root) / run_id
    builder.build(feature_root, output_dir, videos, feature_set)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
