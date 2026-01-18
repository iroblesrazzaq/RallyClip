#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.io.config import load_config  # noqa: E402
from training.pose.yolo_hdf5 import YoloExtractConfig, YoloHdf5Extractor  # noqa: E402

VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv"]


def _format_conf(conf: float) -> str:
    text = f"{conf:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _format_time(value: Optional[float]) -> str:
    if value is None:
        return "full"
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _match_video_from_annotation(csv_path: Path, raw_dir: Path) -> Optional[str]:
    base = csv_path.name
    if base.endswith(".csv"):
        base = base[:-4]
    if Path(base).suffix in VIDEO_EXTS and (raw_dir / base).exists():
        return base
    for ext in VIDEO_EXTS:
        candidate = raw_dir / f"{base}{ext}"
        if candidate.exists():
            return candidate.name
    return None


def _list_annotated_videos(raw_dir: Path, ann_dir: Path) -> List[str]:
    videos = []
    for csv_path in ann_dir.glob("*.csv"):
        match = _match_video_from_annotation(csv_path, raw_dir)
        if match:
            videos.append(match)
        else:
            logging.warning("No video found for annotation: %s", csv_path.name)
    return videos


def _list_all_videos(raw_dir: Path) -> List[str]:
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend([p.name for p in raw_dir.glob(f"*{ext}")])
    return videos


def _resolve_videos(
    mode: str,
    raw_dir: Path,
    ann_dir: Path,
    explicit: Optional[Iterable[str]],
) -> List[str]:
    if explicit:
        return list(explicit)
    if mode == "annotated":
        return _list_annotated_videos(raw_dir, ann_dir)
    if mode == "all":
        return _list_all_videos(raw_dir)
    if mode == "list":
        return []
    raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract YOLO outputs to HDF5")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["annotated", "all", "list"], help="Video selection mode")
    parser.add_argument("--video", action="append", help="Explicit video list (repeatable)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    data_root = Path(config.get("data_root", "data")).expanduser().resolve()

    yolo_cfg = config.get("yolo", {})
    extract_cfg = config.get("extract", {})

    mode = args.mode or extract_cfg.get("mode", "annotated")
    explicit_videos = args.video or extract_cfg.get("videos")

    raw_dir = data_root / "raw_videos"
    ann_dir = data_root / "annotations"

    videos = _resolve_videos(mode, raw_dir, ann_dir, explicit_videos)
    if not videos:
        raise SystemExit("No videos to process")

    start_time = float(extract_cfg.get("start_time", 0))
    duration = extract_cfg.get("duration")
    duration_val = None if duration in (None, "", "null") else float(duration)

    overwrite = bool(args.overwrite or extract_cfg.get("overwrite", False))
    resume = bool(args.resume or extract_cfg.get("resume", True))

    conf = float(yolo_cfg.get("conf", 0.25))
    model_path = str(yolo_cfg.get("model", "yolov8s-pose.pt"))
    model_dir = yolo_cfg.get("model_dir", "models")
    device = yolo_cfg.get("device")
    batch_size = yolo_cfg.get("batch_size")

    extractor = YoloHdf5Extractor(
        YoloExtractConfig(
            model_path=model_path,
            conf=conf,
            model_dir=str(model_dir) if model_dir else None,
            device=device,
            batch_size=batch_size,
        )
    )

    conf_tag = _format_conf(conf)
    model_tag = Path(model_path).name
    output_root = data_root / "pose_data" / "raw" / f"yolo={model_tag}" / f"conf={conf_tag}"

    for video_name in videos:
        video_path = Path(video_name)
        if not video_path.is_absolute():
            video_path = raw_dir / video_name
        if not video_path.exists():
            logging.warning("Video not found: %s", video_path)
            continue
        stem = video_path.stem
        out_name = f"{stem}__start{_format_time(start_time)}__dur{_format_time(duration_val)}.h5"
        output_path = output_root / out_name
        extractor.extract(
            video_path=video_path,
            output_path=output_path,
            start_time=start_time,
            duration=duration_val,
            overwrite=overwrite,
            resume=resume,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
