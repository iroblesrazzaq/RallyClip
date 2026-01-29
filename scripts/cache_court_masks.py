#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.courts.cache import CourtMaskCache  # noqa: E402
from training.io.config import load_config  # noqa: E402
from training.io.videos import resolve_videos  # noqa: E402
from training.paths import annotations_dir, raw_videos_dir, resolve_data_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache court masks for videos")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["annotated", "all", "list"], help="Video selection mode")
    parser.add_argument("--video", action="append", help="Explicit video list (repeatable)")
    parser.add_argument("--force", action="store_true", help="Force recompute even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    data_root = resolve_data_root(config)

    court_cfg = config.get("court", {})
    mode = args.mode or court_cfg.get("mode", "annotated")
    explicit = args.video or court_cfg.get("videos")

    raw_dir = raw_videos_dir(data_root)
    ann_dir = annotations_dir(data_root)

    videos = resolve_videos(mode, raw_dir, ann_dir, explicit)
    if not videos:
        raise SystemExit("No videos to process")

    cache = CourtMaskCache(
        model_path=court_cfg.get("model_path", "yolov8s.pt"),
        target_time=int(court_cfg.get("target_time", 60)),
    )
    force = bool(args.force or court_cfg.get("force", False))

    for video in videos:
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = raw_dir / video
        if not video_path.exists():
            logging.warning("Video not found: %s", video_path)
            continue
        result = cache.get_or_create(data_root, video_path, force=force)
        status = "success" if result.success else "failed"
        logging.info("Court cache %s: %s", status, video_path.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
