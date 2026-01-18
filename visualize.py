#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from training.io.config import load_config  # noqa: E402
from training.viz.stages import render_stage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RallyClip training visualization")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--stage", required=True, choices=["yolo", "court", "preproc"], help="Visualization stage")
    parser.add_argument("--video", action="append", help="Video path(s) to visualize (repeatable)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    videos = args.video or config.get("videos")
    if not videos:
        raise SystemExit("No videos provided; use --video or set videos in config")

    render_stage(args.stage, config, videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
