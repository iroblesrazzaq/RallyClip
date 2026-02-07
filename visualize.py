#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from training.io.config import load_config  # noqa: E402
from training.viz.stages import render_dataset_sequence, render_stage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RallyClip training visualization")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["yolo", "court", "court_image", "preproc", "dataset_seq"],
        help="Visualization stage",
    )
    parser.add_argument("--video", action="append", help="Video path(s) to visualize (repeatable)")
    parser.add_argument("--run-id", help="Override run_id for output directory")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: full)")
    parser.add_argument("--dataset-run-id", help="Dataset run id under data/datasets for dataset_seq stage")
    parser.add_argument("--dataset-split", default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--sequence-index", type=int, default=0, help="Sequence index within split for dataset_seq")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on rendered frames for dataset_seq")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    if args.run_id:
        config["run_id"] = args.run_id

    if args.stage == "dataset_seq":
        dataset_run_id = args.dataset_run_id or config.get("run_id")
        if not dataset_run_id:
            raise SystemExit("dataset_seq requires --dataset-run-id or config run_id")
        output_path = render_dataset_sequence(
            config=config,
            dataset_run_id=str(dataset_run_id),
            split=args.dataset_split,
            sequence_index=int(args.sequence_index),
            output_run_id=args.run_id,
            max_frames=args.max_frames,
        )
        logging.info("Wrote dataset sequence visualization: %s", output_path)
        return 0

    videos = args.video or config.get("videos")
    if not videos:
        raise SystemExit("No videos provided; use --video or set videos in config")

    render_stage(args.stage, config, videos, start_time=args.start_time, duration=args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
