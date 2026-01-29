#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.io.annotations import csv_to_json, write_annotations_json  # noqa: E402
from training.io.videos import match_video_from_annotation  # noqa: E402
from training.paths import annotations_dir, raw_videos_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CSV annotations to JSON")
    parser.add_argument("--data-root", default="data", help="Data root containing raw_videos and annotations")
    parser.add_argument("--csv", action="append", help="Explicit CSV file(s) to convert")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    data_root = Path(args.data_root).expanduser().resolve()
    ann_dir = annotations_dir(data_root)
    raw_dir = raw_videos_dir(data_root)

    csv_files = [Path(p) for p in (args.csv or ann_dir.glob("*.csv"))]
    if not csv_files:
        raise SystemExit("No CSV files found")

    for csv_path in csv_files:
        if not csv_path.exists():
            logging.warning("CSV not found: %s", csv_path)
            continue
        video_name = match_video_from_annotation(csv_path, raw_dir)
        if not video_name:
            logging.warning("No video found for CSV: %s", csv_path.name)
            continue
        video_path = raw_dir / video_name
        data = csv_to_json(csv_path, video_path)
        output_path = ann_dir / f"{video_path.name}.json"
        write_annotations_json(data, output_path)
        logging.info("Wrote %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
