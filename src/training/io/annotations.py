from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def load_annotations_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def csv_to_json(csv_path: Path, video_path: Path) -> Dict:
    segments: List[Dict[str, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [c.strip().lower() for c in (reader.fieldnames or [])]
        reader.fieldnames = fieldnames
        for row in reader:
            try:
                start = float(row["start_time"])
                end = float(row["end_time"])
            except (KeyError, ValueError, TypeError):
                continue
            segments.append({"start_time": start, "end_time": end, "label": "in_play"})

    return {
        "video_path": str(video_path),
        "segments": segments,
        "metadata": {},
    }


def write_annotations_json(data: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
