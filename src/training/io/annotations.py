from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_annotations_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return normalize_annotations(data, video_path=Path(data.get("video_path", path.stem)))


def csv_to_json(csv_path: Path, video_path: Path, *, metadata: Optional[Dict] = None) -> Dict:
    segments: List[Dict[str, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [c.strip().lower() for c in (reader.fieldnames or [])]
        if not fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        reader.fieldnames = fieldnames
        if "start_time" not in fieldnames or "end_time" not in fieldnames:
            raise ValueError(
                f"CSV must have start_time,end_time columns (got {fieldnames}): {csv_path}"
            )
        for row_num, row in enumerate(reader, start=2):
            raw_start = (row.get("start_time") or "").strip()
            raw_end = (row.get("end_time") or "").strip()
            if not raw_start and not raw_end:
                continue
            try:
                start = float(raw_start)
                end = float(raw_end)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Unparseable time on row {row_num} of {csv_path}: "
                    f"start_time={raw_start!r} end_time={raw_end!r}. "
                    "Expected float seconds (not HH:MM:SS.mmm)."
                ) from exc
            if start >= end:
                raise ValueError(
                    f"Invalid segment on row {row_num} of {csv_path}: "
                    f"start_time={start} >= end_time={end}"
                )
            segments.append({"start_time": start, "end_time": end, "label": "in_play"})

    return normalize_annotations(
        {
            "video_path": str(video_path),
            "segments": segments,
            "metadata": dict(metadata or {}),
        },
        video_path=video_path,
    )


def normalize_annotations(data: Dict, *, video_path: Path) -> Dict:
    segments = list(data.get("segments") or [])
    cleaned: List[Dict] = []
    for i, seg in enumerate(segments):
        try:
            start = float(seg["start_time"])
            end = float(seg["end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid segment[{i}] in annotations for {video_path}") from exc
        if start >= end:
            raise ValueError(
                f"Invalid segment[{i}] for {video_path}: start_time={start} >= end_time={end}"
            )
        cleaned.append(
            {
                "start_time": start,
                "end_time": end,
                "label": seg.get("label", "in_play"),
            }
        )
    cleaned.sort(key=lambda s: (s["start_time"], s["end_time"]))
    for i in range(1, len(cleaned)):
        if cleaned[i]["start_time"] < cleaned[i - 1]["end_time"]:
            raise ValueError(
                f"Overlapping segments for {video_path}: "
                f"{cleaned[i - 1]} overlaps {cleaned[i]}"
            )

    metadata = dict(data.get("metadata") or {})
    ignore_before = metadata.get("ignore_before_s", data.get("ignore_before_s"))
    if ignore_before is not None:
        ignore_before = float(ignore_before)
        if ignore_before < 0:
            raise ValueError(f"ignore_before_s must be >= 0 for {video_path}")
        metadata["ignore_before_s"] = ignore_before

    return {
        "video_path": str(data.get("video_path") or video_path),
        "segments": cleaned,
        "metadata": metadata,
    }


def write_annotations_json(data: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
