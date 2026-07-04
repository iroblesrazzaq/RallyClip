from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional

from .contracts import FrameSegment, Interval


def frame_segments_to_intervals(segments: Iterable[FrameSegment], fps: float) -> List[Interval]:
    return [(start / float(fps), end / float(fps)) for start, end in segments]


def read_point_intervals(csv_path: Optional[Path]) -> List[Interval]:
    if csv_path is None or not csv_path.exists():
        return []
    intervals: List[Interval] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [field.strip().lower() for field in reader.fieldnames]
        for row in reader:
            try:
                start = float(row.get("start_time", ""))
                end = float(row.get("end_time", ""))
            except (TypeError, ValueError):
                continue
            if end > start:
                intervals.append((start, end))
    return sorted(intervals, key=lambda item: (item[0], item[1]))


def point_duration(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in intervals)

