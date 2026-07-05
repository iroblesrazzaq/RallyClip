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


def write_point_intervals(csv_path: Path, intervals: Iterable[Interval]) -> None:
    """Write point intervals in the segments.csv contract (3-decimal seconds).

    Writes via a temp file + replace so readers never observe a partial CSV.
    """
    tmp_path = csv_path.with_name(csv_path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["start_time", "end_time"])
        for start, end in intervals:
            writer.writerow([f"{start:.3f}", f"{end:.3f}"])
    tmp_path.replace(csv_path)


def point_duration(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in intervals)

