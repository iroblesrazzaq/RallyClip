from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick RallyClip e2e quality-test windows from GT JSONs.")
    parser.add_argument("annotations", nargs="+", type=Path, help="Ground-truth JSON files")
    parser.add_argument("--duration", type=int, default=180, help="Window duration in seconds")
    parser.add_argument("--min-points", type=int, default=8, help="Minimum fully-contained GT points")
    parser.add_argument("--step", type=int, default=5, help="Candidate start-time step in seconds")
    args = parser.parse_args()

    entries = [pick_window(path, args.duration, args.min_points, args.step) for path in args.annotations]
    print(json.dumps(entries, indent=2, ensure_ascii=False))
    return 0


def pick_window(path: Path, duration_s: int, min_points: int, step_s: int) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        (float(segment["start_time"]), float(segment["end_time"]))
        for segment in data.get("segments", [])
        if float(segment["end_time"]) > float(segment["start_time"])
    ]
    if not segments:
        raise ValueError(f"{path}: no valid segments")

    last_end = max(end for _, end in segments)
    midpoint = last_end / 2.0
    best: tuple[float, int, list[tuple[float, float]]] | None = None
    low = max(0, int(midpoint) - 900)
    high = int(midpoint) + 900
    for start in range(low, high + 1, step_s):
        end = start + duration_s
        straddles = any(seg_start < start < seg_end or seg_start < end < seg_end for seg_start, seg_end in segments)
        if straddles:
            continue
        contained = [(seg_start, seg_end) for seg_start, seg_end in segments if seg_start >= start and seg_end <= end]
        if len(contained) < min_points:
            continue
        distance = abs((start + duration_s / 2.0) - midpoint)
        if best is None or distance < best[0]:
            best = (distance, start, contained)

    if best is None:
        raise ValueError(f"{path}: no {duration_s}s window with at least {min_points} non-straddling segments")

    _, start, contained = best
    video_name = path.name.removesuffix(".json")
    return {
        "id": slugify(Path(video_name).stem),
        "video": video_name,
        "gt": path.name,
        "start_time_s": start,
        "duration_s": duration_s,
        "gt_points_in_window": len(contained),
    }


def slugify(value: str) -> str:
    out = []
    last_was_sep = False
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
            last_was_sep = False
        elif not last_was_sep:
            out.append("_")
            last_was_sep = True
    return "".join(out).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
