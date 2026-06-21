"""Cut a short golden e2e clip out of a real training match.

Extracts a contiguous window [t0, t1] from a source match video (reusing the
tested ``segment_video`` cutter) and offsets the point annotations into
clip-local time, writing both into ``data/e2e/<name>/``:

    data/e2e/<name>/clip.mp4     # the cut window, re-encoded
    data/e2e/<name>/golden.json  # {source, t0, t1, fps, points: [[s, e], ...]}

Only points that fall fully inside [t0, t1] are kept (clean golden boundaries).
The output dir lives under data/ which is gitignored — the footage is private,
so the golden e2e test self-skips wherever this clip is absent (CI), and anyone
with the training data can regenerate it.

Example:
    python scripts/make_e2e_clip.py \
      --source "data/raw_videos/Aditi Narayan ｜ Matchplay_utr6_genF_courtH_IN_angleMED_zoomMED.mp4" \
      --annotations "annotations/Aditi Narayan ｜ Matchplay_utr6_genF_courtH_IN_angleMED_zoomMED.mp4.csv" \
      --t0 0 --t1 130 --name aditi_5pts
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from segmentation.segment import segment_video  # noqa: E402


def read_annotations(csv_path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)  # start_time,end_time(,)
        for row in reader:
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                continue
            points.append((float(row[0]), float(row[1])))
    return points


def clip_annotations(points: list[tuple[float, float]], t0: float, t1: float) -> list[tuple[float, float]]:
    """Keep points fully inside [t0, t1], offset into clip-local time."""
    return [(s - t0, e - t0) for s, e in points if s >= t0 and e <= t1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="source match mp4")
    ap.add_argument("--annotations", required=True, type=Path, help="point annotation csv (start_time,end_time)")
    ap.add_argument("--t0", required=True, type=float, help="window start (seconds, source time)")
    ap.add_argument("--t1", required=True, type=float, help="window end (seconds, source time)")
    ap.add_argument("--name", required=True, help="fixture name (subdir under --out-dir)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "e2e", help="e2e data root (gitignored)")
    args = ap.parse_args()

    if not args.source.is_file():
        ap.error(f"source not found: {args.source}")
    if not args.annotations.is_file():
        ap.error(f"annotations not found: {args.annotations}")
    if args.t1 <= args.t0:
        ap.error("--t1 must be greater than --t0")

    all_points = read_annotations(args.annotations)
    local_points = clip_annotations(all_points, args.t0, args.t1)
    if not local_points:
        ap.error(f"no annotated points fall fully within [{args.t0}, {args.t1}] — pick a window with points")

    out_dir = (args.out_dir / args.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = out_dir / "clip.mp4"
    golden_path = out_dir / "golden.json"

    print(f"Cutting [{args.t0}, {args.t1}]s ({args.t1 - args.t0:.0f}s) -> {clip_path}")
    segment_video(str(args.source), [(args.t0, args.t1)], str(clip_path))

    golden = {
        "source": args.source.name,
        "t0": args.t0,
        "t1": args.t1,
        "duration": args.t1 - args.t0,
        "n_points": len(local_points),
        "points": [[round(s, 3), round(e, 3)] for s, e in local_points],
    }
    golden_path.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    print(f"Wrote {golden_path}  ({len(local_points)} points, {clip_path.stat().st_size / 1e6:.1f} MB clip)")
    for s, e in local_points:
        print(f"   point  {s:7.2f} - {e:7.2f}  ({e - s:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
