#!/usr/bin/env python3
"""Convert new_data/labels CSV+meta → RallyClip annotation JSON.

Writes annotations/{video.name}.json with segments + metadata.ignore_before_s.
Never copies labels_meta.json into annotations/ (any *.json there is treated
as an annotation by the pipeline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.io.annotations import csv_to_json, write_annotations_json  # noqa: E402


def convert_one(csv_path: Path, meta_path: Path | None, out_path: Path) -> dict:
    video_name = csv_path.name  # e.g. <hash>.mp4.csv → used as video_path basename rule
    # Training annotation naming: annotations/{video.name}.json where video.name is e.g. foo.mp4
    if video_name.endswith(".csv"):
        video_basename = video_name[: -len(".csv")]
    else:
        video_basename = video_name

    metadata: dict = {}
    if meta_path is not None and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        ignore = meta.get("ignore_before_s")
        if ignore is not None:
            metadata["ignore_before_s"] = float(ignore)
        metadata["formula_version"] = meta.get("formula_version")
        metadata["mode"] = meta.get("mode")
        metadata["session"] = meta.get("session")

    data = csv_to_json(csv_path, Path(video_basename), metadata=metadata)
    write_annotations_json(data, out_path)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=ROOT.parent / "new_data" / "labels",
        help="Directory containing <hash>.mp4.csv and <hash>.labels_meta.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination annotations/ directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotation JSON files",
    )
    args = parser.parse_args()

    labels_dir: Path = args.labels_dir.expanduser().resolve()
    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top-level only — never ingest new_data/labels/rallies_excluded/.
    csvs = sorted(p for p in labels_dir.glob("*.mp4.csv") if p.parent == labels_dir)
    if not csvs:
        raise SystemExit(f"No *.mp4.csv files in {labels_dir}")

    converted = 0
    skipped = 0
    for csv_path in csvs:
        stem = csv_path.name[: -len(".mp4.csv")]
        meta_path = labels_dir / f"{stem}.labels_meta.json"
        out_path = out_dir / f"{stem}.mp4.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        data = convert_one(csv_path, meta_path if meta_path.exists() else None, out_path)
        ignore = data.get("metadata", {}).get("ignore_before_s")
        print(
            f"wrote {out_path.name}: segments={len(data['segments'])} "
            f"ignore_before_s={ignore}"
        )
        converted += 1

    print(f"done: converted={converted} skipped={skipped} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
