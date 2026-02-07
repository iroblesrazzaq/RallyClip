#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _format_seconds(value: float) -> str:
    return f"{value:.6f}"


def _to_seconds(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return text
    try:
        return _format_seconds(float(text))
    except ValueError:
        pass

    parts = text.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = float(parts[1])
    else:
        raise ValueError(f"Unsupported timestamp format: {text}")

    total_seconds = (hours * 3600.0) + (minutes * 60.0) + seconds
    return _format_seconds(total_seconds)


def fix_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    for row in rows[1:]:
        if len(row) < 2:
            continue
        row[0] = _to_seconds(row[0])
        row[1] = _to_seconds(row[1])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize annotation CSV timestamps to seconds")
    parser.add_argument("--csv", action="append", required=True, help="CSV file path to update in-place")
    args = parser.parse_args()

    for csv_path in args.csv:
        path = Path(csv_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        fix_csv(path)
        print(f"Fixed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
