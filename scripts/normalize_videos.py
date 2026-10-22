#!/usr/bin/env python3
"""Normalize source videos to 1280x720 @ 5 fps into data_root/raw_videos/."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.normalize.normalize import NormalizeConfig, normalize_video  # noqa: E402
from training.paths import raw_videos_dir, resolve_data_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML config for data_root")
    parser.add_argument("--data-root", type=Path, help="Override data_root")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Source video files or directories",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.data_root:
        data_root = args.data_root.expanduser().resolve()
    elif args.config:
        from training.io.config import load_config

        data_root = resolve_data_root(load_config(args.config))
    else:
        data_root = (ROOT / "data").resolve()

    out_dir = raw_videos_dir(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = NormalizeConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        overwrite=args.overwrite,
    )

    sources: list[Path] = []
    for item in args.inputs:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            for ext in (".mp4", ".mov", ".avi", ".mkv"):
                sources.extend(sorted(path.rglob(f"*{ext}")))
        else:
            sources.append(path)

    if not sources:
        raise SystemExit("No source videos found")

    for src in sources:
        if "__flip_h" in src.stem:
            logging.info("Skipping flipped variant: %s", src)
            continue
        dst = out_dir / src.name
        normalize_video(src, dst, cfg)

    logging.info("Normalized %d videos into %s", len(sources), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
