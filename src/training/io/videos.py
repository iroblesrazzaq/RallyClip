from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv"]


def match_video_from_annotation(annotation_path: Path, raw_dir: Path) -> Optional[str]:
    base = annotation_path.name
    if base.endswith(".csv"):
        base = base[:-4]
    elif base.endswith(".json"):
        base = base[:-5]
    if Path(base).suffix in VIDEO_EXTS and (raw_dir / base).exists():
        return base
    for ext in VIDEO_EXTS:
        candidate = raw_dir / f"{base}{ext}"
        if candidate.exists():
            return candidate.name
    return None


def list_annotated_videos(raw_dir: Path, ann_dir: Path) -> List[str]:
    videos: List[str] = []
    # Prefer JSON annotations; fall back to CSV for backward compatibility.
    annotation_files = sorted(ann_dir.glob("*.json")) or sorted(ann_dir.glob("*.csv"))

    for ann_path in annotation_files:
        match = match_video_from_annotation(ann_path, raw_dir)
        if match:
            videos.append(match)
        else:
            logging.warning("No video found for annotation: %s", ann_path.name)
    return sorted(set(videos))


def list_all_videos(raw_dir: Path) -> List[str]:
    videos: List[str] = []
    for ext in VIDEO_EXTS:
        videos.extend([p.name for p in raw_dir.glob(f"*{ext}")])
    return videos


def resolve_videos(
    mode: str,
    raw_dir: Path,
    ann_dir: Path,
    explicit: Optional[Iterable[str]],
) -> List[str]:
    if explicit:
        return list(explicit)
    if mode == "annotated":
        return list_annotated_videos(raw_dir, ann_dir)
    if mode == "all":
        return list_all_videos(raw_dir)
    if mode == "list":
        return []
    raise ValueError(f"Unknown mode: {mode}")
