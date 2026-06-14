from __future__ import annotations

import logging
import subprocess
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


def flipped_video_name(video_name: str, suffix: str = "__flip_h") -> str:
    path = Path(video_name)
    return f"{path.stem}{suffix}{path.suffix}"


def is_flipped_video(video_name: str, suffix: str = "__flip_h") -> bool:
    return Path(video_name).stem.endswith(suffix)


def flipped_video_output_path(
    video_path: Path,
    *,
    source_root: Path,
    output_root: Path,
    suffix: str = "__flip_h",
) -> Path:
    relative_path = video_path.relative_to(source_root)
    return output_root / relative_path.parent / flipped_video_name(relative_path.name, suffix=suffix)


def create_flipped_videos(
    *,
    raw_dir: Path,
    output_dir: Path,
    videos: Iterable[str],
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
    suffix: str = "__flip_h",
) -> List[Path]:
    created: List[Path] = []
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for video in videos:
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = raw_dir / video
        video_path = video_path.resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_path = flipped_video_output_path(
            video_path,
            source_root=raw_dir,
            output_root=output_dir,
            suffix=suffix,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            logging.info("Skipping existing flipped video: %s", output_path)
            created.append(output_path)
            continue

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y" if overwrite else "-n",
            "-i",
            str(video_path),
            "-vf",
            "hflip",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        logging.info("Created flipped video: %s", output_path)
        created.append(output_path)

    return created
