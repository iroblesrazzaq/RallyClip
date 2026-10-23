"""Canonical video normalization: 1280x720 @ 5 fps via ffmpeg.

Everything downstream (court detector, PlayerAssigner, YOLO extract) assumes
this resolution/fps contract. Labels are in wall-clock seconds, so the
re-encode must preserve timeline (same duration, switches at the same times).
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

CANONICAL_WIDTH = 1280
CANONICAL_HEIGHT = 720
CANONICAL_FPS = 5.0


@dataclass
class NormalizeConfig:
    width: int = CANONICAL_WIDTH
    height: int = CANONICAL_HEIGHT
    fps: float = CANONICAL_FPS
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    overwrite: bool = False
    crf: int = 18
    preset: str = "veryfast"


def probe_video(path: Path, *, ffprobe_bin: str = "ffprobe") -> Tuple[int, int, float, float]:
    """Return (width, height, fps, duration_s)."""
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream in {path}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    if isinstance(rate, str) and "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / float(den) if float(den) else 0.0
    else:
        fps = float(rate)
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    return width, height, fps, duration


def is_canonical(path: Path, cfg: NormalizeConfig, *, fps_tol: float = 0.05) -> bool:
    try:
        width, height, fps, _ = probe_video(path, ffprobe_bin=cfg.ffprobe_bin)
    except Exception:
        return False
    return (
        width == cfg.width
        and height == cfg.height
        and abs(fps - cfg.fps) <= fps_tol
    )


def normalize_video(
    src: Path,
    dst: Path,
    cfg: Optional[NormalizeConfig] = None,
) -> Path:
    """Re-encode src → dst as width×height @ fps. Atomic replace into dst."""
    cfg = cfg or NormalizeConfig()
    src = src.resolve()
    dst = dst.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source video not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not cfg.overwrite and is_canonical(dst, cfg):
        logger.info("Skipping existing canonical video: %s", dst)
        return dst

    tmp = dst.with_name(f".{dst.name}.tmp.{uuid4().hex}.mp4")
    try:
        cmd = [
            cfg.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=disable,fps={cfg.fps}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            cfg.preset,
            "-crf",
            str(cfg.crf),
            "-pix_fmt",
            "yuv420p",
            str(tmp),
        ]
        subprocess.run(cmd, check=True)
        if not is_canonical(tmp, cfg):
            width, height, fps, _ = probe_video(tmp, ffprobe_bin=cfg.ffprobe_bin)
            raise RuntimeError(
                f"Normalized output is not canonical: {tmp} "
                f"got {width}x{height}@{fps}, expected {cfg.width}x{cfg.height}@{cfg.fps}"
            )
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    logger.info("Normalized %s → %s", src.name, dst)
    return dst
