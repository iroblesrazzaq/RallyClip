"""Preflight validation for input videos, shared by the CLI and GUI.

Opens a candidate video once and checks it is usable *before* the expensive
pose/preprocess/inference passes, so bad input fails fast with a clean,
user-facing message instead of crashing deep in the pipeline.

Policy (see docs):
  - accept any container PyAV can decode (mp4/mov/mkv/webm/...), validated by
    content, not file extension;
  - reject files that can't be opened or have no video stream;
  - reject resolution below 720p (a real quality floor: the v1 features are
    raw-pixel and the model trained on 720p — upscaling sub-720p can't recover
    keypoint detail YOLO never had);
  - reject footage too short for the model's window (seq_len / fps seconds).

Higher resolutions (1080p/4K) are accepted here and rescaled into the model's
reference space during preprocessing (see preprocessing.data_preprocessor).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av

MIN_HEIGHT = 720


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float


class VideoValidationError(ValueError):
    """An input video failed preflight validation.

    The message is user-facing: callers should surface it directly (clean error,
    no traceback) rather than letting it propagate.
    """


def probe_video(path) -> VideoInfo:
    """Open the file and read video-stream metadata. Raises VideoValidationError
    if it cannot be opened or has no video stream."""
    name = Path(path).name
    try:
        container = av.open(str(path))
    except Exception as exc:  # av raises a variety of errors for bad input
        raise VideoValidationError(f"'{name}' could not be opened as a video file.") from exc
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise VideoValidationError(f"'{name}' has no video stream.")
        cc = stream.codec_context
        width = int(getattr(cc, "width", 0) or 0)
        height = int(getattr(cc, "height", 0) or 0)
        rate = stream.average_rate or stream.base_rate
        fps = float(rate) if rate else 0.0
        duration_s = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_s = float(container.duration) * float(av.time_base)
    finally:
        container.close()
    return VideoInfo(width=width, height=height, fps=fps, duration_s=duration_s)


def validate_video(path, *, seq_len: int, fps: float, min_height: int = MIN_HEIGHT) -> VideoInfo:
    """Validate an input video for the pipeline. Returns its VideoInfo, or raises
    VideoValidationError (clean, user-facing message) for unreadable / no-stream /
    sub-min-height / too-short input."""
    info = probe_video(path)  # raises on unreadable / no video stream
    name = Path(path).name
    if info.width <= 0 or info.height <= 0:
        raise VideoValidationError(f"Could not read the video dimensions of '{name}'.")
    if info.height < min_height:
        raise VideoValidationError(
            f"Video resolution {info.width}x{info.height} is below the {min_height}p minimum. "
            f"Please use footage at least {min_height}p."
        )
    min_seconds = (seq_len / fps) if fps and fps > 0 else 0.0
    if min_seconds > 0 and 0 < info.duration_s < min_seconds:
        raise VideoValidationError(
            f"Video is too short ({info.duration_s:.0f}s); the model needs at least "
            f"{min_seconds:.0f}s of footage."
        )
    return info
