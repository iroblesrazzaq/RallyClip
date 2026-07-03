"""PyAV-backed single-frame video access.

Runtime video decode goes through PyAV so the app depends on exactly one
ffmpeg stack for video IO; OpenCV stays an image-processing library only.
Mirrors the cv2.VideoCapture seek-then-read pattern: frame indices address
frames at the stream's average rate.
"""

from __future__ import annotations

from typing import Optional

import av
import numpy as np


class VideoFrameReader:
    """Random access to decoded BGR frames by frame index."""

    def __init__(self, path: str):
        self._container = av.open(str(path))
        if not self._container.streams.video:
            self._container.close()
            raise RuntimeError(f"Failed to open video: {path}")
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"

        rate = self._stream.average_rate or self._stream.guessed_rate
        self.fps: float = float(rate) if rate else 30.0
        if self.fps <= 0:
            self.fps = 30.0

        total = int(self._stream.frames or 0)
        if total <= 0:
            duration_s = 0.0
            if self._stream.duration is not None and self._stream.time_base is not None:
                duration_s = float(self._stream.duration * self._stream.time_base)
            elif self._container.duration is not None:
                duration_s = float(self._container.duration) * float(av.time_base)
            total = int(duration_s * self.fps)
        self.total_frames: int = total

    def read_frame_at_index(self, frame_num: int) -> Optional[np.ndarray]:
        """Decode the frame at frame_num (BGR ndarray), or None when unavailable."""
        target_s = max(0.0, frame_num / self.fps)
        # Seek to the keyframe at/before the target, then decode forward.
        time_base = self._stream.time_base
        try:
            self._container.seek(int(target_s / time_base), stream=self._stream, backward=True)
        except av.AVError:
            return None
        # Accept the first frame whose timestamp reaches the target (within
        # half a frame), matching cv2's read-after-positioning behavior.
        threshold_s = target_s - 0.5 / self.fps
        for frame in self._container.decode(self._stream):
            frame_time = frame.time
            if frame_time is None or frame_time >= threshold_s:
                return frame.to_ndarray(format="bgr24")
        return None

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
