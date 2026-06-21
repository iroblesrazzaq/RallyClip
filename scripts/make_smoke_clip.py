"""Generate a small synthetic MP4 for CI pipeline smoke tests.

The clip must be longer than the model's inference window (seq_len=100 frames
at 5 fps = 20 s), so we default to 30 s. mpeg4 is used instead of h264 so the
script never depends on a libx264-enabled PyAV build; the decode side handles
mpeg4-in-mp4 everywhere.

Usage: python scripts/make_smoke_clip.py <output.mp4> [duration_seconds]
"""

from __future__ import annotations

import sys
from pathlib import Path

import av
import numpy as np

# 720p by default so the clip clears the pipeline's 720p input minimum.
WIDTH, HEIGHT, FPS = 1280, 720, 10


def make_clip(out_path: Path, duration_s: float = 30.0, width: int = WIDTH, height: int = HEIGHT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(duration_s * FPS)
    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(total_frames):
            img = np.full((height, width, 3), 96, dtype=np.uint8)
            # A moving rectangle gives the encoder and pose model something
            # non-constant to chew on.
            x = (i * 7) % (width - 60)
            y = (i * 3) % (height - 90)
            img[y : y + 90, x : x + 60] = (200, 180, 40)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    out_path = Path(sys.argv[1])
    make_clip(out_path, duration)
    print(f"Wrote {out_path} ({duration:.0f}s @ {FPS}fps, {WIDTH}x{HEIGHT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
