from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from training.normalize.normalize import NormalizeConfig, is_canonical, normalize_video, probe_video


def _write_color_switch_clip(path: Path, *, width=1920, height=1080, fps=60.0, duration_s=2.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    n = int(duration_s * fps)
    switch_frame = n // 2
    for i in range(n):
        color = (0, 0, 255) if i < switch_frame else (0, 255, 0)  # BGR red then green
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_normalize_outputs_canonical_720p5fps(tmp_path):
    src = tmp_path / "src.mp4"
    dst = tmp_path / "out.mp4"
    _write_color_switch_clip(src)
    normalize_video(src, dst, NormalizeConfig(overwrite=True))
    width, height, fps, duration = probe_video(dst)
    assert width == 1280
    assert height == 720
    assert abs(fps - 5.0) < 0.1
    assert abs(duration - 2.0) < 0.25
    assert is_canonical(dst, NormalizeConfig())


def test_normalize_preserves_timeline_switch(tmp_path):
    src = tmp_path / "src.mp4"
    dst = tmp_path / "out.mp4"
    _write_color_switch_clip(src, duration_s=2.0, fps=60.0)
    normalize_video(src, dst, NormalizeConfig(overwrite=True))

    cap = cv2.VideoCapture(str(dst))
    assert cap.isOpened()
    out_fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    greens = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Green dominance score in BGR.
        greens.append(float(frame[:, :, 1].mean()) - float(frame[:, :, 2].mean()))
    cap.release()
    assert greens
    # First half should be red-dominant (score < 0), second half green-dominant (> 0).
    switch_idx = next((i for i, g in enumerate(greens) if g > 0), None)
    assert switch_idx is not None
    switch_t = switch_idx / out_fps
    assert abs(switch_t - 1.0) <= 0.25, f"switch_t={switch_t}, greens={greens}"
