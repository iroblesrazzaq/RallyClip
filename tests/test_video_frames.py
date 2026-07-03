"""Unit tests for the PyAV-backed VideoFrameReader."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("av")
np = pytest.importorskip("numpy")

from runtime.video_frames import VideoFrameReader  # noqa: E402

CLIP = Path(__file__).resolve().parent / "fixtures" / "golden_cli" / "clip.mp4"

pytestmark = pytest.mark.skipif(not CLIP.is_file(), reason="fixture clip absent")


def test_reader_reports_stream_properties():
    with VideoFrameReader(str(CLIP)) as reader:
        assert 59.0 < reader.fps < 61.0
        assert reader.total_frames == 1440


def test_read_frame_at_index_returns_bgr_frame():
    with VideoFrameReader(str(CLIP)) as reader:
        frame = reader.read_frame_at_index(int(reader.fps * 10))
        assert frame is not None
        assert frame.shape == (720, 1280, 3)
        assert frame.dtype == np.uint8


def test_reader_seeks_backward_after_forward_read():
    with VideoFrameReader(str(CLIP)) as reader:
        late = reader.read_frame_at_index(int(reader.fps * 20))
        early = reader.read_frame_at_index(0)
        assert late is not None and early is not None
        assert not np.array_equal(late, early)


def test_read_past_end_returns_none():
    with VideoFrameReader(str(CLIP)) as reader:
        assert reader.read_frame_at_index(reader.total_frames + 10_000) is None


def test_open_missing_file_raises():
    with pytest.raises(Exception):
        VideoFrameReader(str(CLIP.parent / "does-not-exist.mp4"))
