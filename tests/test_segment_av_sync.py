"""Audio/video sync regression tests for segment_video (issue #21).

Video and audio are quantised on different grids, so naive per-stream
re-stamping accumulates drift across concatenated segments. These tests cut
many segments from a synthesized A/V clip and assert the output audio length
stays locked to the video length.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from segmentation.segment import segment_video  # noqa: E402

FPS = 30
RATE = 44100
DURATION_S = 30


@pytest.fixture(scope="module")
def av_clip(tmp_path_factory) -> Path:
    """Synthesized clip: gray frames + 440 Hz sine, 30s, 30fps, 44.1kHz AAC."""
    path = tmp_path_factory.mktemp("avsync") / "clip.mp4"
    container = av.open(str(path), "w")
    v = container.add_stream("libx264", rate=FPS)
    v.width, v.height, v.pix_fmt = 320, 240, "yuv420p"
    a = container.add_stream("aac", rate=RATE)
    a.layout = "mono"

    for i in range(DURATION_S * FPS):
        img = np.full((240, 320, 3), (i * 3) % 255, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for pkt in v.encode(frame):
            container.mux(pkt)

    samples_per_chunk = 1024
    t0 = 0
    while t0 < DURATION_S * RATE:
        n = min(samples_per_chunk, DURATION_S * RATE - t0)
        ts = (np.arange(t0, t0 + n) / RATE) * 2 * math.pi * 440
        chunk = (np.sin(ts) * 0.3).astype(np.float32).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="mono")
        frame.sample_rate = RATE
        frame.pts = t0
        frame.time_base = Fraction(1, RATE)
        for pkt in a.encode(frame):
            container.mux(pkt)
        t0 += n

    for pkt in v.encode():
        container.mux(pkt)
    for pkt in a.encode():
        container.mux(pkt)
    container.close()
    return path


def _stream_durations(path: Path) -> tuple[float, float]:
    with av.open(str(path)) as container:
        v = container.streams.video[0]
        a = container.streams.audio[0]
        v_dur = float(v.duration * v.time_base)
        a_dur = float(a.duration * a.time_base)
    return v_dur, a_dur


def test_many_segments_do_not_accumulate_av_drift(av_clip, tmp_path):
    # 10 short cuts: naive independent re-timing drifts by up to ~1 frame per
    # boundary; the locked timeline must stay within one AAC frame overall.
    intervals = [(s, s + 1.7) for s in np.arange(1.0, 29.0, 2.8)]
    out = tmp_path / "out.mp4"
    segment_video(str(av_clip), [(float(s), float(e)) for s, e in intervals], str(out))

    v_dur, a_dur = _stream_durations(out)
    assert abs(v_dur - a_dur) <= (1024 / RATE) + 0.005, (
        f"audio/video durations diverged: video={v_dur:.4f}s audio={a_dur:.4f}s"
    )
    expected = sum(e - s for s, e in intervals)
    assert v_dur == pytest.approx(expected, abs=len(intervals) * (1.5 / FPS))


def test_single_segment_stays_in_sync(av_clip, tmp_path):
    out = tmp_path / "out.mp4"
    segment_video(str(av_clip), [(2.0, 12.0)], str(out))
    v_dur, a_dur = _stream_durations(out)
    assert abs(v_dur - a_dur) <= (1024 / RATE) + 0.005
    assert v_dur == pytest.approx(10.0, abs=0.2)


def test_overlapping_intervals_are_merged_not_duplicated(av_clip, tmp_path):
    out = tmp_path / "out.mp4"
    segment_video(str(av_clip), [(1.0, 5.0), (4.0, 8.0)], str(out))
    v_dur, _ = _stream_durations(out)
    # Merged [1, 8) is ~7s; duplication would yield ~8s.
    assert v_dur == pytest.approx(7.0, abs=0.2)


def test_video_only_input_still_works(av_clip, tmp_path):
    video_only = tmp_path / "mute.mp4"
    with av.open(str(video_only), "w") as dst:
        out_v = dst.add_stream("libx264", rate=FPS)
        out_v.width, out_v.height, out_v.pix_fmt = 320, 240, "yuv420p"
        for i in range(10 * FPS):
            img = np.full((240, 320, 3), (i * 3) % 255, dtype=np.uint8)
            for pkt in out_v.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                dst.mux(pkt)
        for pkt in out_v.encode():
            dst.mux(pkt)

    out = tmp_path / "out.mp4"
    segment_video(str(video_only), [(1.0, 3.0), (5.0, 7.0)], str(out))
    with av.open(str(out)) as container:
        assert container.streams.video
        assert not container.streams.audio
