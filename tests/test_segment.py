from __future__ import annotations

from fractions import Fraction

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from segmentation.segment import _in_interval, load_intervals, segment_video


def _make_clip(path, seconds=12, fps=10, with_audio=True, sample_rate=48000):
    """Generate a tiny test clip (solid color frames + optional 440Hz tone)."""
    container = av.open(str(path), "w")
    try:
        v = container.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = 320, 240, "yuv420p"
        a = None
        if with_audio:
            a = container.add_stream("aac", rate=sample_rate)
            a.layout = "stereo"

        for i in range(seconds * fps):
            arr = np.full((240, 320, 3), i % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = i
            for pkt in v.encode(frame):
                container.mux(pkt)
        for pkt in v.encode():
            container.mux(pkt)

        if a is not None:
            chunk = 1024
            for start in range(0, seconds * sample_rate, chunk):
                n = min(chunk, seconds * sample_rate - start)
                tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(start, start + n) / sample_rate)).astype("float32")
                af = av.AudioFrame.from_ndarray(np.stack([tone, tone]), format="fltp", layout="stereo")
                af.sample_rate = sample_rate
                af.pts = start
                af.time_base = Fraction(1, sample_rate)
                for pkt in a.encode(af):
                    container.mux(pkt)
            for pkt in a.encode():
                container.mux(pkt)
    finally:
        container.close()


def _make_audio_only(path, seconds=2, sample_rate=48000):
    """Generate a clip with an audio stream but no video stream."""
    container = av.open(str(path), "w")
    try:
        a = container.add_stream("aac", rate=sample_rate)
        a.layout = "stereo"
        for start in range(0, seconds * sample_rate, 1024):
            n = min(1024, seconds * sample_rate - start)
            silence = np.zeros(n, dtype="float32")
            af = av.AudioFrame.from_ndarray(np.stack([silence, silence]), format="fltp", layout="stereo")
            af.sample_rate = sample_rate
            af.pts = start
            af.time_base = Fraction(1, sample_rate)
            for pkt in a.encode(af):
                container.mux(pkt)
        for pkt in a.encode():
            container.mux(pkt)
    finally:
        container.close()


def _streams_and_duration(path):
    with av.open(str(path)) as c:
        kinds = {s.type for s in c.streams}
        duration = (c.duration or 0) / av.time_base
    return kinds, duration


def test_load_intervals(tmp_path):
    csv_path = tmp_path / "segs.csv"
    csv_path.write_text("start_time,end_time\n5.0,7.0\n1.0,2.0\nbad,row\n", encoding="utf-8")
    assert load_intervals(str(csv_path)) == [(1.0, 2.0), (5.0, 7.0)]  # sorted, bad row skipped


def test_segment_no_intervals_raises(tmp_path):
    with pytest.raises(ValueError):
        segment_video(str(tmp_path / "in.mp4"), [], str(tmp_path / "out.mp4"))


def test_in_interval_boundaries():
    intervals = [(1.0, 2.0), (5.0, 7.0)]
    starts = [1.0, 5.0]
    assert _in_interval(1.5, intervals, starts, 1e-6)
    assert _in_interval(1.0, intervals, starts, 1e-6)  # inclusive start
    assert _in_interval(7.0, intervals, starts, 1e-6)  # inclusive end
    assert not _in_interval(3.0, intervals, starts, 1e-6)  # in the gap
    assert not _in_interval(0.5, intervals, starts, 1e-6)  # before first


def test_segment_carries_audio_and_concatenates(tmp_path):
    src = tmp_path / "src.mp4"
    try:
        _make_clip(src, seconds=12, with_audio=True)
    except Exception as exc:  # encoder not available in this build
        pytest.skip(f"cannot encode test clip: {exc}")
    out = tmp_path / "out.mp4"

    segment_video(str(src), [(2.0, 4.0), (7.0, 9.0)], str(out))  # 4s total

    kinds, duration = _streams_and_duration(out)
    assert "video" in kinds and "audio" in kinds  # audio carried through (topic 3)
    assert duration == pytest.approx(4.0, abs=0.3)  # frame-accurate concat


def test_segment_video_only_input(tmp_path):
    src = tmp_path / "src_noaudio.mp4"
    try:
        _make_clip(src, seconds=8, with_audio=False)
    except Exception as exc:
        pytest.skip(f"cannot encode test clip: {exc}")
    out = tmp_path / "out.mp4"

    segment_video(str(src), [(1.0, 3.0)], str(out))  # 2s

    kinds, duration = _streams_and_duration(out)
    assert kinds == {"video"}  # no audio stream, no crash
    assert duration == pytest.approx(2.0, abs=0.3)


def test_segment_no_video_stream_raises_without_leaving_a_file(tmp_path):
    src = tmp_path / "audio_only.mp4"
    try:
        _make_audio_only(src)
    except Exception as exc:
        pytest.skip(f"cannot encode test clip: {exc}")
    out = tmp_path / "out.mp4"

    with pytest.raises(RuntimeError):
        segment_video(str(src), [(0.5, 1.0)], str(out))
    assert not out.exists()  # no corrupt/zero-byte output left behind
