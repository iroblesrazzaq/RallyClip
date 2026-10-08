from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from segmentation import segment as segment_module
from segmentation.segment import (
    _in_interval,
    _keyframe_padded_intervals,
    _stream_copy_video,
    load_intervals,
    segment_video,
    segment_video_sources,
    timeline_intervals_by_source,
)
from segmentation import proxy as proxy_module
from segmentation.proxy import create_analysis_proxy, probe_video_sources


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


def test_videotoolbox_bitrate_scales_with_resolution_fps_and_crf():
    default_1080p60 = segment_module._videotoolbox_bitrate(1920, 1080, Fraction(60, 1), 20)
    default_4k60 = segment_module._videotoolbox_bitrate(3840, 2160, Fraction(60, 1), 20)

    assert default_1080p60 == 7_464_960
    assert default_4k60 == 29_859_840
    assert segment_module._videotoolbox_bitrate(1920, 1080, Fraction(60, 1), 14) == 14_929_920
    assert segment_module._videotoolbox_bitrate(1920, 1080, Fraction(60, 1), 26) == 3_732_480


def test_videotoolbox_failure_retries_cleanly_with_libx264(tmp_path, monkeypatch):
    output = tmp_path / "out.mp4"
    calls = []

    monkeypatch.setattr(
        segment_module,
        "_video_encoder_candidates",
        lambda: (segment_module.VIDEOTOOLBOX_ENCODER, segment_module.SOFTWARE_ENCODER),
    )

    def fake_encode(input_video, intervals, output_path, *, eps, crf, video_encoder):
        calls.append((intervals, video_encoder))
        if video_encoder == segment_module.VIDEOTOOLBOX_ENCODER:
            Path(output_path).write_bytes(b"incomplete hardware output")
            raise av.error.ExternalError(1, "hardware encoder failed")
        assert not Path(output_path).exists()
        Path(output_path).write_bytes(b"software output")

    monkeypatch.setattr(segment_module, "_segment_video_with_encoder", fake_encode)

    segment_video("input.mp4", [(4.0, 8.0), (1.0, 5.0)], str(output))

    assert calls == [
        ([(1.0, 8.0)], segment_module.VIDEOTOOLBOX_ENCODER),
        ([(1.0, 8.0)], segment_module.SOFTWARE_ENCODER),
    ]
    assert output.read_bytes() == b"software output"


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


def _make_video_only_clip(path, seconds=8, fps=10, gop=1):
    """Video-only H.264 clip with a fixed keyframe interval."""
    container = av.open(str(path), "w")
    try:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = 320, 240, "yuv420p"
        video.options = {"g": str(gop), "keyint_min": str(gop)}
        for i in range(seconds * fps):
            frame = av.VideoFrame.from_ndarray(np.full((240, 320, 3), i % 256, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
    finally:
        container.close()


def test_keyframe_pads_refuse_ranges_closer_than_two_pads():
    with pytest.raises(RuntimeError, match="overlap"):
        _keyframe_padded_intervals([(1.0, 2.0), (2.5, 3.5)], 1.0)

    assert _keyframe_padded_intervals([(1.0, 2.0), (8.0, 9.0)], 1.0) == [
        (1.0, 2.0, 0.0, 3.0),
        (8.0, 9.0, 7.0, 10.0),
    ]


def test_close_video_only_points_fall_back_instead_of_repeating(tmp_path):
    src = tmp_path / "intra.mp4"
    try:
        _make_video_only_clip(src, seconds=8, fps=10, gop=1)
    except Exception as exc:
        pytest.skip(f"cannot encode test clip: {exc}")
    out = tmp_path / "out.mp4"

    segment_video(str(src), [(1.0, 2.0), (2.5, 3.5)], str(out))

    kinds, duration = _streams_and_duration(out)
    assert kinds == {"video"}
    # Frame-accurate cuts are 1s + 1s. A padded remux would repeat the overlap
    # and land well above 2s.
    assert duration == pytest.approx(2.0, abs=0.35)


def test_keyframe_aligned_cut_stream_copies_without_extra_footage(tmp_path):
    src = tmp_path / "intra.mp4"
    try:
        _make_video_only_clip(src, seconds=8, fps=10, gop=1)
    except Exception as exc:
        pytest.skip(f"cannot encode test clip: {exc}")
    out = tmp_path / "out.mp4"

    _stream_copy_video(str(src), [(2.0, 3.0)], str(out))

    kinds, duration = _streams_and_duration(out)
    assert kinds == {"video"}
    assert duration == pytest.approx(1.0, abs=0.35)


def test_keyframe_overshoot_refuses_stream_copy(tmp_path):
    src = tmp_path / "gop.mp4"
    try:
        _make_video_only_clip(src, seconds=8, fps=10, gop=40)
    except Exception as exc:
        pytest.skip(f"cannot encode test clip: {exc}")
    out = tmp_path / "out.mp4"

    with pytest.raises(RuntimeError, match="outside"):
        _stream_copy_video(str(src), [(0.2, 0.8), (2.0, 2.5)], str(out), keyframe_pad_s=0.3)


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


def test_proxy_silence_uses_bounded_frames(tmp_path, monkeypatch):
    sizes = []
    real_frame = av.AudioFrame

    def bounded_frame(*args, **kwargs):
        samples = kwargs.get("samples")
        if samples is None and len(args) >= 3:
            samples = args[2]
        sizes.append(int(samples or 0))
        if samples and int(samples) > 4096:
            raise AssertionError(f"silence frame too large: {samples}")
        return real_frame(*args, **kwargs)

    monkeypatch.setattr(proxy_module.av, "AudioFrame", bounded_frame)
    src = tmp_path / "silent.mp4"
    try:
        _make_clip(src, seconds=1, fps=10, with_audio=False)
    except Exception as exc:
        pytest.skip(f"cannot encode test clip: {exc}")

    create_analysis_proxy(probe_video_sources([src]), tmp_path / "proxy.mp4", fps=10)

    assert sizes
    assert max(sizes) <= 4096


def _make_delayed_audio_clip(path, seconds=2, fps=10, delay_s=1.0, sample_rate=48000):
    """Video from t=0 and a tone whose timestamps begin at delay_s."""
    container = av.open(str(path), "w")
    try:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = 320, 240, "yuv420p"
        audio = container.add_stream("aac", rate=sample_rate)
        audio.layout = "stereo"
        for i in range(seconds * fps):
            frame = av.VideoFrame.from_ndarray(np.full((240, 320, 3), 32, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        start_sample = int(delay_s * sample_rate)
        chunk = 1024
        for start in range(start_sample, seconds * sample_rate, chunk):
            n = min(chunk, seconds * sample_rate - start)
            tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(n) / sample_rate)).astype("float32")
            frame = av.AudioFrame.from_ndarray(np.stack([tone, tone]), format="fltp", layout="stereo")
            frame.sample_rate = sample_rate
            frame.pts = start
            frame.time_base = Fraction(1, sample_rate)
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    finally:
        container.close()


def _audio_rms_between(path, start_s, end_s):
    pieces = []
    with av.open(str(path)) as container:
        stream = next(item for item in container.streams if item.type == "audio")
        cursor = 0.0
        for frame in container.decode(stream):
            if frame.pts is not None and frame.time_base is not None:
                cursor = float(frame.pts * frame.time_base)
            duration = frame.samples / float(frame.sample_rate)
            if cursor + duration <= start_s:
                cursor += duration
                continue
            if cursor >= end_s:
                break
            pieces.append(frame.to_ndarray().astype("float32").reshape(-1))
            cursor += duration
    if not pieces:
        return 0.0
    stacked = np.concatenate(pieces)
    return float(np.sqrt(np.mean(np.square(stacked))))


def test_proxy_keeps_a_late_audio_start(tmp_path):
    src = tmp_path / "late.mp4"
    try:
        _make_delayed_audio_clip(src)
    except Exception as exc:
        pytest.skip(f"cannot encode delayed-audio clip: {exc}")
    output = tmp_path / "proxy.mp4"

    create_analysis_proxy(probe_video_sources([src]), output, fps=10)

    assert _audio_rms_between(output, 0.0, 0.6) < 0.02
    assert _audio_rms_between(output, 1.3, 1.8) > 0.02


def test_proxy_keeps_audio_on_a_later_clip(tmp_path):
    first = tmp_path / "001.mp4"
    second = tmp_path / "002.mp4"
    try:
        _make_clip(first, seconds=1, fps=10, with_audio=True)
        _make_clip(second, seconds=1, fps=10, with_audio=True)
    except Exception as exc:
        pytest.skip(f"cannot encode test clips: {exc}")
    output = tmp_path / "proxy.mp4"

    create_analysis_proxy(probe_video_sources([first, second]), output, fps=10)

    assert _audio_rms_between(output, 0.15, 0.7) > 0.02
    assert _audio_rms_between(output, 1.15, 1.7) > 0.02


def test_proxy_and_export_reject_multiple_video_tracks():
    class _Stream:
        def __init__(self, kind):
            self.type = kind

    tracks = [_Stream("video"), _Stream("video"), _Stream("audio")]
    with pytest.raises(ValueError, match="video tracks"):
        proxy_module._one_track(tracks, "video", "clip.mp4")
    with pytest.raises(RuntimeError, match="video tracks"):
        segment_module._only_track(type("Box", (), {"streams": tracks})(), "video", "clip.mp4")
    with pytest.raises(RuntimeError, match="audio tracks"):
        segment_module._only_track(
            type("Box", (), {"streams": [_Stream("video"), _Stream("audio"), _Stream("audio")]})(),
            "audio",
            "clip.mp4",
        )


def test_timeline_intervals_split_cleanly_across_chunk_boundary():
    sources = [
        {"path": "one.mp4", "start_s": 0.0, "duration_s": 10.0},
        {"path": "two.mp4", "start_s": 10.0, "duration_s": 8.0},
    ]

    assert timeline_intervals_by_source(sources, [(8.5, 11.25), (15.0, 16.0)]) == [
        ("one.mp4", [(8.5, 10.0)]),
        ("two.mp4", [(0.0, 1.25), (5.0, 6.0)]),
    ]


def test_segment_video_sources_keeps_source_timeline_order(tmp_path):
    first = tmp_path / "001.mp4"
    second = tmp_path / "002.mp4"
    try:
        _make_clip(first, seconds=4, with_audio=True)
        _make_clip(second, seconds=4, with_audio=True)
    except Exception as exc:
        pytest.skip(f"cannot encode test clips: {exc}")
    output = tmp_path / "combined.mp4"
    sources = [
        {"path": str(first), "start_s": 0.0, "duration_s": 4.0},
        {"path": str(second), "start_s": 4.0, "duration_s": 4.0},
    ]

    segment_video_sources(sources, [(3.0, 5.0), (6.0, 7.0)], str(output))

    kinds, duration = _streams_and_duration(output)
    assert kinds == {"video", "audio"}
    assert duration == pytest.approx(3.0, abs=0.35)


def test_analysis_proxy_is_continuous_and_returns_exact_offsets(tmp_path):
    first = tmp_path / "clip_1.mp4"
    second = tmp_path / "clip_2.mp4"
    try:
        _make_clip(first, seconds=2, fps=10, with_audio=True)
        _make_clip(second, seconds=2, fps=10, with_audio=True)
    except Exception as exc:
        pytest.skip(f"cannot encode test clips: {exc}")
    output = tmp_path / "proxy.mp4"

    manifest = create_analysis_proxy(probe_video_sources([first, second]), output, fps=10)

    kinds, duration = _streams_and_duration(output)
    assert kinds == {"video", "audio"}
    assert duration == pytest.approx(4.0, abs=0.35)
    assert manifest[0]["start_s"] == 0.0
    assert manifest[1]["start_s"] == pytest.approx(manifest[0]["duration_s"], abs=0.001)


def test_segment_video_sources_preserves_timing_across_different_frame_rates(tmp_path):
    first = tmp_path / "10fps.mp4"
    second = tmp_path / "15fps.mp4"
    try:
        _make_clip(first, seconds=4, fps=10, with_audio=True)
        _make_clip(second, seconds=4, fps=15, with_audio=True)
    except Exception as exc:
        pytest.skip(f"cannot encode test clips: {exc}")
    output = tmp_path / "mixed-rate.mp4"
    sources = [
        {"path": str(first), "start_s": 0.0, "duration_s": 4.0},
        {"path": str(second), "start_s": 4.0, "duration_s": 4.0},
    ]

    segment_video_sources(sources, [(3.0, 5.0)], str(output))

    kinds, duration = _streams_and_duration(output)
    assert kinds == {"video", "audio"}
    assert duration == pytest.approx(2.0, abs=0.35)
