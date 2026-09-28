import bisect
import csv
import logging
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import av

# libx264 quality. Lower = higher quality/larger; ~20 is near visually lossless and keeps the
# output bitrate in the neighbourhood of a typical H.264 source (vs the old mp4v blow-up).
DEFAULT_CRF = 20
VIDEOTOOLBOX_ENCODER = "h264_videotoolbox"
SOFTWARE_ENCODER = "libx264"

# VideoToolbox has no libx264-style CRF mode. Target a visually strong bitrate
# from the source pixel rate, then scale it with the caller's CRF preference.
# At the default CRF this is ~7.5 Mbps for 1080p60 and ~30 Mbps for 4K60.
VIDEOTOOLBOX_BITS_PER_PIXEL_FRAME = 0.06
MIN_VIDEOTOOLBOX_BITRATE = 1_000_000
MAX_VIDEOTOOLBOX_BITRATE = 80_000_000


def load_intervals(csv_path: str) -> List[Tuple[float, float]]:
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [c.strip().lower() for c in reader.fieldnames]
        if 'start_time' not in reader.fieldnames or 'end_time' not in reader.fieldnames:
            raise ValueError("CSV must contain 'start_time' and 'end_time' columns")
        intervals = []
        for row in reader:
            try:
                start = float(row['start_time'])
                end = float(row['end_time'])
                intervals.append((start, end))
            except (ValueError, KeyError, TypeError):
                continue
        return sorted(intervals, key=lambda x: (x[0], x[1]))


def _interval_index(t: float, intervals: List[Tuple[float, float]], starts: List[float], eps: float) -> int:
    """Index of the kept interval containing time t (seconds), or -1."""
    k = bisect.bisect_right(starts, t + eps) - 1
    if k < 0:
        return -1
    start_t, end_t = intervals[k]
    return k if (t + eps) >= start_t and (t - eps) <= end_t else -1


def _in_interval(t: float, intervals: List[Tuple[float, float]], starts: List[float], eps: float) -> bool:
    """True if input time t (seconds) falls within any kept interval."""
    return _interval_index(t, intervals, starts, eps) >= 0


def _video_encoder_candidates() -> tuple[str, ...]:
    """Return the preferred H.264 encoders for this host, in retry order."""
    if sys.platform == "darwin" and VIDEOTOOLBOX_ENCODER in av.codecs_available:
        return (VIDEOTOOLBOX_ENCODER, SOFTWARE_ENCODER)
    return (SOFTWARE_ENCODER,)


def _videotoolbox_bitrate(width: int, height: int, rate: Fraction, crf: int) -> int:
    """Map the software CRF preference to a bounded VideoToolbox bitrate."""
    fps = max(1.0, float(rate))
    crf_scale = 2.0 ** ((DEFAULT_CRF - int(crf)) / 6.0)
    target = int(round(width * height * fps * VIDEOTOOLBOX_BITS_PER_PIXEL_FRAME * crf_scale))
    return max(MIN_VIDEOTOOLBOX_BITRATE, min(MAX_VIDEOTOOLBOX_BITRATE, target))


def segment_video(
    input_video: str,
    intervals: List[Tuple[float, float]],
    output_path: str,
    eps: float = 1e-6,
    crf: int = DEFAULT_CRF,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> None:
    """Cut kept intervals and concatenate them into one frame-accurate MP4.

    Apple hosts try the dedicated VideoToolbox H.264 encoder first. If the
    hardware session cannot be opened or fails while encoding, the incomplete
    file is discarded and the exact same cut is retried with libx264.
    """
    merged = _merge_intervals(intervals, eps)

    output = Path(output_path)
    try:
        _stream_copy_video(input_video, merged, output_path)
        if progress_callback is not None:
            progress_callback(100)
        return
    except (RuntimeError, ValueError, av.FFmpegError) as exc:
        output.unlink(missing_ok=True)
        logging.info("Stream-copy export unavailable; using frame-accurate encode: %s", exc)
    encoders = _video_encoder_candidates()
    for encoder in encoders:
        try:
            encode_kwargs = {}
            if progress_callback is not None:
                encode_kwargs["progress_callback"] = progress_callback
            _segment_video_with_encoder(
                input_video,
                merged,
                output_path,
                eps=eps,
                crf=crf,
                video_encoder=encoder,
                **encode_kwargs,
            )
            return
        except Exception as exc:
            output.unlink(missing_ok=True)
            can_fallback = (
                encoder == VIDEOTOOLBOX_ENCODER
                and SOFTWARE_ENCODER in encoders
                and isinstance(exc, (av.FFmpegError, ValueError))
            )
            if not can_fallback:
                raise
            logging.warning(
                "VideoToolbox export failed (%s); retrying with libx264.",
                exc,
            )


def _keyframe_padded_intervals(
    intervals: Sequence[Tuple[float, float]],
    keyframe_pad_s: float,
) -> List[Tuple[float, float, float, float]]:
    """Return (cut start, cut end, search start, search end) for each interval.

    The search window is only where we look for keyframes. Copied packets must
    still stay inside the requested cut; overlapping search windows are refused
    because they would remux the same packets twice.
    """
    windows = []
    for start, end in intervals:
        start_f = float(start)
        end_f = float(end)
        windows.append(
            (start_f, end_f, max(0.0, start_f - keyframe_pad_s), end_f + keyframe_pad_s)
        )
    windows.sort()
    for previous, current in zip(windows, windows[1:]):
        if current[2] <= previous[3]:
            raise RuntimeError("Keyframe-padded intervals overlap; refusing stream copy")
    return windows


def _only_track(container, kind: str, path: str):
    """Return the only track of this kind. Audio may be absent; video may not."""
    name = Path(path).name
    matches = [stream for stream in container.streams if stream.type == kind]
    if kind == "video" and len(matches) != 1:
        raise RuntimeError(f"'{name}' has {len(matches)} video tracks; expected one.")
    if kind == "audio" and len(matches) > 1:
        raise RuntimeError(f"'{name}' has {len(matches)} audio tracks; expected at most one.")
    return matches[0] if matches else None


def _stream_copy_video(
    input_video: str,
    intervals: Sequence[Tuple[float, float]],
    output_path: str,
    *,
    keyframe_pad_s: float = 1.0,
) -> None:
    """Remux keyframe-bounded intervals without decoding or re-encoding."""
    padded = _keyframe_padded_intervals(intervals, keyframe_pad_s)
    with av.open(input_video) as source:
        video = _only_track(source, "video", input_video)
        if any(stream.type == "audio" for stream in source.streams):
            raise RuntimeError("Stream-copy export requires video-only input for timestamp safety")
        streams = [stream for stream in source.streams if stream.type in {"video", "audio"}]
        with av.open(output_path, "w") as output:
            output_streams = {}
            for stream in streams:
                codec_name = stream.codec_context.name
                if stream.type == "video":
                    out_stream = output.add_stream(codec_name, rate=stream.average_rate)
                    out_stream.width = stream.codec_context.width
                    out_stream.height = stream.codec_context.height
                    out_stream.pix_fmt = stream.codec_context.format.name
                else:
                    out_stream = output.add_stream(codec_name, rate=stream.codec_context.sample_rate)
                    out_stream.layout = stream.layout
                out_stream.codec_context.extradata = stream.codec_context.extradata
                output_streams[stream.index] = out_stream
            output_time = 0.0
            copied_until = None
            for requested_start, requested_end, _start, _end in padded:
                source.seek(int(requested_start / video.time_base), stream=video, backward=True)
                packets = []
                start_time = None
                end_keyframe_seen = False
                for packet in source.demux(streams):
                    shown = packet.pts if packet.pts is not None else packet.dts
                    if shown is None:
                        continue
                    packet_time = float(shown * packet.time_base)
                    if packet.stream.type == "video" and packet.is_keyframe:
                        if start_time is None:
                            start_time = packet_time
                            if start_time < requested_start - 1e-3:
                                raise RuntimeError(
                                    "Stream copy would include footage outside the selected interval"
                                )
                        elif packet_time >= requested_end - 1e-3:
                            end_keyframe_seen = True
                    if start_time is None:
                        continue
                    # The keyframe at the cut end closes the GOP and is not copied.
                    if end_keyframe_seen:
                        break
                    if packet_time < requested_start - 1e-3 or packet_time >= requested_end - 1e-3:
                        continue
                    packets.append(packet)
                if start_time is None or not end_keyframe_seen or not packets:
                    raise RuntimeError("Could not find keyframe-bounded export interval")
                segment_end = max(
                    float(
                        (
                            (packet.pts if packet.pts is not None else packet.dts)
                            + (packet.duration or 0)
                        )
                        * packet.time_base
                    )
                    for packet in packets
                )
                if copied_until is not None and start_time <= copied_until:
                    raise RuntimeError("Keyframe-bounded intervals overlap; refusing stream copy")
                if segment_end > requested_end + 0.05:
                    raise RuntimeError("Stream copy would include footage outside the selected interval")
                copied_until = segment_end
                segment_duration = max(0.0, segment_end - start_time)
                for packet in packets:
                    out_stream = output_streams[packet.stream.index]
                    source_time_base = packet.time_base
                    target_time_base = out_stream.time_base or source_time_base
                    source_pts = packet.pts
                    source_dts = packet.dts
                    packet.stream = out_stream
                    if source_pts is not None:
                        packet.pts = int(round((float(source_pts * source_time_base) - start_time + output_time) / target_time_base))
                    if source_dts is not None:
                        packet.dts = int(round((float(source_dts * source_time_base) - start_time + output_time) / target_time_base))
                    packet.time_base = target_time_base
                    output.mux(packet)
                output_time += segment_duration
            output.close()


def _merge_intervals(
    intervals: Sequence[Tuple[float, float]], eps: float = 1e-6
) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for start_t, end_t in sorted(intervals, key=lambda x: (x[0], x[1])):
        start_t = float(start_t)
        end_t = float(end_t)
        if end_t <= start_t:
            continue
        if merged and start_t <= merged[-1][1] + eps:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_t))
        else:
            merged.append((start_t, end_t))
    if not merged:
        raise ValueError("No intervals provided")
    return merged


def timeline_intervals_by_source(
    sources: Sequence[dict], intervals: Sequence[Tuple[float, float]], eps: float = 1e-6
) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Map continuous-match intervals onto local timestamps in source chunks."""
    mapped: List[Tuple[str, List[Tuple[float, float]]]] = []
    normalized = _merge_intervals(intervals, eps)
    for source in sources:
        path = str(source["path"])
        offset = float(source["start_s"])
        duration = float(source["duration_s"])
        source_end = offset + duration
        local: List[Tuple[float, float]] = []
        for start_t, end_t in normalized:
            start = max(start_t, offset)
            end = min(end_t, source_end)
            if end > start + eps:
                local.append((max(0.0, start - offset), min(duration, end - offset)))
        if local:
            mapped.append((path, local))
    if not mapped:
        raise ValueError("No point intervals overlap the source files")
    return mapped


def segment_video_sources(
    sources: Sequence[dict],
    intervals: Sequence[Tuple[float, float]],
    output_path: str,
    eps: float = 1e-6,
    crf: int = DEFAULT_CRF,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> None:
    """Cut a continuous timeline backed by consecutive camera files."""
    selections = timeline_intervals_by_source(sources, intervals, eps)
    output = Path(output_path)
    if len(selections) == 1:
        input_video, local_intervals = selections[0]
        try:
            _stream_copy_video(input_video, local_intervals, output_path)
            if progress_callback is not None:
                progress_callback(100)
            return
        except (RuntimeError, ValueError, av.FFmpegError) as exc:
            output.unlink(missing_ok=True)
            logging.info("Stream-copy export unavailable; using frame-accurate encode: %s", exc)
    encoders = _video_encoder_candidates()
    for encoder in encoders:
        try:
            encode_kwargs = {}
            if progress_callback is not None:
                encode_kwargs["progress_callback"] = progress_callback
            _segment_video_sources_with_encoder(
                selections,
                output_path,
                eps=eps,
                crf=crf,
                video_encoder=encoder,
                **encode_kwargs,
            )
            return
        except Exception as exc:
            output.unlink(missing_ok=True)
            can_fallback = (
                encoder == VIDEOTOOLBOX_ENCODER
                and SOFTWARE_ENCODER in encoders
                and isinstance(exc, (av.FFmpegError, ValueError))
            )
            if not can_fallback:
                raise
            logging.warning("VideoToolbox multi-source export failed (%s); retrying with libx264.", exc)


def _segment_video_with_encoder(
    input_video: str,
    intervals: List[Tuple[float, float]],
    output_path: str,
    *,
    eps: float,
    crf: int,
    video_encoder: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> None:
    """Encode one already-normalized interval set with one selected codec.

    Kept video/audio frames are re-stamped onto a single gap-free timeline so
    clips play back-to-back in sync. Audio is optional; inputs without an audio
    stream produce a clean video-only file.
    """
    _segment_video_sources_with_encoder(
        [(input_video, intervals)],
        output_path,
        eps=eps,
        crf=crf,
        video_encoder=video_encoder,
        progress_callback=progress_callback,
    )


def _segment_video_sources_with_encoder(
    selections: Sequence[Tuple[str, List[Tuple[float, float]]]],
    output_path: str,
    *,
    eps: float,
    crf: int,
    video_encoder: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> None:
    """Encode local intervals from ordered sources into one gap-free output."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    first_path = selections[0][0]
    in_container = av.open(first_path)
    try:
        in_v = _only_track(in_container, "video", first_path)
        # Keep software decode parallel while VideoToolbox handles encode. This
        # also benefits the libx264 fallback on high-frame-rate inputs.
        try:
            in_v.thread_type = "AUTO"
        except Exception:
            pass
        in_a = _only_track(in_container, "audio", first_path)
        # Open the output only after validating the input, so a no-video input doesn't
        # leave a zero-byte file behind and a failed open doesn't leak in_container.
        out_container = av.open(output_path, 'w')
    except Exception:
        in_container.close()
        raise

    try:
        rate = in_v.average_rate or Fraction(30, 1)
        video_tb = Fraction(1, 1) / rate  # 1/fps; kept frames get sequential pts in this base
        out_v = out_container.add_stream(video_encoder, rate=rate)
        out_v.width = in_v.codec_context.width
        out_v.height = in_v.codec_context.height
        out_v.pix_fmt = 'yuv420p'
        if video_encoder == VIDEOTOOLBOX_ENCODER:
            out_v.bit_rate = _videotoolbox_bitrate(out_v.width, out_v.height, rate, crf)
            logging.info(
                "Exporting with VideoToolbox H.264 at %.1f Mbps.",
                out_v.bit_rate / 1_000_000,
            )
        else:
            out_v.options = {'crf': str(crf)}

        out_a = fifo = None
        if in_a is not None:
            out_a = out_container.add_stream('aac', rate=in_a.codec_context.sample_rate)
            out_a.layout = in_a.layout
            fifo = av.AudioFifo()

        audio_pts = 0  # cumulative output samples => gap-free audio timeline
        total_frames = sum(
            max(1, int(round((end_t - start_t) * float(rate))))
            for _, intervals in selections
            for start_t, end_t in intervals
        )
        encoded_frames = 0
        if progress_callback is not None:
            progress_callback(0)

        def drain_audio(flush: bool = False) -> None:
            nonlocal audio_pts
            frame_size = out_a.frame_size or 1024
            while fifo.samples >= frame_size or (flush and fifo.samples > 0):
                take = frame_size if fifo.samples >= frame_size else fifo.samples
                a_frame = fifo.read(take)
                if a_frame is None:
                    break
                a_frame.pts = audio_pts
                a_frame.time_base = Fraction(1, out_a.rate)
                audio_pts += a_frame.samples
                for pkt in out_a.encode(a_frame):
                    out_container.mux(pkt)

        def _write_silence(n_samples: int) -> None:
            silence = av.AudioFrame(format=out_a.format.name, layout=out_a.layout, samples=n_samples)
            silence.sample_rate = out_a.rate
            for plane in silence.planes:
                plane.update(bytes(plane.buffer_size))
            silence.pts = None
            fifo.write(silence)

        def reconcile_audio(video_frames_total: int) -> None:
            """Lock the audio timeline to the video timeline (issue #21).

            Video and audio are quantised on different grids (1/fps frames vs
            ~1024-sample AAC frames), so per-segment length mismatches would
            otherwise accumulate into lip-sync drift across concatenated
            segments. After each segment, pad (silence) or trim the pending
            FIFO samples so total audio == emitted video frames / fps. The
            FIFO is sample-granular, so the encoder still receives clean
            fixed-size frames and the residual error is bounded (<~25 ms at a
            cut point), not additive.
            """
            target_total = round(video_frames_total * out_a.rate / float(rate))
            diff = target_total - (audio_pts + fifo.samples)
            if diff > 0:
                _write_silence(diff)
            elif diff < 0:
                fifo.read(min(-diff, fifo.samples))
            drain_audio()

        v_index = 0
        for source_index, (input_video, intervals) in enumerate(selections):
            if source_index:
                in_container.close()
                in_container = av.open(input_video)
                in_v = _only_track(in_container, "video", input_video)
                try:
                    in_v.thread_type = "AUTO"
                except Exception:
                    pass
                if (
                    in_v.codec_context.width != out_v.width
                    or in_v.codec_context.height != out_v.height
                ):
                    raise ValueError("All match chunks must use the same resolution")
                in_a = _only_track(in_container, "audio", input_video)
                if (out_a is None) != (in_a is None):
                    raise ValueError("All match chunks must either include audio or omit it")

            decode_streams = [s for s in (in_v, in_a) if s is not None]
            for start_t, end_t in intervals:
                in_container.seek(int(start_t / in_v.time_base), stream=in_v, backward=True)
                interval_base_index = v_index
                target_interval_frames = max(1, int(round((end_t - start_t) * float(rate))))
                last_local_video_index = -1
                resampler = None
                if in_a is not None:
                    resampler = av.AudioResampler(format=out_a.format, layout=out_a.layout, rate=out_a.rate)
                v_done = False
                a_done = in_a is None
                for frame in in_container.decode(*decode_streams):
                    if v_done and a_done:
                        break
                    if frame.pts is None:
                        continue
                    t = float(frame.pts * frame.time_base)
                    if isinstance(frame, av.VideoFrame):
                        if v_done or t < start_t - eps:
                            continue
                        if t > end_t + eps:
                            v_done = True
                            continue
                        local_video_index = int(round((t - start_t) * float(rate)))
                        if (
                            local_video_index <= last_local_video_index
                            or local_video_index >= target_interval_frames
                        ):
                            continue
                        last_local_video_index = local_video_index
                        frame.pts = interval_base_index + local_video_index
                        frame.time_base = video_tb
                        frame.pict_type = av.video.frame.PictureType.NONE
                        encoded_frames += 1
                        if progress_callback is not None and (
                            encoded_frames == 1
                            or encoded_frames % max(1, total_frames // 100) == 0
                        ):
                            progress_callback(min(99, round(encoded_frames * 100 / total_frames)))
                        for pkt in out_v.encode(frame):
                            out_container.mux(pkt)
                    elif out_a is not None and isinstance(frame, av.AudioFrame):
                        if a_done or t < start_t - eps:
                            continue
                        if t > end_t + eps:
                            a_done = True
                            continue
                        frame.pts = None
                        for r_frame in resampler.resample(frame):
                            r_frame.pts = None
                            fifo.write(r_frame)
                v_index = interval_base_index + target_interval_frames
                if out_a is not None:
                    for r_frame in resampler.resample(None):
                        r_frame.pts = None
                        fifo.write(r_frame)
                    reconcile_audio(v_index)

        # Flush the encoders (audio already reconciled per segment).
        for pkt in out_v.encode():
            out_container.mux(pkt)
        if out_a is not None:
            drain_audio(flush=True)
            for pkt in out_a.encode():
                out_container.mux(pkt)
        if progress_callback is not None:
            progress_callback(100)
    finally:
        out_container.close()
        in_container.close()
