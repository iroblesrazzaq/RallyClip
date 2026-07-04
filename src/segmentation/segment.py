import bisect
import csv
import os
from fractions import Fraction
from typing import List, Tuple

import av

# libx264 quality. Lower = higher quality/larger; ~20 is near visually lossless and keeps the
# output bitrate in the neighbourhood of a typical H.264 source (vs the old mp4v blow-up).
DEFAULT_CRF = 20


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


def segment_video(input_video: str, intervals: List[Tuple[float, float]], output_path: str,
                  eps: float = 1e-6, crf: int = DEFAULT_CRF) -> None:
    """Cut the kept intervals out of input_video and concatenate them into one output file.

    Frame-accurate, re-encoded with libx264 (CRF) for video and AAC for audio, so the output
    carries sound and stays near the source bitrate. Kept video/audio frames are re-stamped onto
    a single gap-free timeline (sequential per stream) so the concatenated clips play back-to-back
    in sync. Audio is optional — inputs without an audio stream produce a clean video-only file.
    """
    if not intervals:
        raise ValueError("No intervals provided")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Merge overlapping/touching intervals so per-segment processing keeps
    # each source frame at most once (matching the old single-pass filter).
    merged: List[Tuple[float, float]] = []
    for start_t, end_t in sorted(intervals, key=lambda x: (x[0], x[1])):
        if merged and start_t <= merged[-1][1] + eps:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_t))
        else:
            merged.append((start_t, end_t))
    intervals = merged

    in_container = av.open(input_video)
    try:
        in_v = next((s for s in in_container.streams if s.type == 'video'), None)
        if in_v is None:
            raise RuntimeError(f"No video stream found in {input_video}")
        in_a = next((s for s in in_container.streams if s.type == 'audio'), None)
        # Open the output only after validating the input, so a no-video input doesn't
        # leave a zero-byte file behind and a failed open doesn't leak in_container.
        out_container = av.open(output_path, 'w')
    except Exception:
        in_container.close()
        raise

    try:
        rate = in_v.average_rate or Fraction(30, 1)
        video_tb = Fraction(1, 1) / rate  # 1/fps; kept frames get sequential pts in this base
        out_v = out_container.add_stream('libx264', rate=rate)
        out_v.width = in_v.codec_context.width
        out_v.height = in_v.codec_context.height
        out_v.pix_fmt = 'yuv420p'
        out_v.options = {'crf': str(crf)}

        out_a = fifo = None
        if in_a is not None:
            out_a = out_container.add_stream('aac', rate=in_a.codec_context.sample_rate)
            out_a.layout = in_a.layout
            fifo = av.AudioFifo()

        audio_pts = 0  # cumulative output samples => gap-free audio timeline

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

        decode_streams = [s for s in (in_v, in_a) if s is not None]
        v_index = 0
        # Process one kept interval at a time (seek + decode forward) so each
        # segment's audio can be reconciled against exactly that segment's
        # video frames before the next segment starts.
        for start_t, end_t in intervals:
            in_container.seek(int(start_t / in_v.time_base), stream=in_v, backward=True)
            resampler = None
            if in_a is not None:
                # Fresh resampler per segment: flushing (resample(None)) ends a
                # resampler's stream, and audio from disjoint cuts must not be
                # rate-interpolated across the gap.
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
                    frame.pts = v_index
                    frame.time_base = video_tb
                    frame.pict_type = av.video.frame.PictureType.NONE
                    v_index += 1
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
    finally:
        out_container.close()
        in_container.close()
