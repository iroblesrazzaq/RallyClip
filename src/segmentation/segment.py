import bisect
import csv
import logging
import math
import os
import sys
from fractions import Fraction
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

import av

# libx264 quality. Lower = higher quality/larger; ~20 is near visually lossless and keeps the
# output bitrate in the neighbourhood of a typical H.264 source (vs the old mp4v blow-up).
DEFAULT_CRF = 20
VIDEOTOOLBOX = "h264_videotoolbox"
ProgressCallback = Callable[[int], None]
T = TypeVar("T")


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


def video_encoders() -> List[str]:
    """H.264 encoders to try in order: VideoToolbox on macOS (far faster for 4K), then libx264."""
    if sys.platform == "darwin" and VIDEOTOOLBOX in av.codecs_available:
        return [VIDEOTOOLBOX, "libx264"]
    return ["libx264"]


def add_video_stream(container, encoder: str, rate: Fraction, width: int, height: int, crf: int):
    stream = container.add_stream(encoder, rate=rate)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    if encoder == VIDEOTOOLBOX:
        # VideoToolbox has no CRF. ~0.1 bit/pixel/frame is close to libx264 CRF 20
        # (~6 Mb/s at 1080p30); every +6 CRF halves it.
        stream.bit_rate = int(width * height * float(rate) * 0.1 * 2 ** ((DEFAULT_CRF - crf) / 6))
    else:
        stream.options = {"crf": str(crf)}
    return stream


def with_encoder_fallback(write: Callable[[str], T]) -> T:
    """Run write(encoder) with the fastest encoder, falling back to libx264 on an FFmpeg error."""
    encoders = video_encoders()
    for encoder in encoders[:-1]:
        try:
            return write(encoder)
        except av.FFmpegError as exc:
            logging.warning("%s encode failed (%s); retrying with %s", encoder, exc, encoders[-1])
    return write(encoders[-1])


class AudioTrack:
    """AAC output stream whose length callers pin to the video timeline.

    Video and audio are quantised on different grids (1/fps frames vs ~1024-sample
    AAC frames). Padding or trimming to the video length after each cut (issue #21)
    keeps the error bounded (<~25 ms) instead of drifting across concatenated cuts.
    """

    def __init__(self, container, rate: int, layout) -> None:
        self.container = container
        self.stream = container.add_stream("aac", rate=rate)
        self.stream.layout = layout
        self.rate = int(rate)
        self.fifo = av.AudioFifo()
        self.encoded = 0

    @property
    def written(self) -> int:
        return self.encoded + self.fifo.samples

    def resampler(self):
        return av.AudioResampler(format=self.stream.format, layout=self.stream.layout, rate=self.rate)

    def write(self, frames) -> None:
        for frame in frames:
            frame.pts = None
            self.fifo.write(frame)
        self._drain()

    def pad_to(self, total: int) -> None:
        """Pad with silence, or trim pending samples, so the track holds `total` samples."""
        missing = total - self.written
        while missing > 0:
            # One-second frames: a long silent clip must not allocate one huge frame.
            n = min(missing, self.rate)
            silence = av.AudioFrame(format=self.stream.format.name, layout=self.stream.layout, samples=n)
            silence.sample_rate = self.rate
            for plane in silence.planes:
                plane.update(bytes(plane.buffer_size))
            self.fifo.write(silence)
            missing -= n
            self._drain()
        if missing < 0:
            self.fifo.read(min(-missing, self.fifo.samples))
        self._drain()

    def close(self) -> None:
        self._drain(flush=True)
        for packet in self.stream.encode():
            self.container.mux(packet)

    def _drain(self, flush: bool = False) -> None:
        size = self.stream.frame_size or 1024
        while self.fifo.samples >= size or (flush and self.fifo.samples > 0):
            frame = self.fifo.read(min(size, self.fifo.samples))
            frame.pts = self.encoded
            frame.time_base = Fraction(1, self.rate)
            self.encoded += frame.samples
            for packet in self.stream.encode(frame):
                self.container.mux(packet)


def _merge_intervals(intervals: Sequence[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    """Merge overlapping/touching intervals so each source frame is kept at most once."""
    merged: List[Tuple[float, float]] = []
    for start_t, end_t in sorted(intervals):
        if merged and start_t <= merged[-1][1] + eps:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_t))
        else:
            merged.append((start_t, end_t))
    return merged


def timeline_pieces(sources: Sequence[dict], intervals: Sequence[Tuple[float, float]],
                    eps: float = 1e-6) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Split timeline intervals into (source path, source-local intervals), in source order.

    Each source dict carries `path`, `start_s` and `duration_s` on the shared timeline;
    an interval that crosses a clip boundary is cut into one piece per clip.
    """
    pieces = []
    for source in sources:
        start = float(source["start_s"])
        end = start + float(source["duration_s"])
        local = [
            (max(s, start) - start, min(e, end) - start)
            for s, e in intervals
            if min(e, end) - max(s, start) > eps
        ]
        if local:
            pieces.append((str(source["path"]), local))
    return pieces


def segment_video(input_video: str, intervals: List[Tuple[float, float]], output_path: str,
                  eps: float = 1e-6, crf: int = DEFAULT_CRF,
                  progress_callback: Optional[ProgressCallback] = None) -> None:
    """Cut the kept intervals out of input_video and concatenate them into one output file."""
    source = {"path": input_video, "start_s": 0.0, "duration_s": math.inf}
    segment_video_sources([source], intervals, output_path, eps=eps, crf=crf,
                          progress_callback=progress_callback)


def segment_video_sources(sources: Sequence[dict], intervals: List[Tuple[float, float]],
                          output_path: str, eps: float = 1e-6, crf: int = DEFAULT_CRF,
                          progress_callback: Optional[ProgressCallback] = None) -> None:
    """Cut timeline intervals out of consecutive source clips into one output file.

    Frame-accurate, re-encoded as H.264 (VideoToolbox on macOS, else libx264 CRF) plus
    AAC. Kept frames are re-stamped onto one gap-free timeline so the cuts play
    back-to-back in sync. A clip without audio contributes silence when another clip
    has audio; all-silent sources produce a video-only file. progress_callback, when
    given, receives 0-99 as kept footage is encoded.
    """
    if not intervals:
        raise ValueError("No intervals provided")
    pieces = timeline_pieces(sources, _merge_intervals(intervals, eps), eps)
    if not pieces:
        raise ValueError("No intervals fall inside the source video")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    # Probe every input before opening the output, so a bad input doesn't leave a file behind.
    video = audio = None
    for path, _ in pieces:
        with av.open(path) as container:
            in_v = next((s for s in container.streams if s.type == 'video'), None)
            if in_v is None:
                raise RuntimeError(f"No video stream found in {path}")
            in_a = next((s for s in container.streams if s.type == 'audio'), None)
            if video is None:
                video = (in_v.average_rate or Fraction(30, 1), in_v.codec_context.width, in_v.codec_context.height)
            if audio is None and in_a is not None:
                audio = (in_a.codec_context.sample_rate, in_a.layout.name)
    with_encoder_fallback(
        lambda encoder: _write_cuts(pieces, output_path, encoder, crf, eps, video, audio, progress_callback)
    )


def _write_cuts(pieces, output_path, encoder, crf, eps, video, audio, progress_callback) -> None:
    rate, width, height = video
    video_tb = Fraction(1, 1) / rate
    total_s = sum(e - s for _, local in pieces for s, e in local) or 1.0
    done_s = 0.0
    reported = -1
    out = av.open(output_path, 'w')
    try:
        out_v = add_video_stream(out, encoder, rate, width, height, crf)
        track = AudioTrack(out, *audio) if audio else None
        v_index = 0  # kept frames so far => sequential video pts
        for path, local in pieces:
            with av.open(path) as container:
                in_v = next(s for s in container.streams if s.type == 'video')
                in_a = next((s for s in container.streams if s.type == 'audio'), None) if track else None
                streams = [s for s in (in_v, in_a) if s is not None]
                # Times are measured from the first video frame, as analysis counts
                # frames from it (matters for MPEG-TS clips that don't start at 0).
                origin = float(in_v.start_time * in_v.time_base) if in_v.start_time is not None else 0.0
                # One cut at a time (seek + decode forward) so each cut's audio is
                # reconciled against exactly that cut's video before the next starts.
                for start_t, end_t in local:
                    container.seek(int((start_t + origin) / in_v.time_base), stream=in_v, backward=True)
                    # Fresh resampler per cut: audio from disjoint cuts must not be
                    # rate-interpolated across the gap.
                    resampler = track.resampler() if in_a is not None else None
                    v_done, a_done = False, in_a is None
                    for frame in container.decode(*streams):
                        if v_done and a_done:
                            break
                        if frame.pts is None:
                            continue
                        is_video = isinstance(frame, av.VideoFrame)
                        t = float(frame.pts * frame.time_base) - origin
                        if (v_done if is_video else a_done) or t < start_t - eps:
                            continue
                        if t > end_t + eps:
                            if is_video:
                                v_done = True
                            else:
                                a_done = True
                            continue
                        if not is_video:
                            frame.pts = None
                            track.write(resampler.resample(frame))
                            continue
                        if (frame.width, frame.height) != (width, height):
                            frame = frame.reformat(width=width, height=height)
                        frame.pts = v_index
                        frame.time_base = video_tb
                        frame.pict_type = av.video.frame.PictureType.NONE
                        v_index += 1
                        for packet in out_v.encode(frame):
                            out.mux(packet)
                        if progress_callback is not None:
                            percent = min(99, int(100 * (done_s + max(0.0, t - start_t)) / total_s))
                            if percent != reported:
                                reported = percent
                                progress_callback(percent)
                    if resampler is not None:
                        track.write(resampler.resample(None))
                    if track is not None:
                        track.pad_to(round(v_index * track.rate / float(rate)))
                    done_s += end_t - start_t
        for packet in out_v.encode():
            out.mux(packet)
        if track is not None:
            track.close()
    finally:
        out.close()
