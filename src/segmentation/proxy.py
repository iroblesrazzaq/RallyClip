"""One continuous analysis proxy for a folder of consecutive camera clips."""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional, Sequence

import av

from .segment import DEFAULT_CRF, MAX_AUDIO_AHEAD_S, AudioTrack, add_video_stream, with_encoder_fallback

PROXY_AUDIO_RATE = 48000


def create_analysis_proxy(
    paths: Sequence[Path],
    output_path: Path,
    *,
    fps: float,
    max_height: int = 720,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> list[dict]:
    """Concatenate clips into one constant-frame-rate proxy and return each clip's timeline span.

    Spans come from the frames actually decoded, not container metadata, so a
    truncated clip shortens the timeline instead of leaving a padded gap. The proxy
    always has a stereo AAC track; clips without audio contribute silence.
    progress_callback receives 0-100 across all clips.
    """
    if not paths:
        raise ValueError("No source videos provided")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return with_encoder_fallback(
        lambda encoder: _write_proxy(paths, output_path, encoder, fps, max_height, progress_callback)
    )


def _duration_s(container, stream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    return float(container.duration or 0) / av.time_base


def _write_proxy(paths, output_path, encoder, fps, max_height, progress_callback) -> list[dict]:
    fps = float(fps)
    with av.open(str(paths[0])) as first:
        cc = first.streams.video[0].codec_context
        scale = min(1.0, max_height / float(cc.height))
        width = max(2, int(cc.width * scale) // 2 * 2)
        height = max(2, int(cc.height * scale) // 2 * 2)
    video_tb = Fraction(1, 1) / Fraction(fps).limit_denominator(1001)
    tolerance = PROXY_AUDIO_RATE // 50  # 20 ms
    manifest: list[dict] = []
    frame_index = 0
    reported = -1
    out = av.open(str(output_path), "w")
    try:
        out_v = add_video_stream(out, encoder, 1 / video_tb, width, height, DEFAULT_CRF + 3)
        track = AudioTrack(out, PROXY_AUDIO_RATE, "stereo")

        def encode(frame, index: int) -> None:
            frame.pts = index
            frame.time_base = video_tb
            frame.pict_type = av.video.frame.PictureType.NONE
            for packet in out_v.encode(frame):
                out.mux(packet)

        for clip_index, path in enumerate(paths):
            path = Path(path)
            start_frame = frame_index
            chunk_audio_start = round(start_frame * PROXY_AUDIO_RATE / fps)
            last = -1  # last clip-local frame slot written
            decoded = False
            end = 0  # clip length in proxy frames, from decoded frame end times
            previous = None
            with av.open(str(path)) as container:
                in_v = container.streams.video[0]
                in_v.thread_type = "AUTO"
                in_a = next((s for s in container.streams if s.type == "audio"), None)
                streams = [s for s in (in_v, in_a) if s is not None]
                origin = float(in_v.start_time * in_v.time_base) if in_v.start_time is not None else 0.0
                source_rate = in_v.average_rate or in_v.base_rate  # as the folder preflight reads it
                frame_s = 1.0 / float(source_rate) if source_rate else 1.0 / fps
                expected_s = _duration_s(container, in_v)
                resampler = track.resampler() if in_a is not None else None
                # Audio is encoded only up to the decoded video span: a truncated clip's
                # audio past its last picture would otherwise play over the next clip,
                # and the track can trim only its unencoded reserve.
                pending: deque = deque()

                def release(until_s: float) -> None:
                    while pending and pending[0][0] < until_s:
                        at, audio = pending.popleft()
                        gap = chunk_audio_start + round(at * PROXY_AUDIO_RATE) - track.written
                        if gap < -tolerance:
                            continue  # overlaps audio already written
                        if gap > tolerance:
                            track.pad_to(track.written + gap)
                        audio.pts = None
                        track.write(resampler.resample(audio))

                for frame in container.decode(*streams):
                    if frame.pts is None:
                        continue
                    t = float(frame.pts * frame.time_base) - origin
                    if isinstance(frame, av.AudioFrame):
                        pending.append((t, frame))
                        while t - pending[0][0] > MAX_AUDIO_AHEAD_S:
                            pending.popleft()
                        release(end / fps)
                        continue
                    decoded = True
                    index = int(t * fps + 0.5)
                    end = max(end, int((t + frame_s) * fps + 0.5))
                    release(end / fps)
                    if index <= last or index >= end:
                        continue  # source is faster than the proxy rate
                    frame = frame.reformat(width=width, height=height, format="yuv420p")
                    # Hold the previous picture across timestamp gaps so the proxy stays
                    # constant-frame-rate: analysis maps proxy frame N to N / fps.
                    for slot in range(last + 1, index + 1):
                        encode(frame if slot == index or previous is None else previous, start_frame + slot)
                    previous, last = frame, index
                    if progress_callback is not None and expected_s > 0:
                        percent = int(100 * (clip_index + min(1.0, t / expected_s)) / len(paths))
                        if percent != reported:
                            reported = percent
                            progress_callback(min(99, percent))
                release(end / fps)
                if resampler is not None:
                    track.write(resampler.resample(None))
            if not decoded:
                raise ValueError(f"No decodable video frames found in {path.name}")
            if previous is None:
                end = 0  # shorter than half a proxy frame: the clip adds no time
            # Show the final picture for its full duration, so the next clip starts
            # where it really begins instead of drifting a frame per boundary.
            for slot in range(last + 1, end):
                encode(previous, start_frame + slot)
            last = max(last, end - 1)
            frame_index = start_frame + last + 1
            track.pad_to(round(frame_index * PROXY_AUDIO_RATE / fps))
            manifest.append(
                {
                    "path": str(path.resolve()),
                    "name": path.name,
                    "start_s": start_frame / fps,
                    "duration_s": (last + 1) / fps,
                }
            )
        for packet in out_v.encode():
            out.mux(packet)
        track.close()
    finally:
        out.close()
    if progress_callback is not None:
        progress_callback(100)
    return manifest
