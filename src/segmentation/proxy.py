from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence

import av

from .segment import (
    DEFAULT_CRF,
    VIDEOTOOLBOX_ENCODER,
    _video_encoder_candidates,
    _videotoolbox_bitrate,
)

ProgressCallback = Callable[[int, int, str, int], None]


def _one_track(streams: Sequence, kind: str, name: str):
    """Return the only track of kind, or None when audio is absent."""
    matches = [stream for stream in streams if stream.type == kind]
    if kind == "video" and len(matches) != 1:
        raise ValueError(f"'{name}' has {len(matches)} video tracks; expected one.")
    if kind == "audio" and len(matches) > 1:
        raise ValueError(f"'{name}' has {len(matches)} audio tracks; expected at most one.")
    return matches[0] if matches else None


def probe_video_sources(source_paths: Sequence[Path]) -> list[dict]:
    """Return a continuous-timeline manifest for ordered camera chunks."""
    sources: list[dict] = []
    offset = 0.0
    for path in source_paths:
        with av.open(str(path)) as container:
            stream = _one_track(container.streams, "video", path.name)
            _one_track(container.streams, "audio", path.name)
            duration = 0.0
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration:
                duration = float(container.duration) / float(av.time_base)
            if duration <= 0:
                raise ValueError(f"Could not determine duration of {path.name}")
            rate = stream.average_rate or Fraction(30, 1)
            sources.append(
                {
                    "path": str(path.resolve()),
                    "name": path.name,
                    "start_s": offset,
                    "duration_s": duration,
                    "width": int(stream.codec_context.width),
                    "height": int(stream.codec_context.height),
                    "fps": float(rate),
                }
            )
            offset += duration
    return sources


def create_analysis_proxy(
    sources: Sequence[dict],
    output_path: Path,
    *,
    max_width: int = 1280,
    max_height: int = 720,
    fps: int = 30,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Create one 720p continuous proxy while preserving source-time offsets.

    The returned manifest uses the proxy's exact per-chunk durations, avoiding
    cumulative boundary drift when timestamps are mapped back to the originals.
    """
    if not sources:
        raise ValueError("No source videos provided")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for encoder in _video_encoder_candidates():
        output_path.unlink(missing_ok=True)
        try:
            return _create_proxy_with_encoder(
                sources,
                output_path,
                encoder=encoder,
                max_width=max_width,
                max_height=max_height,
                fps=fps,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            if encoder != VIDEOTOOLBOX_ENCODER or not isinstance(exc, (av.FFmpegError, ValueError)):
                raise
            logging.warning("VideoToolbox proxy creation failed (%s); retrying with libx264.", exc)
    raise RuntimeError("Could not create analysis proxy")


def _create_proxy_with_encoder(
    sources: Sequence[dict],
    output_path: Path,
    *,
    encoder: str,
    max_width: int,
    max_height: int,
    fps: int,
    progress_callback: ProgressCallback | None,
) -> list[dict]:
    first = sources[0]
    ratio = min(max_width / float(first["width"]), max_height / float(first["height"]), 1.0)
    width = max(2, int(round(float(first["width"]) * ratio)) // 2 * 2)
    height = max(2, int(round(float(first["height"]) * ratio)) // 2 * 2)
    rate = Fraction(fps, 1)
    video_tb = Fraction(1, fps)
    output = av.open(str(output_path), "w", options={"movflags": "+faststart"})
    out_v = output.add_stream(encoder, rate=rate)
    out_v.width = width
    out_v.height = height
    out_v.pix_fmt = "yuv420p"
    if encoder == VIDEOTOOLBOX_ENCODER:
        out_v.bit_rate = _videotoolbox_bitrate(width, height, rate, DEFAULT_CRF + 3)
    else:
        out_v.options = {"crf": str(DEFAULT_CRF + 3), "preset": "veryfast"}

    out_a = output.add_stream("aac", rate=48000)
    out_a.layout = "stereo"
    fifo = av.AudioFifo()
    audio_pts = 0
    video_index = 0
    manifest: list[dict] = []

    def drain_audio(flush: bool = False) -> None:
        nonlocal audio_pts
        frame_size = out_a.frame_size or 1024
        while fifo.samples >= frame_size or (flush and fifo.samples > 0):
            take = frame_size if fifo.samples >= frame_size else fifo.samples
            frame = fifo.read(take)
            if frame is None:
                break
            frame.pts = audio_pts
            frame.time_base = Fraction(1, out_a.rate)
            audio_pts += frame.samples
            for packet in out_a.encode(frame):
                output.mux(packet)

    def write_silence(samples: int) -> None:
        # One AudioFrame for a multi-hour gap can allocate hundreds of megabytes.
        chunk = out_a.frame_size or 1024
        remaining = int(samples)
        while remaining > 0:
            take = min(chunk, remaining)
            frame = av.AudioFrame(format=out_a.format.name, layout=out_a.layout, samples=take)
            frame.sample_rate = out_a.rate
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            fifo.write(frame)
            remaining -= take
            drain_audio()

    first_video_t = None
    chunk_audio_start = 0.0

    def write_timed_audio(source_time: float | None, converted: av.AudioFrame) -> None:
        """Place source audio on the proxy timeline, padding a late start with silence."""
        if source_time is not None and first_video_t is not None:
            written = (audio_pts + fifo.samples) / float(out_a.rate) - chunk_audio_start
            gap = (source_time - first_video_t) - written
            if gap > 0.02:
                write_silence(max(0, int(round(gap * out_a.rate))))
            elif gap < -0.02:
                return
        fifo.write(converted)
        drain_audio()

    try:
        for source_index, source in enumerate(sources):
            source_start_frame = video_index
            chunk_audio_start = source_start_frame / fps
            expected_duration = float(source["duration_s"])
            target_chunk_frames = max(1, int(round(expected_duration * fps)))
            encoded_chunk_frames = 0
            path = Path(str(source["path"]))
            if progress_callback:
                progress_callback(source_index, len(sources), path.name, 0)
            with av.open(str(path)) as container:
                in_v = _one_track(container.streams, "video", path.name)
                in_a = _one_track(container.streams, "audio", path.name)
                try:
                    in_v.thread_type = "AUTO"
                except Exception:
                    pass
                resampler = (
                    av.AudioResampler(format=out_a.format, layout=out_a.layout, rate=out_a.rate)
                    if in_a is not None
                    else None
                )
                first_video_t = None
                last_output_local_index = -1
                held_audio: list[tuple[float | None, av.AudioFrame]] = []
                streams = [item for item in (in_v, in_a) if item is not None]

                def take_audio(frame: av.AudioFrame) -> None:
                    source_time = None
                    if frame.pts is not None and frame.time_base is not None:
                        source_time = float(frame.pts * frame.time_base)
                    frame.pts = None
                    for converted in resampler.resample(frame):
                        converted.pts = None
                        if first_video_t is None:
                            held_audio.append((source_time, converted))
                        else:
                            write_timed_audio(source_time, converted)

                for frame in container.decode(*streams):
                    if isinstance(frame, av.VideoFrame):
                        if frame.pts is None:
                            continue
                        t = float(frame.pts * frame.time_base)
                        if first_video_t is None:
                            first_video_t = t
                            pending_audio = held_audio[:]
                            held_audio.clear()
                            for source_time, converted in pending_audio:
                                write_timed_audio(source_time, converted)
                        local_t = max(0.0, t - first_video_t)
                        local_index = int(local_t * fps + 1e-6)
                        if local_index <= last_output_local_index or local_index >= target_chunk_frames:
                            continue
                        last_output_local_index = local_index
                        frame = frame.reformat(width=width, height=height, format="yuv420p")
                        frame.pts = source_start_frame + local_index
                        frame.time_base = video_tb
                        frame.pict_type = av.video.frame.PictureType.NONE
                        encoded_chunk_frames += 1
                        for packet in out_v.encode(frame):
                            output.mux(packet)
                        if progress_callback and encoded_chunk_frames % (fps * 2) == 0:
                            percent = min(99, int((local_t / max(expected_duration, 0.001)) * 100))
                            progress_callback(source_index, len(sources), path.name, percent)
                    elif isinstance(frame, av.AudioFrame):
                        take_audio(frame)
                if resampler is not None:
                    for converted in resampler.resample(None):
                        converted.pts = None
                        fifo.write(converted)

            if encoded_chunk_frames <= 0:
                raise ValueError(f"No decodable video frames found in {path.name}")
            video_index = source_start_frame + target_chunk_frames
            chunk_duration = target_chunk_frames / fps
            target_audio = round(video_index * out_a.rate / fps)
            audio_diff = target_audio - (audio_pts + fifo.samples)
            if audio_diff > 0:
                write_silence(audio_diff)
            elif audio_diff < 0:
                fifo.read(min(-audio_diff, fifo.samples))
            drain_audio()
            manifest.append(
                {
                    **source,
                    "start_s": source_start_frame / fps,
                    "duration_s": chunk_duration,
                }
            )
            if progress_callback:
                progress_callback(source_index, len(sources), path.name, 100)

        for packet in out_v.encode():
            output.mux(packet)
        drain_audio(flush=True)
        for packet in out_a.encode():
            output.mux(packet)
    finally:
        output.close()
    return manifest
