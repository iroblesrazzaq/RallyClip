from __future__ import annotations

import logging
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence

import av

from .segment import (
    DEFAULT_CRF,
    SOFTWARE_ENCODER,
    VIDEOTOOLBOX_ENCODER,
    _video_encoder_candidates,
    _videotoolbox_bitrate,
)

ProgressCallback = Callable[[int, int, str, int], None]
AVCONVERT_PATH = Path("/usr/bin/avconvert")
AVCONVERT_PRESET = "PresetAppleM4V720pHD"


def probe_video_sources(source_paths: Sequence[Path]) -> list[dict]:
    """Return a continuous-timeline manifest for ordered camera chunks."""
    sources: list[dict] = []
    offset = 0.0
    for path in source_paths:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise ValueError(f"No video stream found in {path.name}")
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
    if _can_use_avconvert(sources):
        try:
            return _create_proxy_with_avconvert(
                sources,
                output_path,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            if exc.__class__.__name__ in {"PipelineCancelled", "PoseExtractionCancelled"}:
                raise
            logging.warning("Native Apple proxy creation failed (%s); using PyAV fallback.", exc)
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


def _can_use_avconvert(sources: Sequence[dict]) -> bool:
    if sys.platform != "darwin" or not AVCONVERT_PATH.is_file():
        return False
    return any(int(source.get("width") or 0) > 1280 or int(source.get("height") or 0) > 720 for source in sources)


def _create_proxy_with_avconvert(
    sources: Sequence[dict],
    output_path: Path,
    *,
    progress_callback: ProgressCallback | None,
) -> list[dict]:
    """Use AVFoundation's media-engine transcode, then packet-join the parts."""
    parts: list[Path] = []
    try:
        for index, source in enumerate(sources):
            part = output_path.parent / f".analysis-proxy-{index:04d}.m4v"
            part.unlink(missing_ok=True)
            parts.append(part)
            command = [
                str(AVCONVERT_PATH),
                "--source",
                str(source["path"]),
                "--preset",
                AVCONVERT_PRESET,
                "--output",
                str(part),
                "--replace",
                "--progress",
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            captured: list[str] = []
            fragment = ""
            try:
                assert process.stdout is not None
                while True:
                    char = process.stdout.read(1)
                    if char == "" and process.poll() is not None:
                        break
                    if char not in {"\r", "\n", ""}:
                        fragment += char
                        continue
                    if fragment:
                        captured.append(fragment)
                        match = re.search(r"([0-9]+(?:\.[0-9]+)?)% complete", fragment)
                        if match and progress_callback:
                            progress_callback(index, len(sources), str(source["name"]), int(float(match.group(1))))
                        fragment = ""
                return_code = process.wait()
                if return_code != 0 or not part.exists():
                    detail = " ".join(captured[-3:]).strip()
                    raise RuntimeError(f"avconvert failed for {source['name']}: {detail or return_code}")
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                raise
            if progress_callback:
                progress_callback(index, len(sources), str(source["name"]), 100)
        return _remux_proxy_parts(parts, sources, output_path)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


def _remux_proxy_parts(parts: Sequence[Path], sources: Sequence[dict], output_path: Path) -> list[dict]:
    """Join compatible AVFoundation proxy parts by rewriting timestamps only."""
    if len(parts) != len(sources):
        raise ValueError("Proxy part/source count mismatch")
    output_path.unlink(missing_ok=True)
    output = av.open(str(output_path), "w", options={"movflags": "+faststart"})
    out_streams = {}
    timeline = 0.0
    manifest: list[dict] = []
    try:
        for part, source_info in zip(parts, sources):
            original_duration = float(source_info["duration_s"])
            chunk_start = timeline
            with av.open(str(part)) as source:
                selected = [stream for stream in source.streams if stream.type in {"video", "audio"}]
                if not any(stream.type == "video" for stream in selected):
                    raise ValueError(f"No video stream in proxy for {source_info['name']}")
                for stream in selected:
                    if stream.type not in out_streams:
                        out_streams[stream.type] = output.add_stream_from_template(stream)
                first_pts: dict[int, int] = {}
                buffered = []
                for packet in source.demux(*selected):
                    if packet.pts is None or packet.dts is None or packet.duration is None:
                        continue
                    if packet.stream.type not in out_streams:
                        continue
                    buffered.append(packet)
                    previous = first_pts.get(packet.stream.index)
                    first_pts[packet.stream.index] = packet.pts if previous is None else min(previous, packet.pts)
                for packet in buffered:
                    origin = first_pts[packet.stream.index]
                    relative_start = float((packet.pts - origin) * packet.time_base)
                    if relative_start >= original_duration:
                        continue
                    offset_ticks = round(chunk_start / float(packet.time_base))
                    packet.pts = packet.pts - origin + offset_ticks
                    packet.dts = packet.dts - origin + offset_ticks
                    packet.stream = out_streams[packet.stream.type]
                    output.mux(packet)
            manifest.append(
                {
                    **source_info,
                    "start_s": chunk_start,
                    "duration_s": original_duration,
                }
            )
            timeline += original_duration
    finally:
        output.close()
    return manifest


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

    try:
        for source_index, source in enumerate(sources):
            source_start_frame = video_index
            expected_duration = float(source["duration_s"])
            target_chunk_frames = max(1, int(round(expected_duration * fps)))
            encoded_chunk_frames = 0
            path = Path(str(source["path"]))
            if progress_callback:
                progress_callback(source_index, len(sources), path.name, 0)
            with av.open(str(path)) as container:
                in_v = next((item for item in container.streams if item.type == "video"), None)
                in_a = next((item for item in container.streams if item.type == "audio"), None)
                if in_v is None:
                    raise ValueError(f"No video stream found in {path.name}")
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
                streams = [item for item in (in_v, in_a) if item is not None]
                for frame in container.decode(*streams):
                    if isinstance(frame, av.VideoFrame):
                        if frame.pts is None:
                            continue
                        t = float(frame.pts * frame.time_base)
                        if first_video_t is None:
                            first_video_t = t
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
                        frame.pts = None
                        for converted in resampler.resample(frame):
                            converted.pts = None
                            fifo.write(converted)
                        drain_audio()
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
