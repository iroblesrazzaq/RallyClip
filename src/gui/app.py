from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from flask import Flask, jsonify, request, send_file
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "rallyclip gui requires Flask. Reinstall with `pip install .`."
    ) from exc

import numpy as np
import av

from runtime.assets import candidate_roots, resolve_asset
from extraction.pose_extractor import PoseExtractionCancelled, PoseExtractor
from features.feature_engineer import FeatureEngineer
from infer import (
    extract_segments_from_binary,
    gaussian_filter1d,
    hysteresis_threshold,
    load_scaler_asset,
    load_model_from_checkpoint,
    run_windowed_inference_average_onnx_stream,
    run_windowed_inference_average_torch_stream,
    write_segments_csv,
)
from preprocessing.data_preprocessor import DataPreprocessor
from runtime.defaults import build_gui_defaults
from runtime.device import (
    apply_pose_device,
    detect_available_devices,
    prefer_cpu_over_mps_for_pose,
    resolve_auto_device,
)
from runtime.paths import resolve_frontend_dir
from runtime.video_validation import VideoValidationError, validate_video
from segmentation.segment import load_intervals, segment_video

JobDict = Dict[str, Any]
FIXED_YOLO_MODEL = "yolov8n-pose.pt"


STATIC_DIR = resolve_frontend_dir()


def _frozen_data_root() -> Optional[Path]:
    """In packaged builds, keep user data out of the bundle and the CWD."""
    if getattr(sys, "frozen", False):
        return (Path.home() / "RallyClip").resolve()
    return None


def _default_jobs_dir() -> Path:
    """Pick a jobs/output root inside the RallyClip install if possible; fallback to CWD."""
    frozen_root = _frozen_data_root()
    if frozen_root is not None:
        return frozen_root / "jobs"
    for root in candidate_roots():
        root_path = Path(root).resolve()
        if (root_path / "models").exists() or (root_path / "gui").exists():
            return (root_path / "RallyClipJobs").resolve()
    return (Path.cwd() / "RallyClipJobs").resolve()


def _default_output_dir() -> Path:
    frozen_root = _frozen_data_root()
    if frozen_root is not None:
        return frozen_root / "output_videos"
    for root in candidate_roots():
        root_path = Path(root).resolve()
        if (root_path / "models").exists() or (root_path / "gui").exists():
            return (root_path / "output_videos").resolve()
    return (Path.cwd() / "output_videos").resolve()


def _default_csv_dir() -> Path:
    frozen_root = _frozen_data_root()
    if frozen_root is not None:
        return frozen_root / "output_csvs"
    for root in candidate_roots():
        root_path = Path(root).resolve()
        if (root_path / "models").exists() or (root_path / "gui").exists():
            return (root_path / "output_csvs").resolve()
    return (Path.cwd() / "output_csvs").resolve()


def _default_library_dir() -> Path:
    """Persistent library of segmented matches (one folder per match). Survives
    restarts; the GUI's default view reads from here."""
    frozen_root = _frozen_data_root()
    if frozen_root is not None:
        return frozen_root / "library"
    for root in candidate_roots():
        root_path = Path(root).resolve()
        if (root_path / "models").exists() or (root_path / "gui").exists():
            return (root_path / "RallyClipLibrary").resolve()
    return (Path.cwd() / "RallyClipLibrary").resolve()


def _keep_jobs() -> bool:
    return os.environ.get("RALLYCLIP_KEEP_JOBS", "").strip().lower() in {"1", "true", "yes"}


def _sweep_old_jobs(max_age_hours: int = 24) -> None:
    if _keep_jobs():
        return
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    try:
        for child in JOBS_DIR.iterdir():
            try:
                if child.is_dir() and datetime.fromtimestamp(child.stat().st_mtime) < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                continue
    except Exception:
        pass


# Created lazily on first job (upload mkdirs with parents=True); importing the
# module must not write to disk.
JOBS_DIR = _default_jobs_dir()
DEFAULT_OUTPUT_DIR = _default_output_dir()
DEFAULT_CSV_DIR = _default_csv_dir()
LIBRARY_DIR = _default_library_dir()

def _load_default_config() -> Dict[str, Any]:
    try:
        cfg = build_gui_defaults()
    except Exception as exc:
        logging.warning("Could not load manifest defaults (%s); using minimal fallback.", exc)
        cfg = {
            "write_csv": True,
            "segment_video": True,
            "yolo_size": "nano",
            "yolo_weights": FIXED_YOLO_MODEL,
            "yolo_device": None,
            "fps": 5.0,
            "seq_len": 100,
            "overlap": 50,
            "sigma": 1.0,
            "low": 0.45,
            "high": 0.7,
            "min_dur_sec": 1.0,
            "conf": 0.25,
            "imgsz": 960,
            "feature_set": "v1",
            "screen_width": 1280,
            "screen_height": 720,
            "start_time": 0,
            "duration": 999999,
        }
    cfg["output_name"] = None
    # scaler_path is resolved fresh at job-run time via resolve_asset;
    # model_path is retained from build_gui_defaults as a validated startup hint.
    cfg["scaler_path"] = None
    cfg.pop("available_devices", None)
    cfg.pop("auto_device", None)
    return cfg


DEFAULT_CONFIG: Dict[str, Any] = _load_default_config()

ADVANCED_WARNINGS = {
    "fps": "Changing fps will break model expectations; keep at 5.0 unless you retrain.",
    "seq_len": "Sequence length is tied to training; keep at 100.",
    "overlap": "Overlap tunes throughput vs smoothness; default 50 is recommended.",
    "low": "Lowering thresholds increases sensitivity and false positives.",
    "high": "Raising thresholds decreases sensitivity but may miss points.",
    "min_dur_sec": "Shorter durations may create noisy/short segments.",
    "imgsz": "YOLO image size affects pose recall and runtime; v0.3.1 was exported with 960.",
}

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB — matches UI cap
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                re.compile(r"http://127\.0\.0\.1(:\d+)?$"),
                re.compile(r"http://localhost(:\d+)?$"),
            ],
        }
    },
)

_LOCAL_ORIGIN_RE = re.compile(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$")
_LOCAL_HOSTS = {"127.0.0.1", "localhost"}


@app.before_request
def _reject_cross_origin_writes():
    """Block drive-by POSTs from non-local web pages (and DNS-rebinding hosts).

    Multipart form POSTs skip CORS preflight, so CORS alone does not stop a
    malicious website from submitting jobs with attacker-chosen output paths.
    """
    if request.method not in {"POST", "PUT", "DELETE", "PATCH"}:
        return None
    origin = request.headers.get("Origin")
    if origin and not _LOCAL_ORIGIN_RE.match(origin):
        return jsonify({"error": "Forbidden origin"}), 403
    host = (request.host or "").rsplit(":", 1)[0]
    if host not in _LOCAL_HOSTS:
        return jsonify({"error": "Forbidden host"}), 403
    return None


jobs_lock = threading.Lock()
jobs: Dict[str, JobDict] = {}
preview_locks: Dict[str, threading.Lock] = {}
preview_jobs: Dict[str, str] = {}
_MEMORY_PROCESS = None
active_preview_item_id: Optional[str] = None
last_preview_cache_prune = 0.0
PREVIEW_WINDOW_DURATION_S = 8.0
PREVIEW_WINDOW_MIN_DURATION_S = 5.0
PREVIEW_WINDOW_MAX_DURATION_S = 90.0
PREVIEW_WINDOW_WIDTH = 640
PREVIEW_WINDOW_FPS = 30
PREVIEW_CACHE_TTL_SECONDS = 24 * 60 * 60
PREVIEW_TRANSCODE_CONCURRENCY = 2
preview_transcode_semaphore = threading.BoundedSemaphore(PREVIEW_TRANSCODE_CONCURRENCY)


class PipelineCancelled(Exception):
    """Raised when a job is cancelled mid-flight."""


def _ensure_job_dir(job_id: str) -> Path:
    job_dir = (JOBS_DIR / job_id).resolve()
    jobs_root = JOBS_DIR.resolve()
    if jobs_root not in job_dir.parents and job_dir != jobs_root:
        raise ValueError(f"Invalid job id: {job_id!r}")
    return job_dir


def _new_library_id() -> str:
    """Sortable, unique library item id (timestamp + short random suffix)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def _library_item_dir(item_id: str) -> Path:
    """Resolve a library item folder, rejecting ids that escape LIBRARY_DIR."""
    item_dir = (LIBRARY_DIR / item_id).resolve()
    root = LIBRARY_DIR.resolve()
    if root not in item_dir.parents:
        raise ValueError(f"Invalid library id: {item_id!r}")
    return item_dir


def _write_thumbnail(video_path: Path, thumb_path: Path, max_width: int = 480) -> bool:
    """Grab the first frame of the segmented video as a JPEG thumbnail. cv2 is
    imported lazily (it's already loaded by court detection during a job)."""
    try:
        import cv2  # lazy: avoids the libGL import at GUI startup

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                arr = frame.to_ndarray(format="bgr24")
                h, w = arr.shape[:2]
                if w > max_width:
                    arr = cv2.resize(arr, (max_width, max(1, int(h * max_width / w))))
                cv2.imwrite(str(thumb_path), arr)
                return True
    except Exception:
        logging.warning("Could not write thumbnail for %s", video_path, exc_info=True)
    return False


def _rss_mb() -> Optional[float]:
    """Return current resident memory when psutil is available."""
    global _MEMORY_PROCESS
    try:
        if _MEMORY_PROCESS is None:
            import psutil  # type: ignore

            _MEMORY_PROCESS = psutil.Process(os.getpid())
        return float(_MEMORY_PROCESS.memory_info().rss) / 1e6
    except Exception:
        return None


def _peak_rss_mb() -> Optional[float]:
    """Return peak resident memory from resource on Unix-like platforms."""
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return value / 1e6
        return value / 1000.0
    except Exception:
        return None


def _log_memory(label: str, *, job_id: Optional[str] = None, **fields: Any) -> None:
    rss = _rss_mb()
    peak = _peak_rss_mb()
    parts = [f"event=memory", f"label={label}"]
    if job_id:
        parts.append(f"job_id={job_id}")
    if rss is not None:
        parts.append(f"rss_mb={rss:.1f}")
    else:
        parts.append("rss_mb=unavailable")
    if peak is not None:
        parts.append(f"peak_rss_mb={peak:.1f}")
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        elif isinstance(value, str):
            parts.append(f"{key}={json.dumps(value)}")
        else:
            parts.append(f"{key}={value}")
    logging.getLogger("rallyclip.memory").info(" ".join(parts))


def _preview_lock(item_id: str) -> threading.Lock:
    with jobs_lock:
        lock = preview_locks.get(item_id)
        if lock is None:
            lock = threading.Lock()
            preview_locks[item_id] = lock
        return lock


def _ffmpeg_executable() -> Optional[str]:
    env_path = os.environ.get("RALLYCLIP_FFMPEG_PATH")
    candidates = [
        env_path,
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def _write_web_preview_ffmpeg(
    source_path: Path,
    preview_path: Path,
    max_width: int = 640,
    *,
    start_s: float = 0.0,
    duration_s: Optional[float] = None,
    fps: int = 15,
    video_bitrate: str = "700k",
) -> bool:
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        return False

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = preview_path.with_suffix(".tmp.webm")
    if tmp_path.exists():
        tmp_path.unlink()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if start_s > 0:
        command.extend(["-ss", f"{start_s:.3f}"])
    command.extend(
        [
            "-i",
            str(source_path),
        ]
    )
    if duration_s is not None:
        command.extend(["-t", f"{duration_s:.3f}"])
    command.extend(
        [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"scale={max_width}:-2:force_original_aspect_ratio=decrease,fps={fps}",
            "-c:v",
            "libvpx",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-b:v",
            video_bitrate,
            "-maxrate",
            video_bitrate,
            "-bufsize",
            "1800k",
            "-threads",
            "0",
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            "-ac",
            "2",
            "-ar",
            "48000",
            str(tmp_path),
        ]
    )
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        logging.warning("ffmpeg preview transcode failed for %s", source_path, exc_info=True)
        return False

    tmp_path.replace(preview_path)
    return True


def _write_web_preview(
    source_path: Path,
    preview_path: Path,
    max_width: int = 640,
    *,
    start_s: float = 0.0,
    duration_s: Optional[float] = None,
    fps: int = 15,
    video_bitrate: str = "700k",
) -> None:
    """Transcode a browser-safe WebM preview for QtWebEngine.

    The packaged QtWebEngine build can parse MP4 metadata/audio but does not
    decode H.264 video, so the in-app viewer needs a VP8/Opus preview cache.
    """
    if _write_web_preview_ffmpeg(
        source_path,
        preview_path,
        max_width=max_width,
        start_s=start_s,
        duration_s=duration_s,
        fps=fps,
        video_bitrate=video_bitrate,
    ):
        return

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = preview_path.with_suffix(".tmp.webm")
    if tmp_path.exists():
        tmp_path.unlink()

    in_container = av.open(str(source_path))
    try:
        in_v = next((s for s in in_container.streams if s.type == "video"), None)
        if in_v is None:
            raise RuntimeError(f"No video stream found in {source_path}")
        in_a = next((s for s in in_container.streams if s.type == "audio"), None)
        out_container = av.open(str(tmp_path), "w")
    except Exception:
        in_container.close()
        raise

    try:
        width = int(in_v.codec_context.width)
        height = int(in_v.codec_context.height)
        if width > max_width:
            ratio = max_width / width
            width = max(2, int(round(width * ratio)) // 2 * 2)
            height = max(2, int(round(height * ratio)) // 2 * 2)
        else:
            width = max(2, width // 2 * 2)
            height = max(2, height // 2 * 2)

        if start_s > 0:
            in_container.seek(int(start_s * av.time_base), any_frame=False, backward=True)
        end_s = start_s + duration_s if duration_s is not None else None

        source_rate = in_v.average_rate or Fraction(30, 1)
        rate = min(source_rate, Fraction(fps, 1))
        video_tb = Fraction(1, 1) / rate
        out_v = out_container.add_stream("libvpx", rate=rate)
        out_v.width = width
        out_v.height = height
        out_v.pix_fmt = "yuv420p"
        out_v.options = {
            "deadline": "realtime",
            "cpu-used": "8",
            "crf": "12",
            "b:v": video_bitrate,
        }

        out_a = resampler = fifo = None
        if in_a is not None:
            out_a = out_container.add_stream("libopus", rate=48000)
            out_a.layout = "stereo"
            resampler = av.AudioResampler(format=out_a.format, layout=out_a.layout, rate=out_a.rate)
            fifo = av.AudioFifo()

        video_index = 0
        audio_pts = 0

        def drain_audio(flush: bool = False) -> None:
            nonlocal audio_pts
            if out_a is None or fifo is None:
                return
            frame_size = out_a.frame_size or 960
            while fifo.samples >= frame_size or (flush and fifo.samples > 0):
                take = frame_size if fifo.samples >= frame_size else fifo.samples
                a_frame = fifo.read(take)
                if a_frame is None:
                    break
                a_frame.pts = audio_pts
                a_frame.time_base = Fraction(1, out_a.rate)
                audio_pts += a_frame.samples
                for packet in out_a.encode(a_frame):
                    out_container.mux(packet)

        decode_streams = [s for s in (in_v, in_a) if s is not None]
        next_video_t = 0.0
        frame_step = 1.0 / float(rate)
        for frame in in_container.decode(*decode_streams):
            if isinstance(frame, av.VideoFrame):
                if frame.pts is not None:
                    frame_t = float(frame.pts * frame.time_base)
                    if frame_t + 1e-6 < start_s:
                        continue
                    if end_s is not None and frame_t >= end_s:
                        break
                else:
                    frame_t = start_s + (video_index / float(rate))
                if frame.pts is not None:
                    if frame_t + 1e-6 < next_video_t:
                        continue
                    next_video_t = frame_t + frame_step
                frame = frame.reformat(width=width, height=height, format="yuv420p")
                frame.pts = video_index
                frame.time_base = video_tb
                frame.pict_type = av.video.frame.PictureType.NONE
                video_index += 1
                for packet in out_v.encode(frame):
                    out_container.mux(packet)
            elif out_a is not None and isinstance(frame, av.AudioFrame):
                if frame.pts is not None:
                    audio_t = float(frame.pts * frame.time_base)
                    if audio_t + (frame.samples / frame.sample_rate) < start_s:
                        continue
                    if end_s is not None and audio_t >= end_s:
                        continue
                frame.pts = None
                for r_frame in resampler.resample(frame):
                    fifo.write(r_frame)
                drain_audio()

        for packet in out_v.encode():
            out_container.mux(packet)
        if out_a is not None:
            for r_frame in resampler.resample(None):
                fifo.write(r_frame)
            drain_audio(flush=True)
            for packet in out_a.encode():
                out_container.mux(packet)
    finally:
        out_container.close()
        in_container.close()

    tmp_path.replace(preview_path)


def _ensure_web_preview(item_id: str, source_path: Path) -> Path:
    item_dir = _library_item_dir(item_id)
    preview_path = item_dir / "preview.webm"
    if preview_path.exists() and preview_path.stat().st_mtime >= source_path.stat().st_mtime:
        return preview_path
    lock = _preview_lock(item_id)
    with lock:
        if preview_path.exists() and preview_path.stat().st_mtime >= source_path.stat().st_mtime:
            return preview_path
        with preview_transcode_semaphore:
            _write_web_preview(source_path, preview_path)
        return preview_path


def _web_preview_path(item_id: str) -> Path:
    return _library_item_dir(item_id) / "preview.webm"


def _preview_windows_dir(item_id: str) -> Path:
    return _library_item_dir(item_id) / "preview_windows"


def _preview_window_values(source_path: Path, start_s: float, duration_s: Optional[float] = None) -> tuple[float, float]:
    del duration_s
    source_duration = _estimate_duration_seconds(source_path)
    chunk_s = PREVIEW_WINDOW_DURATION_S
    safe_start = max(0.0, start_s)
    if source_duration > 0:
        safe_start = min(safe_start, max(0.0, source_duration - 0.1))
    start = int(safe_start // chunk_s) * chunk_s
    duration = chunk_s
    if source_duration > 0:
        duration = min(duration, max(0.1, source_duration - start))
    return round(start, 3), round(duration, 3)


def _preview_window_path(item_id: str, start_s: float, duration_s: float) -> Path:
    start_ms = int(round(start_s * 1000))
    duration_ms = int(round(duration_s * 1000))
    return _preview_windows_dir(item_id) / f"{start_ms:012d}_{duration_ms:06d}.webm"


def _preview_window_job_key(item_id: str, start_s: float, duration_s: float) -> str:
    return f"{item_id}:window:{int(round(start_s * 1000))}:{int(round(duration_s * 1000))}"


def _preview_window_ready(item_id: str, source_path: Path, start_s: float, duration_s: float) -> bool:
    preview_path = _preview_window_path(item_id, start_s, duration_s)
    return preview_path.exists() and preview_path.stat().st_mtime >= source_path.stat().st_mtime


def _preview_window_state(item_id: str, source_path: Path, start_s: float, duration_s: float) -> str:
    if _preview_window_ready(item_id, source_path, start_s, duration_s):
        return "ready"
    key = _preview_window_job_key(item_id, start_s, duration_s)
    with jobs_lock:
        state = preview_jobs.get(key, "missing")
        if state == "ready":
            preview_jobs.pop(key, None)
            return "missing"
        return state


def _clear_preview_job_state(item_id: str) -> None:
    prefix = f"{item_id}:window:"
    with jobs_lock:
        for key in list(preview_jobs):
            if key == item_id or key.startswith(prefix):
                preview_jobs.pop(key, None)


def _is_preview_cache_active(item_id: str) -> bool:
    with jobs_lock:
        active = active_preview_item_id
    return active is None or active == item_id


def _unlink_if_old(path: Path, cutoff: float) -> bool:
    try:
        if path.exists() and path.stat().st_mtime < cutoff:
            path.unlink()
            return True
    except Exception:
        logging.debug("Could not prune preview cache file %s", path, exc_info=True)
    return False


def _prune_preview_cache(active_item_id: Optional[str], *, now: Optional[float] = None) -> None:
    if not LIBRARY_DIR.exists():
        return
    cutoff = (time.time() if now is None else now) - PREVIEW_CACHE_TTL_SECONDS
    try:
        children = [child for child in LIBRARY_DIR.iterdir() if child.is_dir()]
    except Exception:
        logging.debug("Could not scan preview cache root %s", LIBRARY_DIR, exc_info=True)
        return

    for item_dir in children:
        item_id = item_dir.name
        windows_dir = item_dir / "preview_windows"
        full_preview = item_dir / "preview.webm"

        if active_item_id is not None and item_id != active_item_id:
            if windows_dir.exists():
                shutil.rmtree(windows_dir, ignore_errors=True)
            try:
                full_preview.unlink(missing_ok=True)
            except Exception:
                logging.debug("Could not remove inactive preview cache %s", full_preview, exc_info=True)
            _clear_preview_job_state(item_id)
            continue

        removed_any = _unlink_if_old(full_preview, cutoff)
        if windows_dir.exists():
            try:
                for cached in windows_dir.iterdir():
                    if cached.is_file() and cached.suffix in {".webm", ".tmp"}:
                        removed_any = _unlink_if_old(cached, cutoff) or removed_any
                if not any(windows_dir.iterdir()):
                    windows_dir.rmdir()
            except Exception:
                logging.debug("Could not prune active preview cache %s", windows_dir, exc_info=True)
        if removed_any:
            _clear_preview_job_state(item_id)


def _activate_preview_cache(item_id: str, *, force: bool = False) -> None:
    global active_preview_item_id, last_preview_cache_prune
    now = time.time()
    with jobs_lock:
        changed = active_preview_item_id != item_id
        active_preview_item_id = item_id
        should_prune = force or changed or (now - last_preview_cache_prune) >= 60.0
        if should_prune:
            last_preview_cache_prune = now
    if should_prune:
        _prune_preview_cache(item_id, now=now)


def _ensure_preview_window(item_id: str, source_path: Path, start_s: float, duration_s: float) -> Path:
    preview_path = _preview_window_path(item_id, start_s, duration_s)
    if preview_path.exists() and preview_path.stat().st_mtime >= source_path.stat().st_mtime:
        return preview_path
    lock = _preview_lock(_preview_window_job_key(item_id, start_s, duration_s))
    with lock:
        if preview_path.exists() and preview_path.stat().st_mtime >= source_path.stat().st_mtime:
            return preview_path
        with preview_transcode_semaphore:
            _write_web_preview(
                source_path,
                preview_path,
                max_width=PREVIEW_WINDOW_WIDTH,
                start_s=start_s,
                duration_s=duration_s,
                fps=PREVIEW_WINDOW_FPS,
                video_bitrate="1200k",
            )
        return preview_path


def _prepare_preview_window_background(item_id: str, source_path: Path, start_s: float, duration_s: float) -> None:
    key = _preview_window_job_key(item_id, start_s, duration_s)
    with jobs_lock:
        preview_jobs[key] = "processing"
    try:
        preview_path = _ensure_preview_window(item_id, source_path, start_s, duration_s)
        if not _is_preview_cache_active(item_id):
            try:
                preview_path.unlink(missing_ok=True)
            finally:
                with jobs_lock:
                    preview_jobs.pop(key, None)
            return
        with jobs_lock:
            preview_jobs[key] = "ready"
    except Exception:
        with jobs_lock:
            preview_jobs[key] = "error"
        logging.warning("Could not prepare preview window for %s at %.3fs", item_id, start_s, exc_info=True)


def _start_preview_window_background(item_id: str, source_path: Path, start_s: float, duration_s: float) -> str:
    state = _preview_window_state(item_id, source_path, start_s, duration_s)
    if state in {"ready", "processing"}:
        return state
    threading.Thread(
        target=_prepare_preview_window_background,
        args=(item_id, source_path, start_s, duration_s),
        daemon=True,
        name=f"rallyclip-preview-window-{item_id}",
    ).start()
    return "processing"


def _web_preview_ready(item_id: str, source_path: Path) -> bool:
    preview_path = _web_preview_path(item_id)
    return preview_path.exists() and preview_path.stat().st_mtime >= source_path.stat().st_mtime


def _web_preview_state(item_id: str, source_path: Path) -> str:
    if _web_preview_ready(item_id, source_path):
        return "ready"
    with jobs_lock:
        return preview_jobs.get(item_id, "missing")


def _prepare_web_preview_background(item_id: str, source_path: Path) -> None:
    with jobs_lock:
        preview_jobs[item_id] = "processing"
    try:
        preview_path = _ensure_web_preview(item_id, source_path)
        if not _is_preview_cache_active(item_id):
            try:
                preview_path.unlink(missing_ok=True)
            finally:
                with jobs_lock:
                    preview_jobs.pop(item_id, None)
            return
        with jobs_lock:
            preview_jobs[item_id] = "ready"
    except Exception:
        with jobs_lock:
            preview_jobs[item_id] = "error"
        logging.warning("Could not prepare web preview for %s", item_id, exc_info=True)


def _start_web_preview_background(item_id: str, source_path: Path) -> str:
    state = _web_preview_state(item_id, source_path)
    if state in {"ready", "processing"}:
        return state
    threading.Thread(
        target=_prepare_web_preview_background,
        args=(item_id, source_path),
        daemon=True,
        name=f"rallyclip-preview-{item_id}",
    ).start()
    return "processing"


def _read_library_items() -> list[Dict[str, Any]]:
    """List saved matches (newest first). An item needs meta.json + a source video."""
    items: list[Dict[str, Any]] = []
    if not LIBRARY_DIR.exists():
        return items
    for child in LIBRARY_DIR.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists() or not ((child / "source.mp4").exists() or (child / "video.mp4").exists()):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta["id"] = child.name  # trust the folder name, not the file contents
        meta["has_csv"] = (child / "segments.csv").exists()
        meta["has_thumbnail"] = (child / "thumb.jpg").exists()
        meta["has_export"] = (child / "export.mp4").exists()
        items.append(meta)
    items.sort(key=lambda m: m.get("created_ts", 0), reverse=True)
    return items


def _persist_library_item(
    *,
    upload_path: Path,
    base_name: str,
    segments: list[tuple[int, int]],
    intervals_sec: list[tuple[float, float]],
    fps: float,
    job: JobDict,
) -> tuple[str, Path, Path]:
    """Write one saved match folder, removing it again if any write fails.

    The library stores the full source video plus the CSV. Cut/stitched exports
    are generated lazily by the export endpoint, not during analysis.
    """
    library_id = _new_library_id()
    item_dir = _library_item_dir(library_id)
    item_dir.mkdir(parents=True, exist_ok=True)
    try:
        csv_out = item_dir / "segments.csv"
        write_segments_csv(segments, str(csv_out), fps=fps, overwrite=True)
        source_out = item_dir / "source.mp4"
        shutil.copy2(upload_path, source_out)
        _set_step(job, "output", "in_progress", 70)
        _write_thumbnail(source_out, item_dir / "thumb.jpg")
        full_duration_s = _estimate_duration_seconds(source_out)
        meta = {
            "id": library_id,
            "name": base_name,
            "source_name": upload_path.name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "created_ts": time.time(),
            "duration_s": round(full_duration_s, 2) if full_duration_s > 0 else 0.0,
            "point_duration_s": round(sum(e - s for s, e in intervals_sec), 2),
            "n_segments": len(segments),
        }
        (item_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if intervals_sec:
            start_s, duration_s = _preview_window_values(source_out, intervals_sec[0][0], PREVIEW_WINDOW_DURATION_S)
            _start_preview_window_background(library_id, source_out, start_s, duration_s)
        return library_id, source_out, csv_out
    except Exception:
        shutil.rmtree(item_dir, ignore_errors=True)
        raise


def _new_job_state(job_id: str, cfg: Dict[str, Any]) -> JobDict:
    return {
        "id": job_id,
        "status": "in_progress",
        "error": None,
        "cancelled": False,
        "config": cfg,
        "steps": {
            "pose": {"status": "waiting", "progress": 0},
            "preprocess": {"status": "waiting", "progress": 0},
            "feature": {"status": "waiting", "progress": 0},
            "inference": {"status": "waiting", "progress": 0},
            "output": {"status": "waiting", "progress": 0},
        },
        "weights": None,
        "eta_seconds": None,
        "library_id": None,
        "pose_t0": None,
        "paths": {
            "upload": None,
            "raw_npz": None,
            "preprocessed_npz": None,
            "features_npz": None,
            "csv": None,
            "video": None,
            "job_dir": str(JOBS_DIR / job_id),
        },
        "thread": None,
    }


def _set_step(job: JobDict, step: str, status: str, progress: int) -> None:
    job["steps"][step]["status"] = status
    job["steps"][step]["progress"] = int(max(0, min(100, progress)))


def _check_cancel(job: JobDict) -> None:
    if job.get("cancelled"):
        job["status"] = "cancelled"
        raise PipelineCancelled("Job cancelled")


def _pick_port(preferred: Optional[list[int]] = None) -> int:
    choices = preferred or [8000, 5173]
    for port in choices:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _safe_open_browser(port: int) -> None:
    time.sleep(1.5)
    try:
        webbrowser.open(f"http://127.0.0.1:{port}/")
    except Exception:
        pass


# Keys the browser may override. Everything else (manifest-pinned inference
# params, model/artifact paths) keeps the server-side default even though the
# frontend round-trips the full defaults payload.
_CLIENT_KEYS = {
    "output_name",
    "yolo_device",
    "write_csv",
    "segment_video",
    "low",
    "high",
    "min_dur_sec",
    "start_time",
    "duration",
}


def _normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {**DEFAULT_CONFIG}
    cfg.update(
        {k: v for k, v in (raw or {}).items() if v is not None and k in _CLIENT_KEYS}
    )
    cfg["yolo_size"] = "nano"
    cfg["yolo_weights"] = FIXED_YOLO_MODEL
    return cfg


def _resolve_yolo_weights(cfg: Dict[str, Any]) -> str:
    path = resolve_asset(
        None,
        env_var="RALLYCLIP_YOLO_MODEL_PATH",
        relatives=[
            f"models/{FIXED_YOLO_MODEL}",
            FIXED_YOLO_MODEL,
        ],
        description="RallyClip YOLO pose model (nano)",
    )
    return str(path)


def _resolve_model_paths(cfg: Dict[str, Any]) -> tuple[Path, Path]:
    model_path = resolve_asset(
        cfg.get("model_path"),
        env_var="RALLYCLIP_MODEL_PATH",
        relatives=[
            "models/rallyclip_v0.3.1/model.onnx",
            "models/lstm_300_v0.1.pth",
            "checkpoints/seq_len300/best_model.pth",
        ],
        description="RallyClip model artifact (ONNX or PyTorch checkpoint)",
    )
    scaler_path = resolve_asset(
        cfg.get("scaler_path"),
        env_var="RALLYCLIP_SCALER_PATH",
        relatives=[
            "models/rallyclip_v0.3.1/scaler.json",
            "models/scaler_300_v0.1.joblib",
            "data/seq_len_300/scaler.joblib",
        ],
        description="RallyClip scaler artifact (JSON or joblib)",
    )
    return model_path, scaler_path


def _estimate_duration_seconds(video_path: Path) -> float:
    try:
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                return 0.0
            stream = container.streams.video[0]
            if getattr(stream, "duration", None) and getattr(stream, "time_base", None):
                return float(stream.duration * stream.time_base)
            if container.duration:
                return float(container.duration) * float(av.time_base)
    except Exception:
        return 0.0
    return 0.0


def _pose_weight(duration_seconds: float) -> float:
    minutes = max(0.0, duration_seconds) / 60.0
    if minutes <= 5.0:
        return 0.90
    if minutes >= 120.0:
        return 0.98
    # linear ramp between 5min and 120min
    return 0.90 + (min(120.0, max(5.0, minutes)) - 5.0) / (115.0) * 0.08


def _compute_weights(duration_seconds: float) -> Dict[str, float]:
    pose_w = _pose_weight(duration_seconds)
    remaining = max(0.0, 1.0 - pose_w)
    other = remaining / 4.0 if remaining else 0.0
    return {
        "pose": pose_w,
        "preprocess": other,
        "feature": other,
        "inference": other,
        "output": other,
    }


def _estimate_stream_window_count(num_frames: int, sequence_length: int, overlap: int) -> Optional[int]:
    try:
        n = int(num_frames)
        length = int(sequence_length)
        ov = int(overlap)
    except Exception:
        return None
    if n < length or length <= 0 or ov < 0 or ov >= length:
        return None
    step = length - ov
    count = ((n - length) // step) + 1
    last_start = (count - 1) * step
    if last_start + length < n:
        count += 1
    return max(1, count)


def _run_pipeline(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return

    cfg = job["config"]
    try:
        upload_path = Path(job["paths"]["upload"])
        job_dir = Path(job["paths"]["job_dir"])
        job_dir.mkdir(parents=True, exist_ok=True)
        raw_output_name = cfg.get("output_name") or upload_path.stem
        base_name = Path(str(raw_output_name)).name or upload_path.stem
        pipeline_start = time.perf_counter()
        last_memory_log = 0.0
        _log_memory("job_start", job_id=job_id, video=upload_path.name)

        duration_seconds = _estimate_duration_seconds(upload_path)
        if duration_seconds <= 0:
            cfg_duration = float(cfg.get("duration") or 0)
            duration_seconds = cfg_duration if 0 < cfg_duration < 999999 else 600.0
        elif cfg.get("duration") and cfg["duration"] > 0:
            duration_seconds = min(duration_seconds, float(cfg["duration"]))
        weights = _compute_weights(duration_seconds)
        job["weights"] = weights

        yolo_weights = _resolve_yolo_weights(cfg)
        if cfg.get("yolo_device"):
            pose_device = apply_pose_device(
                str(cfg["yolo_device"]), model_path=yolo_weights, set_env=False
            )
        else:
            pose_device = apply_pose_device(None, model_path=yolo_weights, set_env=False)

        model_path, scaler_path = _resolve_model_paths(cfg)
        models_dir = None
        frozen_root = _frozen_data_root()
        if frozen_root is not None:
            models_dir = str(frozen_root / "models")
        else:
            for root in candidate_roots():
                candidate = Path(root) / "models"
                if candidate.exists():
                    models_dir = str(candidate.resolve())
                    break

        _check_cancel(job)
        if str(cfg.get("feature_set", "v1")) != "v1":
            raise RuntimeError(
                f"Unsupported feature_set '{cfg.get('feature_set')}'. Only 'v1' is implemented."
            )

        pre = DataPreprocessor(
            screen_width=int(cfg["screen_width"]),
            screen_height=int(cfg["screen_height"]),
            save_court_masks=False,
            yolo_model_path=yolo_weights,
            conf=float(cfg["conf"]),
            yolo_device=pose_device,
        )
        _check_cancel(job)
        _set_step(job, "pose", "in_progress", 1)
        _check_cancel(job)
        court_mask, _ = pre.compute_court_mask(str(upload_path))
        # Court mask detection has no inner progress hooks; tick so the bar
        # visibly moves before pose extraction starts reporting.
        _set_step(job, "pose", "in_progress", 3)
        _log_memory("after_court_mask", job_id=job_id, elapsed_s=time.perf_counter() - pipeline_start)

        _check_cancel(job)
        extractor = PoseExtractor(
            model_dir=models_dir,
            model_path=yolo_weights,
            imgsz=int(cfg["imgsz"]),
            device=pose_device,
        )

        def pose_progress(frac: float, meta: Optional[Dict[str, Any]] = None) -> None:
            nonlocal last_memory_log
            if job.get("cancelled"):
                raise PoseExtractionCancelled("Job cancelled during pose extraction")
            _set_step(job, "pose", "in_progress", int(3 + max(0.0, min(1.0, frac)) * 96))
            if meta:
                frames_seen = meta.get("frames_seen", meta.get("frames_done", 0))
                frames_total = meta.get("frames_total", 1)
                # prefer FPS derived from frames_seen to mirror tqdm ETA
                smoothed_fps = max(1e-3, meta.get("smoothed_seen_fps", meta.get("smoothed_proc_fps", 0.0)))
                remaining_frames = max(0, frames_total - frames_seen)
                pose_eta = remaining_frames / smoothed_fps
                # Tail buffer: 10s minimum, 60s max, scaled by minutes
                tail = max(10.0, min(60.0, (duration_seconds / 60.0) * 5.0))
                job["eta_seconds"] = pose_eta + tail
                job["pose_eta_seconds"] = pose_eta
                job["pose_throughput_fps"] = smoothed_fps
                now = time.perf_counter()
                if now - last_memory_log >= 10.0 or frac >= 1.0:
                    last_memory_log = now
                    _log_memory(
                        "pose_progress",
                        job_id=job_id,
                        elapsed_s=now - pipeline_start,
                        frames_seen=frames_seen,
                        frames_total=frames_total,
                        progress=frac,
                        pose_fps=smoothed_fps,
                    )

        # Streaming hand-off: pose -> preprocess -> features chain through their generators, so
        # pose_data and the preprocessed records are produced-and-discarded one frame at a time
        # (no intermediate NPZ, no full-length pose/preprocess buffers). The pose progress
        # callback drives the bar while the chain is consumed. Native source resolution gives an
        # identity rescale at 720p, exactly as preprocess_single_video did internally.
        src_height, src_width, _ = pre._source_frame_shape(str(upload_path))
        pose_stream = extractor.iter_pose_frames(
            video_path=str(upload_path),
            confidence_threshold=float(cfg["conf"]),
            start_time_seconds=int(cfg["start_time"]),
            duration_seconds=int(cfg["duration"]),
            target_fps=int(cfg["fps"]),
            imgsz=int(cfg["imgsz"]),
            annotations_csv=None,
            progress_callback=pose_progress,
        )
        preprocessed_stream = pre.iter_preprocess_frames(pose_stream, court_mask, src_width, src_height)
        fe = FeatureEngineer(
            screen_width=int(cfg["screen_width"]),
            screen_height=int(cfg["screen_height"]),
            target_fps=float(cfg["fps"]),
        )
        feature_stream = fe.iter_build_features(preprocessed_stream)

        _check_cancel(job)
        _set_step(job, "preprocess", "in_progress", 1)
        _set_step(job, "feature", "in_progress", 1)
        _set_step(job, "inference", "in_progress", 5)
        scaler = load_scaler_asset(str(scaler_path))
        _log_memory("before_streaming_inference", job_id=job_id, elapsed_s=time.perf_counter() - pipeline_start)
        estimated_feature_rows = max(0, int(round(duration_seconds * float(cfg["fps"]))) - 1)
        estimated_windows = _estimate_stream_window_count(
            estimated_feature_rows,
            int(cfg["seq_len"]),
            int(cfg["overlap"]),
        )
        feature_rows_seen = 0
        last_stream_progress = 0.0

        def scaled_feature_rows():
            nonlocal feature_rows_seen, last_stream_progress
            for feature_vector, _target in feature_stream:
                _check_cancel(job)
                feature_rows_seen += 1
                now = time.perf_counter()
                if estimated_feature_rows > 0 and (now - last_stream_progress >= 2.0 or feature_rows_seen >= estimated_feature_rows):
                    last_stream_progress = now
                    frac = min(1.0, feature_rows_seen / float(estimated_feature_rows))
                    stage_progress = int(1 + frac * 94)
                    _set_step(job, "preprocess", "in_progress", stage_progress)
                    _set_step(job, "feature", "in_progress", stage_progress)
                    if estimated_windows is None:
                        _set_step(job, "inference", "in_progress", int(5 + frac * 80))
                row = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
                yield scaler.transform(row)[0].astype(np.float32)

        def infer_progress(frac: float) -> None:
            _set_step(job, "inference", "in_progress", int(5 + max(0.0, min(1.0, frac)) * 90))
        if model_path.suffix.lower() == ".onnx":
            avg_probs = run_windowed_inference_average_onnx_stream(
                str(model_path),
                scaled_feature_rows(),
                sequence_length=int(cfg["seq_len"]),
                overlap=int(cfg["overlap"]),
                progress_callback=infer_progress,
                total_windows=estimated_windows,
            )
        else:
            model, device = load_model_from_checkpoint(str(model_path), return_logits=False)
            avg_probs = run_windowed_inference_average_torch_stream(
                model,
                device,
                scaled_feature_rows(),
                sequence_length=int(cfg["seq_len"]),
                overlap=int(cfg["overlap"]),
                progress_callback=infer_progress,
                total_windows=estimated_windows,
            )
        _set_step(job, "pose", "completed", 100)
        _set_step(job, "preprocess", "completed", 100)
        _set_step(job, "feature", "completed", 100)
        _log_memory("after_streaming_inference", job_id=job_id, elapsed_s=time.perf_counter() - pipeline_start)
        smoothed_probs = gaussian_filter1d(avg_probs.astype(np.float32), sigma=float(cfg["sigma"]))
        min_duration_frames = int(round(max(0.0, float(cfg["min_dur_sec"])) * float(cfg["fps"])))
        binary_pred = hysteresis_threshold(
            smoothed_probs,
            low=float(cfg["low"]),
            high=float(cfg["high"]),
            min_duration=min_duration_frames,
        )
        segments = extract_segments_from_binary(binary_pred)
        _set_step(job, "inference", "completed", 100)

        _check_cancel(job)
        _set_step(job, "output", "in_progress", 5)
        intervals_sec = [
            (start_idx / float(cfg["fps"]), end_idx / float(cfg["fps"]))
            for start_idx, end_idx in segments
        ]
        # Persist a found match as one library item (full source + csv + thumb +
        # meta in a single folder): survives restarts, deletes atomically. No
        # segments -> nothing worth saving, so no item is created.
        library_id = None
        if intervals_sec:
            library_id, source_out, csv_out = _persist_library_item(
                upload_path=upload_path,
                base_name=base_name,
                segments=segments,
                intervals_sec=intervals_sec,
                fps=float(cfg["fps"]),
                job=job,
            )
            job["paths"]["video"] = str(source_out)
            job["paths"]["csv"] = str(csv_out)
        job["library_id"] = library_id
        _set_step(job, "output", "completed", 100)
        job["status"] = "completed"
        job["eta_seconds"] = 0.0
        _log_memory(
            "job_completed",
            job_id=job_id,
            elapsed_s=time.perf_counter() - pipeline_start,
            segments=len(segments),
            library_id=library_id,
        )
        # No intermediate NPZs are written anymore (stages hand off in memory), so there
        # is nothing to clean up between stages.
        # Remove uploaded input and optionally job dir if outputs are elsewhere and keep flag not set
        if not _keep_jobs():
            upload_path.unlink(missing_ok=True)
            all_outputs = [job.get("paths", {}).get("video"), job.get("paths", {}).get("csv")]
            outputs_in_job_dir = False
            for p in all_outputs:
                if not p:
                    continue
                try:
                    if Path(p).resolve().is_relative_to(job_dir):
                        outputs_in_job_dir = True
                        break
                except Exception:
                    continue
            if job_dir.exists() and not outputs_in_job_dir:
                shutil.rmtree(job_dir, ignore_errors=True)
    except PoseExtractionCancelled:
        _log_memory("job_cancelled", job_id=job_id)
        job["status"] = "cancelled"
        job["error"] = None
        job["eta_seconds"] = 0.0
        if not _keep_jobs():
            try:
                Path(job["paths"].get("upload", "")).unlink(missing_ok=True)
            except Exception:
                pass
            try:
                shutil.rmtree(job.get("paths", {}).get("job_dir", ""), ignore_errors=True)
            except Exception:
                pass
    except PipelineCancelled:
        _log_memory("job_cancelled", job_id=job_id)
        job["status"] = "cancelled"
        job["error"] = None
        job["eta_seconds"] = 0.0
        if not _keep_jobs():
            try:
                Path(job["paths"].get("upload", "")).unlink(missing_ok=True)
            except Exception:
                pass
            try:
                shutil.rmtree(job.get("paths", {}).get("job_dir", ""), ignore_errors=True)
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - runtime safety
        _log_memory("job_failed", job_id=job_id, error=type(exc).__name__)
        if job.get("status") != "cancelled":
            job["status"] = "failed"
            job["error"] = str(exc)
        if not _keep_jobs():
            try:
                Path(job["paths"].get("upload", "")).unlink(missing_ok=True)
            except Exception:
                pass
            try:
                shutil.rmtree(job.get("paths", {}).get("job_dir", ""), ignore_errors=True)
            except Exception:
                pass


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/health", methods=["GET"])
def health() -> tuple[Any, int]:
    return jsonify({"status": "ok"}), 200


@app.route("/api/config/defaults", methods=["GET"])
def config_defaults() -> tuple[Any, int]:
    defaults = {**DEFAULT_CONFIG}
    # Server-internal values; the frontend never edits these (see _CLIENT_KEYS)
    # and absolute server paths don't belong in the browser payload.
    for key in ("model_path", "artifact_dir", "scaler_path", "yolo_weights"):
        defaults.pop(key, None)
    available = detect_available_devices()
    # Report the device the pipeline will actually use on "Auto" (auto-MPS is
    # downgraded to CPU for pose models).
    auto_device = prefer_cpu_over_mps_for_pose(
        resolve_auto_device(), _resolve_yolo_weights(DEFAULT_CONFIG), warn=False
    )
    return jsonify(
        {
            "defaults": defaults,
            "yolo_model": FIXED_YOLO_MODEL,
            "warnings": ADVANCED_WARNINGS,
            "available_devices": available,
            "auto_device": auto_device,
        }
    ), 200


@app.route("/api/upload-and-start", methods=["POST"])
def upload_and_start():
    if "video" not in request.files:
        return jsonify({"error": "Missing file field 'video'"}), 400
    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    # Accept any container PyAV can decode; validate by content below, not by
    # extension. secure_filename strips non-ASCII; the saved name's extension
    # doesn't affect decoding, so just ensure a safe, non-empty filename.
    filename = secure_filename(file.filename) or "input.mp4"
    if "." not in filename:
        filename = f"{filename}.mp4"

    try:
        cfg_raw = json.loads(request.form.get("config", "{}") or "{}")
    except json.JSONDecodeError:
        cfg_raw = {}
    cfg = _normalize_config(cfg_raw)

    job_id = str(uuid.uuid4())
    try:
        job_dir = _ensure_job_dir(job_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    job_dir.mkdir(parents=True, exist_ok=True)
    upload_path = job_dir / filename
    file.save(str(upload_path))

    # Preflight before spawning the worker so bad input is rejected immediately,
    # not surfaced as a "failed" job minutes later.
    try:
        validate_video(upload_path, seq_len=int(cfg["seq_len"]), fps=float(cfg["fps"]))
    except VideoValidationError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400

    state = _new_job_state(job_id, cfg)
    state["paths"]["upload"] = str(upload_path)
    worker = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    state["thread"] = worker
    with jobs_lock:
        jobs[job_id] = state
    worker.start()
    return jsonify({"job_id": job_id}), 200


@app.route("/api/progress/<job_id>", methods=["GET"])
def get_progress(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id"}), 404
    return jsonify(
        {
            "status": job["status"],
            "steps": job["steps"],
            "error": job.get("error"),
            "weights": job.get("weights"),
            "eta_seconds": job.get("eta_seconds"),
            "pose_eta_seconds": job.get("pose_eta_seconds"),
            "pose_throughput_fps": job.get("pose_throughput_fps"),
            "library_id": job.get("library_id"),
        }
    ), 200


@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id"}), 404
    if job["status"] == "in_progress":
        job["cancelled"] = True
        job["status"] = "cancelled"
    return jsonify({"status": job["status"]}), 200


@app.route("/api/library", methods=["GET"])
def library_list():
    return jsonify({"items": _read_library_items()}), 200


def _resolve_library_file(item_id: str, filename: str) -> Optional[Path]:
    """Resolve a file inside a library item, or None if the id/file is invalid."""
    try:
        item_dir = _library_item_dir(item_id)
    except ValueError:
        return None
    path = item_dir / filename
    return path if path.exists() else None


def _resolve_library_source(item_id: str) -> Optional[Path]:
    """Resolve the full source video for a library item.

    source.mp4 is the current storage contract. video.mp4 is accepted only as a
    legacy fallback for library items created before lazy export existed.
    """
    try:
        item_dir = _library_item_dir(item_id)
    except ValueError:
        return None
    for filename in ("source.mp4", "video.mp4"):
        path = item_dir / filename
        if path.exists():
            return path
    return None


@app.route("/api/library/<item_id>/thumbnail", methods=["GET"])
def library_thumbnail(item_id: str):
    path = _resolve_library_file(item_id, "thumb.jpg")
    if path is None:
        return jsonify({"error": "Thumbnail not available"}), 404
    return send_file(str(path), mimetype="image/jpeg")


@app.route("/api/library/<item_id>/video", methods=["GET"])
def library_video(item_id: str):
    source_path = _resolve_library_source(item_id)
    csv_path = _resolve_library_file(item_id, "segments.csv")
    if source_path is None:
        return jsonify({"error": "Video not available"}), 404
    if csv_path is None:
        # Legacy items may only have an already-cut video.mp4.
        if source_path.name == "video.mp4":
            return send_file(str(source_path), as_attachment=True, download_name=f"{item_id}_segmented.mp4")
        return jsonify({"error": "CSV not available"}), 404

    try:
        item_dir = _library_item_dir(item_id)
        export_path = item_dir / "export.mp4"
        needs_export = not export_path.exists()
        if not needs_export:
            export_mtime = export_path.stat().st_mtime
            needs_export = source_path.stat().st_mtime > export_mtime or csv_path.stat().st_mtime > export_mtime
        if needs_export:
            intervals = load_intervals(str(csv_path))
            if not intervals:
                return jsonify({"error": "No point intervals available"}), 404
            segment_video(str(source_path), intervals, str(export_path))
        return send_file(str(export_path), as_attachment=True, download_name=f"{item_id}_segmented.mp4")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Could not export library video %s", item_id)
        return jsonify({"error": f"Could not export video: {exc}"}), 500


@app.route("/api/library/<item_id>/preview", methods=["GET"])
def library_video_preview(item_id: str):
    path = _resolve_library_source(item_id)
    if path is None:
        return jsonify({"error": "Video not available"}), 404
    _activate_preview_cache(item_id)
    if not _web_preview_ready(item_id, path):
        state = _start_web_preview_background(item_id, path)
        return jsonify({"status": state}), 202
    try:
        preview_path = _web_preview_path(item_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Could not prepare library preview %s", item_id)
        return jsonify({"error": f"Could not prepare preview: {exc}"}), 500
    return send_file(str(preview_path), mimetype="video/webm")


@app.route("/api/library/<item_id>/preview/status", methods=["GET"])
def library_video_preview_status(item_id: str):
    path = _resolve_library_source(item_id)
    if path is None:
        return jsonify({"error": "Video not available"}), 404
    _activate_preview_cache(item_id)
    state = _start_web_preview_background(item_id, path)
    preview_path = _web_preview_path(item_id)
    payload = {
        "status": state,
        "ready": state == "ready",
        "preview_url": f"/api/library/{item_id}/preview" if state == "ready" else None,
    }
    if preview_path.exists():
        payload["bytes"] = preview_path.stat().st_size
    return jsonify(payload), 200


def _preview_window_from_request(source_path: Path) -> tuple[float, float]:
    try:
        start_s = float(request.args.get("start", "0"))
        duration_arg = request.args.get("duration")
        duration_s = float(duration_arg) if duration_arg is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid preview window") from exc
    return _preview_window_values(source_path, start_s, duration_s)


def _preview_window_payload(item_id: str, source_path: Path, start_s: float, duration_s: float, state: str) -> Dict[str, Any]:
    source_duration = _estimate_duration_seconds(source_path)
    preview_path = _preview_window_path(item_id, start_s, duration_s)
    payload: Dict[str, Any] = {
        "status": state,
        "ready": state == "ready",
        "start": start_s,
        "duration": duration_s,
        "source_duration": round(source_duration, 3) if source_duration > 0 else None,
        "preview_url": (
            f"/api/library/{item_id}/preview/window?start={start_s:.3f}&duration={duration_s:.3f}"
            if state == "ready"
            else None
        ),
    }
    if preview_path.exists():
        payload["bytes"] = preview_path.stat().st_size
    return payload


@app.route("/api/library/<item_id>/preview/window/status", methods=["GET"])
def library_video_preview_window_status(item_id: str):
    path = _resolve_library_source(item_id)
    if path is None:
        return jsonify({"error": "Video not available"}), 404
    _activate_preview_cache(item_id)
    try:
        start_s, duration_s = _preview_window_from_request(path)
        state = _start_preview_window_background(item_id, path, start_s, duration_s)
        return jsonify(_preview_window_payload(item_id, path, start_s, duration_s, state)), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/library/<item_id>/preview/window", methods=["GET"])
def library_video_preview_window(item_id: str):
    path = _resolve_library_source(item_id)
    if path is None:
        return jsonify({"error": "Video not available"}), 404
    _activate_preview_cache(item_id)
    try:
        start_s, duration_s = _preview_window_from_request(path)
        if not _preview_window_ready(item_id, path, start_s, duration_s):
            state = _start_preview_window_background(item_id, path, start_s, duration_s)
            return jsonify(_preview_window_payload(item_id, path, start_s, duration_s, state)), 202
        preview_path = _preview_window_path(item_id, start_s, duration_s)
        return send_file(str(preview_path), mimetype="video/webm")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Could not prepare preview window %s", item_id)
        return jsonify({"error": f"Could not prepare preview window: {exc}"}), 500


@app.route("/api/library/<item_id>/segments", methods=["GET"])
def library_segments(item_id: str):
    path = _resolve_library_file(item_id, "segments.csv")
    if path is None:
        return jsonify({"error": "CSV not available"}), 404
    try:
        intervals = load_intervals(str(path))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(
        {
            "segments": [{"start": start, "end": end} for start, end in intervals],
            "point_duration_s": round(sum(end - start for start, end in intervals), 3),
        }
    ), 200


@app.route("/api/library/<item_id>/csv", methods=["GET"])
def library_csv(item_id: str):
    path = _resolve_library_file(item_id, "segments.csv")
    if path is None:
        return jsonify({"error": "CSV not available"}), 404
    return send_file(str(path), as_attachment=True, download_name=f"{item_id}_segments.csv")


@app.route("/api/library/<item_id>", methods=["DELETE"])
def library_delete(item_id: str):
    try:
        item_dir = _library_item_dir(item_id)
    except ValueError:
        return jsonify({"error": "Invalid id"}), 400
    if not item_dir.exists():
        return jsonify({"error": "Unknown library id"}), 404
    shutil.rmtree(item_dir, ignore_errors=True)
    return jsonify({"status": "deleted"}), 200


def _configure_gui_logging() -> None:
    verbose = os.environ.get("RALLYCLIP_GUI_VERBOSE", "").strip().lower() in {"1", "true", "yes"}
    log_level = logging.INFO if verbose else logging.ERROR
    if not getattr(_configure_gui_logging, "_configured", False):
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(formatter)
        root_logger.addHandler(console)

        log_dir = Path(os.environ.get("RALLYCLIP_LOG_DIR") or (_frozen_data_root() or Path.cwd()) / "logs")
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "rallyclip.log", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.getLogger("rallyclip.memory").info("event=log_ready path=%s", log_dir / "rallyclip.log")
        except Exception:
            logging.getLogger(__name__).exception("Could not configure RallyClip file logging")

        _configure_gui_logging._configured = True  # type: ignore[attr-defined]
    else:
        logging.getLogger().setLevel(logging.INFO)
    for name in ("werkzeug", "flask.app"):
        logging.getLogger(name).setLevel(log_level)
    if not verbose:
        os.environ.setdefault("RALLYCLIP_NO_TQDM", "1")
        try:
            import flask.cli  # type: ignore
            flask.cli.show_server_banner = lambda *args, **kwargs: None  # noqa: E731
        except Exception:
            pass


def _choose_gui_port(port: Optional[int] = None) -> int:
    preferred_ports: list[int] = []
    env_port = os.environ.get("RALLYCLIP_GUI_PORT")
    if port:
        preferred_ports.append(int(port))
    if env_port:
        try:
            preferred_ports.append(int(env_port))
        except ValueError:
            pass
    preferred_ports.extend([8000, 5173])
    return _pick_port(preferred_ports)


def start_backend_thread(port: Optional[int] = None) -> tuple[int, threading.Thread]:
    """Start the Flask backend on localhost and return (port, thread)."""
    _configure_gui_logging()
    threading.Thread(target=_sweep_old_jobs, daemon=True, name="rallyclip-job-sweep").start()
    chosen_port = _choose_gui_port(port)

    def _serve() -> None:
        app.run(host="127.0.0.1", port=chosen_port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=_serve, daemon=True, name="rallyclip-gui-backend")
    thread.start()
    return chosen_port, thread


def launch(port: Optional[int] = None) -> int:
    _configure_gui_logging()
    threading.Thread(target=_sweep_old_jobs, daemon=True, name="rallyclip-job-sweep").start()
    chosen_port = _choose_gui_port(port)
    threading.Thread(target=_safe_open_browser, args=(chosen_port,), daemon=True).start()
    app.logger.info("Starting GUI on http://127.0.0.1:%s", chosen_port)
    try:
        app.run(host="127.0.0.1", port=chosen_port, debug=False, use_reloader=False, threaded=True)
    except Exception:  # pragma: no cover - runtime safety
        app.logger.exception("GUI server crashed")
        return 1
    return 0


def main() -> int:
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
