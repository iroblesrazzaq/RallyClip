from __future__ import annotations

import json
import logging
import math
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
import csv
from datetime import datetime, timedelta
from importlib import metadata as importlib_metadata
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

try:
    from flask import Flask, jsonify, request, send_file
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "rallyclip gui requires Flask. Reinstall with `pip install .`."
    ) from exc

from runtime.assets import candidate_roots, resolve_asset
from runtime.defaults import build_gui_defaults
from runtime.paths import resolve_frontend_dir
from rallyclip_core.intervals import read_point_intervals
from rallyclip_core.library import SavedMatchStore, new_item_id
from rallyclip_core.playback import build_playback_manifest, playback_manifest_payload

JobDict = Dict[str, Any]
FIXED_YOLO_MODEL = "yolov8n-pose.pt"
GITHUB_REPO = "iroblesrazzaq/RallyClip"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases"
GITHUB_LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
UPDATE_CHECK_CACHE_SECONDS = 6 * 60 * 60

# Test seams and lazy runtime slots. These names intentionally exist at module
# import time so tests can monkeypatch the pipeline without importing Torch,
# Ultralytics, PyAV, or Numpy during replay-only startup.
PoseExtractionCancelled = None
PoseExtractor = None
FeatureEngineer = None
DataPreprocessor = None
extract_segments_from_binary = None
gaussian_filter1d = None
hysteresis_threshold = None
load_scaler_asset = None
load_model_from_checkpoint = None
run_windowed_inference_average_onnx_stream = None
run_windowed_inference_average_torch_stream = None
write_segments_csv = None
apply_pose_device = None
segment_video = None

_ANALYSIS_RUNTIME = None
_ANALYSIS_RUNTIME_LOCK = threading.Lock()
_RUNTIME_STATUS: Dict[str, Any] = {
    "state": "cold",
    "available_devices": ["cpu"],
    "auto_device": "cpu",
    "error": None,
    "loaded_at": None,
}
_RUNTIME_WARMUP_THREAD: Optional[threading.Thread] = None
_UPDATE_STATUS_CACHE: Dict[str, Any] = {"checked_at": 0.0, "payload": None}
_UPDATE_STATUS_LOCK = threading.Lock()


STATIC_DIR = resolve_frontend_dir()


def _frozen_data_root() -> Optional[Path]:
    """In packaged builds, keep user data out of the bundle and the CWD."""
    if getattr(sys, "frozen", False):
        return (Path.home() / "RallyClip").resolve()
    return None


def _default_jobs_dir() -> Path:
    """Pick a jobs/output root inside the RallyClip install if possible; fallback to CWD."""
    if os.environ.get("RALLYCLIP_JOBS_DIR"):
        return Path(os.environ["RALLYCLIP_JOBS_DIR"]).expanduser().resolve()
    frozen_root = _frozen_data_root()
    if frozen_root is not None:
        return frozen_root / "jobs"
    for root in candidate_roots():
        root_path = Path(root).resolve()
        if (root_path / "models").exists() or (root_path / "gui").exists():
            return (root_path / "RallyClipJobs").resolve()
    return (Path.cwd() / "RallyClipJobs").resolve()


def _default_output_dir() -> Path:
    if os.environ.get("RALLYCLIP_OUTPUT_DIR"):
        return Path(os.environ["RALLYCLIP_OUTPUT_DIR"]).expanduser().resolve()
    frozen_root = _frozen_data_root()
    if frozen_root is not None:
        return frozen_root / "output_videos"
    for root in candidate_roots():
        root_path = Path(root).resolve()
        if (root_path / "models").exists() or (root_path / "gui").exists():
            return (root_path / "output_videos").resolve()
    return (Path.cwd() / "output_videos").resolve()


def _default_csv_dir() -> Path:
    if os.environ.get("RALLYCLIP_CSV_DIR"):
        return Path(os.environ["RALLYCLIP_CSV_DIR"]).expanduser().resolve()
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
    if os.environ.get("RALLYCLIP_LIBRARY_DIR"):
        return Path(os.environ["RALLYCLIP_LIBRARY_DIR"]).expanduser().resolve()
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
PREFERENCES_PATH = (_frozen_data_root() or LIBRARY_DIR.parent) / "preferences.json"

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
            "pipeline_id": "frame_probability_hysteresis",
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
preview_job_errors: Dict[str, str] = {}
_MEMORY_PROCESS = None
active_preview_item_id: Optional[str] = None
last_preview_cache_prune = 0.0
PREVIEW_WINDOW_DURATION_S = 8.0
PREVIEW_WINDOW_MIN_DURATION_S = 5.0
PREVIEW_WINDOW_MAX_DURATION_S = 90.0
PREVIEW_WINDOW_WIDTH = 640
PREVIEW_WINDOW_FPS = 30
PREVIEW_CACHE_TTL_SECONDS = 24 * 60 * 60
PREVIEW_ACTIVE_CACHE_CAP_BYTES = 256 * 1024 * 1024
PREVIEW_GLOBAL_CACHE_CAP_BYTES = 1024 * 1024 * 1024
PREVIEW_TRANSCODE_CONCURRENCY = 2
NATIVE_PLAYBACK_PROXY_FILENAME = "playback_proxy.mp4"
preview_transcode_semaphore = threading.BoundedSemaphore(PREVIEW_TRANSCODE_CONCURRENCY)


class PipelineCancelled(Exception):
    """Raised when a job is cancelled mid-flight."""


def _load_av():
    import av  # noqa: WPS433 - heavy optional import, replay startup must avoid it

    return av


def _load_numpy():
    import numpy as np  # noqa: WPS433 - heavy optional import, analysis-only

    return np


def _load_video_validation_runtime():
    from runtime.video_validation import VideoValidationError, validate_video  # noqa: WPS433

    return SimpleNamespace(VideoValidationError=VideoValidationError, validate_video=validate_video)


def _load_segment_video():
    global segment_video
    if segment_video is not None:
        return segment_video
    from segmentation.segment import segment_video as loaded_segment_video  # noqa: WPS433

    segment_video = loaded_segment_video
    return loaded_segment_video


def _load_device_runtime():
    from runtime.device import (  # noqa: WPS433
        apply_pose_device as loaded_apply_pose_device,
        detect_available_devices,
        prefer_cpu_over_mps_for_pose,
        resolve_auto_device,
    )

    return SimpleNamespace(
        apply_pose_device=loaded_apply_pose_device,
        detect_available_devices=detect_available_devices,
        prefer_cpu_over_mps_for_pose=prefer_cpu_over_mps_for_pose,
        resolve_auto_device=resolve_auto_device,
    )


def _analysis_global(name: str, loader):
    value = globals().get(name)
    if value is not None:
        return value
    value = loader()
    globals()[name] = value
    return value


def _get_analysis_runtime() -> SimpleNamespace:
    """Load the analysis stack on demand, never during replay-only startup."""
    global _ANALYSIS_RUNTIME
    with _ANALYSIS_RUNTIME_LOCK:
        if _ANALYSIS_RUNTIME is not None:
            return _ANALYSIS_RUNTIME
        _set_runtime_status("warming", error=None)
        try:
            from extraction.pose_extractor import (  # noqa: WPS433
                PoseExtractionCancelled as loaded_pose_cancelled,
                PoseExtractor as loaded_pose_extractor,
            )
            from features.feature_engineer import FeatureEngineer as loaded_feature_engineer  # noqa: WPS433
            from infer import (  # noqa: WPS433
                extract_segments_from_binary as loaded_extract_segments_from_binary,
                gaussian_filter1d as loaded_gaussian_filter1d,
                hysteresis_threshold as loaded_hysteresis_threshold,
                load_model_from_checkpoint as loaded_load_model_from_checkpoint,
                load_scaler_asset as loaded_load_scaler_asset,
                run_windowed_inference_average_onnx_stream as loaded_onnx_stream,
                run_windowed_inference_average_torch_stream as loaded_torch_stream,
                write_segments_csv as loaded_write_segments_csv,
            )
            from preprocessing.data_preprocessor import DataPreprocessor as loaded_data_preprocessor  # noqa: WPS433

            device_runtime = _load_device_runtime()
            runtime = SimpleNamespace(
                np=_load_numpy(),
                PoseExtractionCancelled=_analysis_global("PoseExtractionCancelled", lambda: loaded_pose_cancelled),
                PoseExtractor=_analysis_global("PoseExtractor", lambda: loaded_pose_extractor),
                FeatureEngineer=_analysis_global("FeatureEngineer", lambda: loaded_feature_engineer),
                DataPreprocessor=_analysis_global("DataPreprocessor", lambda: loaded_data_preprocessor),
                extract_segments_from_binary=_analysis_global(
                    "extract_segments_from_binary",
                    lambda: loaded_extract_segments_from_binary,
                ),
                gaussian_filter1d=_analysis_global("gaussian_filter1d", lambda: loaded_gaussian_filter1d),
                hysteresis_threshold=_analysis_global("hysteresis_threshold", lambda: loaded_hysteresis_threshold),
                load_scaler_asset=_analysis_global("load_scaler_asset", lambda: loaded_load_scaler_asset),
                load_model_from_checkpoint=_analysis_global(
                    "load_model_from_checkpoint",
                    lambda: loaded_load_model_from_checkpoint,
                ),
                run_windowed_inference_average_onnx_stream=_analysis_global(
                    "run_windowed_inference_average_onnx_stream",
                    lambda: loaded_onnx_stream,
                ),
                run_windowed_inference_average_torch_stream=_analysis_global(
                    "run_windowed_inference_average_torch_stream",
                    lambda: loaded_torch_stream,
                ),
                write_segments_csv=_analysis_global("write_segments_csv", lambda: loaded_write_segments_csv),
                apply_pose_device=_analysis_global("apply_pose_device", lambda: device_runtime.apply_pose_device),
                device_runtime=device_runtime,
            )
            _ANALYSIS_RUNTIME = runtime
            _refresh_runtime_devices(runtime)
            return runtime
        except Exception as exc:
            _set_runtime_status("error", error=str(exc))
            raise


def _runtime_with_injected_globals(runtime: SimpleNamespace) -> SimpleNamespace:
    """Honor test/dev monkeypatches even if the runtime was warmed earlier."""
    for name in (
        "PoseExtractor",
        "FeatureEngineer",
        "DataPreprocessor",
        "extract_segments_from_binary",
        "gaussian_filter1d",
        "hysteresis_threshold",
        "load_scaler_asset",
        "load_model_from_checkpoint",
        "run_windowed_inference_average_onnx_stream",
        "run_windowed_inference_average_torch_stream",
        "write_segments_csv",
        "apply_pose_device",
    ):
        value = globals().get(name)
        if value is not None:
            setattr(runtime, name, value)
    return runtime


def _set_runtime_status(state: str, *, error: Optional[str] = None, **extra: Any) -> None:
    _RUNTIME_STATUS.update({"state": state, "error": error, **extra})


def _refresh_runtime_devices(runtime: Optional[SimpleNamespace] = None) -> None:
    runtime = runtime or _get_analysis_runtime()
    devices = runtime.device_runtime.detect_available_devices()
    auto_device = runtime.device_runtime.prefer_cpu_over_mps_for_pose(
        runtime.device_runtime.resolve_auto_device(),
        _resolve_yolo_weights(DEFAULT_CONFIG),
        warn=False,
    )
    _set_runtime_status(
        "ready",
        error=None,
        available_devices=devices,
        auto_device=auto_device,
        loaded_at=time.time(),
    )


def _runtime_warmup_target() -> None:
    try:
        result = subprocess.run(
            _analysis_warmup_command(),
            cwd=str(Path(__file__).resolve().parents[2]),
            env=_analysis_worker_env(),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        status = None
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "runtime_status":
                status = event.get("status")
        if result.returncode == 0 and isinstance(status, dict):
            _RUNTIME_STATUS.update(status)
        else:
            error = result.stderr.strip() or f"Analysis warmup exited with status {result.returncode}."
            _set_runtime_status("error", error=error)
    except Exception as exc:
        _set_runtime_status("error", error=str(exc))
        logging.warning("Could not warm analysis runtime", exc_info=True)


def _start_runtime_warmup() -> None:
    global _RUNTIME_WARMUP_THREAD
    if _RUNTIME_STATUS.get("state") == "ready":
        return
    if _RUNTIME_WARMUP_THREAD is not None and _RUNTIME_WARMUP_THREAD.is_alive():
        return
    _set_runtime_status("warming", error=None)
    _RUNTIME_WARMUP_THREAD = threading.Thread(
        target=_runtime_warmup_target,
        name="rallyclip-analysis-warmup",
        daemon=True,
    )
    _RUNTIME_WARMUP_THREAD.start()


def _ensure_job_dir(job_id: str) -> Path:
    job_dir = (JOBS_DIR / job_id).resolve()
    jobs_root = JOBS_DIR.resolve()
    if jobs_root not in job_dir.parents and job_dir != jobs_root:
        raise ValueError(f"Invalid job id: {job_id!r}")
    return job_dir


def _library_store() -> SavedMatchStore:
    """Store over the current LIBRARY_DIR (read lazily so tests can patch it)."""
    return SavedMatchStore(root=Path(LIBRARY_DIR))


def _new_library_id() -> str:
    return new_item_id()


def _library_item_dir(item_id: str) -> Path:
    return _library_store().item_dir(item_id)


def _write_thumbnail(video_path: Path, thumb_path: Path, max_width: int = 480) -> bool:
    """Grab the first frame of the segmented video as a JPEG thumbnail. cv2 is
    imported lazily (it's already loaded by court detection during a job)."""
    try:
        import cv2  # lazy: avoids the libGL import at GUI startup
        av = _load_av()

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

    av = _load_av()
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
    source_duration = _estimate_duration_seconds(source_path)
    chunk_s = PREVIEW_WINDOW_DURATION_S
    safe_start = max(0.0, start_s)
    if source_duration > 0:
        safe_start = min(safe_start, max(0.0, source_duration - 0.1))
    start = int(safe_start // chunk_s) * chunk_s
    requested_duration = chunk_s
    requested_end = start + chunk_s
    if duration_s is not None and duration_s > 0:
        requested_duration = max(PREVIEW_WINDOW_MIN_DURATION_S, min(float(duration_s), PREVIEW_WINDOW_MAX_DURATION_S))
        requested_end = safe_start + requested_duration
    duration = max(chunk_s, math.ceil(max(0.1, requested_end - start) / chunk_s) * chunk_s)
    duration = min(duration, PREVIEW_WINDOW_MAX_DURATION_S)
    if source_duration > 0:
        duration = min(duration, max(0.1, source_duration - start))
    return round(start, 3), round(duration, 3)


def _preview_window_path(item_id: str, start_s: float, duration_s: float) -> Path:
    start_ms = int(round(start_s * 1000))
    duration_ms = int(round(duration_s * 1000))
    return _preview_windows_dir(item_id) / f"{start_ms:012d}_{duration_ms:06d}.webm"


def _preview_window_job_key(item_id: str, start_s: float, duration_s: float) -> str:
    return f"{item_id}:window:{int(round(start_s * 1000))}:{int(round(duration_s * 1000))}"


def _preview_window_dependency_mtime(item_id: str, source_path: Path) -> float:
    mtime = source_path.stat().st_mtime
    csv_path = _resolve_library_file(item_id, "segments.csv")
    if csv_path is not None:
        mtime = max(mtime, csv_path.stat().st_mtime)
    return mtime


def _preview_window_ready(item_id: str, source_path: Path, start_s: float, duration_s: float) -> bool:
    preview_path = _preview_window_path(item_id, start_s, duration_s)
    return preview_path.exists() and preview_path.stat().st_mtime >= _preview_window_dependency_mtime(item_id, source_path)


def _preview_window_state(item_id: str, source_path: Path, start_s: float, duration_s: float) -> str:
    if _preview_window_ready(item_id, source_path, start_s, duration_s):
        key = _preview_window_job_key(item_id, start_s, duration_s)
        with jobs_lock:
            preview_jobs.pop(key, None)
            preview_job_errors.pop(key, None)
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
                preview_job_errors.pop(key, None)
        for key in list(preview_job_errors):
            if key == item_id or key.startswith(prefix):
                preview_job_errors.pop(key, None)


def _is_preview_cache_active(item_id: str) -> bool:
    with jobs_lock:
        active = active_preview_item_id
    return active is None or active == item_id


def _is_preview_temp_file(path: Path) -> bool:
    return path.name.endswith(".tmp") or path.name.endswith(".tmp.webm")


def _is_preview_cache_file(path: Path) -> bool:
    return path.is_file() and (
        path.suffix in {".webm", ".tmp"} or _is_preview_temp_file(path)
    )


def _unlink_preview_cache_file(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception:
        logging.debug("Could not prune preview cache file %s", path, exc_info=True)
    return False


def _touch_preview_cache_file(path: Path) -> None:
    try:
        if path.exists():
            os.utime(path, None)
    except Exception:
        logging.debug("Could not touch preview cache file %s", path, exc_info=True)


def _unlink_if_old(path: Path, cutoff: float) -> bool:
    try:
        if path.exists() and path.stat().st_mtime < cutoff:
            return _unlink_preview_cache_file(path)
    except Exception:
        logging.debug("Could not stat preview cache file %s", path, exc_info=True)
    return False


def _preview_cache_records(item_dirs: list[Path]) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for item_dir in item_dirs:
        item_id = item_dir.name
        candidates = [item_dir / "preview.webm"]
        windows_dir = item_dir / "preview_windows"
        if windows_dir.exists():
            try:
                candidates.extend(child for child in windows_dir.iterdir())
            except Exception:
                logging.debug("Could not scan preview windows %s", windows_dir, exc_info=True)
        for path in candidates:
            try:
                if not _is_preview_cache_file(path):
                    continue
                stat = path.stat()
            except Exception:
                continue
            records.append(
                {
                    "path": path,
                    "item_id": item_id,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "is_tmp": _is_preview_temp_file(path),
                }
            )
    return records


def _enforce_preview_cache_cap(
    records: list[Dict[str, Any]],
    cap_bytes: int,
    removed_item_ids: set[str],
    *,
    protected_item_id: Optional[str] = None,
) -> None:
    cap = max(0, int(cap_bytes))
    total = sum(int(record["size"]) for record in records if Path(record["path"]).exists())
    ordered = sorted(records, key=lambda rec: (rec["mtime"], str(rec["path"])))
    for allow_protected in (False, True):
        for record in ordered:
            if total <= cap:
                return
            if protected_item_id is not None and record["item_id"] == protected_item_id and not allow_protected:
                continue
            path = Path(record["path"])
            if not path.exists():
                continue
            size = int(record["size"])
            if _unlink_preview_cache_file(path):
                total -= size
                removed_item_ids.add(str(record["item_id"]))


def _cleanup_empty_preview_dirs(item_dirs: list[Path]) -> None:
    for item_dir in item_dirs:
        windows_dir = item_dir / "preview_windows"
        if not windows_dir.exists():
            continue
        try:
            if not any(windows_dir.iterdir()):
                windows_dir.rmdir()
        except Exception:
            logging.debug("Could not remove empty preview cache dir %s", windows_dir, exc_info=True)


def _prune_preview_cache(active_item_id: Optional[str], *, now: Optional[float] = None) -> None:
    if not LIBRARY_DIR.exists():
        return
    cutoff = (time.time() if now is None else now) - PREVIEW_CACHE_TTL_SECONDS
    try:
        children = [child for child in LIBRARY_DIR.iterdir() if child.is_dir()]
    except Exception:
        logging.debug("Could not scan preview cache root %s", LIBRARY_DIR, exc_info=True)
        return

    removed_item_ids: set[str] = set()
    for record in _preview_cache_records(children):
        path = Path(record["path"])
        item_id = str(record["item_id"])
        if active_item_id is not None and item_id != active_item_id and record["is_tmp"]:
            if _unlink_preview_cache_file(path):
                removed_item_ids.add(item_id)
            continue
        if _unlink_if_old(path, cutoff):
            removed_item_ids.add(item_id)

    records = [record for record in _preview_cache_records(children) if Path(record["path"]).exists()]
    if active_item_id is not None:
        active_records = [record for record in records if record["item_id"] == active_item_id]
        _enforce_preview_cache_cap(active_records, PREVIEW_ACTIVE_CACHE_CAP_BYTES, removed_item_ids)

    records = [record for record in _preview_cache_records(children) if Path(record["path"]).exists()]
    _enforce_preview_cache_cap(
        records,
        PREVIEW_GLOBAL_CACHE_CAP_BYTES,
        removed_item_ids,
        protected_item_id=active_item_id,
    )
    _cleanup_empty_preview_dirs(children)

    for item_id in removed_item_ids:
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
    dependency_mtime = _preview_window_dependency_mtime(item_id, source_path)
    if preview_path.exists() and preview_path.stat().st_mtime >= dependency_mtime:
        return preview_path
    lock = _preview_lock(_preview_window_job_key(item_id, start_s, duration_s))
    with lock:
        dependency_mtime = _preview_window_dependency_mtime(item_id, source_path)
        if preview_path.exists() and preview_path.stat().st_mtime >= dependency_mtime:
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
        preview_job_errors.pop(key, None)
    try:
        preview_path = _ensure_preview_window(item_id, source_path, start_s, duration_s)
        if not _is_preview_cache_active(item_id):
            try:
                preview_path.unlink(missing_ok=True)
            finally:
                with jobs_lock:
                    preview_jobs.pop(key, None)
                    preview_job_errors.pop(key, None)
            return
        with jobs_lock:
            preview_jobs[key] = "ready"
            preview_job_errors.pop(key, None)
    except Exception as exc:
        with jobs_lock:
            preview_jobs[key] = "error"
            preview_job_errors[key] = f"Could not prepare preview window: {exc}"
        logging.warning("Could not prepare preview window for %s at %.3fs", item_id, start_s, exc_info=True)


def _start_preview_window_background(item_id: str, source_path: Path, start_s: float, duration_s: float) -> str:
    state = _preview_window_state(item_id, source_path, start_s, duration_s)
    if state in {"ready", "processing", "error"}:
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
        with jobs_lock:
            preview_jobs.pop(item_id, None)
            preview_job_errors.pop(item_id, None)
        return "ready"
    with jobs_lock:
        return preview_jobs.get(item_id, "missing")


def _prepare_web_preview_background(item_id: str, source_path: Path) -> None:
    with jobs_lock:
        preview_jobs[item_id] = "processing"
        preview_job_errors.pop(item_id, None)
    try:
        preview_path = _ensure_web_preview(item_id, source_path)
        if not _is_preview_cache_active(item_id):
            try:
                preview_path.unlink(missing_ok=True)
            finally:
                with jobs_lock:
                    preview_jobs.pop(item_id, None)
                    preview_job_errors.pop(item_id, None)
            return
        with jobs_lock:
            preview_jobs[item_id] = "ready"
            preview_job_errors.pop(item_id, None)
    except Exception as exc:
        with jobs_lock:
            preview_jobs[item_id] = "error"
            preview_job_errors[item_id] = f"Could not prepare preview: {exc}"
        logging.warning("Could not prepare web preview for %s", item_id, exc_info=True)


def _start_web_preview_background(item_id: str, source_path: Path) -> str:
    state = _web_preview_state(item_id, source_path)
    if state in {"ready", "processing", "error"}:
        return state
    threading.Thread(
        target=_prepare_web_preview_background,
        args=(item_id, source_path),
        daemon=True,
        name=f"rallyclip-preview-{item_id}",
    ).start()
    return "processing"


def _read_library_items() -> list[Dict[str, Any]]:
    return _library_store().list_items()


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
        _get_analysis_runtime().write_segments_csv(segments, str(csv_out), fps=fps, overwrite=True)
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
        "process": None,
    }


def _set_step(job: JobDict, step: str, status: str, progress: int) -> None:
    job["steps"][step]["status"] = status
    job["steps"][step]["progress"] = int(max(0, min(100, progress)))


def _check_cancel(job: JobDict) -> None:
    if job.get("cancelled"):
        job["status"] = "cancelled"
        raise PipelineCancelled("Job cancelled")


def _job_json_copy(job: JobDict) -> JobDict:
    payload = {key: value for key, value in job.items() if key not in {"thread", "process"}}
    return json.loads(json.dumps(payload, default=str))


def _analysis_worker_env() -> Dict[str, str]:
    env = dict(os.environ)
    src_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src_root) if not existing_pythonpath else f"{src_root}{os.pathsep}{existing_pythonpath}"
    env["RALLYCLIP_JOBS_DIR"] = str(JOBS_DIR)
    env["RALLYCLIP_OUTPUT_DIR"] = str(DEFAULT_OUTPUT_DIR)
    env["RALLYCLIP_CSV_DIR"] = str(DEFAULT_CSV_DIR)
    env["RALLYCLIP_LIBRARY_DIR"] = str(LIBRARY_DIR)
    env.setdefault("MPLCONFIGDIR", "/private/tmp/rallyclip-matplotlib")
    return env


def _analysis_worker_command(job_path: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--analysis-worker", str(job_path)]
    return [sys.executable, "-m", "gui.analysis_worker", str(job_path)]


def _analysis_warmup_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--analysis-worker", "--warmup"]
    return [sys.executable, "-m", "gui.analysis_worker", "--warmup"]


def _merge_worker_job(job_id: str, worker_job: Dict[str, Any]) -> None:
    with jobs_lock:
        current = jobs.get(job_id)
        if current is None:
            return
        thread = current.get("thread")
        process = current.get("process")
        cancelled = bool(current.get("cancelled"))
        current.clear()
        current.update(worker_job)
        current["thread"] = thread
        current["process"] = process
        if cancelled:
            # Cancellation is decided parent-side; a late worker snapshot
            # (still "in_progress") must not resurrect the job, or the
            # non-zero exit from terminate() would flip it to "failed".
            current["cancelled"] = True
            current["status"] = "cancelled"


def _run_pipeline_in_worker_process(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        job_payload = _job_json_copy(job)

    job_dir = Path(job_payload["paths"]["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "analysis_job.json"
    stderr_path = job_dir / "analysis_worker.stderr.log"
    job_path.write_text(json.dumps(job_payload), encoding="utf-8")

    process: Optional[subprocess.Popen[str]] = None
    with stderr_path.open("w", encoding="utf-8") as stderr_fh:
        try:
            process = subprocess.Popen(
                _analysis_worker_command(job_path),
                cwd=str(Path(__file__).resolve().parents[2]),
                env=_analysis_worker_env(),
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
                text=True,
                bufsize=1,
            )
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["process"] = process
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning("Ignoring non-JSON analysis worker output: %s", line[:200])
                    continue
                worker_job = event.get("job")
                if isinstance(worker_job, dict):
                    _merge_worker_job(job_id, worker_job)
            return_code = process.wait()
            with jobs_lock:
                current = jobs.get(job_id)
                if current is not None:
                    current["process"] = process
                    if return_code != 0 and current.get("status") == "in_progress":
                        current["status"] = "failed"
                        current["error"] = f"Analysis worker exited with status {return_code}."
        except Exception as exc:
            logging.exception("Analysis worker failed for %s", job_id)
            with jobs_lock:
                current = jobs.get(job_id)
                if current is not None and current.get("status") != "cancelled":
                    current["status"] = "failed"
                    current["error"] = str(exc)
        finally:
            if process is not None:
                with jobs_lock:
                    current = jobs.get(job_id)
                    if current is not None:
                        current["process"] = process


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


def _api_services():
    from rallyclip_api import RallyClipServices  # noqa: WPS433

    return RallyClipServices(
        defaults_provider=lambda: {**DEFAULT_CONFIG},
        runtime_status_provider=lambda: dict(_RUNTIME_STATUS),
        runtime_warmup=_start_runtime_warmup,
        start_job_handler=_start_analysis_job,
        job_status_provider=_job_status_payload,
        cancel_job_handler=_cancel_analysis_job,
        export_handler=_export_library_video,
        playback_manifest_provider=_library_playback_manifest_payload,
        saved_match_store=_library_store(),
    )


# Keys the browser may override. Everything else (manifest-pinned inference
# params, model/artifact paths) keeps the server-side default even though the
# frontend round-trips the full defaults payload.
# write_csv/segment_video are deliberately absent: the GUI job always runs the
# engine with both off (the library owns the CSV; exports are cut lazily), so
# advertising them as client-controllable would be a lie.
_CLIENT_KEYS = {
    "output_name",
    "pipeline_id",
    "yolo_device",
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
        av = _load_av()
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
    pose_cancelled_type = None
    try:
        from rallyclip_core.contracts import ProgressEvent, RunRequest, RuntimeDeps  # noqa: WPS433
        from rallyclip_engine import run_analysis  # noqa: WPS433

        runtime = _runtime_with_injected_globals(_get_analysis_runtime())
        pose_cancelled_type = runtime.PoseExtractionCancelled
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
        model_path, scaler_path = _resolve_model_paths(cfg)
        models_dir = None
        frozen_root = _frozen_data_root()
        if frozen_root is not None:
            models_dir = frozen_root / "models"
        else:
            for root in candidate_roots():
                candidate = Path(root) / "models"
                if candidate.exists():
                    models_dir = candidate.resolve()
                    break

        def progress(event: ProgressEvent) -> None:
            nonlocal last_memory_log
            _set_step(job, event.stage, event.status, event.progress)
            if event.stage == "pose" and event.metadata:
                meta = event.metadata
                frames_seen = meta.get("frames_seen", meta.get("frames_done", 0))
                frames_total = meta.get("frames_total", 1)
                # prefer FPS derived from frames_seen to mirror tqdm ETA
                smoothed_fps = max(1e-3, meta.get("smoothed_seen_fps", meta.get("smoothed_proc_fps", 0.0)))
                pose_eta = max(0, frames_total - frames_seen) / smoothed_fps
                # Tail buffer: 10s minimum, 60s max, scaled by minutes
                tail = max(10.0, min(60.0, (duration_seconds / 60.0) * 5.0))
                job["eta_seconds"] = pose_eta + tail
                job["pose_eta_seconds"] = pose_eta
                job["pose_throughput_fps"] = smoothed_fps
                now = time.perf_counter()
                if now - last_memory_log >= 10.0 or event.progress >= 99:
                    last_memory_log = now
                    _log_memory(
                        "pose_progress",
                        job_id=job_id,
                        elapsed_s=now - pipeline_start,
                        frames_seen=frames_seen,
                        frames_total=frames_total,
                        progress=event.progress,
                        pose_fps=smoothed_fps,
                    )

        _check_cancel(job)
        request = RunRequest(
            video_path=upload_path,
            output_dir=DEFAULT_OUTPUT_DIR,
            output_name=base_name,
            csv_output_dir=DEFAULT_CSV_DIR,
            write_csv=False,
            segment_video=False,
            yolo_weights=yolo_weights,
            yolo_device=cfg.get("yolo_device"),
            model_path=model_path,
            scaler_path=scaler_path,
            pipeline_id=cfg.get("pipeline_id"),
            fps=float(cfg["fps"]),
            seq_len=int(cfg["seq_len"]),
            imgsz=int(cfg["imgsz"]),
            conf=float(cfg["conf"]),
            feature_set=str(cfg.get("feature_set", "v1")),
            screen_width=int(cfg["screen_width"]),
            screen_height=int(cfg["screen_height"]),
            overlap=int(cfg["overlap"]),
            sigma=float(cfg["sigma"]),
            low=float(cfg["low"]),
            high=float(cfg["high"]),
            min_dur_sec=float(cfg["min_dur_sec"]),
            start_time=int(cfg["start_time"]),
            duration=int(cfg["duration"]),
            models_dir=models_dir,
            estimated_duration_s=duration_seconds,
        )
        deps = RuntimeDeps(
            np=runtime.np,
            PoseExtractor=runtime.PoseExtractor,
            DataPreprocessor=runtime.DataPreprocessor,
            FeatureEngineer=runtime.FeatureEngineer,
            load_scaler_asset=runtime.load_scaler_asset,
            load_model_from_checkpoint=runtime.load_model_from_checkpoint,
            run_windowed_inference_average_onnx_stream=runtime.run_windowed_inference_average_onnx_stream,
            run_windowed_inference_average_torch_stream=runtime.run_windowed_inference_average_torch_stream,
            gaussian_filter1d=runtime.gaussian_filter1d,
            hysteresis_threshold=runtime.hysteresis_threshold,
            extract_segments_from_binary=runtime.extract_segments_from_binary,
            write_segments_csv=runtime.write_segments_csv,
            segment_video=_load_segment_video(),
            apply_pose_device=runtime.apply_pose_device,
        )
        result = run_analysis(request, deps=deps, progress_callback=progress, cancel_check=lambda: _check_cancel(job))
        segments = result.frame_segments

        _check_cancel(job)
        _set_step(job, "output", "in_progress", 5)
        intervals_sec = result.intervals_sec
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
        if pose_cancelled_type is not None and isinstance(exc, pose_cancelled_type):
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
            return
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


def _read_preferences() -> Dict[str, Any]:
    if not PREFERENCES_PATH.exists():
        return {}
    try:
        data = json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("Could not read preferences from %s", PREFERENCES_PATH, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _write_preferences(preferences: Dict[str, Any]) -> None:
    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = PREFERENCES_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(preferences, indent=2), encoding="utf-8")
    tmp_path.replace(PREFERENCES_PATH)


def _read_pyproject_version() -> str:
    for root in candidate_roots():
        pyproject = Path(root) / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            version = data.get("project", {}).get("version")
        except Exception:
            continue
        if version:
            return str(version)
    return "0.1.0"


def current_app_version() -> str:
    try:
        return importlib_metadata.version("rallyclip")
    except importlib_metadata.PackageNotFoundError:
        return _read_pyproject_version()


def _version_parts(version: str) -> tuple[int, ...]:
    normalized = str(version or "").strip().lower()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    parts = []
    for piece in re.split(r"[.\-+_]", normalized):
        if not piece:
            continue
        match = re.match(r"(\d+)", piece)
        if match is None:
            break
        parts.append(int(match.group(1)))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def is_newer_version(latest: str, current: str) -> bool:
    return _version_parts(latest) > _version_parts(current)


def _fetch_latest_release() -> Dict[str, Any]:
    request_obj = Request(
        GITHUB_LATEST_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"RallyClip/{current_app_version()}",
        },
    )
    with urlopen(request_obj, timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8"))
    tag = str(payload.get("tag_name") or "").strip()
    return {
        "latest_version": tag[1:] if tag.startswith("v") else tag,
        "latest_tag": tag,
        "release_url": payload.get("html_url") or GITHUB_RELEASES_URL,
        "release_name": payload.get("name") or tag,
    }


def update_status_payload(*, force: bool = False) -> Dict[str, Any]:
    now = time.time()
    with _UPDATE_STATUS_LOCK:
        cached = _UPDATE_STATUS_CACHE.get("payload")
        checked_at = float(_UPDATE_STATUS_CACHE.get("checked_at") or 0.0)
        if (
            not force
            and isinstance(cached, dict)
            and now - checked_at < UPDATE_CHECK_CACHE_SECONDS
        ):
            return dict(cached)

    current = current_app_version()
    payload: Dict[str, Any] = {
        "current_version": current,
        "latest_version": None,
        "latest_tag": None,
        "update_available": False,
        "release_url": GITHUB_RELEASES_URL,
        "release_name": None,
        "checked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "error": None,
    }
    try:
        latest = _fetch_latest_release()
        payload.update(latest)
        payload["update_available"] = bool(
            payload.get("latest_version")
            and is_newer_version(str(payload["latest_version"]), current)
        )
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        payload["error"] = str(exc)
        logging.info("Could not check for RallyClip updates: %s", exc)

    with _UPDATE_STATUS_LOCK:
        _UPDATE_STATUS_CACHE["checked_at"] = now
        _UPDATE_STATUS_CACHE["payload"] = dict(payload)
    return payload


@app.route("/api/preferences/welcome", methods=["GET", "POST"])
def welcome_preferences() -> tuple[Any, int]:
    preferences = _read_preferences()
    if request.method == "POST":
        preferences["welcome_seen"] = True
        try:
            _write_preferences(preferences)
        except Exception as exc:
            logging.warning("Could not write welcome preference", exc_info=True)
            return jsonify({"error": str(exc)}), 500
    return jsonify({"welcome_seen": bool(preferences.get("welcome_seen"))}), 200


@app.route("/api/update/status", methods=["GET"])
def update_status() -> tuple[Any, int]:
    force = request.args.get("force", "").strip().lower() in {"1", "true", "yes"}
    return jsonify(update_status_payload(force=force)), 200


@app.route("/api/update/open", methods=["POST"])
def open_update_page() -> tuple[Any, int]:
    webbrowser.open(GITHUB_RELEASES_URL)
    return jsonify({"opened": True, "release_url": GITHUB_RELEASES_URL}), 200


@app.route("/api/config/defaults", methods=["GET"])
def config_defaults() -> tuple[Any, int]:
    services = _api_services()
    defaults = services.get_defaults()
    # Server-internal values; the frontend never edits these (see _CLIENT_KEYS)
    # and absolute server paths don't belong in the browser payload.
    for key in ("model_path", "artifact_dir", "scaler_path", "yolo_weights"):
        defaults.pop(key, None)
    return jsonify(
        {
            "defaults": defaults,
            "yolo_model": FIXED_YOLO_MODEL,
            "warnings": ADVANCED_WARNINGS,
            "available_devices": _RUNTIME_STATUS.get("available_devices", ["cpu"]),
            "auto_device": _RUNTIME_STATUS.get("auto_device", "cpu"),
            "runtime_state": _RUNTIME_STATUS.get("state", "cold"),
        }
    ), 200


@app.route("/api/runtime/status", methods=["GET"])
def runtime_status() -> tuple[Any, int]:
    return jsonify(_api_services().get_runtime_status()), 200


@app.route("/api/runtime/warmup", methods=["POST"])
def runtime_warmup() -> tuple[Any, int]:
    return jsonify(_api_services().warmup_runtime()), 202


def _start_analysis_job(upload_path: Path, cfg: Dict[str, Any]) -> str:
    """Validate the uploaded video and spawn the analysis worker.

    Raises ValueError (with a user-facing message) when validation fails; the
    job directory containing upload_path is removed on failure.
    """
    job_id = upload_path.parent.name
    # Preflight before spawning the worker so bad input is rejected immediately,
    # not surfaced as a "failed" job minutes later.
    validation = _load_video_validation_runtime()
    try:
        validation.validate_video(upload_path, seq_len=int(cfg["seq_len"]), fps=float(cfg["fps"]))
    except validation.VideoValidationError as exc:
        shutil.rmtree(upload_path.parent, ignore_errors=True)
        raise ValueError(str(exc)) from exc

    state = _new_job_state(job_id, cfg)
    state["paths"]["upload"] = str(upload_path)
    worker = threading.Thread(target=_run_pipeline_in_worker_process, args=(job_id,), daemon=True)
    state["thread"] = worker
    with jobs_lock:
        jobs[job_id] = state
    worker.start()
    return job_id


def _job_status_payload(job_id: str) -> Optional[Dict[str, Any]]:
    """Client-facing job progress payload, or None for an unknown job."""
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return None
    return {
        "status": job["status"],
        "steps": job["steps"],
        "error": job.get("error"),
        "weights": job.get("weights"),
        "eta_seconds": job.get("eta_seconds"),
        "pose_eta_seconds": job.get("pose_eta_seconds"),
        "pose_throughput_fps": job.get("pose_throughput_fps"),
        "library_id": job.get("library_id"),
    }


def _cancel_analysis_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Cancel a running job, or None for an unknown job. Idempotent."""
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return None
    if job["status"] == "in_progress":
        job["cancelled"] = True
        job["status"] = "cancelled"
        process = job.get("process")
        if process is not None and getattr(process, "poll", lambda: None)() is None:
            try:
                process.terminate()
            except Exception:
                logging.debug("Could not terminate analysis worker for %s", job_id, exc_info=True)
    return {"status": job["status"]}


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

    try:
        _api_services().start_job(upload_path, cfg)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"job_id": job_id}), 200


@app.route("/api/progress/<job_id>", methods=["GET"])
def get_progress(job_id: str):
    payload = _api_services().get_job_status(job_id)
    if payload is None:
        return jsonify({"error": "Unknown job id"}), 404
    return jsonify(payload), 200


@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id: str):
    payload = _api_services().cancel_job(job_id)
    if payload is None:
        return jsonify({"error": "Unknown job id"}), 404
    return jsonify(payload), 200


@app.route("/api/library", methods=["GET"])
def library_list():
    return jsonify(_api_services().list_library()), 200


def _resolve_library_file(item_id: str, filename: str) -> Optional[Path]:
    return _library_store().resolve_file(item_id, filename)


def _resolve_library_source(item_id: str) -> Optional[Path]:
    return _library_store().resolve_source(item_id)


def _read_library_meta(item_dir: Path) -> Dict[str, Any]:
    return _library_store().read_meta(item_dir)


def _sorted_point_intervals(csv_path: Optional[Path]) -> list[tuple[float, float]]:
    return read_point_intervals(csv_path)


def _native_playback_proxy_path(item_id: str) -> Path:
    return _library_item_dir(item_id) / NATIVE_PLAYBACK_PROXY_FILENAME


def _native_playback_proxy_ready(source_path: Path, proxy_path: Path) -> bool:
    if not proxy_path.exists():
        return False
    try:
        return proxy_path.stat().st_mtime >= source_path.stat().st_mtime
    except OSError:
        return False


def _native_playback_proxy_state(source_path: Path, proxy_path: Path) -> Dict[str, Any]:
    ready = _native_playback_proxy_ready(source_path, proxy_path)
    return {
        "ready": ready,
        "state": "ready" if ready else "missing",
        "path": str(proxy_path) if ready else None,
    }


def native_playback_descriptor(item_id: str) -> Dict[str, Any]:
    """Return trusted desktop-only playback data for a saved match.

    This helper is intentionally not exposed as an HTTP endpoint: the desktop
    bridge can read local file paths, but browser JavaScript should not.
    """
    item_dir = _library_item_dir(item_id)
    source_path = _resolve_library_source(item_id)
    if source_path is None:
        raise FileNotFoundError("Video not available")
    csv_path = item_dir / "segments.csv"
    meta = _read_library_meta(item_dir)
    intervals = _sorted_point_intervals(csv_path if csv_path.exists() else None)
    try:
        source_duration = float(meta.get("duration_s") or 0.0)
    except (TypeError, ValueError):
        source_duration = 0.0
    if source_duration <= 0:
        source_duration = _estimate_duration_seconds(source_path)
    proxy_path = item_dir / NATIVE_PLAYBACK_PROXY_FILENAME
    manifest = build_playback_manifest(
        source_duration_s=source_duration,
        chunk_duration_s=PREVIEW_WINDOW_DURATION_S,
        point_intervals=intervals,
    )
    manifest_payload = playback_manifest_payload(manifest)
    return {
        "id": item_id,
        "name": str(meta.get("name") or item_id),
        "source_name": meta.get("source_name"),
        "created": meta.get("created"),
        "source_path": str(source_path),
        "source_duration_s": manifest_payload["source_duration_s"],
        "point_intervals": manifest_payload["point_intervals"],
        "point_duration_s": manifest_payload["point_duration_s"],
        "has_csv": csv_path.exists(),
        "has_export": (item_dir / "export.mp4").exists(),
        "csv_url": f"/api/library/{item_id}/csv" if csv_path.exists() else None,
        "export_url": f"/api/library/{item_id}/video",
        "proxy": _native_playback_proxy_state(source_path, proxy_path),
    }


def _native_playback_proxy_command(source_path: Path, output_path: Path) -> list[str]:
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError(
            "Native playback failed and ffmpeg is not available to prepare a playback proxy."
        )
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "scale='min(1280,iw)':-2:force_original_aspect_ratio=decrease,fps=30",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-g",
        "30",
        "-keyint_min",
        "30",
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _write_native_playback_proxy(source_path: Path, proxy_path: Path) -> Path:
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = proxy_path.with_name(f"{proxy_path.stem}.tmp{proxy_path.suffix}")
    tmp_path.unlink(missing_ok=True)
    command = _native_playback_proxy_command(source_path, tmp_path)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        tmp_path.replace(proxy_path)
        return proxy_path
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"Could not prepare playback proxy: {stderr or exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def ensure_native_playback_proxy(item_id: str) -> Dict[str, Any]:
    source_path = _resolve_library_source(item_id)
    if source_path is None:
        raise FileNotFoundError("Video not available")
    proxy_path = _native_playback_proxy_path(item_id)
    if not _native_playback_proxy_ready(source_path, proxy_path):
        _write_native_playback_proxy(source_path, proxy_path)
    return _native_playback_proxy_state(source_path, proxy_path)


def _library_playback_manifest_payload(item_id: str) -> Dict[str, Any]:
    source_path = _resolve_library_source(item_id)
    csv_path = _resolve_library_file(item_id, "segments.csv")
    if source_path is None:
        raise FileNotFoundError("Video not available")
    if csv_path is None:
        raise FileNotFoundError("CSV not available")
    intervals = _sorted_point_intervals(csv_path)
    meta = _read_library_meta(_library_item_dir(item_id))
    try:
        source_duration = float(meta.get("duration_s") or 0.0)
    except (TypeError, ValueError):
        source_duration = 0.0
    if source_duration <= 0:
        source_duration = _estimate_duration_seconds(source_path)
    manifest = build_playback_manifest(
        source_duration_s=source_duration,
        chunk_duration_s=PREVIEW_WINDOW_DURATION_S,
        point_intervals=intervals,
    )
    return playback_manifest_payload(manifest)


@app.route("/api/library/<item_id>/thumbnail", methods=["GET"])
def library_thumbnail(item_id: str):
    path = _resolve_library_file(item_id, "thumb.jpg")
    if path is None:
        return jsonify({"error": "Thumbnail not available"}), 404
    return send_file(str(path), mimetype="image/jpeg")


_export_locks: Dict[str, threading.Lock] = {}
_export_locks_guard = threading.Lock()


def _export_lock(item_id: str) -> threading.Lock:
    with _export_locks_guard:
        return _export_locks.setdefault(item_id, threading.Lock())


def _export_library_video(item_id: str) -> Path:
    """Return the downloadable cut video for a saved match, generating it lazily.

    Raises FileNotFoundError when the item/source/CSV/intervals are missing and
    ValueError for invalid ids or interval data.
    """
    source_path = _resolve_library_source(item_id)
    if source_path is None:
        raise FileNotFoundError("Video not available")
    csv_path = _resolve_library_file(item_id, "segments.csv")
    if csv_path is None:
        # Legacy items may only have an already-cut video.mp4.
        if source_path.name == "video.mp4":
            return source_path
        raise FileNotFoundError("CSV not available")

    item_dir = _library_item_dir(item_id)
    export_path = item_dir / "export.mp4"
    # Serialize per item so simultaneous export requests don't both run the
    # slow re-encode; the loser of the race re-checks freshness and reuses.
    with _export_lock(item_id):
        needs_export = not export_path.exists()
        if not needs_export:
            export_mtime = export_path.stat().st_mtime
            needs_export = source_path.stat().st_mtime > export_mtime or csv_path.stat().st_mtime > export_mtime
        if needs_export:
            intervals = _sorted_point_intervals(csv_path)
            if not intervals:
                raise FileNotFoundError("No point intervals available")
            _load_segment_video()(str(source_path), intervals, str(export_path))
    return export_path


@app.route("/api/library/<item_id>/video", methods=["GET"])
def library_video(item_id: str):
    try:
        export_path = _api_services().export_match(item_id)
        return send_file(str(export_path), as_attachment=True, download_name=f"{item_id}_segmented.mp4")
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
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
        payload = {"status": state, "ready": state == "ready"}
        if state == "error":
            with jobs_lock:
                payload["error"] = preview_job_errors.get(item_id, "Could not prepare preview.")
        return jsonify(payload), 500 if state == "error" else 202
    try:
        preview_path = _web_preview_path(item_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Could not prepare library preview %s", item_id)
        return jsonify({"error": f"Could not prepare preview: {exc}"}), 500
    _touch_preview_cache_file(preview_path)
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
    if state == "error":
        with jobs_lock:
            payload["error"] = preview_job_errors.get(item_id, "Could not prepare preview.")
    if preview_path.exists():
        payload["bytes"] = preview_path.stat().st_size
        if state == "ready":
            _touch_preview_cache_file(preview_path)
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
    key = _preview_window_job_key(item_id, start_s, duration_s)
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
    if state == "error":
        with jobs_lock:
            payload["error"] = preview_job_errors.get(key, "Could not prepare preview window.")
    if preview_path.exists():
        payload["bytes"] = preview_path.stat().st_size
        if state == "ready":
            _touch_preview_cache_file(preview_path)
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
            status_code = 500 if state == "error" else 202
            return jsonify(_preview_window_payload(item_id, path, start_s, duration_s, state)), status_code
        preview_path = _preview_window_path(item_id, start_s, duration_s)
        _touch_preview_cache_file(preview_path)
        return send_file(str(preview_path), mimetype="video/webm")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Could not prepare preview window %s", item_id)
        return jsonify({"error": f"Could not prepare preview window: {exc}"}), 500


@app.route("/api/library/<item_id>/playback", methods=["GET"])
def library_playback(item_id: str):
    try:
        payload = _api_services().get_playback_manifest(item_id)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(payload), 200


@app.route("/api/library/<item_id>/segments", methods=["GET"])
def library_segments(item_id: str):
    path = _resolve_library_file(item_id, "segments.csv")
    if path is None:
        return jsonify({"error": "CSV not available"}), 404
    try:
        intervals = _sorted_point_intervals(path)
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
