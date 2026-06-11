from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timedelta
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

from runtime.assets import YOLO_SIZE_MAP, candidate_roots, resolve_asset
from extraction.pose_extractor import PoseExtractionCancelled, PoseExtractor
from features.feature_engineer import FeatureEngineer
from infer import (
    extract_segments_from_binary,
    gaussian_filter1d,
    hysteresis_threshold,
    load_scaler_asset,
    load_model_from_checkpoint,
    run_windowed_inference_average_onnx,
    run_windowed_inference_average,
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
from segmentation.segment import segment_video

JobDict = Dict[str, Any]


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

def _load_default_config() -> Dict[str, Any]:
    try:
        cfg = build_gui_defaults()
    except Exception as exc:
        logging.warning("Could not load manifest defaults (%s); using minimal fallback.", exc)
        cfg = {
            "write_csv": True,
            "segment_video": True,
            "yolo_size": "small",
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
    cfg["output_dir"] = str(DEFAULT_OUTPUT_DIR)
    cfg["csv_output_dir"] = str(DEFAULT_CSV_DIR)
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


class PipelineCancelled(Exception):
    """Raised when a job is cancelled mid-flight."""


def _ensure_job_dir(job_id: str) -> Path:
    job_dir = (JOBS_DIR / job_id).resolve()
    jobs_root = JOBS_DIR.resolve()
    if jobs_root not in job_dir.parents and job_dir != jobs_root:
        raise ValueError(f"Invalid job id: {job_id!r}")
    return job_dir


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
# output_dir/csv_output_dir are intentionally free-form: this is a local
# single-user app and choosing where outputs land is a feature (e.g. external
# drives). Cross-origin abuse is blocked by _reject_cross_origin_writes.
_CLIENT_KEYS = {
    "output_name",
    "output_dir",
    "csv_output_dir",
    "yolo_size",
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
    return cfg


def _resolve_yolo_weights(cfg: Dict[str, Any]) -> str:
    choice = cfg.get("yolo_size")
    if choice and choice in YOLO_SIZE_MAP:
        return YOLO_SIZE_MAP[choice]
    if choice:
        return str(choice)
    return YOLO_SIZE_MAP["small"]


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
        court_mask, _ = pre.compute_court_mask(str(upload_path))
        # Court mask detection has no inner progress hooks; tick so the bar
        # visibly moves before pose extraction starts reporting.
        _set_step(job, "pose", "in_progress", 3)

        _check_cancel(job)
        extractor = PoseExtractor(
            model_dir=models_dir,
            model_path=yolo_weights,
            imgsz=int(cfg["imgsz"]),
            device=pose_device,
        )

        def pose_progress(frac: float, meta: Optional[Dict[str, Any]] = None) -> None:
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

        raw_npz = extractor.extract_pose_data(
            video_path=str(upload_path),
            confidence_threshold=float(cfg["conf"]),
            start_time_seconds=int(cfg["start_time"]),
            duration_seconds=int(cfg["duration"]),
            target_fps=int(cfg["fps"]),
            imgsz=int(cfg["imgsz"]),
            annotations_csv=None,
            progress_callback=pose_progress,
            output_dir=str(job_dir),
        )
        job["paths"]["raw_npz"] = raw_npz
        _set_step(job, "pose", "completed", 100)

        _check_cancel(job)
        _set_step(job, "preprocess", "in_progress", 5)
        preprocessed_npz = str(job_dir / "preprocessed.npz")
        success_pre = pre.preprocess_single_video(
            raw_npz, str(upload_path), preprocessed_npz, overwrite=True, court_mask=court_mask
        )
        if not success_pre or not Path(preprocessed_npz).exists():
            raise RuntimeError("Preprocessing failed")
        job["paths"]["preprocessed_npz"] = preprocessed_npz
        _set_step(job, "preprocess", "completed", 100)

        _check_cancel(job)
        _set_step(job, "feature", "in_progress", 5)
        fe = FeatureEngineer(
            screen_width=int(cfg["screen_width"]),
            screen_height=int(cfg["screen_height"]),
            target_fps=float(cfg["fps"]),
        )
        features_npz = str(job_dir / "features.npz")
        success_fe = fe.create_features_from_preprocessed(preprocessed_npz, features_npz, overwrite=True)
        if not success_fe or not Path(features_npz).exists():
            raise RuntimeError("Feature engineering failed")
        job["paths"]["features_npz"] = features_npz
        _set_step(job, "feature", "completed", 100)

        _check_cancel(job)
        _set_step(job, "inference", "in_progress", 5)
        with np.load(features_npz) as data:
            features = data["features"].copy()
        scaler = load_scaler_asset(str(scaler_path))
        features = scaler.transform(features)

        def infer_progress(frac: float) -> None:
            _set_step(job, "inference", "in_progress", int(1 + max(0.0, min(1.0, frac)) * 94))
        if model_path.suffix.lower() == ".onnx":
            avg_probs = run_windowed_inference_average_onnx(
                str(model_path),
                features,
                sequence_length=int(cfg["seq_len"]),
                overlap=int(cfg["overlap"]),
                progress_callback=infer_progress,
            )
        else:
            model, device = load_model_from_checkpoint(str(model_path), return_logits=False)
            avg_probs = run_windowed_inference_average(
                model,
                device,
                features,
                sequence_length=int(cfg["seq_len"]),
                overlap=int(cfg["overlap"]),
                progress_callback=infer_progress,
            )
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
        if cfg.get("write_csv"):
            csv_out = Path(cfg["csv_output_dir"] or DEFAULT_CSV_DIR).expanduser() / f"{base_name}_segments.csv"
            csv_out.parent.mkdir(parents=True, exist_ok=True)
            write_segments_csv(segments, str(csv_out), fps=float(cfg["fps"]), overwrite=True)
            job["paths"]["csv"] = str(csv_out)
        video_out_path = None
        if cfg.get("segment_video"):
            video_out = Path(cfg["output_dir"] or DEFAULT_OUTPUT_DIR).expanduser() / f"{base_name}_segmented.mp4"
            video_out.parent.mkdir(parents=True, exist_ok=True)
            intervals_sec = [(start_idx / float(cfg["fps"]), end_idx / float(cfg["fps"])) for start_idx, end_idx in segments]
            if intervals_sec:
                segment_video(str(upload_path), intervals_sec, str(video_out))
            video_out_path = str(video_out)
        job["paths"]["video"] = video_out_path
        _set_step(job, "output", "completed", 100)
        job["status"] = "completed"
        job["eta_seconds"] = 0.0
        # Cleanup intermediates now that outputs are ready
        for key in ("raw_npz", "preprocessed_npz", "features_npz"):
            path = job["paths"].get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
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
            "yolo_sizes": list(YOLO_SIZE_MAP.keys()),
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
    if not file.filename.lower().endswith(".mp4"):
        return jsonify({"error": "Only MP4 files are supported"}), 400
    filename = secure_filename(file.filename)
    # secure_filename strips non-ASCII; don't reject those uploads, rename them.
    if not filename.lower().endswith(".mp4") or filename.lower() == ".mp4":
        filename = "input.mp4"

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


@app.route("/api/download/video/<job_id>", methods=["GET"])
def download_video(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id"}), 404
    video_path = job["paths"].get("video")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video not available"}), 404
    return send_file(video_path, as_attachment=True, download_name=f"{job_id}_segmented.mp4")


@app.route("/api/download/csv/<job_id>", methods=["GET"])
def download_csv(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id"}), 404
    csv_path = job["paths"].get("csv")
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"error": "CSV not available"}), 404
    return send_file(csv_path, as_attachment=True, download_name=f"{job_id}_segments.csv")


def _configure_gui_logging() -> None:
    verbose = os.environ.get("RALLYCLIP_GUI_VERBOSE", "").strip().lower() in {"1", "true", "yes"}
    log_level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
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
