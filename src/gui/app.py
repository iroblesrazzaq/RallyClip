from __future__ import annotations

import json
import logging
import os
import shutil
import socket
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
        "rallyvision gui requires GUI extras. Install with `pip install .[gui]`."
    ) from exc

import joblib
import numpy as np
import scipy.ndimage
import av

from cli.main import YOLO_SIZE_MAP, _candidate_roots, _resolve_asset
from extraction.pose_extractor import PoseExtractionCancelled, PoseExtractor
from features.feature_engineer import FeatureEngineer
from infer import (
    extract_segments_from_binary,
    hysteresis_threshold,
    load_model_from_checkpoint,
    run_windowed_inference_average,
    write_segments_csv,
)
from preprocessing.data_preprocessor import DataPreprocessor
from segmentation.segment import segment_video

JobDict = Dict[str, Any]


def _find_static_dir() -> Path:
    """Locate the frontend bundle relative to common repo roots."""
    rel = Path("apps/gui/frontend")
    for root in _candidate_roots():
        candidate = Path(root) / rel
        if candidate.exists():
            return candidate.resolve()
    return (Path(__file__).resolve().parent / "../../apps/gui/frontend").resolve()


STATIC_DIR = _find_static_dir()


def _default_jobs_dir() -> Path:
    """Pick a jobs/output root inside the RallyVision install if possible; fallback to CWD."""
    for root in _candidate_roots():
        root_path = Path(root).resolve()
        if (root_path / "models").exists() or (root_path / "apps").exists():
            return (root_path / "RallyVisionJobs").resolve()
    return (Path.cwd() / "RallyVisionJobs").resolve()


def _default_output_dir() -> Path:
    for root in _candidate_roots():
        root_path = Path(root).resolve()
        if (root_path / "models").exists() or (root_path / "apps").exists():
            return (root_path / "output_videos").resolve()
    return (Path.cwd() / "output_videos").resolve()


def _default_csv_dir() -> Path:
    for root in _candidate_roots():
        root_path = Path(root).resolve()
        if (root_path / "models").exists() or (root_path / "apps").exists():
            return (root_path / "output_csvs").resolve()
    return (Path.cwd() / "output_csvs").resolve()


def _keep_jobs() -> bool:
    return os.environ.get("RALLYVISION_KEEP_JOBS", "").strip().lower() in {"1", "true", "yes"}


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


JOBS_DIR = _default_jobs_dir()
JOBS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_DIR = _default_output_dir()
DEFAULT_CSV_DIR = _default_csv_dir()

DEFAULT_CONFIG: Dict[str, Any] = {
    "write_csv": True,
    "segment_video": True,
    "yolo_size": "small",
    "yolo_device": None,
    "output_name": None,
    "output_dir": str(DEFAULT_OUTPUT_DIR),
    "csv_output_dir": str(DEFAULT_CSV_DIR),
    "model_path": None,
    "scaler_path": None,
    "fps": 15.0,
    "seq_len": 300,
    "overlap": 150,
    "sigma": 1.5,
    "low": 0.45,
    "high": 0.8,
    "min_dur_sec": 0.5,
    "conf": 0.25,
    "start_time": 0,
    "duration": 999999,
}

ADVANCED_WARNINGS = {
    "fps": "Changing fps will break model expectations; keep at 15.0 unless you retrain.",
    "seq_len": "Sequence length is tied to training; keep at 300.",
    "overlap": "Overlap tunes throughput vs smoothness; default 150 is recommended.",
    "low": "Lowering thresholds increases sensitivity and false positives.",
    "high": "Raising thresholds decreases sensitivity but may miss points.",
    "min_dur_sec": "Shorter durations may create noisy/short segments.",
}

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/")
CORS(app, resources={r"/api/*": {"origins": "*"}})

jobs_lock = threading.Lock()
jobs: Dict[str, JobDict] = {}


class PipelineCancelled(Exception):
    """Raised when a job is cancelled mid-flight."""


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


def _normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {**DEFAULT_CONFIG}
    cfg.update({k: v for k, v in (raw or {}).items() if v is not None})
    return cfg


def _resolve_yolo_weights(cfg: Dict[str, Any]) -> str:
    choice = cfg.get("yolo_size")
    if choice and choice in YOLO_SIZE_MAP:
        return YOLO_SIZE_MAP[choice]
    if choice:
        return str(choice)
    return YOLO_SIZE_MAP["small"]


def _resolve_model_paths(cfg: Dict[str, Any]) -> tuple[Path, Path]:
    model_path = _resolve_asset(
        cfg.get("model_path"),
        env_var="DEEPMATCH_MODEL_PATH",
        relatives=["models/lstm_300_v0.1.pth", "checkpoints/seq_len300/best_model.pth"],
        description="LSTM checkpoint (seq_len=300)",
    )
    scaler_path = _resolve_asset(
        cfg.get("scaler_path"),
        env_var="DEEPMATCH_SCALER_PATH",
        relatives=["models/scaler_300_v0.1.joblib", "data/seq_len_300/scaler.joblib"],
        description="StandardScaler for seq_len=300",
    )
    return model_path, scaler_path


def _estimate_duration_seconds(video_path: Path) -> float:
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if getattr(stream, "duration", None) and getattr(stream, "time_base", None):
                return float(stream.duration * stream.time_base)
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
        base_name = cfg.get("output_name") or upload_path.stem

        duration_seconds = _estimate_duration_seconds(upload_path)
        if duration_seconds <= 0:
            duration_seconds = float(cfg.get("duration", 0) or 0)
        elif cfg.get("duration") and cfg["duration"] > 0:
            duration_seconds = min(duration_seconds, float(cfg["duration"]))
        weights = _compute_weights(duration_seconds)
        job["weights"] = weights

        if cfg.get("yolo_device"):
            os.environ["POSE_DEVICE"] = str(cfg["yolo_device"])

        yolo_weights = _resolve_yolo_weights(cfg)
        model_path, scaler_path = _resolve_model_paths(cfg)
        models_dir = None
        for root in _candidate_roots():
            candidate = Path(root) / "models"
            if candidate.exists():
                models_dir = str(candidate.resolve())
                break

        _check_cancel(job)
        _set_step(job, "pose", "in_progress", 1)
        extractor = PoseExtractor(model_dir=models_dir, model_path=yolo_weights)

        def pose_progress(frac: float, meta: Optional[Dict[str, Any]] = None) -> None:
            if job.get("cancelled"):
                raise PoseExtractionCancelled("Job cancelled during pose extraction")
            _set_step(job, "pose", "in_progress", int(1 + max(0.0, min(1.0, frac)) * 98))
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
            annotations_csv=None,
            progress_callback=pose_progress,
        )
        job["paths"]["raw_npz"] = raw_npz
        _set_step(job, "pose", "completed", 100)

        _check_cancel(job)
        _set_step(job, "preprocess", "in_progress", 5)
        pre = DataPreprocessor(save_court_masks=False)
        preprocessed_npz = str(job_dir / "preprocessed.npz")
        success_pre = pre.preprocess_single_video(raw_npz, str(upload_path), preprocessed_npz, overwrite=True)
        if not success_pre or not Path(preprocessed_npz).exists():
            raise RuntimeError("Preprocessing failed")
        job["paths"]["preprocessed_npz"] = preprocessed_npz
        _set_step(job, "preprocess", "completed", 100)

        _check_cancel(job)
        _set_step(job, "feature", "in_progress", 5)
        fe = FeatureEngineer()
        features_npz = str(job_dir / "features.npz")
        success_fe = fe.create_features_from_preprocessed(preprocessed_npz, features_npz, overwrite=True)
        if not success_fe or not Path(features_npz).exists():
            raise RuntimeError("Feature engineering failed")
        job["paths"]["features_npz"] = features_npz
        _set_step(job, "feature", "completed", 100)

        _check_cancel(job)
        _set_step(job, "inference", "in_progress", 5)
        data = np.load(features_npz)
        features = data["features"]
        scaler = joblib.load(str(scaler_path))
        features = scaler.transform(features)
        model, device = load_model_from_checkpoint(str(model_path), return_logits=False)

        def infer_progress(frac: float) -> None:
            _set_step(job, "inference", "in_progress", int(1 + max(0.0, min(1.0, frac)) * 94))

        avg_probs = run_windowed_inference_average(
            model,
            device,
            features,
            sequence_length=int(cfg["seq_len"]),
            overlap=int(cfg["overlap"]),
            progress_callback=infer_progress,
        )
        smoothed_probs = scipy.ndimage.gaussian_filter1d(avg_probs.astype(np.float32), sigma=float(cfg["sigma"]))
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
            csv_out = Path(cfg["csv_output_dir"] or DEFAULT_CSV_DIR) / f"{base_name}_segments.csv"
            csv_out.parent.mkdir(parents=True, exist_ok=True)
            write_segments_csv(segments, str(csv_out), fps=float(cfg["fps"]), overwrite=True)
            job["paths"]["csv"] = str(csv_out)
        video_out_path = None
        if cfg.get("segment_video"):
            video_out = Path(cfg["output_dir"] or DEFAULT_OUTPUT_DIR) / f"{base_name}_segmented.mp4"
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
    return jsonify(
        {
            "defaults": DEFAULT_CONFIG,
            "yolo_sizes": list(YOLO_SIZE_MAP.keys()),
            "warnings": ADVANCED_WARNINGS,
        }
    ), 200


@app.route("/api/upload-and-start", methods=["POST"])
def upload_and_start():
    if "video" not in request.files:
        return jsonify({"error": "Missing file field 'video'"}), 400
    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".mp4"):
        return jsonify({"error": "Only MP4 files are supported"}), 400

    try:
        cfg_raw = json.loads(request.form.get("config", "{}") or "{}")
    except json.JSONDecodeError:
        cfg_raw = {}
    cfg = _normalize_config(cfg_raw)

    job_id = request.form.get("job_id") or str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
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
    job["cancelled"] = True
    job["status"] = "cancelled"
    return jsonify({"status": "cancelled"}), 200


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


def launch(port: Optional[int] = None) -> int:
    verbose = os.environ.get("RALLYVISION_GUI_VERBOSE", "").strip().lower() in {"1", "true", "yes"}
    log_level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    # Quiet Flask/werkzeug unless verbose
    for name in ("werkzeug", "flask.app"):
        logging.getLogger(name).setLevel(log_level)
    if not verbose:
        # Disable tqdm bars in GUI mode to keep the terminal clean
        os.environ.setdefault("RALLYVISION_NO_TQDM", "1")
        # Suppress the Flask devserver banner
        try:
            import flask.cli  # type: ignore
            flask.cli.show_server_banner = lambda *args, **kwargs: None  # noqa: E731
        except Exception:
            pass
    _sweep_old_jobs()
    preferred_ports: list[int] = []
    env_port = os.environ.get("RALLYVISION_GUI_PORT")
    if port:
        preferred_ports.append(int(port))
    if env_port:
        try:
            preferred_ports.append(int(env_port))
        except ValueError:
            pass
    preferred_ports.extend([8000, 5173])
    chosen_port = _pick_port(preferred_ports)
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
