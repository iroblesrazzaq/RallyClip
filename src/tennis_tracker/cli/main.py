from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import scipy.ndimage

from tennis_tracker.extraction.pose_extractor import PoseExtractor
from tennis_tracker.features.feature_engineer import FeatureEngineer
from tennis_tracker.infer import (
    extract_segments_from_binary,
    hysteresis_threshold,
    load_model_from_checkpoint,
    run_windowed_inference_average,
    write_segments_csv,
)
from tennis_tracker.preprocessing.data_preprocessor import DataPreprocessor
from tennis_tracker.segmentation.segment import segment_video

try:  # Python 3.11+ ships tomllib; fall back to tomli otherwise.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - dependent on interpreter version
    import tomli as tomllib

YOLO_SIZE_MAP = {
    "nano": "yolov8n-pose.pt",
    "small": "yolov8s-pose.pt",
    "medium": "yolov8m-pose.pt",
    "large": "yolov8l-pose.pt",
}


@dataclass
class RunConfig:
    video_path: Path
    output_dir: Path
    output_name: Optional[str]
    csv_output_dir: Path
    write_csv: bool
    segment_video: bool
    yolo_weights: str
    yolo_device: Optional[str]
    model_path: Path
    scaler_path: Path
    fps: float = 15.0
    seq_len: int = 300
    overlap: int = 150
    sigma: float = 1.5
    low: float = 0.45
    high: float = 0.8
    min_dur_sec: float = 0.5
    conf: float = 0.25
    start_time: int = 0
    duration: int = 999999


def _candidate_roots() -> list[Path]:
    """Possible roots where assets might live (repo root, cwd, site-packages)."""
    here = Path(__file__).resolve()
    roots = [Path.cwd()]
    for depth in (2, 3, 4):
        try:
            roots.append(here.parents[depth])
        except IndexError:
            continue
    seen: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.append(r)
    return seen


def _resolve_asset(explicit: Optional[str], env_var: str, relatives: list[str], description: str) -> Path:
    """Resolve a required asset from CLI/config/env/default locations."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at '{path}'")
        return path

    env_val = os.environ.get(env_var)
    if env_val:
        path = Path(env_val).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at '{path}' (from {env_var})")
        return path

    for root in _candidate_roots():
        for rel in relatives:
            candidate = (Path(root) / rel).expanduser()
            if candidate.exists():
                return candidate.resolve()

    roots_str = ", ".join(str(r) for r in _candidate_roots())
    raise FileNotFoundError(
        f"{description} not found. Set via CLI flag, config, or env {env_var}; "
        f"searched relative locations {relatives} under: {roots_str}"
    )


def _load_config_dict(path: Optional[str]) -> Dict[str, Any]:
    if path:
        cfg_path = Path(path).expanduser()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found at '{cfg_path}'")
    else:
        cfg_path = Path("deepmatch.toml")
        if not cfg_path.exists():
            return {}
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def _pick_bool(arg_val: Optional[bool], cfg_val: Optional[Any], default: bool) -> bool:
    if arg_val is not None:
        return bool(arg_val)
    if cfg_val is not None:
        return bool(cfg_val)
    return default


def build_run_config(args: argparse.Namespace) -> RunConfig:
    cfg_dict = _load_config_dict(args.config) if (args.config or Path("deepmatch.toml").exists()) else {}
    cfg_section = cfg_dict.get("run", cfg_dict) if isinstance(cfg_dict, dict) else {}

    def cfg(key: str, default=None):
        return cfg_section.get(key, default) if isinstance(cfg_section, dict) else default

    video_path = args.video or cfg("video_path")
    output_dir = args.output_dir or cfg("output_dir")
    if not video_path or not output_dir:
        raise SystemExit("Please provide both video and output_dir via CLI flags or deepmatch.toml [run] section.")

    output_name = args.output_name or cfg("output_name")
    csv_output_dir_raw = args.csv_output_dir or cfg("csv_output_dir")

    yolo_choice = args.yolo_size or cfg("yolo_model")
    if yolo_choice and yolo_choice in YOLO_SIZE_MAP:
        yolo_weights = YOLO_SIZE_MAP[yolo_choice]
    elif yolo_choice:
        yolo_weights = str(yolo_choice)
    else:
        yolo_weights = YOLO_SIZE_MAP["small"]
    yolo_device = args.yolo_device or cfg("yolo_device")

    write_csv = _pick_bool(args.write_csv, cfg("write_csv"), True)
    segment_video_flag = _pick_bool(args.segment_video, cfg("segment_video"), True)

    model_path = _resolve_asset(
        args.model_path or cfg("model_path"),
        env_var="DEEPMATCH_MODEL_PATH",
        relatives=[
            "models/lstm_300_v0.1.pth",
            "checkpoints/seq_len300/best_model.pth",
        ],
        description="LSTM checkpoint (seq_len=300)",
    )
    scaler_path = _resolve_asset(
        args.scaler_path or cfg("scaler_path"),
        env_var="DEEPMATCH_SCALER_PATH",
        relatives=[
            "models/scaler_300_v0.1.joblib",
            "data/seq_len_300/scaler.joblib",
        ],
        description="StandardScaler for seq_len=300",
    )

    return RunConfig(
        video_path=Path(video_path).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        output_name=output_name,
        csv_output_dir=Path(csv_output_dir_raw).expanduser().resolve() if csv_output_dir_raw else Path(video_path).expanduser().resolve().parent,
        write_csv=write_csv,
        segment_video=segment_video_flag,
        yolo_weights=yolo_weights,
        yolo_device=yolo_device,
        model_path=model_path,
        scaler_path=scaler_path,
        fps=float(args.fps if args.fps is not None else cfg("fps", 15.0)),
        seq_len=int(args.seq_len if args.seq_len is not None else cfg("seq_len", 300)),
        overlap=int(args.overlap if args.overlap is not None else cfg("overlap", 150)),
        sigma=float(args.sigma if args.sigma is not None else cfg("sigma", 1.5)),
        low=float(args.low if args.low is not None else cfg("low", 0.45)),
        high=float(args.high if args.high is not None else cfg("high", 0.8)),
        min_dur_sec=float(args.min_dur_sec if args.min_dur_sec is not None else cfg("min_dur_sec", 0.5)),
        conf=float(args.conf if args.conf is not None else cfg("conf", 0.25)),
        start_time=int(args.start_time if args.start_time is not None else cfg("start_time", 0)),
        duration=int(args.duration if args.duration is not None else cfg("duration", 999999)),
    )


def run_pipeline(cfg: RunConfig) -> int:
    if not cfg.video_path.exists():
        print(f"Error: video file not found at '{cfg.video_path}'")
        return 1

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    base_name = cfg.output_name or cfg.video_path.stem

    if cfg.yolo_device:
        os.environ["POSE_DEVICE"] = cfg.yolo_device
    pose_extractor = PoseExtractor(model_path=cfg.yolo_weights)
    raw_npz = pose_extractor.extract_pose_data(
        video_path=str(cfg.video_path),
        confidence_threshold=float(cfg.conf),
        start_time_seconds=int(cfg.start_time),
        duration_seconds=int(cfg.duration),
        target_fps=int(cfg.fps),
        annotations_csv=None,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        preprocessed_npz = tmp_dir / "preprocessed.npz"
        features_npz = tmp_dir / "features.npz"

        pre = DataPreprocessor(save_court_masks=False)
        pre.preprocess_single_video(raw_npz, str(cfg.video_path), str(preprocessed_npz), overwrite=True)

        fe = FeatureEngineer()
        fe.create_features_from_preprocessed(str(preprocessed_npz), str(features_npz), overwrite=True)

        data = np.load(str(features_npz))
        features = data["features"]
        scaler = joblib.load(str(cfg.scaler_path))
        features = scaler.transform(features)

        model, device = load_model_from_checkpoint(str(cfg.model_path), return_logits=False)
        avg_probs = run_windowed_inference_average(
            model, device, features, sequence_length=int(cfg.seq_len), overlap=int(cfg.overlap)
        )
        smoothed_probs = scipy.ndimage.gaussian_filter1d(avg_probs.astype(np.float32), sigma=float(cfg.sigma))
        min_duration_frames = int(round(max(0.0, float(cfg.min_dur_sec)) * float(cfg.fps)))
        binary_pred = hysteresis_threshold(
            smoothed_probs, low=float(cfg.low), high=float(cfg.high), min_duration=min_duration_frames
        )
        segments = extract_segments_from_binary(binary_pred)

    if cfg.write_csv:
        cfg.csv_output_dir.mkdir(parents=True, exist_ok=True)
        csv_out = cfg.csv_output_dir / f"{base_name}_segments.csv"
        write_segments_csv(segments, str(csv_out), fps=float(cfg.fps), overwrite=True)

    if cfg.segment_video:
        video_out = cfg.output_dir / f"{base_name}_segmented.mp4"
        intervals_sec = [
            (start_idx / float(cfg.fps), end_idx / float(cfg.fps))
            for (start_idx, end_idx) in segments
        ]
        if intervals_sec:
            segment_video(str(cfg.video_path), intervals_sec, str(video_out))

    print(f"✅ Done. Outputs in {cfg.output_dir}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="DeepMatch end-to-end CLI with optional deepmatch.toml config.")
    p.add_argument("--config", help="Path to deepmatch.toml. If omitted, looks for ./deepmatch.toml.")
    p.add_argument("--video", help="Path to input MP4 video")
    p.add_argument("--output-dir", help="Directory to store outputs")
    p.add_argument("--output-name", help="Optional base name for outputs (without extension)")
    p.add_argument("--csv-output-dir", help="Optional directory for CSV output (defaults to video directory)")
    p.add_argument("--model-path", help="Path to LSTM .pth (defaults to checkpoints/seq_len300/best_model.pth if present)")
    p.add_argument("--scaler-path", help="Path to StandardScaler .joblib (defaults to data/seq_len_300/scaler.joblib if present)")
    p.add_argument("--yolo-size", choices=list(YOLO_SIZE_MAP.keys()), help="YOLO pose model size (auto-downloads if needed)")
    p.add_argument("--yolo-device", choices=["cpu", "cuda", "mps"], help="Force YOLO device (overrides POSE_DEVICE env)")

    p.add_argument("--fps", type=float, help="Sampling FPS used during feature creation")
    p.add_argument("--seq-len", type=int, help="Sequence length for inference windows")
    p.add_argument("--overlap", type=int, help="Overlap (frames) between windows")
    p.add_argument("--sigma", type=float, help="Gaussian smoothing sigma")
    p.add_argument("--low", type=float, help="Hysteresis low threshold")
    p.add_argument("--high", type=float, help="Hysteresis high threshold")
    p.add_argument("--min-dur-sec", type=float, help="Minimum segment duration in seconds")
    p.add_argument("--conf", type=float, help="Pose model confidence threshold")
    p.add_argument("--start-time", type=int, help="Start time offset (seconds)")
    p.add_argument("--duration", type=int, help="Max duration to process (seconds)")

    p.add_argument("--write-csv", dest="write_csv", action="store_true", default=None, help="Write segments CSV")
    p.add_argument("--no-csv", dest="write_csv", action="store_false", help="Skip writing segments CSV")
    p.add_argument("--segment-video", dest="segment_video", action="store_true", default=None, help="Write segmented MP4")
    p.add_argument("--no-segment-video", dest="segment_video", action="store_false", help="Skip segmented MP4")
    args = p.parse_args()

    cfg = build_run_config(args)
    return run_pipeline(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
