import argparse
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seq_len300" / "best_model.pth"
DEFAULT_SCALER = PROJECT_ROOT / "data" / "seq_len_300" / "scaler.joblib"


def ensure_asset(path: Path, description: str) -> Path:
    p = path.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Required {description} not found at '{p}'.")
    return p


def run(args: argparse.Namespace) -> int:
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        print(f"Error: Video file not found at '{video_path}'")
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enforce 300-seq model assets; either provided or defaults must exist
    model_path = ensure_asset(Path(args.model_path) if args.model_path else DEFAULT_CHECKPOINT, "LSTM checkpoint (seq_len 300)")
    scaler_path = ensure_asset(Path(args.scaler_path) if args.scaler_path else DEFAULT_SCALER, "StandardScaler for seq_len 300")

    # Use small YOLO v8 pose by default; ultralytics auto-downloads if missing
    yolo_model = str(Path(args.yolo_model).expanduser()) if args.yolo_model else "yolov8s-pose.pt"

    raw_npz = None
    if args.raw_npz:
        raw_npz_path = Path(args.raw_npz).expanduser().resolve()
        if not raw_npz_path.is_file():
            print(f"Error: Raw pose NPZ not found at '{raw_npz_path}'")
            return 1
        raw_npz = str(raw_npz_path)

    pose_extractor = PoseExtractor(model_dir=str(DEFAULT_MODELS_DIR), model_path=yolo_model)
    raw_npz = raw_npz or None
    if raw_npz is None:
        raw_npz = pose_extractor.extract_pose_data(
            video_path=str(video_path),
            confidence_threshold=float(args.conf),
            start_time_seconds=int(args.start_time),
            duration_seconds=int(args.duration),
            target_fps=int(args.fps),
            annotations_csv=None,
        )

    preprocessed_npz = output_dir / "preprocessed.npz"
    pre = DataPreprocessor(save_court_masks=False)
    pre.preprocess_single_video(raw_npz, str(video_path), str(preprocessed_npz), overwrite=bool(args.overwrite))

    features_npz = output_dir / "features.npz"
    fe = FeatureEngineer()
    fe.create_features_from_preprocessed(str(preprocessed_npz), str(features_npz), overwrite=bool(args.overwrite))

    data = np.load(str(features_npz))
    features = data["features"]
    scaler = joblib.load(str(scaler_path))
    features = scaler.transform(features)
    model, device = load_model_from_checkpoint(str(model_path), return_logits=False)
    avg_probs = run_windowed_inference_average(
        model, device, features, sequence_length=int(args.seq_len), overlap=int(args.overlap)
    )
    smoothed_probs = scipy.ndimage.gaussian_filter1d(avg_probs.astype(np.float32), sigma=float(args.sigma))
    min_duration_frames = int(round(max(0.0, float(args.min_dur_sec)) * float(args.fps)))
    binary_pred = hysteresis_threshold(
        smoothed_probs, low=float(args.low), high=float(args.high), min_duration=min_duration_frames
    )
    segments = extract_segments_from_binary(binary_pred)

    # Write CSV only if requested (default: off)
    if args.csv:
        csv_out = output_dir / "segments.csv"
        write_segments_csv(segments, str(csv_out), fps=float(args.fps), overwrite=bool(args.overwrite))

    # Segment video by default; allow opt-out via flag
    if args.segment_video:
        video_out = output_dir / "segmented.mp4"
        intervals_sec = [
            (start_idx / float(args.fps), end_idx / float(args.fps))
            for (start_idx, end_idx) in segments
        ]
        if intervals_sec:
            segment_video(str(video_path), intervals_sec, str(video_out))

    print("Done.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Tennis Tracker end-to-end CLI")
    p.add_argument("--video", required=True, help="Path to input MP4 video")
    p.add_argument("--output-dir", required=True, help="Directory to store outputs")
    # Keep optional but enforce existence via defaults above (seq_len 300 canonical)
    p.add_argument("--model-path", help="Path to LSTM .pth (seq_len 300). Defaults to repo checkpoints.")
    p.add_argument("--scaler-path", help="Path to StandardScaler .joblib (seq_len 300). Defaults to repo data.")
    p.add_argument("--yolo-model", help="YOLO pose weights (e.g., yolov8s-pose.pt). Defaults to small and auto-download.")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--low", type=float, default=0.45)
    p.add_argument("--high", type=float, default=0.8)
    p.add_argument("--min-dur-sec", type=float, default=0.5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--start-time", type=int, default=0)
    p.add_argument("--duration", type=int, default=999999)
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")
    p.add_argument("--raw-npz", default=None, help="Optional: point to existing raw pose npz to skip extraction")
    # Defaults: segment on, csv off
    p.add_argument("--segment-video", dest="segment_video", action="store_true", default=True, help="Write segmented MP4 (default: on)")
    p.add_argument("--no-segment-video", dest="segment_video", action="store_false", help="Disable segmented MP4 output")
    p.add_argument("--csv", action="store_true", default=False, help="Also write segments CSV annotations (default: off)")
    args = p.parse_args()
    return run(args)


