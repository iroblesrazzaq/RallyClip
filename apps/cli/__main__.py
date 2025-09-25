#!/usr/bin/env python3
import argparse
from pathlib import Path

from tennis_tracker.extraction.pose_extractor import PoseExtractor
from tennis_tracker.preprocessing.data_preprocessor import DataPreprocessor
from tennis_tracker.features.feature_engineer import FeatureEngineer
from tennis_tracker.infer import (
    load_model_from_checkpoint,
    run_windowed_inference_average,
    hysteresis_threshold,
    extract_segments_from_binary,
    write_segments_csv,
)
from tennis_tracker.segmentation.segment import segment_video

import numpy as np
import joblib
import scipy.ndimage


def run(args: argparse.Namespace) -> int:
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        print(f"Error: Video file not found at '{video_path}'")
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_file():
        print(f"Error: Model checkpoint not found at '{model_path}'")
        return 1

    scaler_path = Path(args.scaler_path).expanduser().resolve()
    if not scaler_path.is_file():
        print(f"Error: Scaler file not found at '{scaler_path}'")
        return 1

    # yolo_model may be a filename inside models/ or an absolute path
    yolo_model_path = Path(args.yolo_model).expanduser()
    yolo_model = str(yolo_model_path)

    raw_npz = None
    if args.raw_npz:
        raw_npz_path = Path(args.raw_npz).expanduser().resolve()
        if not raw_npz_path.is_file():
            print(f"Error: Raw pose NPZ not found at '{raw_npz_path}'")
            return 1
        raw_npz = str(raw_npz_path)

    # Step 1: Pose extraction (skip if exists and not overwrite)
    pose_extractor = PoseExtractor(model_path=yolo_model)
    raw_npz = raw_npz or None
    if raw_npz is None:
        # create default raw path
        raw_npz = pose_extractor.extract_pose_data(
            video_path=str(video_path),
            confidence_threshold=float(args.conf),
            start_time_seconds=int(args.start_time),
            duration_seconds=int(args.duration),
            target_fps=int(args.fps),
            annotations_csv=None,
        )
    # Step 2: Preprocess
    preprocessed_npz = output_dir / "preprocessed.npz"
    pre = DataPreprocessor(save_court_masks=False)
    pre.preprocess_single_video(raw_npz, str(video_path), str(preprocessed_npz), overwrite=bool(args.overwrite))
    # Step 3: Features
    features_npz = output_dir / "features.npz"
    fe = FeatureEngineer()
    fe.create_features_from_preprocessed(str(preprocessed_npz), str(features_npz), overwrite=bool(args.overwrite))
    # Step 4: Inference
    data = np.load(str(features_npz))
    features = data["features"]
    scaler = joblib.load(str(scaler_path))
    features = scaler.transform(features)
    model, device = load_model_from_checkpoint(str(model_path), return_logits=False)
    avg_probs = run_windowed_inference_average(model, device, features, sequence_length=int(args.seq_len), overlap=int(args.overlap))
    smoothed_probs = scipy.ndimage.gaussian_filter1d(avg_probs.astype(np.float32), sigma=float(args.sigma))
    min_duration_frames = int(round(max(0.0, float(args.min_dur_sec)) * float(args.fps)))
    binary_pred = hysteresis_threshold(smoothed_probs, low=float(args.low), high=float(args.high), min_duration=min_duration_frames)
    segments = extract_segments_from_binary(binary_pred)
    # Optionally write CSV annotations
    if args.csv:
        csv_out = output_dir / "segments.csv"
        write_segments_csv(segments, str(csv_out), fps=float(args.fps), overwrite=bool(args.overwrite))
    # Step 5: Segment video (optional)
    if args.segment_video:
        video_out = output_dir / "segmented.mp4"
        # Convert frame index segments to second-based intervals
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
    p.add_argument("--model-path", required=True, help="Path to LSTM .pth file")
    p.add_argument("--scaler-path", required=True, help="Path to StandardScaler .joblib")
    p.add_argument("--yolo-model", default="yolov8s-pose.pt", help="YOLO pose weights file name in models/")
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
    p.add_argument("--segment-video", action="store_true", help="Also write segmented MP4")
    p.add_argument("--csv", action="store_true", help="Also write segments CSV annotations to output dir")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())


