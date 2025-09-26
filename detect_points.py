#!/usr/bin/env python3
"""
DeepMatch Tennis Point Inference & Segmentation Engine

This script orchestrates the entire inference pipeline, from a raw video
to a final segmented video showing only the points.
"""

import argparse
import tempfile
import os
import numpy as np
import torch
import joblib
import scipy.ndimage
from pathlib import Path

from data_scripts.pose_extractor import PoseExtractor
from data_scripts.data_preprocessor import DataPreprocessor
from data_scripts.feature_engineer import FeatureEngineer
from inference import load_model_from_checkpoint, run_windowed_inference_average, hysteresis_threshold, extract_segments_from_binary, write_segments_csv
from execute_segmentation import segment_video, load_intervals


def main():

    parser = argparse.ArgumentParser(description="DeepMatch Tennis Point Inference & Segmentation Engine")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--model", default="checkpoints/seq_len300/best_model.pth", help="Path to the trained LSTM model checkpoint (.pth).")
    parser.add_argument("--scaler", default="data/seq_len_300/scaler.joblib", help="Path to the trained StandardScaler (.joblib).")
    parser.add_argument("--yolo-model", default="yolov8s-pose.pt", help="YOLOv8 pose model filename in models/ (or full path).")
    parser.add_argument("--output-dir", default="output_videos", help="Directory to save the output CSV and segmented video.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for pose detection.")

    FPS = 15
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = Path(args.video).stem

    # Use a temporary directory for all intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        print("--- Step 1: Extracting Poses ---")
        pose_extractor = PoseExtractor(model_path=args.yolo_model)
        # Note: We run on the full video duration (99999) and without annotations
        raw_npz_path = pose_extractor.extract_pose_data(
            video_path=args.video,
            confidence_threshold=args.conf,
            start_time_seconds=0,
            duration_seconds=99999,
            target_fps=FPS,
            annotations_csv=None
        )

        print("\n--- Step 2: Preprocessing Data ---")
        preprocessor = DataPreprocessor(save_court_masks=False)
        preprocessed_npz_path = os.path.join(temp_dir, f"{base_name}_preprocessed.npz")
        preprocessor.preprocess_single_video(raw_npz_path, args.video, preprocessed_npz_path, overwrite=True)

        print("\n--- Step 3: Engineering Features ---")
        feature_engineer = FeatureEngineer()
        features_npz_path = os.path.join(temp_dir, f"{base_name}_features.npz")
        feature_engineer.create_features_from_preprocessed(preprocessed_npz_path, features_npz_path, overwrite=True)
        
        feature_data = np.load(features_npz_path)
        # We only need annotated frames, which in this case (no annotation CSV) is all of them
        feature_vectors = feature_data['features']

    print("\n--- Step 4: Normalizing Features ---")
    scaler = joblib.load(args.scaler)
    normalized_features = scaler.transform(feature_vectors)

    print("\n--- Step 5: Running Inference ---")
    model, device = load_model_from_checkpoint(args.model, return_logits=False)
    avg_probs = run_windowed_inference_average(model, device, normalized_features, sequence_length=300, overlap=150)

    print("\n--- Step 6: Post-Processing ---")
    smoothed_probs = scipy.ndimage.gaussian_filter1d(avg_probs, sigma=1.5)
    binary_pred = hysteresis_threshold(smoothed_probs, low=0.45, high=0.80, min_duration=int(0.5 * FPS))

    print("\n--- Step 7: Generating Output CSV ---")
    output_csv_path = os.path.join(args.output_dir, f"{base_name}_segments.csv")
    segments = extract_segments_from_binary(binary_pred)
    write_segments_csv(segments, output_csv_path, fps=float(FPS), overwrite=True)

    print("\n--- Step 8: Generating Segmented Video ---")
    output_video_path = os.path.join(args.output_dir, f"{base_name}_segmented.mp4")
    try:
        intervals = load_intervals(output_csv_path)
        if intervals:
            segment_video(args.video, intervals, output_video_path)
            print(f"Segmented video saved to {output_video_path}")
        else:
            print("No points detected, skipping video segmentation.")
    except Exception as e:
        print(f"Error: Could not generate segmented video. {e}")

    print(f"\n✅ MVP Pipeline Complete! Outputs are in '{args.output_dir}'")


if __name__ == "__main__":
    main()
