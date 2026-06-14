#!/usr/bin/env python3
"""Score a classic frame-prob LSTM checkpoint (TennisPointLSTM) on the six-bin
segment metric: stitch val windows per video, gaussian-smooth + hysteresis to get
segments (the shipped postprocess), then bin against GT points.

Example:
    python scripts/eval_six_bin_checkpoint.py \
        --checkpoint .../runs/prod_yolon960_fps5_seq20_mirror_v1/checkpoints/best.pth \
        --val-h5 .../datasets/prod_yolon960_fps5_seq20_mirror_v1/val.h5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infer.inference import gaussian_filter1d, hysteresis_threshold  # noqa: E402
from training.eval.seg_evaluator import _stitch_videos, gt_segments_from_targets  # noqa: E402
from training.metrics.frame import compute_frame_metrics  # noqa: E402
from training.metrics.segments6 import aggregate_six_bin, compute_six_bin  # noqa: E402
from training.models.lstm import TennisPointLSTM  # noqa: E402


def segments_from_binary(binary: np.ndarray, timestamps: np.ndarray):
    return gt_segments_from_targets(binary.astype(np.float32), timestamps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-h5", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--low", type=float, default=0.45)
    parser.add_argument("--high", type=float, default=0.7)
    parser.add_argument("--min-dur-sec", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")

    with h5py.File(Path(args.val_h5).expanduser(), "r") as h5f:
        features = h5f["features"][:]
        targets = h5f["targets"][:].astype(np.float32)
        video_idx = h5f["sequence_video_index"][:]
        frame_idx = h5f["sequence_frame_index"][:]
        timestamps = h5f["sequence_timestamps"][:]

    model = TennisPointLSTM(input_size=features.shape[-1], return_logits=True).to(device)
    state = torch.load(Path(args.checkpoint).expanduser(), map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    probs = np.zeros_like(targets, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, features.shape[0], args.batch_size):
            end = min(start + args.batch_size, features.shape[0])
            feats = torch.from_numpy(features[start:end]).float().to(device)
            probs[start:end] = torch.sigmoid(model(feats)).cpu().numpy()

    # offsets are unused for the hysteresis decode; pass zeros to reuse the stitcher
    zeros = np.zeros_like(probs)
    stitched = _stitch_videos(video_idx, frame_idx, timestamps, targets, probs, zeros, zeros)

    min_dur_frames = int(round(args.min_dur_sec * args.fps))
    per_video = []
    for vid, tl in stitched.items():
        smoothed = gaussian_filter1d(tl["probs"].astype(np.float32), sigma=args.sigma)
        binary = hysteresis_threshold(smoothed, low=args.low, high=args.high, min_duration=min_dur_frames)
        gt_segs = gt_segments_from_targets(tl["targets"], tl["timestamps"])
        pred_segs = segments_from_binary(np.asarray(binary), tl["timestamps"])
        per_video.append(compute_six_bin(gt_segs, pred_segs))

    agg = aggregate_six_bin(per_video)
    frame = compute_frame_metrics(targets.reshape(-1), probs.reshape(-1).astype(np.float64), threshold=0.5)
    print(json.dumps({**frame, **agg}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
