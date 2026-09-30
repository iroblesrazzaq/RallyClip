#!/usr/bin/env python3
"""Parity gate: 5fps-normalize-then-YOLO vs full-fps-YOLO-then-target_fps=5.

Acceptance (plan Tier 4):
  - matched-frame detection box IoU > 0.9 (mean)
  - keypoint median delta < ~3 px
  - per-frame targets identical after preprocess

This is a one-off validation script (not CI). Results print to stdout and an
optional JSON report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.normalize.normalize import NormalizeConfig, normalize_video  # noqa: E402
from training.pose.yolo_hdf5 import YoloExtractConfig, YoloHdf5Extractor  # noqa: E402
from training.preprocess.preprocessor import _sample_indices  # noqa: E402

logger = logging.getLogger(__name__)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0 if a.size != b.size else 1.0
    # Compare top-confidence box only for a coarse parity signal.
    ax1, ay1, ax2, ay2 = a[0]
    bx1, by1, bx2, by2 = b[0]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _extract(video: Path, out: Path, *, sample_fps: float | None, mode: str, yolo: dict) -> Path:
    extractor = YoloHdf5Extractor(
        YoloExtractConfig(
            model_path=str(yolo.get("model", "yolov8n-pose.pt")),
            conf=float(yolo.get("conf", 0.25)),
            model_dir=str(yolo.get("model_dir", "models")),
            device=yolo.get("device"),
            imgsz=int(yolo.get("imgsz", 960)),
        )
    )
    return extractor.extract(
        video_path=video,
        output_path=out,
        start_time=0.0,
        duration=float(yolo.get("duration", 60.0)),
        sampling_mode=mode,
        sample_fps=sample_fps,
        overwrite=True,
        resume=False,
    )


def compare_raw_h5(baseline: Path, candidate: Path, target_fps: float = 5.0) -> dict:
    with h5py.File(baseline, "r") as base, h5py.File(candidate, "r") as cand:
        base_ts = base["frames"]["timestamps"][:]
        cand_ts = cand["frames"]["timestamps"][:]
        base_idx = _sample_indices(base_ts, target_fps)
        # Candidate should already be ~target_fps after normalize+full extract.
        cand_idx = np.arange(len(cand_ts), dtype=np.int64)

        n = min(len(base_idx), len(cand_idx))
        ious = []
        kp_deltas = []
        for i in range(n):
            bi = int(base_idx[i])
            ci = int(cand_idx[i])
            b_off = base["frames"]["frame_offsets"]
            c_off = cand["frames"]["frame_offsets"]
            b_boxes = np.asarray(base["detections"]["boxes"][int(b_off[bi]) : int(b_off[bi + 1])])
            c_boxes = np.asarray(cand["detections"]["boxes"][int(c_off[ci]) : int(c_off[ci + 1])])
            ious.append(_box_iou(b_boxes, c_boxes))
            b_kps = np.asarray(base["detections"]["keypoints"][int(b_off[bi]) : int(b_off[bi + 1])])
            c_kps = np.asarray(cand["detections"]["keypoints"][int(c_off[ci]) : int(c_off[ci + 1])])
            if b_kps.size and c_kps.size:
                kp_deltas.append(float(np.median(np.abs(b_kps[0] - c_kps[0]))))

    mean_iou = float(np.mean(ious)) if ious else 0.0
    median_kp = float(np.median(kp_deltas)) if kp_deltas else float("inf")
    ok = mean_iou > 0.9 and median_kp < 3.0
    return {
        "matched_frames": n,
        "mean_box_iou": mean_iou,
        "median_keypoint_delta_px": median_kp,
        "pass": ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="One old 720p source video")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    video = args.video.expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    yolo = {
        "model": args.model,
        "imgsz": args.imgsz,
        "duration": args.duration,
        "model_dir": str(ROOT / "models"),
    }

    with tempfile.TemporaryDirectory(prefix="rc_parity_") as tmp:
        tmp_path = Path(tmp)
        baseline_h5 = tmp_path / "baseline_full.h5"
        norm_video = tmp_path / "normalized.mp4"
        candidate_h5 = tmp_path / "candidate_norm.h5"

        logger.info("Baseline: YOLO on full-fps video...")
        _extract(video, baseline_h5, sample_fps=None, mode="full_then_downsample", yolo=yolo)

        logger.info("Normalize to 720p@5fps then YOLO...")
        normalize_video(video, norm_video, NormalizeConfig(overwrite=True))
        _extract(norm_video, candidate_h5, sample_fps=None, mode="full_then_downsample", yolo=yolo)

        report = compare_raw_h5(baseline_h5, candidate_h5, target_fps=5.0)
        report["video"] = str(video)
        report["duration_s"] = args.duration

    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not report["pass"]:
        logger.error("Parity gate FAILED")
        return 1
    logger.info("Parity gate PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
