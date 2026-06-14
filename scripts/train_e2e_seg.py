#!/usr/bin/env python3
"""Train the per-frame e2e segment model on an already-built dataset directory.

Example:
    python scripts/train_e2e_seg.py \
        --dataset-dir ~/cs_projects/RallyClip/data/datasets/prod_yolo26n960_fps5_seq20_mirror_v1 \
        --run-dir ~/cs_projects/RallyClip/data/runs/e2e_seg_yolo26_v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from training.train.seg_loop import train_seg  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train e2e segment head")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--diou-weight", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--selection-metric", default="val_loss", choices=["val_loss", "acceptable", "good_weighted"])
    parser.add_argument("--lambda0", type=float, default=0.05)
    parser.add_argument("--lambda1", type=float, default=0.2)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--mu0", type=float, default=0.05)
    parser.add_argument("--mu1", type=float, default=0.4)
    parser.add_argument("--mu2", type=float, default=2.0)
    parser.add_argument("--save-every-n", type=int, default=0)
    parser.add_argument("--head", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--decode-vote", default="mean", choices=["mean", "median"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = {
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "fps": args.fps,
        "pos_weight": args.pos_weight,
        "boundary_weight": args.boundary_weight,
        "diou_weight": args.diou_weight,
        "threshold": args.threshold,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": 0.0005,
        "selection_metric": args.selection_metric,
        "lambda0": args.lambda0,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "mu0": args.mu0,
        "mu1": args.mu1,
        "mu2": args.mu2,
        "save_every_n": args.save_every_n,
        "head": args.head,
        "decode_vote": args.decode_vote,
    }
    train_seg(
        Path(args.dataset_dir).expanduser().resolve(),
        Path(args.run_dir).expanduser().resolve(),
        config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
