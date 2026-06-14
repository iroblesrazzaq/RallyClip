#!/usr/bin/env python3
"""Honest CPU evaluation of a TennisPointSegLSTM checkpoint with a decode sweep.

Loads best.pth, infers head type from the state_dict, evaluates on CPU (the path
that matches the deployable/ONNX artifact — MPS in-loop eval is inflated), and
sweeps decode threshold x vote, printing the six-bin shares for each and the best
by good_weighted (2*good + decent).

Example:
    python scripts/eval_seg_checkpoint.py \
        --run-dir ~/cs_projects/RallyClip/data/runs/e2e_seg_yolo26_v5 \
        --val-h5 ~/cs_projects/RallyClip/data/datasets/prod_yolo26n960_fps5_seq20_mirror_v1/val.h5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from training.eval.seg_evaluator import DecodeConfig, evaluate_seg_model  # noqa: E402
from training.metrics.segments6 import SixBinConfig  # noqa: E402
from training.models.seg_lstm import TennisPointSegLSTM  # noqa: E402
from training.train.seg_loss import E2ESegLoss, SegLossConfig  # noqa: E402

SHARE_KEYS = [
    "share_good",
    "share_decent",
    "share_bad_segmentation",
    "share_poor_recognition",
    "share_false_positive",
    "share_false_negative",
]


def _infer_head(state: dict) -> str:
    return "mlp" if any(k.startswith("reg_head") for k in state) else "linear"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--val-h5", required=True)
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--thresholds", default="0.4,0.5,0.6,0.7")
    parser.add_argument("--votes", default="mean,median")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    ckpt = torch.load(run_dir / "checkpoints" / args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    head = _infer_head(state)
    feature_dim = state["lstm.weight_ih_l0"].shape[1]

    model = TennisPointSegLSTM(input_size=feature_dim, head=head)
    model.load_state_dict(state)
    model.eval()

    criterion = E2ESegLoss(SegLossConfig(fps=args.fps))
    six_bin_cfg = SixBinConfig()
    val_h5 = Path(args.val_h5).expanduser()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    votes = [v.strip() for v in args.votes.split(",")]

    print(f"run={run_dir.name} ckpt={args.checkpoint} epoch={ckpt.get('epoch')} head={head}")
    print(f"{'thr':>4} {'vote':>7} | {'good':>6} {'decent':>6} {'bad':>6} {'poor':>6} {'fp':>6} {'fn':>6} | {'g+d':>6} {'2g+d':>6}")
    best = None
    for vote in votes:
        for thr in thresholds:
            metrics, _ = evaluate_seg_model(
                model, val_h5, torch.device("cpu"), criterion,
                DecodeConfig(threshold=thr, vote=vote), six_bin_cfg,
            )
            g = metrics["share_good"]
            d = metrics["share_decent"]
            gw = 2 * g + d
            row = " ".join(f"{metrics[k]:>6.3f}" for k in SHARE_KEYS)
            print(f"{thr:>4} {vote:>7} | {row} | {g + d:>6.3f} {gw:>6.3f}")
            if best is None or gw > best[0]:
                best = (gw, thr, vote, dict(metrics))

    gw, thr, vote, m = best
    print(f"\nBEST: thr={thr} vote={vote} | good={m['share_good']:.3f} g+d={m['share_good']+m['share_decent']:.3f} "
          f"2g+d={gw:.3f} fp={m['share_false_positive']:.3f} fn={m['share_false_negative']:.3f} bal_acc={m['balanced_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
