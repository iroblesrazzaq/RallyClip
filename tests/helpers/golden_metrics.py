"""Self-contained segment-quality scorer for the golden e2e fixture.

Mirrors the v0.4.0 6-bin metric (docs/segment_eval_metrics.md) but lives here so
the release branch doesn't depend on training-branch code. Greedy IoU matching of
predicted vs ground-truth point intervals, then each matched pair is tiered by
absolute-time boundary error; unmatched GT -> fn, unmatched pred -> fp.

A point counts as "acceptable" if it lands in the good or decent tier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

Interval = Tuple[float, float]


@dataclass(frozen=True)
class SixBinConfig:
    good_out: float = 0.5    # how early a start / how late an end may be (generous side)
    good_in: float = 0.2     # how late a start / how early an end may be (clipping side)
    # decent limits per boundary (2026-10-30): start 2 s early / 0.5 s late, end 1 s early / 3 s late
    decent_start_out: float = 2.0
    decent_start_in: float = 0.5
    decent_end_in: float = 1.0
    decent_end_out: float = 3.0
    bad_seg_iou: float = 0.5
    negligible_iou: float = 0.1


def _iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _within(es: float, ee: float, out_tol: float, in_tol: float,
            end_out: float | None = None, end_in: float | None = None) -> bool:
    # es = pred_start - gt_start (negative = early), ee = pred_end - gt_end (positive = late)
    end_out = out_tol if end_out is None else end_out
    end_in = in_tol if end_in is None else end_in
    return (-out_tol <= es <= in_tol) and (-end_in <= ee <= end_out)


def score_six_bin(pred: List[Interval], gt: List[Interval], cfg: SixBinConfig = SixBinConfig()) -> Dict[str, float]:
    """Return counts + fractions over GT points: good/decent/bad/poor/fn, plus fp count."""
    pairs = []
    gt_max = [0.0] * len(gt)
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            s = _iou(p, g)
            gt_max[gi] = max(gt_max[gi], s)
            if s >= cfg.negligible_iou:
                pairs.append((s, pi, gi))
    matched_p: Dict[int, int] = {}
    matched_g: Dict[int, int] = {}
    for s, pi, gi in sorted(pairs, key=lambda t: -t[0]):
        if pi in matched_p or gi in matched_g:
            continue
        matched_p[pi] = gi
        matched_g[gi] = pi

    bins = {"good": 0, "decent": 0, "bad": 0, "poor": 0, "fn": 0}
    for gi, g in enumerate(gt):
        if gt_max[gi] < cfg.negligible_iou or gi not in matched_g:
            bins["fn"] += 1
            continue
        p = pred[matched_g[gi]]
        es, ee, iou = p[0] - g[0], p[1] - g[1], _iou(p, g)
        if _within(es, ee, cfg.good_out, cfg.good_in):
            bins["good"] += 1
        elif _within(es, ee, cfg.decent_start_out, cfg.decent_start_in, cfg.decent_end_out, cfg.decent_end_in):
            bins["decent"] += 1
        elif iou >= cfg.bad_seg_iou:
            bins["bad"] += 1
        else:
            bins["poor"] += 1

    fp = len(pred) - len(matched_p)
    n = max(1, len(gt))
    return {
        **bins,
        "fp": fp,
        "n_gt": len(gt),
        "n_pred": len(pred),
        "acceptable": bins["good"] + bins["decent"],
        "acceptable_frac": (bins["good"] + bins["decent"]) / n,
        "good_frac": bins["good"] / n,
    }
