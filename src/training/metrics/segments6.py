"""Six-bin segment-level metric (docs/segment_eval_metrics.md).

Flow: IoU cross-matrix -> falses (max IoU < 0.1) -> greedy 1-to-1 matching by
descending IoU (leftover predictions count as false positives, leftover GT as
false negatives) -> tier matched pairs by absolute boundary error, falling back
to IoU gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Interval = Tuple[float, float]

BINS = (
    "good",
    "decent",
    "bad_segmentation",
    "poor_recognition",
    "false_positive",
    "false_negative",
)


@dataclass
class SixBinConfig:
    """Tier limits in seconds. "out" = the clip is longer than the point (start early /
    end late); "in" = the clip cuts into the point (start late / end early).

    Decent has its own limits per boundary (2026-10-30): a clip may start up to 2 s
    early or 0.5 s late, and end up to 1 s early or 3 s late. Extra padding costs a
    viewer little; a cut-off shot costs more. The v1 limits (1.5 s out / 0.5 s in on
    both ends) are DECENT_V1.
    """

    good_out: float = 0.5
    good_in: float = 0.2
    decent_start_out: float = 2.0
    decent_start_in: float = 0.5
    decent_end_in: float = 1.0
    decent_end_out: float = 3.0
    bad_seg_iou: float = 0.5
    negligible_iou: float = 0.1


# Decent limits before 2026-10-30, for numbers comparable with older reports.
DECENT_V1 = dict(decent_start_out=1.5, decent_start_in=0.5, decent_end_in=0.5, decent_end_out=1.5)


def _iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _within(e_start: float, e_end: float, out_tol: float, in_tol: float,
            end_out: float | None = None, end_in: float | None = None) -> bool:
    """e_start/e_end = pred - gt. The end limits default to the start limits."""
    end_out = out_tol if end_out is None else end_out
    end_in = in_tol if end_in is None else end_in
    return (-out_tol <= e_start <= in_tol) and (-end_in <= e_end <= end_out)


def compute_six_bin(
    gt_segments: Sequence[Interval],
    pred_segments: Sequence[Interval],
    cfg: SixBinConfig | None = None,
) -> Dict[str, float]:
    cfg = cfg or SixBinConfig()
    counts = {name: 0 for name in BINS}
    n_split = 0
    n_merge = 0

    # IoU cross-matrix and immediate falses
    pairs: List[Tuple[float, int, int]] = []  # (iou, pred_idx, gt_idx)
    pred_max_iou = [0.0] * len(pred_segments)
    gt_max_iou = [0.0] * len(gt_segments)
    for p_idx, pred in enumerate(pred_segments):
        for g_idx, gt in enumerate(gt_segments):
            score = _iou(pred, gt)
            pred_max_iou[p_idx] = max(pred_max_iou[p_idx], score)
            gt_max_iou[g_idx] = max(gt_max_iou[g_idx], score)
            if score >= cfg.negligible_iou:
                pairs.append((score, p_idx, g_idx))

    candidate_preds = set()
    candidate_gts = set()
    for p_idx, score in enumerate(pred_max_iou):
        if score < cfg.negligible_iou:
            counts["false_positive"] += 1
        else:
            candidate_preds.add(p_idx)
    for g_idx, score in enumerate(gt_max_iou):
        if score < cfg.negligible_iou:
            counts["false_negative"] += 1
        else:
            candidate_gts.add(g_idx)

    # Greedy 1-to-1 matching by descending IoU
    matched_preds: Dict[int, int] = {}
    matched_gts: Dict[int, int] = {}
    for score, p_idx, g_idx in sorted(pairs, key=lambda t: -t[0]):
        if p_idx in matched_preds or g_idx in matched_gts:
            continue
        matched_preds[p_idx] = g_idx
        matched_gts[g_idx] = p_idx

    # Leftovers: overlapping but unmatched
    for p_idx in candidate_preds - set(matched_preds):
        counts["false_positive"] += 1
        n_split += 1
    for g_idx in candidate_gts - set(matched_gts):
        counts["false_negative"] += 1
        n_merge += 1

    # Tier the matched pairs
    boundary_errors: List[Tuple[float, float]] = []
    for p_idx, g_idx in matched_preds.items():
        pred = pred_segments[p_idx]
        gt = gt_segments[g_idx]
        e_start = pred[0] - gt[0]
        e_end = pred[1] - gt[1]
        boundary_errors.append((e_start, e_end))
        iou = _iou(pred, gt)
        if _within(e_start, e_end, cfg.good_out, cfg.good_in):
            counts["good"] += 1
        elif _within(e_start, e_end, cfg.decent_start_out, cfg.decent_start_in,
                     cfg.decent_end_out, cfg.decent_end_in):
            counts["decent"] += 1
        elif iou >= cfg.bad_seg_iou:
            counts["bad_segmentation"] += 1
        else:
            counts["poor_recognition"] += 1

    n_events = len(matched_preds) + counts["false_positive"] + counts["false_negative"]
    result: Dict[str, float] = {f"n_{name}": float(counts[name]) for name in BINS}
    result["n_events"] = float(n_events)
    result["n_gt"] = float(len(gt_segments))
    result["n_pred"] = float(len(pred_segments))
    result["n_split"] = float(n_split)
    result["n_merge"] = float(n_merge)
    for name in BINS:
        result[f"share_{name}"] = counts[name] / n_events if n_events else 0.0
    result["acceptable_rate"] = (
        (counts["good"] + counts["decent"]) / len(gt_segments) if gt_segments else 0.0
    )
    if boundary_errors:
        starts = sorted(e[0] for e in boundary_errors)
        ends = sorted(e[1] for e in boundary_errors)
        result["median_start_err"] = starts[len(starts) // 2]
        result["median_end_err"] = ends[len(ends) // 2]
    return result


def aggregate_six_bin(per_video: List[Dict[str, float]]) -> Dict[str, float]:
    """Sum counts across videos and recompute shares."""
    if not per_video:
        return {}
    summed: Dict[str, float] = {}
    count_keys = [f"n_{name}" for name in BINS] + ["n_events", "n_gt", "n_pred", "n_split", "n_merge"]
    for key in count_keys:
        summed[key] = float(sum(m.get(key, 0.0) for m in per_video))
    n_events = summed["n_events"]
    for name in BINS:
        summed[f"share_{name}"] = summed[f"n_{name}"] / n_events if n_events else 0.0
    summed["acceptable_rate"] = (
        (summed["n_good"] + summed["n_decent"]) / summed["n_gt"] if summed["n_gt"] else 0.0
    )
    return summed
