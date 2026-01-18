from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _segments_from_binary(binary: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for idx, val in enumerate(binary):
        if val and not in_seg:
            in_seg = True
            start = idx
        elif not val and in_seg:
            segments.append((start, idx))
            in_seg = False
    if in_seg:
        segments.append((start, len(binary)))
    return segments


def _iou(seg_a: Tuple[int, int], seg_b: Tuple[int, int]) -> float:
    a0, a1 = seg_a
    b0, b1 = seg_b
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return (inter / union) if union > 0 else 0.0


def compute_segment_metrics(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_pred_bin = np.asarray(y_pred_bin).astype(int)

    true_segments = _segments_from_binary(y_true_bin)
    pred_segments = _segments_from_binary(y_pred_bin)

    matched_true = set()
    matched_pred = set()
    ious: List[float] = []

    for p_idx, pred in enumerate(pred_segments):
        best_iou = 0.0
        best_t = None
        for t_idx, true in enumerate(true_segments):
            if t_idx in matched_true:
                continue
            score = _iou(pred, true)
            if score > best_iou:
                best_iou = score
                best_t = t_idx
        if best_t is not None and best_iou >= iou_threshold:
            matched_pred.add(p_idx)
            matched_true.add(best_t)
            ious.append(best_iou)

    precision = (len(matched_pred) / len(pred_segments)) if pred_segments else 0.0
    recall = (len(matched_true) / len(true_segments)) if true_segments else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    coverage = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
        "mean_iou": mean_iou,
        "coverage": coverage,
        "specificity": specificity,
    }
