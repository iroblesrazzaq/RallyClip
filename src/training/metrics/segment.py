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


def compute_time_segment_metrics(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    timestamps: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_pred_bin = np.asarray(y_pred_bin).astype(int)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if y_true_bin.shape != y_pred_bin.shape or y_true_bin.shape != timestamps.shape:
        raise ValueError("y_true_bin, y_pred_bin, and timestamps must have the same shape")
    if timestamps.size == 0:
        return {
            "segment_precision": 0.0,
            "segment_recall": 0.0,
            "segment_f1": 0.0,
            "mean_iou": 0.0,
            "coverage": 0.0,
            "specificity": 0.0,
        }

    frame_intervals = _frame_intervals_from_timestamps(timestamps)
    true_segments = _time_segments_from_binary(y_true_bin, frame_intervals)
    pred_segments = _time_segments_from_binary(y_pred_bin, frame_intervals)

    matched_true = set()
    matched_pred = set()
    ious: List[float] = []

    for p_idx, pred in enumerate(pred_segments):
        best_iou = 0.0
        best_t = None
        for t_idx, true in enumerate(true_segments):
            if t_idx in matched_true:
                continue
            score = _interval_iou(pred, true)
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

    positive_total = 0.0
    negative_total = 0.0
    positive_overlap = 0.0
    negative_overlap = 0.0
    for (start, end), y_true, y_pred in zip(frame_intervals, y_true_bin, y_pred_bin):
        duration = max(0.0, float(end - start))
        if y_true == 1:
            positive_total += duration
            if y_pred == 1:
                positive_overlap += duration
        else:
            negative_total += duration
            if y_pred == 0:
                negative_overlap += duration

    coverage = (positive_overlap / positive_total) if positive_total > 0 else 0.0
    specificity = (negative_overlap / negative_total) if negative_total > 0 else 0.0

    return {
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
        "mean_iou": mean_iou,
        "coverage": coverage,
        "specificity": specificity,
    }


def compute_weighted_segment_score(
    metrics: Dict[str, Any],
    *,
    segment_recall_weight: float = 0.4,
    coverage_weight: float = 0.4,
    specificity_weight: float = 0.2,
) -> float:
    return (
        float(metrics.get("segment_recall", 0.0)) * segment_recall_weight
        + float(metrics.get("coverage", 0.0)) * coverage_weight
        + float(metrics.get("specificity", 0.0)) * specificity_weight
    )


def _frame_intervals_from_timestamps(timestamps: np.ndarray) -> List[Tuple[float, float]]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    n = int(timestamps.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [(float(timestamps[0]), float(timestamps[0]) + 1.0)]

    boundaries = np.empty(n + 1, dtype=np.float64)
    boundaries[1:-1] = (timestamps[:-1] + timestamps[1:]) / 2.0
    first_gap = max(1e-6, float(timestamps[1] - timestamps[0]))
    last_gap = max(1e-6, float(timestamps[-1] - timestamps[-2]))
    boundaries[0] = float(timestamps[0]) - first_gap / 2.0
    boundaries[-1] = float(timestamps[-1]) + last_gap / 2.0

    intervals: List[Tuple[float, float]] = []
    for idx in range(n):
        start = float(boundaries[idx])
        end = float(boundaries[idx + 1])
        if end <= start:
            end = start + 1e-6
        intervals.append((start, end))
    return intervals


def _time_segments_from_binary(
    binary: np.ndarray,
    frame_intervals: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    in_seg = False
    start = 0.0
    for idx, val in enumerate(np.asarray(binary).astype(int)):
        if val and not in_seg:
            in_seg = True
            start = frame_intervals[idx][0]
        elif not val and in_seg:
            segments.append((start, frame_intervals[idx - 1][1]))
            in_seg = False
    if in_seg:
        segments.append((start, frame_intervals[-1][1]))
    return segments


def _interval_iou(seg_a: Tuple[float, float], seg_b: Tuple[float, float]) -> float:
    a0, a1 = seg_a
    b0, b1 = seg_b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return (inter / union) if union > 0 else 0.0
