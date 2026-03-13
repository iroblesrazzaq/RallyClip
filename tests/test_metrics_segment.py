from __future__ import annotations

import numpy as np

from training.metrics.segment import (
    _segments_from_binary,
    compute_segment_metrics,
    compute_time_point_classification_metrics,
    compute_time_segment_metrics,
    compute_weighted_segment_score,
)


def test_segments_from_binary():
    binary = np.array([0, 1, 1, 0, 1], dtype=int)
    assert _segments_from_binary(binary) == [(1, 3), (4, 5)]


def test_segment_metrics_basic():
    y_true = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=int)
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0], dtype=int)
    metrics = compute_segment_metrics(y_true, y_pred, iou_threshold=0.5)
    assert metrics["segment_precision"] == 1.0
    assert metrics["segment_recall"] == 1.0
    assert metrics["segment_f1"] == 1.0
    assert metrics["coverage"] == 0.6
    assert metrics["specificity"] == 1.0
    assert abs(metrics["mean_iou"] - 0.5833) < 1e-3


def test_weighted_segment_score():
    metrics = {"segment_recall": 0.8, "coverage": 0.7, "specificity": 0.9}
    score = compute_weighted_segment_score(metrics, segment_recall_weight=0.4, coverage_weight=0.4, specificity_weight=0.2)
    assert abs(score - 0.78) < 1e-9


def test_time_segment_metrics_basic():
    timestamps = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_pred = np.array([0, 1, 0, 0], dtype=int)
    metrics = compute_time_segment_metrics(y_true, y_pred, timestamps, iou_threshold=0.6)
    assert metrics["segment_precision"] == 0.0
    assert metrics["segment_recall"] == 0.0
    assert metrics["segment_f1"] == 0.0
    assert metrics["coverage"] == 0.5
    assert metrics["specificity"] == 1.0


def test_time_point_classification_metrics_basic():
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    y_true = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    y_pred = np.array([0, 1, 0, 0, 0, 1], dtype=int)
    metrics = compute_time_point_classification_metrics(
        y_true,
        y_pred,
        timestamps,
        iou_threshold=0.5,
        well_coverage_threshold=0.9,
    )
    assert metrics["total_true_points"] == 2
    assert metrics["total_pred_points"] == 2
    assert metrics["well_classified_points"] == 0
    assert metrics["cut_off_points"] == 1
    assert metrics["missed_points"] == 1
    assert metrics["false_detected_points"] == 1
    assert metrics["unmatched_predicted_points"] == 1
    assert metrics["well_classified_rate"] == 0.0
    assert metrics["cut_off_rate"] == 0.5
    assert metrics["missed_rate"] == 0.5
    assert metrics["false_detected_rate"] == 0.5
    assert metrics["unmatched_predicted_rate"] == 0.5
