from __future__ import annotations

import numpy as np

from training.metrics.segment import _segments_from_binary, compute_segment_metrics


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
