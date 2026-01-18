from __future__ import annotations

import numpy as np

from training.metrics.frame import compute_frame_metrics


def test_frame_metrics_basic():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.6])
    metrics = compute_frame_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["auroc"] is not None


def test_frame_metrics_auroc_none():
    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3])
    metrics = compute_frame_metrics(y_true, y_prob)
    assert metrics["auroc"] is None
