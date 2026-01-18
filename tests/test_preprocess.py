from __future__ import annotations

import numpy as np

from training.preprocess.preprocessor import _build_targets, _filter_by_court, _sample_indices


def test_sample_indices_deterministic():
    timestamps = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float64)
    sampled = _sample_indices(timestamps, target_fps=1.0)
    assert sampled.tolist() == [0, 4]


def test_build_targets_segments():
    annotations = {
        "segments": [
            {"start_time": 0.4, "end_time": 0.6},
            {"start_time": 1.0, "end_time": 1.1},
        ]
    }
    timestamps = np.array([0.0, 0.5, 0.6, 0.7, 1.05], dtype=np.float64)
    targets = _build_targets(timestamps, annotations)
    assert targets.tolist() == [0, 1, 1, 0, 1]


def test_filter_by_court_mask():
    boxes = np.array([[0, 0, 2, 2], [2, 2, 4, 4]], dtype=np.float32)
    box_conf = np.array([0.9, 0.8], dtype=np.float32)
    keypoints = np.zeros((2, 17, 2), dtype=np.float32)
    keypoint_conf = np.ones((2, 17), dtype=np.float32)

    mask_all_ones = np.ones((5, 5), dtype=np.uint8)
    filtered = _filter_by_court(boxes, box_conf, keypoints, keypoint_conf, mask_all_ones)
    assert filtered["boxes"].shape[0] == 0

    mask_all_zeros = np.zeros((5, 5), dtype=np.uint8)
    filtered = _filter_by_court(boxes, box_conf, keypoints, keypoint_conf, mask_all_zeros)
    assert filtered["boxes"].shape[0] == 2
