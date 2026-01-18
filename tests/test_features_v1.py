from __future__ import annotations

import numpy as np

from training.features.v1 import FeatureSetV1


def test_feature_dim_matches_vector():
    feature_set = FeatureSetV1()
    per_player = FeatureSetV1.feature_dim() // 2

    near = {"exists": False}
    far = {"exists": False}
    vector = feature_set.build_feature_vector(near, far, None, None, None, dt=1.0)
    assert vector.shape[0] == FeatureSetV1.feature_dim()
    assert vector.shape[0] == per_player * 2


def test_feature_vector_box_conf_positions():
    feature_set = FeatureSetV1()
    per_player = FeatureSetV1.feature_dim() // 2

    near = {
        "exists": True,
        "box": np.array([0.0, 0.0, 10.0, 20.0], dtype=np.float32),
        "keypoints": np.zeros((17, 2), dtype=np.float32),
        "conf": np.ones((17,), dtype=np.float32),
        "box_conf": 0.75,
    }
    far = {
        "exists": True,
        "box": np.array([100.0, 100.0, 120.0, 140.0], dtype=np.float32),
        "keypoints": np.zeros((17, 2), dtype=np.float32),
        "conf": np.ones((17,), dtype=np.float32),
        "box_conf": 0.25,
    }
    vector = feature_set.build_feature_vector(near, far, near, far, {"near": (0.0, 0.0), "far": (0.0, 0.0)}, dt=1.0)
    assert vector[0] == 1.0
    assert vector[per_player - 1] == 0.75
    assert vector[-1] == 0.25
