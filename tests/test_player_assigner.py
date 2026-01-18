from __future__ import annotations

import numpy as np

from training.preprocess.player_assigner import PlayerAssigner


def _make_data(boxes):
    boxes = np.array(boxes, dtype=np.float32)
    keypoints = np.zeros((len(boxes), 17, 2), dtype=np.float32)
    confs = np.ones((len(boxes), 17), dtype=np.float32)
    box_conf = np.ones((len(boxes),), dtype=np.float32)
    return {
        "boxes": boxes,
        "keypoints": keypoints,
        "keypoint_conf": confs,
        "box_conf": box_conf,
    }


def test_assign_empty():
    assigner = PlayerAssigner()
    data = {
        "boxes": np.empty((0, 4), dtype=np.float32),
        "keypoints": np.empty((0, 17, 2), dtype=np.float32),
        "keypoint_conf": np.empty((0, 17), dtype=np.float32),
        "box_conf": np.empty((0,), dtype=np.float32),
    }
    players = assigner.assign(data)
    assert np.all(players["near_box"] == -1)
    assert np.all(players["far_box"] == -1)


def test_assign_near_far():
    assigner = PlayerAssigner()
    data = _make_data([[100, 100, 200, 200], [600, 300, 700, 600]])
    players = assigner.assign(data)
    near_box = players["near_box"][0]
    far_box = players["far_box"][0]
    assert np.allclose(near_box, np.array([600, 300, 700, 600], dtype=np.float32))
    assert np.allclose(far_box, np.array([100, 100, 200, 200], dtype=np.float32))


def test_merge_edge_zone():
    assigner = PlayerAssigner()
    data = _make_data([[10, 10, 110, 210], [20, 20, 120, 220]])
    players = assigner.assign(data)
    near_box = players["near_box"][0]
    assert np.allclose(near_box, np.array([10, 10, 120, 220], dtype=np.float32))
    assert np.all(players["far_box"] == -1)
