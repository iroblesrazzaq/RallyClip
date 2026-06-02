from __future__ import annotations

import numpy as np

from helpers.module_stubs import import_data_preprocessor_with_stubs
from helpers.runtime_fixtures import fake_yolo_result, write_raw_pose_npz


def _disable_court_detection(monkeypatch, DataPreprocessor):
    monkeypatch.setattr(DataPreprocessor, "generate_court_mask", lambda self, video_path: None)


def test_runtime_preprocessor_propagates_box_conf_to_assigned_players(tmp_path, monkeypatch):
    preprocessor_module = import_data_preprocessor_with_stubs(monkeypatch)
    DataPreprocessor = preprocessor_module.DataPreprocessor
    _disable_court_detection(monkeypatch, DataPreprocessor)
    raw_path = write_raw_pose_npz(tmp_path / "raw_pose.npz", [fake_yolo_result(0)])
    output_path = tmp_path / "preprocessed.npz"

    preprocessor = DataPreprocessor(save_court_masks=False)
    assert preprocessor.preprocess_single_video(str(raw_path), str(tmp_path / "input.mp4"), str(output_path), overwrite=True)

    with np.load(output_path, allow_pickle=True) as data:
        near = data["near_players"][0]
        far = data["far_players"][0]

    expected_box_conf = fake_yolo_result(0).boxes.conf.numpy()
    assert near["box_conf"] == expected_box_conf[0]
    assert far["box_conf"] == expected_box_conf[1]


def test_runtime_preprocessor_assigns_near_and_far_deterministically(monkeypatch):
    preprocessor_module = import_data_preprocessor_with_stubs(monkeypatch)
    DataPreprocessor = preprocessor_module.DataPreprocessor
    result = fake_yolo_result(0)
    preprocessor = DataPreprocessor(save_court_masks=False)

    assigned = preprocessor.assign_players(
        {
            "boxes": result.boxes.xyxy.numpy(),
            "box_conf": result.boxes.conf.numpy(),
            "keypoints": result.keypoints.xy.numpy(),
            "conf": result.keypoints.conf.numpy(),
            "annotation_status": 0,
        }
    )

    assert assigned["near_player"]["box"][3] == result.boxes.xyxy.numpy()[0, 3]
    assert assigned["far_player"]["box"][3] == result.boxes.xyxy.numpy()[1, 3]
    assert assigned["near_player"]["box_conf"] == result.boxes.conf.numpy()[0]
    assert assigned["far_player"]["box_conf"] == result.boxes.conf.numpy()[1]


def test_runtime_preprocessor_closes_npz_file_handles(tmp_path, monkeypatch):
    preprocessor_module = import_data_preprocessor_with_stubs(monkeypatch)
    DataPreprocessor = preprocessor_module.DataPreprocessor
    _disable_court_detection(monkeypatch, DataPreprocessor)
    frames = [
        {
            "boxes": fake_yolo_result(0).boxes.xyxy.numpy(),
            "box_conf": fake_yolo_result(0).boxes.conf.numpy(),
            "keypoints": fake_yolo_result(0).keypoints.xy.numpy(),
            "conf": fake_yolo_result(0).keypoints.conf.numpy(),
            "annotation_status": 0,
        }
    ]
    opened = []

    class TrackingNpz:
        def __init__(self):
            self.closed = False

        def __getitem__(self, key):
            assert key == "frames"
            return np.asarray(frames, dtype=object)

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def fake_load(*_args, **_kwargs):
        npz = TrackingNpz()
        opened.append(npz)
        return npz

    monkeypatch.setattr(preprocessor_module.np, "load", fake_load)

    preprocessor = DataPreprocessor(save_court_masks=False)
    assert preprocessor.preprocess_single_video(
        str(tmp_path / "raw_pose.npz"),
        str(tmp_path / "input.mp4"),
        str(tmp_path / "preprocessed.npz"),
        overwrite=True,
    )

    assert opened
    assert all(npz.closed for npz in opened)
