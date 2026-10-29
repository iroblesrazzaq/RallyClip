from __future__ import annotations

import types

import numpy as np
import pytest

from helpers.module_stubs import import_data_preprocessor_with_stubs
from helpers.runtime_fixtures import fake_yolo_result, write_raw_pose_npz


def test_runtime_preprocessor_propagates_box_conf_to_assigned_players(tmp_path, monkeypatch):
    preprocessor_module = import_data_preprocessor_with_stubs(monkeypatch)
    DataPreprocessor = preprocessor_module.DataPreprocessor
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


class _FakeCourt:
    def __init__(self, mask, frame_size=(1280, 720)):
        self._mask = mask
        self.frame_size = frame_size

    def out_mask(self):
        if isinstance(self._mask, Exception):
            raise self._mask
        return self._mask


def test_compute_court_mask_uses_court_model_mask(tmp_path, monkeypatch):
    module = import_data_preprocessor_with_stubs(monkeypatch)
    detected = np.zeros((720, 1280), dtype=np.uint8)
    detected[:120, :] = 255  # a plausible non-empty 'out' mask

    pre = module.DataPreprocessor(save_court_masks=False)
    mask, meta = pre.compute_court_mask(str(tmp_path / "v.mp4"), _FakeCourt(detected))

    assert meta["detected"] is True and meta["source"] == "detected"
    assert np.array_equal(mask, detected)


@pytest.mark.parametrize("court_mask", [None, RuntimeError("bad points")])
def test_compute_court_mask_default_at_court_frame_size(tmp_path, monkeypatch, court_mask):
    pytest.importorskip("cv2")
    module = import_data_preprocessor_with_stubs(monkeypatch)
    pre = module.DataPreprocessor(save_court_masks=False)
    mask, meta = pre.compute_court_mask(str(tmp_path / "v.mp4"), _FakeCourt(court_mask, (1920, 1080)))

    assert meta["detected"] is False and meta["source"] == "default"
    assert mask.shape == (1080, 1920)


def test_compute_court_mask_falls_back_to_default(tmp_path, monkeypatch):
    # No court model result -> default mask at the probed (here: fallback) frame size.
    pytest.importorskip("cv2")
    module = import_data_preprocessor_with_stubs(monkeypatch)
    pre = module.DataPreprocessor(save_court_masks=False)
    mask, meta = pre.compute_court_mask(str(tmp_path / "missing.mp4"))

    assert meta["detected"] is False and meta["source"] == "default"
    assert mask is not None and mask.shape == (720, 1280)
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.any(mask) and not np.all(mask)  # a real court envelope, not all-in / all-out


def test_compute_court_mask_fallback_uses_native_video_shape(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    module = import_data_preprocessor_with_stubs(monkeypatch)
    monkeypatch.setattr(module, "probe_video", lambda _: types.SimpleNamespace(width=1920, height=1080))
    pre = module.DataPreprocessor(save_court_masks=False)
    mask, meta = pre.compute_court_mask(str(tmp_path / "input.mp4"))

    assert meta["detected"] is False and meta["source"] == "default"
    assert mask.shape == (1080, 1920)


def test_preprocess_rescales_from_native_video_shape(tmp_path, monkeypatch):
    preprocessor_module = import_data_preprocessor_with_stubs(monkeypatch)
    DataPreprocessor = preprocessor_module.DataPreprocessor
    monkeypatch.setattr(preprocessor_module, "probe_video", lambda _: types.SimpleNamespace(width=1920, height=1080))
    raw_path = write_raw_pose_npz(tmp_path / "raw_pose.npz", [fake_yolo_result(0)])
    output_path = tmp_path / "preprocessed.npz"

    preprocessor = DataPreprocessor(save_court_masks=False)
    assert preprocessor.preprocess_single_video(str(raw_path), str(tmp_path / "input.mp4"), str(output_path), overwrite=True)

    with np.load(output_path, allow_pickle=True) as data:
        near = data["near_players"][0]

    np.testing.assert_allclose(near["box"], np.array([66.6667, 266.6667, 93.3333, 333.3333]), rtol=1e-4)


def test_default_court_mask_resizes_to_frame(monkeypatch):
    pytest.importorskip("cv2")
    module = import_data_preprocessor_with_stubs(monkeypatch)
    pre = module.DataPreprocessor(save_court_masks=False)
    resized = pre._load_default_court_mask((360, 640, 3))

    assert resized.shape == (360, 640)
    assert set(np.unique(resized)).issubset({0, 255})
