from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from training.preprocess.preprocessor import UNLABELED_TARGET, _build_targets, _filter_by_court, _sample_indices


def test_sample_indices_deterministic():
    timestamps = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float64)
    sampled = _sample_indices(timestamps, target_fps=1.0)
    assert sampled.tolist() == [0, 4]


def test_sample_indices_identity_at_matching_fps():
    timestamps = np.arange(0, 2.0, 0.2, dtype=np.float64)  # 5 fps
    sampled = _sample_indices(timestamps, target_fps=5.0)
    assert sampled.tolist() == list(range(len(timestamps)))
    assert len(set(sampled.tolist())) == len(sampled)


def test_build_targets_segments():
    annotations = {
        "segments": [
            {"start_time": 0.4, "end_time": 0.6},
            {"start_time": 1.0, "end_time": 1.1},
        ],
        "metadata": {},
    }
    timestamps = np.array([0.0, 0.5, 0.6, 0.7, 1.05], dtype=np.float64)
    targets = _build_targets(timestamps, annotations)
    assert targets.tolist() == [0, 1, 1, 0, 1]


def test_build_targets_unsorted_matches_sorted():
    timestamps = np.array([0.0, 0.5, 1.5, 2.5], dtype=np.float64)
    sorted_ann = {
        "segments": [
            {"start_time": 0.4, "end_time": 0.6},
            {"start_time": 2.0, "end_time": 3.0},
        ],
        "metadata": {},
    }
    unsorted_ann = {
        "segments": [
            {"start_time": 2.0, "end_time": 3.0},
            {"start_time": 0.4, "end_time": 0.6},
        ],
        "metadata": {},
    }
    assert _build_targets(timestamps, sorted_ann).tolist() == _build_targets(timestamps, unsorted_ann).tolist()


def test_build_targets_ignore_prefix():
    annotations = {
        "segments": [{"start_time": 2.0, "end_time": 3.0}],
        "metadata": {"ignore_before_s": 1.5},
    }
    timestamps = np.array([0.0, 1.0, 1.5, 2.5, 3.5], dtype=np.float64)
    targets = _build_targets(timestamps, annotations)
    assert targets.tolist() == [UNLABELED_TARGET, UNLABELED_TARGET, 0, 1, 0]


def test_build_targets_overlap_errors():
    annotations = {
        "segments": [
            {"start_time": 0.0, "end_time": 2.0},
            {"start_time": 1.0, "end_time": 3.0},
        ],
        "metadata": {},
    }
    with pytest.raises(ValueError, match="Overlapping"):
        _build_targets(np.array([0.5, 1.5]), annotations)


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


def _write_pose_h5(path, n_frames, width, height):
    import h5py

    with h5py.File(path, "w") as h5f:
        h5f.attrs["width"], h5f.attrs["height"] = width, height
        frames = h5f.create_group("frames")
        frames.create_dataset("frame_index", data=np.arange(n_frames, dtype=np.int64))
        frames.create_dataset("timestamps", data=np.arange(n_frames, dtype=np.float64) / 5.0)
        frames.create_dataset("frame_offsets", data=np.zeros(n_frames + 1, dtype=np.int64))
        det = h5f.create_group("detections")
        for name, shape in {"boxes": (4,), "box_conf": (), "keypoints": (17, 2), "keypoint_conf": (17,)}.items():
            det.create_dataset(name, shape=(0, *shape), dtype="f4")


def test_crop_sidecar_of_another_frame_size_is_rejected(tmp_path):
    # extract_crop_poses.py always works at 1920x1080; on a 1280x720 root its
    # boxes would land in the wrong place, so preprocessing must refuse it.
    import json

    from training.preprocess.preprocessor import Hdf5Preprocessor, PreprocessConfig

    raw, crop = tmp_path / "raw.h5", tmp_path / "crop.h5"
    _write_pose_h5(raw, 10, 1280, 720)
    _write_pose_h5(crop, 10, 1920, 1080)
    annotations = tmp_path / "clip.mp4.json"
    annotations.write_text(json.dumps({"video_path": "clip.mp4", "segments": [{"start_time": 0.2, "end_time": 1.0}]}))
    pre = Hdf5Preprocessor(
        PreprocessConfig(
            target_fps=5.0, save_court_masks=False, court_model_path="", court_target_time=None,
            court_enabled=False, expect_width=1280, expect_height=720,
        )
    )
    with pytest.raises(ValueError, match="1920x1080 but raw poses are 1280x720"):
        pre.preprocess(tmp_path, raw, tmp_path / "clip.mp4", annotations, tmp_path / "out.h5", crop_h5_path=crop)
