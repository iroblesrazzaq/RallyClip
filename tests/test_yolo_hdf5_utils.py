from __future__ import annotations

import pytest

from pathlib import Path

import h5py
import numpy as np

try:
    yolo_module = pytest.importorskip("training.pose.yolo_hdf5")
except Exception as exc:
    pytest.skip(f"Skipping yolo_hdf5 tests: {exc}", allow_module_level=True)


def test_append_rows_resizes(tmp_path):
    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as h5f:
        dset = h5f.create_dataset("vals", shape=(0, 2), maxshape=(None, 2), dtype="f4")
        yolo_module._append_rows(dset, np.array([[1.0, 2.0]], dtype=np.float32))
        yolo_module._append_rows(dset, np.array([[3.0, 4.0]], dtype=np.float32))
        assert dset.shape == (2, 2)
        assert dset[1, 0] == 3.0


def test_validate_metadata_mismatch(tmp_path):
    h5_path = tmp_path / "meta.h5"
    with h5py.File(h5_path, "w") as h5f:
        h5f.attrs["video_path"] = "video.mp4"
        h5f.attrs["start_time"] = 0.0
        h5f.attrs["duration"] = -1.0
        h5f.attrs["yolo_model"] = "model.pt"
        h5f.attrs["conf"] = 0.25
        h5f.create_group("frames")
        h5f.create_group("detections")

    with h5py.File(h5_path, "r+") as h5f:
        with pytest.raises(ValueError):
            yolo_module.YoloHdf5Extractor._validate_metadata(
                h5f,
                video_path=Path("other.mp4"),
                start_time=0.0,
                duration=None,
                model_path="model.pt",
                conf=0.25,
            )
