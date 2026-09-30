from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from training.dataset.hdf5_dataset import Hdf5SequenceDataset
from training.train.loop import _default_pos_weight


def test_default_pos_weight_from_25pct_positive(tmp_path):
    path = tmp_path / "train.h5"
    # 4 sequences × 4 frames; 4 positives out of 16 => 25% => pos_weight=3.0
    features = np.zeros((4, 4, 8), dtype=np.float32)
    targets = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("features", data=features)
        h5f.create_dataset("targets", data=targets)

    ds = Hdf5SequenceDataset(path)
    assert abs(_default_pos_weight(ds) - 3.0) < 1e-6
