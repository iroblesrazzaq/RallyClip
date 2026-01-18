from __future__ import annotations

import h5py
import numpy as np

from training.dataset.hdf5_dataset import Hdf5SequenceDataset


def test_hdf5_dataset_loads(tmp_path):
    h5_path = tmp_path / "dataset.h5"
    features = np.random.rand(2, 3, 4).astype(np.float32)
    targets = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("features", data=features)
        h5f.create_dataset("targets", data=targets)

    dataset = Hdf5SequenceDataset(h5_path)
    assert len(dataset) == 2
    x, y = dataset[0]
    assert x.shape == (3, 4)
    assert y.shape == (3,)
