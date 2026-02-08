from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class Hdf5SequenceDataset(Dataset):
    def __init__(self, h5_path: Path) -> None:
        self.h5_path = h5_path
        self._h5: Optional[h5py.File] = None
        self._features_ds: Optional[h5py.Dataset] = None
        self._targets_ds: Optional[h5py.Dataset] = None

        with h5py.File(h5_path, "r") as h5f:
            self._length = int(h5f["features"].shape[0])
            self._feature_dim = int(h5f["features"].shape[-1])

    def __len__(self) -> int:
        return self._length

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        self._ensure_open()
        assert self._features_ds is not None
        assert self._targets_ds is not None
        features = np.asarray(self._features_ds[idx], dtype=np.float32)
        targets = np.asarray(self._targets_ds[idx], dtype=np.float32)
        return torch.from_numpy(features), torch.from_numpy(targets)

    def _ensure_open(self) -> None:
        if self._h5 is not None:
            return
        self._h5 = h5py.File(self.h5_path, "r")
        self._features_ds = self._h5["features"]
        self._targets_ds = self._h5["targets"]

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._features_ds = None
        self._targets_ds = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        # File handles are not picklable and each worker should reopen independently.
        state["_h5"] = None
        state["_features_ds"] = None
        state["_targets_ds"] = None
        return state
