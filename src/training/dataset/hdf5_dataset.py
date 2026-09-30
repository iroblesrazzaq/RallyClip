from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Datasets up to this many bytes (features, float32) are decompressed once into
# RAM. The build writes gzip with chunks that span many sequences, so per-item
# reads decompress hugely overlapping data (~200x slower than a single bulk read);
# holding the arrays in memory removes that entirely. Above the cap we fall back
# to lazy per-item reads so an outsized dataset can't exhaust memory.
IN_MEMORY_MAX_BYTES = 6 * 1024**3


class Hdf5SequenceDataset(Dataset):
    def __init__(self, h5_path: Path) -> None:
        self.h5_path = h5_path
        self._h5: Optional[h5py.File] = None
        self._features_ds: Optional[h5py.Dataset] = None
        self._targets_ds: Optional[h5py.Dataset] = None
        self._features_mem: Optional[np.ndarray] = None
        self._targets_mem: Optional[np.ndarray] = None

        with h5py.File(h5_path, "r") as h5f:
            feats = h5f["features"]
            self._length = int(feats.shape[0])
            self._feature_dim = int(feats.shape[-1])
            if feats.dtype.itemsize * int(np.prod(feats.shape)) <= IN_MEMORY_MAX_BYTES:
                # One bulk (decompress-once) read instead of per-item chunk thrash.
                self._features_mem = np.asarray(feats[:], dtype=np.float32)
                self._targets_mem = np.asarray(h5f["targets"][:], dtype=np.float32)

    def __len__(self) -> int:
        return self._length

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        if self._features_mem is not None:
            return (
                torch.from_numpy(self._features_mem[idx]),
                torch.from_numpy(self._targets_mem[idx]),
            )
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
