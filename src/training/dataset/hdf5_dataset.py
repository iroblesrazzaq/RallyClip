from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import torch
from torch.utils.data import Dataset


class Hdf5SequenceDataset(Dataset):
    def __init__(self, h5_path: Path) -> None:
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as h5f:
            self.features = torch.tensor(h5f["features"][:], dtype=torch.float32)
            self.targets = torch.tensor(h5f["targets"][:], dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
