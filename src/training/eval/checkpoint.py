from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from training.dataset.hdf5_dataset import Hdf5SequenceDataset
from training.eval.evaluator import SegmentEvalConfig, evaluate_model
from training.models.lstm import TennisPointLSTM


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    device_str: str | None,
    threshold: float,
    segment_cfg: SegmentEvalConfig,
    fps: float,
    pos_weight: float,
) -> Tuple[Dict[str, float], float]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = Hdf5SequenceDataset(dataset_path)
    device = _resolve_device(device_str)
    model = TennisPointLSTM(input_size=dataset.features.shape[-1], return_logits=True).to(device)

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return evaluate_model(model, loader, device, threshold, segment_cfg, fps, criterion)


def _resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
