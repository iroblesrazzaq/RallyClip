from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from training.dataset.hdf5_dataset import Hdf5SequenceDataset
from training.eval.evaluator import SegmentEvalConfig, evaluate_model
from training.models.lstm import TennisPointLSTM

logger = logging.getLogger(__name__)


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
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    hidden_size, num_layers, bidirectional = _infer_lstm_shape(state_dict)
    model = TennisPointLSTM(
        input_size=dataset.feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2,
        bidirectional=bidirectional,
        return_logits=True,
    ).to(device)
    model.load_state_dict(state_dict)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return evaluate_model(model, loader, device, threshold, segment_cfg, fps, criterion)


def _resolve_device(device: str | None) -> torch.device:
    if device:
        requested = str(device).lower()
        if requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("Requested device 'mps' is unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infer_lstm_shape(state_dict: Dict[str, Any]) -> Tuple[int, int, bool]:
    weight_ih_keys = sorted(k for k in state_dict if k.startswith("lstm.weight_ih_l"))
    if not weight_ih_keys:
        raise ValueError("Checkpoint state_dict does not contain LSTM weights")

    num_layers = max(int(key.split("l")[1].split(".")[0].replace("_reverse", "")) for key in weight_ih_keys) + 1
    bidirectional = any("_reverse" in key for key in weight_ih_keys)
    hidden_size = int(state_dict[weight_ih_keys[0]].shape[0] // 4)
    return hidden_size, num_layers, bidirectional
