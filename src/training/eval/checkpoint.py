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
    model = build_model_from_checkpoint(ckpt, feature_dim=dataset.feature_dim).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return evaluate_model(model, loader, device, threshold, segment_cfg, fps, criterion)


def build_model_from_checkpoint(ckpt: Dict[str, Any], *, feature_dim: int) -> TennisPointLSTM:
    """Build TennisPointLSTM from stored arch keys (no state-dict archaeology)."""
    arch = ckpt.get("arch") if isinstance(ckpt.get("arch"), dict) else {}
    input_size = int(arch.get("input_size") or ckpt.get("input_size") or feature_dim)
    hidden_size = int(arch.get("hidden_size") or ckpt.get("hidden_size") or 128)
    num_layers = int(arch.get("num_layers") or ckpt.get("num_layers") or 2)
    bidirectional = bool(arch.get("bidirectional", ckpt.get("bidirectional", True)))
    return TennisPointLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        return_logits=True,
    )


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
