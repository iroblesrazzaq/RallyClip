from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.metrics.frame import compute_frame_metrics
from training.metrics.segment import compute_segment_metrics
from infer.inference import gaussian_filter1d, hysteresis_threshold


@dataclass
class SegmentEvalConfig:
    low: float = 0.45
    high: float = 0.8
    sigma: float = 1.5
    min_dur_sec: float = 0.5


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    segment_cfg: SegmentEvalConfig,
    fps: float,
    criterion: torch.nn.Module,
) -> Tuple[Dict[str, float], float]:
    model.eval()
    all_probs = []
    all_targets = []
    seg_metrics = []
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            total_loss += float(loss.item())
            batches += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            all_probs.append(probs.reshape(-1))
            all_targets.append(targets_np.reshape(-1))

            min_dur_frames = int(round(segment_cfg.min_dur_sec * fps))
            for seq_prob, seq_target in zip(probs, targets_np):
                smoothed = gaussian_filter1d(seq_prob.astype(np.float32), sigma=segment_cfg.sigma)
                binary = hysteresis_threshold(
                    smoothed,
                    low=segment_cfg.low,
                    high=segment_cfg.high,
                    min_duration=min_dur_frames,
                )
                seg_metrics.append(compute_segment_metrics(seq_target.astype(int), binary.astype(int)))

    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_true = np.concatenate(all_targets) if all_targets else np.array([])

    frame_metrics = compute_frame_metrics(y_true, y_prob, threshold=threshold) if y_true.size else {}
    avg_seg = _average_segment_metrics(seg_metrics)

    metrics = {**frame_metrics, **avg_seg}
    avg_loss = total_loss / max(batches, 1)
    return metrics, avg_loss


def _average_segment_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    out = {}
    for key in keys:
        vals = [m.get(key, 0.0) for m in metrics_list]
        out[key] = float(np.mean(vals))
    return out
