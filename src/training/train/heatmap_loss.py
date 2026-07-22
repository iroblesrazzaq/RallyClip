"""Boundary-heatmap segment loss (BSN-family startness/endness maps).

Each frame predicts (pointness logit, startness logit, endness logit). The two
boundary maps are trained against a soft Gaussian bump centred on the true
start/end frame of every point, so — unlike the anchor-free offset regression in
seg_loss.py — *every* frame gets a boundary gradient, including the negative
(off-point) regions the offset loss leaves with pointness-only signal.

Two heatmap loss modes:
  - "bce"   (default): balanced binary cross-entropy against the soft Gaussian
            target (target treated as a soft label). Positive-region frames are
            up-weighted by pos_weight so the mostly-zero background can't swamp
            the loss. Simple and hard to misconfigure — the BSN TEM recipe.
  - "focal": CenterNet penalty-reduced focal loss, down-weighting easy
            background by p**alpha and near-peak frames by (1 - target)**beta.
            More knobs; opt-in.

Targets are derived on the fly from the per-frame binary point targets [B, T],
mirroring how seg_loss.compute_offset_targets derives its offsets — no dataset
or builder change is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeatmapLossConfig:
    fps: float = 5.0
    sigma_seconds: float = 0.5  # 1σ ≈ "good" tol (0.5s), 2σ ≈ "decent" tol (1.5s)
    pos_weight: float = 3.0  # pointness BCE (same default as e2e_seg)
    cls_weight: float = 1.0
    start_weight: float = 1.0
    end_weight: float = 1.0
    heatmap_loss: str = "bce"  # "bce" | "focal"
    # "bce" mode: frames whose Gaussian target exceeds this get pos_weight up-weighting.
    heatmap_pos_threshold: float = 0.1
    heatmap_pos_weight: float = 20.0
    # "focal" mode (CenterNet):
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    # soft-argmax time term: collapse each boundary's heatmap to a predicted time
    # and penalize squared error to the true boundary time (seconds). Opt-in via
    # time_weight (0 = off). Directly optimizes the decoded boundary time (absolute
    # seconds), which the per-frame shape loss only constrains indirectly.
    time_weight: float = 0.0
    time_window_frames: Optional[int] = None  # None => round(2 * sigma_frames)
    time_temperature: float = 1.0


def _shift_right(x: torch.Tensor) -> torch.Tensor:
    """x shifted one frame later (frame t-1 lands at t); frame 0 gets False/0."""
    pad = torch.zeros_like(x[:, :1])
    return torch.cat([pad, x[:, :-1]], dim=1)


def _shift_left(x: torch.Tensor) -> torch.Tensor:
    pad = torch.zeros_like(x[:, :1])
    return torch.cat([x[:, 1:], pad], dim=1)


def boundary_markers(targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """First frame of each positive run (start marker) and last frame (end marker),
    both bool [B, T]. Same primitive as seg_loss (`pos & ~prev`), plus the mirror
    for run ends."""
    pos = targets > 0.5
    start_marker = pos & ~_shift_right(pos)
    end_marker = pos & ~_shift_left(pos)
    return start_marker, end_marker


def _dist_to_nearest(marker: torch.Tensor) -> torch.Tensor:
    """Per frame: distance in frames to the nearest True in `marker`, either
    direction. Frames with no marker anywhere get a large sentinel distance.
    Vectorized: distance-to-nearest-on-the-left via a forward cummax over marked
    indices, distance-to-nearest-on-the-right via the same on the flipped tensor,
    then elementwise min."""
    batch, length = marker.shape
    device = marker.device
    idx = torch.arange(length, device=device).unsqueeze(0).expand(batch, length)
    big = length + 1

    def dist_left(m: torch.Tensor) -> torch.Tensor:
        # last marked index at or before each position (or -big if none yet)
        marked_idx = torch.where(m, idx, torch.full_like(idx, -big))
        last = torch.cummax(marked_idx, dim=1).values
        return idx - last  # huge where no marker seen yet

    left = dist_left(marker)
    right = dist_left(marker.flip(1)).flip(1)
    dist = torch.minimum(left, right)
    return dist.clamp(max=big).to(torch.float32)


def gaussian_target(dist_frames: torch.Tensor, sigma_frames: float) -> torch.Tensor:
    """exp(-0.5 (d/σ)^2). d is distance-in-frames to the nearest boundary, so with
    multiple boundaries in a window each frame takes the nearest (== union of the
    per-boundary Gaussians, since a Gaussian is monotone in distance)."""
    sigma = max(float(sigma_frames), 1e-6)
    return torch.exp(-0.5 * (dist_frames / sigma) ** 2)


def build_heatmap_targets(
    targets: torch.Tensor, cfg: HeatmapLossConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(start_heatmap, end_heatmap) soft Gaussian targets in [0, 1], shape [B, T]."""
    start_marker, end_marker = boundary_markers(targets)
    sigma_frames = cfg.sigma_seconds * cfg.fps
    start_t = gaussian_target(_dist_to_nearest(start_marker), sigma_frames)
    end_t = gaussian_target(_dist_to_nearest(end_marker), sigma_frames)
    return start_t, end_t


def _balanced_bce(logits: torch.Tensor, target: torch.Tensor, cfg: HeatmapLossConfig) -> torch.Tensor:
    """BCE against a soft target, with near-boundary frames up-weighted so the
    mostly-zero background does not dominate."""
    per_frame = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = torch.where(
        target >= cfg.heatmap_pos_threshold,
        torch.full_like(target, cfg.heatmap_pos_weight),
        torch.ones_like(target),
    )
    return (per_frame * weight).sum() / weight.sum().clamp_min(1.0)


def _penalty_reduced_focal(logits: torch.Tensor, target: torch.Tensor, cfg: HeatmapLossConfig) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss for a Gaussian-splatted target.
    Peak frames (target==1): (1-p)^α log p. Others: (1-target)^β p^α log(1-p).
    Normalized by the number of peak (target==1) frames."""
    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    pos_mask = target >= 1.0 - 1e-6
    pos_loss = (1.0 - p) ** cfg.focal_alpha * torch.log(p)
    neg_loss = (1.0 - target) ** cfg.focal_beta * p ** cfg.focal_alpha * torch.log(1.0 - p)
    pos_term = (pos_loss * pos_mask.float()).sum()
    neg_term = (neg_loss * (~pos_mask).float()).sum()
    n_pos = pos_mask.float().sum().clamp_min(1.0)
    return -(pos_term + neg_term) / n_pos


def soft_argmax_time_loss(
    logits: torch.Tensor, marker: torch.Tensor, fps: float, window_frames: int, temperature: float = 1.0
) -> torch.Tensor:
    """For each True in `marker` (a true boundary frame at index c), soft-argmax the
    heatmap logits over the window [c-W, c+W] to a predicted frame time, and penalize
    the squared time error to c in seconds. Fully vectorized: pad by W with -inf so
    every frame has a full centered window, softmax each, weighted-sum the absolute
    frame indices, then masked-mean the squared per-boundary error over markers.

    At init (flat logits over a window symmetric about c) the soft-argmax returns c,
    so the term starts near zero and only grows as the peak drifts off the boundary."""
    b, t = logits.shape
    device, dtype = logits.device, logits.dtype
    w = int(window_frames)
    if w < 1 or marker.sum() == 0:
        return logits.sum() * 0.0
    neg = torch.finfo(dtype).min
    padded = torch.nn.functional.pad(logits, (w, w), value=neg)  # [b, t + 2w]
    win = padded.unfold(dimension=1, size=2 * w + 1, step=1)  # [b, t, 2w+1], window per center
    offsets = torch.arange(-w, w + 1, device=device, dtype=dtype)  # [2w+1]
    centers = torch.arange(t, device=device, dtype=dtype).unsqueeze(1)  # [t, 1]
    abs_idx = centers + offsets.unsqueeze(0)  # [t, 2w+1] absolute frame index per slot
    weights = torch.softmax(win / temperature, dim=2)  # -inf pads -> 0 weight
    t_pred = (weights * abs_idx.unsqueeze(0)).sum(dim=2)  # [b, t] predicted boundary frame
    target_frame = torch.arange(t, device=device, dtype=dtype).unsqueeze(0)  # center == true boundary
    err_sec = (t_pred - target_frame) / fps
    sq = err_sec * err_sec
    m = marker.to(dtype)
    return (sq * m).sum() / m.sum().clamp_min(1.0)


def _heatmap_term(logits: torch.Tensor, target: torch.Tensor, cfg: HeatmapLossConfig) -> torch.Tensor:
    if cfg.heatmap_loss == "focal":
        return _penalty_reduced_focal(logits, target, cfg)
    if cfg.heatmap_loss == "bce":
        return _balanced_bce(logits, target, cfg)
    raise ValueError(f"Unknown heatmap_loss: {cfg.heatmap_loss!r} (expected bce | focal)")


class E2EHeatmapLoss(nn.Module):
    def __init__(self, cfg: HeatmapLossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("_pos_weight", torch.tensor([cfg.pos_weight]))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)

    def forward(
        self,
        pointness_logits: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        start_t, end_t = build_heatmap_targets(targets, cfg)

        loss_cls = self.bce(pointness_logits, targets)
        loss_start = _heatmap_term(start_logits, start_t, cfg)
        loss_end = _heatmap_term(end_logits, end_t, cfg)

        loss_time_start = pointness_logits.sum() * 0.0
        loss_time_end = pointness_logits.sum() * 0.0
        if cfg.time_weight > 0.0:
            sigma_frames = cfg.sigma_seconds * cfg.fps
            w = cfg.time_window_frames if cfg.time_window_frames is not None else max(1, round(2 * sigma_frames))
            start_marker, end_marker = boundary_markers(targets)
            # Drop boundaries touching the window edge: their true time may lie off-window
            # (truncated run), so the center frame isn't a trustworthy target.
            start_marker = start_marker.clone(); start_marker[:, 0] = False
            end_marker = end_marker.clone(); end_marker[:, -1] = False
            loss_time_start = soft_argmax_time_loss(start_logits, start_marker, cfg.fps, w, cfg.time_temperature)
            loss_time_end = soft_argmax_time_loss(end_logits, end_marker, cfg.fps, w, cfg.time_temperature)

        total = (
            cfg.cls_weight * loss_cls
            + cfg.start_weight * loss_start
            + cfg.end_weight * loss_end
            + cfg.time_weight * (loss_time_start + loss_time_end)
        )
        components = {
            "loss_cls": float(loss_cls.item()),
            "loss_start": float(loss_start.item()),
            "loss_end": float(loss_end.item()),
            "loss_time_start": float(loss_time_start.item()),
            "loss_time_end": float(loss_time_end.item()),
        }
        return total, components
