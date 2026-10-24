"""Boundary-heatmap segment loss (BSN-family startness/endness maps).

Each frame predicts (pointness logit, startness logit, endness logit). The two
boundary maps are trained against a soft Gaussian bump centred on the true
start/end frame of every point, so *every* frame gets a boundary gradient,
including the off-point regions a pointness-only loss leaves without one.

Two heatmap loss modes:
  - "bce"   (default): balanced binary cross-entropy against the soft Gaussian
            target (target treated as a soft label). Positive-region frames are
            up-weighted by pos_weight so the mostly-zero background can't swamp
            the loss. Simple and hard to misconfigure — the BSN TEM recipe.
  - "focal": CenterNet penalty-reduced focal loss, down-weighting easy
            background by p**alpha and near-peak frames by (1 - target)**beta.
            More knobs; opt-in.

Targets are derived on the fly from the per-frame binary point targets [B, T],
so no dataset or builder change is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeatmapLossConfig:
    fps: float = 5.0
    sigma_seconds: float = 0.5  # 1σ ≈ "good" tol (0.5s), 2σ ≈ "decent" tol (1.5s)
    # Asymmetric boundary targets: sigma for the OUTSIDE-the-point side (before a
    # start / after an end). None => symmetric (sigma_seconds both sides). A fatter
    # outside tail biases the soft-argmax decode outward — cutting into the point
    # (late start / early end) ruins a clip, while over-including dead time is
    # nearly free; the val pad sweep measured a systematic ~0.5s late-start bias
    # this is designed to absorb in a learned, per-boundary way.
    sigma_out_seconds: Optional[float] = None
    pos_weight: float = 3.0  # pointness BCE
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

    @classmethod
    def from_train_cfg(cls, train_cfg: Dict[str, Any], fps: float, pos_weight: float) -> "HeatmapLossConfig":
        sigma_out = train_cfg.get("heatmap_sigma_out_seconds")
        return cls(
            fps=float(fps),
            sigma_seconds=float(train_cfg.get("heatmap_sigma_seconds", 0.5)),
            sigma_out_seconds=float(sigma_out) if sigma_out is not None else None,
            pos_weight=float(pos_weight),
            cls_weight=float(train_cfg.get("heatmap_cls_weight", 1.0)),
            start_weight=float(train_cfg.get("heatmap_start_weight", 1.0)),
            end_weight=float(train_cfg.get("heatmap_end_weight", 1.0)),
            heatmap_loss=str(train_cfg.get("heatmap_loss", "bce")),
            heatmap_pos_threshold=float(train_cfg.get("heatmap_pos_threshold", 0.1)),
            heatmap_pos_weight=float(train_cfg.get("heatmap_pos_weight", 20.0)),
            focal_alpha=float(train_cfg.get("heatmap_focal_alpha", 2.0)),
            focal_beta=float(train_cfg.get("heatmap_focal_beta", 4.0)),
        )


def _shift_right(x: torch.Tensor) -> torch.Tensor:
    """x shifted one frame later (frame t-1 lands at t); frame 0 gets False/0."""
    pad = torch.zeros_like(x[:, :1])
    return torch.cat([pad, x[:, :-1]], dim=1)


def _shift_left(x: torch.Tensor) -> torch.Tensor:
    pad = torch.zeros_like(x[:, :1])
    return torch.cat([x[:, 1:], pad], dim=1)


def boundary_markers(targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """First frame of each positive run (start marker) and last frame (end marker),
    both bool [B, T].

    A run cut by the window edge gets a marker at frame 0 / T-1 even though its
    true boundary lies outside the window; those edge markers are still trained
    on (a known approximation, kept for parity with the shipped model)."""
    pos = targets > 0.5
    start_marker = pos & ~_shift_right(pos)
    end_marker = pos & ~_shift_left(pos)
    return start_marker, end_marker


def _dist_to_nearest_directional(marker: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per frame: (dist_after, dist_before) in frames — distance to the nearest
    True at-or-before the frame (frame is AFTER that marker) and to the nearest
    True at-or-after it (frame is BEFORE that marker). Frames with no marker on a
    side get a large sentinel. Vectorized via forward cummax over marked indices
    (and the same on the flipped tensor for the other direction)."""
    batch, length = marker.shape
    device = marker.device
    idx = torch.arange(length, device=device).unsqueeze(0).expand(batch, length)
    big = length + 1

    def dist_left(m: torch.Tensor) -> torch.Tensor:
        # last marked index at or before each position (or -big if none yet)
        marked_idx = torch.where(m, idx, torch.full_like(idx, -big))
        last = torch.cummax(marked_idx, dim=1).values
        return idx - last  # huge where no marker seen yet

    after = dist_left(marker)
    before = dist_left(marker.flip(1)).flip(1)
    return (
        after.clamp(max=big).to(torch.float32),
        before.clamp(max=big).to(torch.float32),
    )


def _dist_to_nearest(marker: torch.Tensor) -> torch.Tensor:
    """Per frame: distance in frames to the nearest True in `marker`, either
    direction (min of the two directional distances)."""
    after, before = _dist_to_nearest_directional(marker)
    return torch.minimum(after, before)


def gaussian_target(dist_frames: torch.Tensor, sigma_frames: float) -> torch.Tensor:
    """exp(-0.5 (d/σ)^2). d is distance-in-frames to the nearest boundary, so with
    multiple boundaries in a window each frame takes the nearest (== union of the
    per-boundary Gaussians, since a Gaussian is monotone in distance)."""
    sigma = max(float(sigma_frames), 1e-6)
    return torch.exp(-0.5 * (dist_frames / sigma) ** 2)


def build_heatmap_targets(
    targets: torch.Tensor, cfg: HeatmapLossConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(start_heatmap, end_heatmap) soft Gaussian targets in [0, 1], shape [B, T].

    Symmetric by default. With sigma_out_seconds set, the OUTSIDE-the-point side
    (before a start / after an end) uses sigma_out and the inside uses
    sigma_seconds — the peak stays at the labeled boundary (target==1 there), but
    the fatter outside tail pulls the soft-argmax refinement outward."""
    start_marker, end_marker = boundary_markers(targets)
    sigma_in = cfg.sigma_seconds * cfg.fps
    if cfg.sigma_out_seconds is None:
        start_t = gaussian_target(_dist_to_nearest(start_marker), sigma_in)
        end_t = gaussian_target(_dist_to_nearest(end_marker), sigma_in)
        return start_t, end_t

    sigma_out = cfg.sigma_out_seconds * cfg.fps
    # start marker: frames AFTER it are inside the point, frames BEFORE it outside.
    s_after, s_before = _dist_to_nearest_directional(start_marker)
    start_t = torch.maximum(
        gaussian_target(s_after, sigma_in), gaussian_target(s_before, sigma_out)
    )
    # end marker: frames AFTER it are outside the point, frames BEFORE it inside.
    e_after, e_before = _dist_to_nearest_directional(end_marker)
    end_t = torch.maximum(
        gaussian_target(e_after, sigma_out), gaussian_target(e_before, sigma_in)
    )
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

        total = cfg.cls_weight * loss_cls + cfg.start_weight * loss_start + cfg.end_weight * loss_end
        components = {
            "loss_cls": float(loss_cls.item()),
            "loss_start": float(loss_start.item()),
            "loss_end": float(loss_end.item()),
        }
        return total, components
