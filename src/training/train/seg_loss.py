"""End-to-end segment loss (docs/segment_eval_metrics.md, per-frame anchor-free form).

Each frame predicts (pointness logit, distance-to-start, distance-to-end) in seconds.
The loss combines:
  - BCE-with-pos-weight on pointness (dense FN/FP signal; no matching at train time),
  - an asymmetric two-knee hinge on the regressed boundary errors of in-point frames
    (zero inside the Good zone, slope l1 to the Decent edge, slope l2 beyond;
    inside-direction slopes mu >= lambda since cutting the point is worse),
  - a 1D DIoU term so the gradient keeps pointing at the GT even at zero overlap.

Runs truncated by the window edge have unknowable true offsets on the cut side; those
sides are masked out of the regression terms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class BoundaryTolerances:
    good_out: float = 0.5
    good_in: float = 0.2
    decent_out: float = 1.5
    decent_in: float = 0.5


@dataclass
class SegLossConfig:
    fps: float = 5.0
    pos_weight: float = 3.0
    cls_weight: float = 1.0
    boundary_weight: float = 1.0
    diou_weight: float = 0.25
    lambda0: float = 0.05  # outside slope, inside Good zone (gentle pull to exact time)
    lambda1: float = 0.2  # outside slope, Good -> Decent
    lambda2: float = 1.0  # outside slope, beyond Decent
    mu0: float = 0.05  # inside slope, inside Good zone (gentle pull to exact time)
    mu1: float = 0.4  # inside slope, Good -> Decent
    mu2: float = 2.0  # inside slope, beyond Decent
    tolerances: BoundaryTolerances = field(default_factory=BoundaryTolerances)


def _dist_to_run_start(pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per frame: frames since the start of its run of 1s, and whether that run
    starts after index 0 (i.e. its true start is inside the window)."""
    batch, length = pos.shape
    idx = torch.arange(length, device=pos.device).unsqueeze(0).expand(batch, length)
    prev = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=pos.device), pos[:, :-1]], dim=1)
    run_start = pos & ~prev
    marker = torch.where(run_start, idx + 1, torch.zeros_like(idx))  # 1-based so index 0 is distinguishable
    start1 = torch.cummax(marker, dim=1).values
    dist = idx - (start1 - 1)
    interior = start1 > 1
    return dist, interior


def compute_offset_targets(
    targets: torch.Tensor, fps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """From per-frame binary targets [B, T], compute per-frame distance-to-run-start
    and distance-to-run-end in seconds, plus validity masks (in-point AND the
    corresponding boundary not truncated by the window edge)."""
    pos = targets > 0.5
    ds_frames, start_interior = _dist_to_run_start(pos)
    de_frames_rev, end_interior_rev = _dist_to_run_start(pos.flip(1))
    de_frames = de_frames_rev.flip(1)
    end_interior = end_interior_rev.flip(1)

    d_start = ds_frames.float() / fps
    d_end = de_frames.float() / fps
    valid_start = pos & start_interior
    valid_end = pos & end_interior
    return d_start, d_end, valid_start, valid_end, pos


def two_knee(x: torch.Tensor, knee: float, slope1: float, slope2: float) -> torch.Tensor:
    """Piecewise-linear penalty for x >= 0: slope1 up to `knee`, slope2 beyond."""
    return slope1 * torch.clamp(x, max=knee) + slope2 * torch.relu(x - knee)


def three_knee(
    x: torch.Tensor, knee1: float, knee2: float, slope0: float, slope1: float, slope2: float
) -> torch.Tensor:
    """Piecewise-linear penalty for x >= 0, minimized at x=0: slope0 in [0, knee1]
    (the Good zone — a gentle pull to the exact time), slope1 in [knee1, knee2]
    (Good->Decent), slope2 beyond knee2."""
    return (
        slope0 * torch.clamp(x, max=knee1)
        + slope1 * torch.clamp(torch.relu(x - knee1), max=knee2 - knee1)
        + slope2 * torch.relu(x - knee2)
    )


def boundary_hinge(
    out_violation: torch.Tensor,
    in_violation: torch.Tensor,
    cfg: SegLossConfig,
) -> torch.Tensor:
    # Measure error from the exact boundary (0), not from the Good-zone edge, so a
    # tiny slope inside Good keeps pulling toward the true time without ever
    # outweighing the Good->Decent and beyond-Decent penalties.
    tol = cfg.tolerances
    o = torch.relu(out_violation)
    i = torch.relu(in_violation)
    out_term = three_knee(o, tol.good_out, tol.decent_out, cfg.lambda0, cfg.lambda1, cfg.lambda2)
    in_term = three_knee(i, tol.good_in, tol.decent_in, cfg.mu0, cfg.mu1, cfg.mu2)
    return out_term + in_term


def diou_1d(
    ds_pred: torch.Tensor,
    de_pred: torch.Tensor,
    ds_true: torch.Tensor,
    de_true: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """DIoU between intervals [t-ds, t+de] sharing the anchor frame t."""
    inter = torch.relu(torch.minimum(ds_pred, ds_true) + torch.minimum(de_pred, de_true))
    union = (ds_pred + de_pred) + (ds_true + de_true) - inter
    iou = inter / (union + eps)
    center_dist = ((de_pred - ds_pred) - (de_true - ds_true)).abs() / 2.0
    enclose = torch.maximum(ds_pred, ds_true) + torch.maximum(de_pred, de_true)
    return iou - (center_dist / (enclose + eps)) ** 2


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum()
    if count == 0:
        return values.sum() * 0.0
    return (values * mask.float()).sum() / count.float()


class E2ESegLoss(nn.Module):
    def __init__(self, cfg: SegLossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("_pos_weight", torch.tensor([cfg.pos_weight]))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)

    def forward(
        self,
        logits: torch.Tensor,
        d_start_pred: torch.Tensor,
        d_end_pred: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        ds_true, de_true, valid_start, valid_end, _ = compute_offset_targets(targets, cfg.fps)

        loss_cls = self.bce(logits, targets)

        # start: e_s = pred_start - gt_start = ds_true - ds_pred (early start => e_s < 0 => outside)
        e_s = ds_true - d_start_pred
        # end: e_e = pred_end - gt_end = de_pred - de_true (late end => e_e > 0 => outside)
        e_e = d_end_pred - de_true
        loss_start = _masked_mean(boundary_hinge(-e_s, e_s, cfg), valid_start)
        loss_end = _masked_mean(boundary_hinge(e_e, -e_e, cfg), valid_end)

        both = valid_start & valid_end
        loss_diou = _masked_mean(1.0 - diou_1d(d_start_pred, d_end_pred, ds_true, de_true), both)

        total = (
            cfg.cls_weight * loss_cls
            + cfg.boundary_weight * (loss_start + loss_end)
            + cfg.diou_weight * loss_diou
        )
        components = {
            "loss_cls": float(loss_cls.item()),
            "loss_start": float(loss_start.item()),
            "loss_end": float(loss_end.item()),
            "loss_diou": float(loss_diou.item()),
        }
        return total, components
