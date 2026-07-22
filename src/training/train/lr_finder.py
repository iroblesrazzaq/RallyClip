"""LR range test (Smith / fastai): one short pass that exponentially ramps the
learning rate per mini-batch while recording the smoothed *training* loss, so the
loss-vs-LR curve reveals the usable LR band. Val loss is not used — it isn't
computed per batch. The suggested peak LR is the min-loss LR divided by 10 (the
standard heuristic: the actual loss minimum sits at the edge of divergence, and
min-loss/10 approximates the steepest-descent 'elbow').

The test MUTATES the passed model/optimizer state — run it on a throwaway model,
then build a fresh one for real training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
from torch.utils.data import DataLoader

# loss_step(model, features, targets) -> scalar loss tensor (already on device)
LossStep = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class LRRangeResult:
    lrs: List[float]
    losses: List[float]  # smoothed
    suggested_lr: float  # min-loss LR / 10
    min_loss_lr: float


def lr_range_test(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    loss_step: LossStep,
    device: torch.device,
    min_lr: float = 1e-7,
    max_lr: float = 1.0,
    num_steps: int = 100,
    smooth: float = 0.05,
    diverge_factor: float = 4.0,
) -> LRRangeResult:
    mult = (max_lr / min_lr) ** (1.0 / max(num_steps - 1, 1))
    lr = min_lr
    lrs: List[float] = []
    losses: List[float] = []
    avg = 0.0
    best = float("inf")

    model.train()
    it = iter(loader)
    for step in range(num_steps):
        try:
            features, targets = next(it)
        except StopIteration:
            it = iter(loader)
            features, targets = next(it)
        for group in optimizer.param_groups:
            group["lr"] = lr
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_step(model, features, targets)
        loss.backward()
        optimizer.step()

        value = float(loss.item())
        if not torch.isfinite(torch.tensor(value)):
            break
        avg = value if step == 0 else smooth * value + (1.0 - smooth) * avg
        # bias-correct the EMA so early steps aren't dragged toward 0
        smoothed = avg / (1.0 - (1.0 - smooth) ** (step + 1))
        lrs.append(lr)
        losses.append(smoothed)
        best = min(best, smoothed)
        if step > 0 and smoothed > diverge_factor * best:
            break
        lr *= mult

    i_min = int(min(range(len(losses)), key=lambda i: losses[i])) if losses else 0
    min_loss_lr = lrs[i_min] if lrs else min_lr
    return LRRangeResult(lrs=lrs, losses=losses, suggested_lr=min_loss_lr / 10.0, min_loss_lr=min_loss_lr)
