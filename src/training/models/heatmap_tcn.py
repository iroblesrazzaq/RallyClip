from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class _Chomp1d(nn.Module):
    """Trim padding so a dilated conv keeps its input length exactly."""

    def __init__(self, left: int, right: int) -> None:
        super().__init__()
        self.left = int(left)
        self.right = int(right)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        end = x.shape[-1] - self.right if self.right > 0 else None
        return x[..., self.left:end]


class _TemporalBlock(nn.Module):
    """Residual block: two dilated convs, LayerNorm + ReLU + dropout after each.

    Non-causal (centred): padding is split evenly on both sides, so frame t sees
    an equal span of past and future — the convolutional analogue of the
    bidirectional LSTM this replaces. LayerNorm (not BatchNorm) keeps train/eval
    behaviour identical, which matters here because the train loop evaluates on a
    CPU-reloaded copy of the model (MPS/CPU LSTM eval divergence, see seg_loop).
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad_total = (kernel_size - 1) * dilation
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad_total, dilation=dilation)
        self.chomp1 = _Chomp1d(pad_left, pad_right)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad_total, dilation=dilation)
        self.chomp2 = _Chomp1d(pad_left, pad_right)
        self.norm2 = nn.LayerNorm(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _unit(self, x, conv, chomp, norm):
        y = chomp(conv(x))                      # (B, C, T)
        y = norm(y.transpose(1, 2)).transpose(1, 2)  # LayerNorm over channels
        return self.dropout(self.relu(y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._unit(x, self.conv1, self.chomp1, self.norm1)
        y = self._unit(y, self.conv2, self.chomp2, self.norm2)
        return x + y  # identity residual (channels constant throughout the stack)


class TennisPointHeatmapTCN(nn.Module):
    """Dilated temporal-convnet backbone for the boundary-heatmap head.

    Same contract as TennisPointHeatmapLSTM / TennisPointHeatmapGRU: consumes
    (B, T, input_size) and returns three per-frame raw logits
    (pointness, startness, endness), sigmoid applied downstream.

    Receptive field with `levels` residual blocks at dilations 1,2,4,... and two
    convs per block:  RF = 1 + 2*(kernel_size-1)*(2**levels - 1).
    Defaults (k=3, levels=5) give RF = 125 frames = 25s @5fps, so every frame
    sees the whole 20s training window — matching the BiLSTM's full-window view
    at roughly half the parameters.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,     # channel width (named hidden_size for factory parity)
        levels: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.2,
        head: str = "mlp",
        stem_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.levels = levels
        self.kernel_size = kernel_size
        self.head = head
        self.stem_hidden = stem_hidden

        # Projection from features to the trunk width; keeps the residual stack
        # at constant channels so every block can use an identity skip.
        #
        # stem_hidden=None (default): a single 1x1 conv, i.e. one affine map per
        # frame. With stem_hidden=S: two 1x1 convs with ReLU between, so the
        # per-frame projection becomes nonlinear (input -> S -> hidden_size).
        # Both are pointwise -- no temporal mixing happens before the blocks.
        if stem_hidden is None:
            self.input_proj = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        else:
            self.input_proj = nn.Sequential(
                nn.Conv1d(input_size, stem_hidden, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(stem_hidden, hidden_size, kernel_size=1),
                nn.ReLU(),
            )
        self.blocks = nn.ModuleList(
            [_TemporalBlock(hidden_size, kernel_size, 2 ** i, dropout) for i in range(levels)]
        )

        if head == "mlp":
            self.cls_head = nn.Linear(hidden_size, 1)
            self.heatmap_head = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )
        elif head == "linear":
            self.fc = nn.Linear(hidden_size, 3)
        else:
            raise ValueError(f"Unknown head: {head}")
        self.dropout = nn.Dropout(dropout)

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.levels - 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(x.transpose(1, 2))  # (B, C, T)
        for block in self.blocks:
            h = block(h)
        h = self.dropout(h.transpose(1, 2))     # (B, T, C)
        if self.head == "mlp":
            pointness = self.cls_head(h).squeeze(-1)
            hm = self.heatmap_head(h)
            startness = hm[..., 0]
            endness = hm[..., 1]
        else:
            out = self.fc(h)
            pointness = out[..., 0]
            startness = out[..., 1]
            endness = out[..., 2]
        return pointness, startness, endness
