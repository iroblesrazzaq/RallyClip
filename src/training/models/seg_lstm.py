from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TennisPointSegLSTM(nn.Module):
    """Anchor-free per-frame segment head: same BiLSTM backbone as TennisPointLSTM,
    but each frame predicts (pointness logit, distance-to-start, distance-to-end)
    in seconds. Offsets go through softplus so they are non-negative."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        head: str = "linear",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.head = head

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_size = hidden_size * 2 if bidirectional else hidden_size
        if head == "mlp":
            # classification stays a linear readout (keeps the representation
            # shaped by the dense frame loss); the boundary regression gets its
            # own small nonlinear head so the trunk need not encode time linearly.
            self.cls_head = nn.Linear(output_size, 1)
            self.reg_head = nn.Sequential(
                nn.Linear(output_size, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )
        elif head == "linear":
            self.fc = nn.Linear(output_size, 3)
        else:
            raise ValueError(f"Unknown head: {head}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        if self.head == "mlp":
            logits = self.cls_head(lstm_out).squeeze(-1)
            reg = self.reg_head(lstm_out)
            d_start = F.softplus(reg[..., 0])
            d_end = F.softplus(reg[..., 1])
        else:
            out = self.fc(lstm_out)
            logits = out[..., 0]
            d_start = F.softplus(out[..., 1])
            d_end = F.softplus(out[..., 2])
        return logits, d_start, d_end
