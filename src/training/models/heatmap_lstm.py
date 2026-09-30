from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TennisPointHeatmapLSTM(nn.Module):
    """Boundary-heatmap per-frame head: same BiLSTM backbone as TennisPointSegLSTM
    and TennisPointLSTM, but each frame predicts three raw logits:
      (pointness, startness, endness).
    Sigmoid is applied downstream (in the loss / evaluator), matching the logit
    convention of the other heads. startness/endness are trained against a soft
    Gaussian bump centred on the true start/end boundary frame; pointness is the
    same dense point/no-point signal the other heads carry (used for frame
    metrics, the hybrid decode's segment detection, and the optional decode gate)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        head: str = "mlp",
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
            # Pointness stays a linear readout (shaped by the dense frame loss);
            # the two boundary heatmaps get a small shared nonlinear head so the
            # trunk need not encode boundary sharpness linearly.
            self.cls_head = nn.Linear(output_size, 1)
            self.heatmap_head = nn.Sequential(
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
            pointness = self.cls_head(lstm_out).squeeze(-1)
            hm = self.heatmap_head(lstm_out)
            startness = hm[..., 0]
            endness = hm[..., 1]
        else:
            out = self.fc(lstm_out)
            pointness = out[..., 0]
            startness = out[..., 1]
            endness = out[..., 2]
        return pointness, startness, endness
