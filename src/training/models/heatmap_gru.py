from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TennisPointHeatmapGRU(nn.Module):
    """GRU-backbone twin of TennisPointHeatmapLSTM: same 3-logit per-frame output
    (pointness, startness, endness), same head variants, but a bidirectional GRU
    trunk instead of an LSTM. GRU has 3 gates vs the LSTM's 4, so ~25% fewer
    recurrent parameters — the motivation is less overfitting on the small corpus.
    Sigmoid is applied downstream (loss / evaluator), matching the logit convention
    of the other heads."""

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

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_size = hidden_size * 2 if bidirectional else hidden_size
        if head == "mlp":
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
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        if self.head == "mlp":
            pointness = self.cls_head(gru_out).squeeze(-1)
            hm = self.heatmap_head(gru_out)
            startness = hm[..., 0]
            endness = hm[..., 1]
        else:
            out = self.fc(gru_out)
            pointness = out[..., 0]
            startness = out[..., 1]
            endness = out[..., 2]
        return pointness, startness, endness
