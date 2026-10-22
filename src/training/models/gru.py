from __future__ import annotations

import torch
import torch.nn as nn

class TennisPointGRU(nn.Module):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        return_logits: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.return_logits = return_logits

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(output_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        out = self.fc(gru_out).squeeze(-1)
        if not self.return_logits:
            out = torch.sigmoid(out)
        return out

