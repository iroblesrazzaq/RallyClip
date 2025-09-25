import torch
import torch.nn as nn


class TennisPointLSTM(nn.Module):
    def __init__(self, input_size: int = 360, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, bidirectional: bool = True, return_logits: bool = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.return_logits = return_logits
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        if not self.return_logits:
            output = torch.sigmoid(output)
        return output


