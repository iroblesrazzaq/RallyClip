from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # training-side; runtime install has no torch

from training.models.lstm import TennisPointLSTM


def test_lstm_output_shape_logits():
    model = TennisPointLSTM(input_size=10, return_logits=True)
    x = torch.randn(2, 5, 10)
    out = model(x)
    assert out.shape == (2, 5)


def test_lstm_output_sigmoid():
    model = TennisPointLSTM(input_size=10, return_logits=False)
    x = torch.randn(2, 5, 10)
    out = model(x)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)
