from __future__ import annotations

from pathlib import Path

import torch

from training.eval.checkpoint import build_model_from_checkpoint
from training.models.lstm import TennisPointLSTM
from training.train.loop import _save_checkpoint


def test_checkpoint_roundtrip_uses_arch_keys(tmp_path):
    model = TennisPointLSTM(input_size=16, hidden_size=32, num_layers=1, bidirectional=True, return_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "best.pth"
    _save_checkpoint(path, model, optimizer, epoch=1, metrics={"val_loss": 0.1})

    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["arch"]["input_size"] == 16
    assert ckpt["arch"]["hidden_size"] == 32
    assert ckpt["arch"]["bidirectional"] is True

    rebuilt = build_model_from_checkpoint(ckpt, feature_dim=16)
    rebuilt.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    rebuilt.eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        a = model(x)
        b = rebuilt(x)
    assert torch.allclose(a, b)


def test_bidirectional_default_not_false_when_missing():
    # Archaeology used to default bidirectional=False — that must not resurface.
    ckpt = {
        "model_state_dict": TennisPointLSTM(input_size=8, bidirectional=True).state_dict(),
        "arch": {"input_size": 8, "hidden_size": 128, "num_layers": 2, "bidirectional": True},
    }
    model = build_model_from_checkpoint(ckpt, feature_dim=8)
    assert model.bidirectional is True
