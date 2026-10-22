"""TCN heatmap backbone + hybrid-decode false-positive filtering."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.eval.heatmap_evaluator import (  # noqa: E402
    HeatmapDecodeConfig,
    decode_hybrid,
)
from training.models.heatmap_lstm import TennisPointHeatmapLSTM  # noqa: E402
from training.models.heatmap_tcn import TennisPointHeatmapTCN  # noqa: E402
from training.train.heatmap_loop import build_heatmap_model  # noqa: E402

INPUT_SIZE, SEQ_LEN = 362, 100


def _tcn(**kw):
    return TennisPointHeatmapTCN(input_size=INPUT_SIZE, hidden_size=64, **kw)


def test_forward_shapes_match_the_three_logit_contract():
    model = _tcn()
    p, s, e = model(torch.randn(4, SEQ_LEN, INPUT_SIZE))
    for out in (p, s, e):
        assert out.shape == (4, SEQ_LEN)
        assert torch.isfinite(out).all()


def test_receptive_field_covers_the_training_window():
    # RF = 1 + 2*(k-1)*(2**levels - 1); defaults must span the 100-frame (20s) window.
    model = _tcn(levels=5, kernel_size=3)
    assert model.receptive_field == 1 + 2 * 2 * (2 ** 5 - 1) == 125
    assert model.receptive_field >= SEQ_LEN
    assert _tcn(levels=4, kernel_size=3).receptive_field == 61


def test_is_non_causal_so_it_sees_the_future_like_the_bilstm():
    """The backbone it replaces is bidirectional; a causal TCN would be a
    handicap. Perturbing frame t must move outputs on BOTH sides of t."""
    torch.manual_seed(0)
    model = _tcn().eval()
    x = torch.randn(1, SEQ_LEN, INPUT_SIZE)
    with torch.no_grad():
        base = model(x)[0]
        bumped = x.clone()
        bumped[:, 60, :] += 5.0
        after = model(bumped)[0]
    delta = (base - after).abs()
    assert delta[:, 50].max() > 0, "must influence earlier frames (non-causal)"
    assert delta[:, 70].max() > 0, "must influence later frames"


def test_uses_fewer_parameters_than_the_bilstm_it_replaces():
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    tcn = n(_tcn())
    lstm = n(TennisPointHeatmapLSTM(input_size=INPUT_SIZE, hidden_size=64))
    assert tcn < lstm / 1.5, f"expected a large param saving, got {tcn} vs {lstm}"


def test_factory_dispatches_tcn_and_rejects_unknown_backbones():
    assert isinstance(build_heatmap_model("tcn", INPUT_SIZE, "mlp", 64, 5, 3),
                      TennisPointHeatmapTCN)
    assert isinstance(build_heatmap_model("lstm", INPUT_SIZE, "mlp", 64),
                      TennisPointHeatmapLSTM)
    with pytest.raises(ValueError, match="tcn"):
        build_heatmap_model("nope", INPUT_SIZE, "mlp", 64)


# --- hybrid decode: smoothing + duration floor ------------------------------

def _tracks(n=60):
    """Two clean point runs plus a short spurious burst between them.

    The burst is 3 frames (0.6s), not 1: a single-frame run collapses to
    s == e and is already dropped by decode_hybrid's `e > s` guard, so it would
    never exercise the duration floor.
    """
    point = np.zeros(n, dtype=np.float32)
    point[5:20] = 0.9        # real point
    point[30:33] = 0.9       # 3-frame spurious burst -> ~0.6s segment
    point[40:55] = 0.9       # real point
    start = np.zeros(n, dtype=np.float32); start[[5, 30, 40]] = 1.0
    end = np.zeros(n, dtype=np.float32); end[[19, 32, 54]] = 1.0
    ts = np.arange(n, dtype=np.float64) / 5.0  # 5 fps
    return point, start, end, ts


def _cfg(**kw):
    base = dict(mode="hybrid", threshold=0.5, sigma_frames=2.5,
                min_duration_sec=0.3, max_duration_sec=60.0)
    base.update(kw)
    return HeatmapDecodeConfig(**base)


def test_new_filters_default_to_no_ops():
    """Defaults must reproduce the previous hybrid behaviour bit-for-bit,
    so existing runs stay comparable."""
    point, start, end, ts = _tracks()
    cfg = _cfg()
    assert cfg.smooth_sigma_frames is None
    assert cfg.hybrid_min_duration_sec == 0.0
    segs = decode_hybrid(point, start, end, ts, cfg)
    assert len(segs) == 3, "unfiltered decode keeps the spurious short burst"


def test_duration_floor_drops_the_spurious_spike_but_keeps_real_points():
    point, start, end, ts = _tracks()
    segs = decode_hybrid(point, start, end, ts, _cfg(hybrid_min_duration_sec=1.0))
    assert len(segs) == 2
    for s, e in segs:
        assert (e - s) >= 1.0


def test_smoothing_suppresses_an_isolated_burst():
    """Smoothing pushes a narrow burst below threshold while a genuine
    multi-second run stays well above it."""
    point, start, end, ts = _tracks()
    segs = decode_hybrid(point, start, end, ts, _cfg(smooth_sigma_frames=2.5))
    assert len(segs) == 2, "a 3-frame burst should not survive smoothing"


def test_duration_floor_applies_after_merging():
    """Adjacent fragments that together form a real point must not be dropped
    individually — the floor is applied to merged intervals."""
    n = 40
    point = np.zeros(n, dtype=np.float32)
    point[10:15] = 0.9
    point[15:20] = 0.9  # contiguous with the above -> one merged run
    start = np.zeros(n, dtype=np.float32); start[10] = 1.0
    end = np.zeros(n, dtype=np.float32); end[19] = 1.0
    ts = np.arange(n, dtype=np.float64) / 5.0
    segs = decode_hybrid(point, start, end, ts, _cfg(hybrid_min_duration_sec=1.5))
    assert len(segs) == 1
