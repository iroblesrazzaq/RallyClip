from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from training.eval.heatmap_evaluator import (
    HeatmapDecodeConfig,
    _pick_peaks,
    decode_heatmap_segments,
)
from training.train.heatmap_loss import (
    E2EHeatmapLoss,
    HeatmapLossConfig,
    _dist_to_nearest,
    boundary_markers,
    build_heatmap_targets,
    gaussian_target,
    soft_argmax_time_loss,
)

FPS = 5.0


# ---- target construction ----------------------------------------------------

def test_boundary_markers_basic():
    targets = torch.zeros(1, 10)
    targets[0, 2:6] = 1.0  # run at frames 2..5
    start, end = boundary_markers(targets)
    assert start[0, 2] and not start[0, 3]
    assert end[0, 5] and not end[0, 4]
    # single-frame run: start and end coincide
    t2 = torch.zeros(1, 6)
    t2[0, 3] = 1.0
    s2, e2 = boundary_markers(t2)
    assert s2[0, 3] and e2[0, 3]


def test_dist_to_nearest_single_and_multiple():
    # single marker at index 4
    marker = torch.zeros(1, 10, dtype=torch.bool)
    marker[0, 4] = True
    d = _dist_to_nearest(marker)
    assert d[0, 4].item() == 0.0
    assert d[0, 0].item() == 4.0
    assert d[0, 9].item() == 5.0
    # two markers: each frame takes the nearer one
    marker = torch.zeros(1, 10, dtype=torch.bool)
    marker[0, 2] = True
    marker[0, 8] = True
    d = _dist_to_nearest(marker)
    assert d[0, 2].item() == 0.0 and d[0, 8].item() == 0.0
    assert d[0, 5].item() == 3.0  # 3 from frame 2, 3 from frame 8
    assert d[0, 4].item() == 2.0  # nearer to 2
    # no marker anywhere -> large sentinel everywhere
    marker = torch.zeros(1, 5, dtype=torch.bool)
    d = _dist_to_nearest(marker)
    assert (d[0] >= 5.0).all()


def test_gaussian_target_shape():
    sigma_frames = 2.5
    dist = torch.tensor([[0.0, sigma_frames, 2 * sigma_frames, 100.0]])
    g = gaussian_target(dist, sigma_frames)
    assert g[0, 0].item() == pytest.approx(1.0)
    assert g[0, 1].item() == pytest.approx(np.exp(-0.5), abs=1e-5)  # 1σ
    assert g[0, 2].item() == pytest.approx(np.exp(-2.0), abs=1e-5)  # 2σ
    assert g[0, 3].item() < 1e-6
    # monotone decay
    assert g[0, 0] > g[0, 1] > g[0, 2] > g[0, 3]


def test_build_heatmap_targets_peaks_at_boundaries():
    targets = torch.zeros(1, 20)
    targets[0, 5:12] = 1.0  # run 5..11
    cfg = HeatmapLossConfig(fps=FPS, sigma_seconds=0.5)
    start_t, end_t = build_heatmap_targets(targets, cfg)
    assert start_t[0, 5].item() == pytest.approx(1.0)  # peak at run start
    assert end_t[0, 11].item() == pytest.approx(1.0)  # peak at run end
    assert start_t[0, 11].item() < 0.2  # low far from the start boundary
    assert end_t[0, 5].item() < 0.2


# ---- loss -------------------------------------------------------------------

def test_loss_runs_and_is_finite():
    targets = torch.zeros(2, 30)
    targets[0, 4:12] = 1.0
    targets[1, 10:14] = 1.0
    targets[1, 20:27] = 1.0
    cfg = HeatmapLossConfig(fps=FPS)
    loss_fn = E2EHeatmapLoss(cfg)
    p = torch.randn(2, 30)
    s = torch.randn(2, 30)
    e = torch.randn(2, 30)
    total, comps = loss_fn(p, s, e, targets)
    assert torch.isfinite(total)
    assert set(comps) == {"loss_cls", "loss_start", "loss_end", "loss_time_start", "loss_time_end"}
    assert all(np.isfinite(v) for v in comps.values())


@pytest.mark.parametrize("mode", ["bce", "focal"])
def test_loss_lower_for_correct_predictions(mode):
    targets = torch.zeros(1, 40)
    targets[0, 10:25] = 1.0
    cfg = HeatmapLossConfig(fps=FPS, heatmap_loss=mode)
    loss_fn = E2EHeatmapLoss(cfg)
    start_t, end_t = build_heatmap_targets(targets, cfg)

    def logit(x):
        return torch.log(x.clamp(1e-4, 1 - 1e-4) / (1 - x.clamp(1e-4, 1 - 1e-4)))

    # "good" prediction matches the Gaussian targets and the pointness label
    good_p = logit(targets * 0.9 + 0.05)
    good_s = logit(start_t * 0.9 + 0.02)
    good_e = logit(end_t * 0.9 + 0.02)
    good, _ = loss_fn(good_p, good_s, good_e, targets)
    # "bad" prediction: everything near 0.5 logits ~ 0
    bad = torch.zeros(1, 40)
    bad_total, _ = loss_fn(bad, bad.clone(), bad.clone(), targets)
    assert good.item() < bad_total.item()


# ---- soft-argmax time loss --------------------------------------------------

def test_time_loss_zero_when_peak_on_boundary():
    # sharp startness peak exactly on the true boundary frame -> ~0 time loss
    T = 20
    logits = torch.full((1, T), -5.0)
    logits[0, 8] = 10.0
    marker = torch.zeros(1, T, dtype=torch.bool)
    marker[0, 8] = True
    loss = soft_argmax_time_loss(logits, marker, fps=FPS, window_frames=5)
    assert loss.item() < 1e-3


def test_time_loss_grows_and_gradient_pulls_peak_to_boundary():
    T = 20
    true = 8
    marker = torch.zeros(1, T, dtype=torch.bool)
    marker[0, true] = True
    # soft peak placed 2 frames late -> nonzero loss (gentle logits so the
    # soft-argmax gradient isn't saturated to ~0 by a one-hot softmax)
    off = torch.zeros(1, T)
    off[0, true + 2] = 2.0
    loss_off = soft_argmax_time_loss(off, marker, fps=FPS, window_frames=5)
    assert loss_off.item() > 0.01
    # a gradient step on the logits should reduce the time loss
    logits = off.clone().requires_grad_(True)
    l = soft_argmax_time_loss(logits, marker, fps=FPS, window_frames=5)
    l.backward()
    stepped = (logits - 5.0 * logits.grad).detach()
    l2 = soft_argmax_time_loss(stepped, marker, fps=FPS, window_frames=5)
    assert l2.item() < loss_off.item()


def test_time_loss_flat_window_is_centered():
    # uniform logits over a window symmetric about the boundary -> soft-argmax == boundary
    T = 21
    logits = torch.zeros(1, T)
    marker = torch.zeros(1, T, dtype=torch.bool)
    marker[0, 10] = True
    loss = soft_argmax_time_loss(logits, marker, fps=FPS, window_frames=5)
    assert loss.item() < 1e-6


def test_time_weight_adds_to_total():
    targets = torch.zeros(1, 30)
    targets[0, 8:20] = 1.0
    p = torch.randn(1, 30); s = torch.randn(1, 30); e = torch.randn(1, 30)
    base = E2EHeatmapLoss(HeatmapLossConfig(fps=FPS, time_weight=0.0))(p, s, e, targets)[0]
    witht = E2EHeatmapLoss(HeatmapLossConfig(fps=FPS, time_weight=1.0))(p, s, e, targets)[0]
    assert not torch.isclose(base, witht)  # time term changes the total when enabled


# ---- decode -----------------------------------------------------------------

def test_pick_peaks_nms():
    prob = np.array([0.0, 0.9, 0.85, 0.1, 0.0, 0.8, 0.0])
    # frames 1 and 2 are adjacent high peaks -> NMS keeps the stronger (1)
    peaks = _pick_peaks(prob, threshold=0.3, nms_frames=2)
    assert 1 in peaks
    assert 2 not in peaks
    assert 5 in peaks


def test_decode_hybrid_single_point():
    n = 30
    ts = np.arange(n, dtype=np.float64) / FPS
    pointness = np.zeros(n)
    pointness[8:20] = 0.9  # run 8..19
    start_p = np.zeros(n)
    end_p = np.zeros(n)
    start_p[8] = 0.95
    end_p[19] = 0.95
    cfg = HeatmapDecodeConfig(mode="hybrid", threshold=0.5, sigma_frames=2.5)
    segs = decode_heatmap_segments(pointness, start_p, end_p, ts, cfg)
    assert len(segs) == 1
    s, e = segs[0]
    assert s == pytest.approx(8 / FPS, abs=1.0 / FPS)
    assert e == pytest.approx(19 / FPS, abs=1.0 / FPS)


def test_decode_peakpair_two_points_no_cross_pairing():
    n = 60
    ts = np.arange(n, dtype=np.float64) / FPS
    pointness = np.zeros(n)
    start_p = np.zeros(n)
    end_p = np.zeros(n)
    # point A: start 5, end 15 ; point B: start 30, end 45
    start_p[5] = 0.9
    end_p[15] = 0.9
    start_p[30] = 0.9
    end_p[45] = 0.9
    pointness[5:16] = 0.9
    pointness[30:46] = 0.9
    cfg = HeatmapDecodeConfig(mode="peakpair", peak_threshold=0.3, sigma_frames=2.5,
                              min_duration_sec=0.3, max_duration_sec=60.0)
    segs = decode_heatmap_segments(pointness, start_p, end_p, ts, cfg)
    assert len(segs) == 2
    (s0, e0), (s1, e1) = segs
    assert s0 == pytest.approx(5 / FPS, abs=1.0 / FPS)
    assert e0 == pytest.approx(15 / FPS, abs=1.0 / FPS)
    assert s1 == pytest.approx(30 / FPS, abs=1.0 / FPS)
    assert e1 == pytest.approx(45 / FPS, abs=1.0 / FPS)


def test_decode_peakpair_max_duration_rejects():
    n = 60
    ts = np.arange(n, dtype=np.float64) / FPS
    pointness = np.zeros(n)
    start_p = np.zeros(n)
    end_p = np.zeros(n)
    start_p[2] = 0.9
    end_p[50] = 0.9  # 48 frames = 9.6s apart
    cfg = HeatmapDecodeConfig(mode="peakpair", peak_threshold=0.3, sigma_frames=2.5,
                              min_duration_sec=0.3, max_duration_sec=5.0)
    segs = decode_heatmap_segments(pointness, start_p, end_p, ts, cfg)
    assert segs == []
