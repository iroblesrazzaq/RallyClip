from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from training.eval.seg_evaluator import decode_segments, gt_segments_from_targets
from training.metrics.segments6 import SixBinConfig, compute_six_bin
from training.train.seg_loss import (
    E2ESegLoss,
    SegLossConfig,
    boundary_hinge,
    compute_offset_targets,
    diou_1d,
)

FPS = 5.0


def test_offset_targets_basic():
    # run at frames 2..5 in a 10-frame window (interior on both sides)
    targets = torch.zeros(1, 10)
    targets[0, 2:6] = 1.0
    ds, de, vs, ve, pos = compute_offset_targets(targets, FPS)
    assert pos[0, 2] and not pos[0, 1]
    assert ds[0, 2].item() == pytest.approx(0.0)
    assert ds[0, 5].item() == pytest.approx(3 / FPS)
    assert de[0, 2].item() == pytest.approx(3 / FPS)
    assert de[0, 5].item() == pytest.approx(0.0)
    assert vs[0, 2:6].all() and ve[0, 2:6].all()
    assert not vs[0, 0] and not ve[0, 9]


def test_offset_targets_truncated_runs():
    # run touching the left edge: start side masked, end side valid
    targets = torch.zeros(1, 10)
    targets[0, 0:4] = 1.0
    _, _, vs, ve, _ = compute_offset_targets(targets, FPS)
    assert not vs[0, 0:4].any()
    assert ve[0, 0:4].all()

    # run touching the right edge: end side masked
    targets = torch.zeros(1, 10)
    targets[0, 7:10] = 1.0
    _, _, vs, ve, _ = compute_offset_targets(targets, FPS)
    assert vs[0, 7:10].all()
    assert not ve[0, 7:10].any()


def test_boundary_hinge_zones():
    cfg = SegLossConfig()  # lambda0=mu0=0.05, lambda1=0.2, lambda2=1.0, mu1=0.4, mu2=2.0
    # e_s = pred_start - gt_start; hinge args: (outside violation, inside violation)
    def start_loss(e_s: float) -> float:
        e = torch.tensor([e_s])
        return float(boundary_hinge(-e, e, cfg).item())

    # exactly on the boundary: zero loss
    assert start_loss(0.0) == pytest.approx(0.0, abs=1e-6)
    # inside the Good zone now carries the gentle lambda0/mu0 pull toward exact time
    assert start_loss(-0.5) == pytest.approx(cfg.lambda0 * 0.5, abs=1e-6)  # outside edge of Good
    assert start_loss(0.2) == pytest.approx(cfg.mu0 * 0.2, abs=1e-6)  # inside edge of Good
    # outside direction (early), within Decent: lambda0 over Good + lambda1 beyond
    assert start_loss(-1.0) == pytest.approx(cfg.lambda0 * 0.5 + cfg.lambda1 * 0.5, abs=1e-6)
    # beyond Decent: lambda0 over Good + lambda1 over the gap + lambda2 beyond
    assert start_loss(-2.0) == pytest.approx(
        cfg.lambda0 * 0.5 + cfg.lambda1 * 1.0 + cfg.lambda2 * 0.5, abs=1e-6
    )
    # inside direction (late start), within Decent: mu0 over Good + mu1 beyond
    assert start_loss(0.4) == pytest.approx(cfg.mu0 * 0.2 + cfg.mu1 * 0.2, abs=1e-6)
    # the Good-zone pull is tiny next to a clearly-beyond-Decent error
    assert start_loss(-0.5) < 0.1 * start_loss(-2.5)
    # inside is punished harder than outside at equal violation
    assert start_loss(0.7) > start_loss(-0.7)


def test_diou_graded_at_zero_overlap():
    # same-length intervals, increasing distance: DIoU keeps decreasing even after IoU hits 0
    ds_true = torch.tensor([1.0, 1.0, 1.0])
    de_true = torch.tensor([1.0, 1.0, 1.0])
    # predictions sliding right: small overlap, zero overlap (near), zero overlap (far)
    ds_pred = torch.tensor([-0.5, -1.5, -4.0])  # negative ds = starts after anchor (no overlap on left)
    de_pred = torch.tensor([2.5, 3.5, 6.0])
    scores = diou_1d(ds_pred, de_pred, ds_true, de_true)
    assert scores[0] > scores[1] > scores[2]


def test_loss_runs_and_perfect_prediction_is_cheap():
    cfg = SegLossConfig()
    criterion = E2ESegLoss(cfg)
    targets = torch.zeros(2, 20)
    targets[0, 5:12] = 1.0
    targets[1, 0:4] = 1.0  # truncated at left edge
    ds, de, _, _, _ = compute_offset_targets(targets, cfg.fps)
    logits = torch.where(targets > 0.5, torch.tensor(8.0), torch.tensor(-8.0))
    loss_perfect, comps = criterion(logits, ds, de, targets)
    assert comps["loss_start"] == pytest.approx(0.0, abs=1e-5)
    assert comps["loss_end"] == pytest.approx(0.0, abs=1e-5)
    loss_bad, _ = criterion(logits, ds + 3.0, de + 3.0, targets)
    assert float(loss_bad) > float(loss_perfect)


def test_six_bin_tiers():
    cfg = SixBinConfig()
    gt = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0), (70.0, 80.0), (100.0, 110.0)]
    preds = [
        (9.9, 20.3),    # good
        (29.0, 40.4),   # decent (start 1.0 early)
        (47.0, 63.0),   # iou = 10/16 = 0.625 -> bad_segmentation
        (74.0, 81.0),   # iou = 6/11 = 0.545... boundary start_err=+4 -> bad_segmentation
        (200.0, 210.0), # false positive (no overlap)
    ]
    # gt[4] (100,110) has no matching pred -> false negative
    result = compute_six_bin(gt, preds, cfg)
    assert result["n_good"] == 1
    assert result["n_decent"] == 1
    assert result["n_bad_segmentation"] == 2
    assert result["n_false_positive"] == 1
    assert result["n_false_negative"] == 1
    assert result["n_events"] == 6
    assert result["share_good"] == pytest.approx(1 / 6)


def test_six_bin_poor_recognition_and_split():
    cfg = SixBinConfig()
    gt = [(10.0, 30.0)]
    # two predictions overlapping one GT: best match wins, the other becomes FP (split)
    preds = [(9.5, 18.0), (22.0, 30.5)]
    result = compute_six_bin(gt, preds, cfg)
    assert result["n_false_positive"] == 1
    assert result["n_split"] == 1
    # winner: iou = 8.5/20.5 < 0.5 -> poor recognition
    assert result["n_poor_recognition"] == 1


def test_decode_and_gt_segments():
    ts = np.arange(10, dtype=np.float64) / FPS
    targets = np.zeros(10)
    targets[3:7] = 1.0
    gt = gt_segments_from_targets(targets, ts)
    assert gt == [(3 / FPS, 6 / FPS)]

    probs = np.zeros(10)
    probs[3:7] = 0.9
    d_start = np.zeros(10)
    d_end = np.zeros(10)
    for i in range(3, 7):
        d_start[i] = (i - 3) / FPS
        d_end[i] = (6 - i) / FPS
    segs = decode_segments(probs, d_start, d_end, ts, threshold=0.5)
    assert len(segs) == 1
    assert segs[0][0] == pytest.approx(3 / FPS, abs=1e-6)
    assert segs[0][1] == pytest.approx(6 / FPS, abs=1e-6)
