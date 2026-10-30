from __future__ import annotations

import pytest

from helpers.golden_metrics import score_six_bin
from training.metrics.segments6 import DECENT_V1, SixBinConfig, compute_six_bin

GT = (100.0, 120.0)


def _tier(pred, cfg=None):
    m = compute_six_bin([GT], [pred], cfg)
    return next(k for k in ("good", "decent", "bad_segmentation", "poor_recognition") if m[f"n_{k}"])


@pytest.mark.parametrize(
    "pred,tier",
    [
        ((100.0, 120.0), "good"),
        ((98.0, 120.0), "decent"),  # start 2 s early
        ((97.9, 120.0), "bad_segmentation"),
        ((100.5, 120.0), "decent"),  # start 0.5 s late
        ((100.6, 120.0), "bad_segmentation"),
        ((100.0, 119.0), "decent"),  # end 1 s early
        ((100.0, 118.9), "bad_segmentation"),
        ((100.0, 123.0), "decent"),  # end 3 s late
        ((100.0, 123.1), "bad_segmentation"),
        ((98.0, 123.0), "decent"),  # both ends at the outside limit
    ],
)
def test_decent_limits_per_boundary(pred, tier):
    assert _tier(pred) == tier


def test_golden_helper_agrees():
    for pred in [(98.0, 123.0), (97.9, 120.0), (100.0, 119.0), (100.0, 118.9), (100.6, 120.0)]:
        a = compute_six_bin([GT], [pred])
        b = score_six_bin([pred], [GT])
        assert a["n_decent"] == b["decent"] and a["n_bad_segmentation"] == b["bad"]


def test_v1_limits_still_available():
    v1 = SixBinConfig(**DECENT_V1)
    assert _tier((98.5, 121.5), v1) == "decent"
    assert _tier((100.0, 119.0), v1) == "bad_segmentation"
    assert _tier((100.0, 119.0)) == "decent"
