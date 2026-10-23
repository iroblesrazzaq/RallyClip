"""Feature set v2 (redundancy-pruned, N-slot) and the far-crop slot rules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.features.registry import FeatureRegistry  # noqa: E402
from training.features.v1 import FeatureSetV1  # noqa: E402
from training.features.v2 import FeatureSetV2, FeatureSetV2NearFar  # noqa: E402
from training.preprocess.preprocessor import (  # noqa: E402
    _crop_slots,
    _default_court_mask,
    _feet_on_court,
)

K = 17


def _player(box, exists=True):
    return {
        "exists": exists,
        "box": np.asarray(box, np.float32),
        "keypoints": np.zeros((K, 2), np.float32),
        "conf": np.ones(K, np.float32),
        "box_conf": 0.9,
    }


# --- layout -----------------------------------------------------------------

def test_v2_drops_exactly_the_redundant_magnitude_groups():
    """speed, accel_mag, keypoint_speed(17), keypoint_accel_mag(17) = 36 dims,
    each exactly the norm of a vector already present."""
    assert FeatureSetV1.feature_dim() // 2 - FeatureSetV2.per_slot_dim() == 36
    assert FeatureSetV2.per_slot_dim() == 145


def test_slot_counts_give_the_expected_widths():
    assert FeatureSetV2().feature_dim() == 580          # near, far, crop0, crop1
    assert FeatureSetV2NearFar().feature_dim() == 290   # baseline


def test_v1_is_unchanged():
    """v2 must be purely additive; existing v1 artifacts stay reproducible."""
    assert FeatureSetV1.feature_dim() == 362


def test_registry_exposes_all_three():
    r = FeatureRegistry()
    assert r.get("v1") is FeatureSetV1
    assert r.get("v2") is FeatureSetV2
    assert r.get("v2_nearfar") is FeatureSetV2NearFar
    with pytest.raises(KeyError):
        r.get("nope")


# --- vector construction ----------------------------------------------------

def test_absent_slots_get_the_not_observed_sentinel():
    f = FeatureSetV2()
    vec = f.build_feature_vector({"near": _player([10, 20, 30, 80])}, None, None, 0.2)
    per = FeatureSetV2.per_slot_dim()
    assert vec.shape == (580,)
    assert vec[0] == 1.0                       # near present
    for i in (1, 2, 3):                        # far, crop0, crop1 absent
        assert vec[i * per] == 0.0
        assert np.all(vec[i * per + 1:(i + 1) * per] == -1.0)


def test_velocity_is_zero_without_a_previous_frame():
    f = FeatureSetV2()
    vec = f.build_feature_vector({"near": _player([0, 0, 10, 100])}, None, None, 0.2)
    assert vec[7] == 0.0 and vec[8] == 0.0     # velocity x,y


def test_velocity_uses_the_previous_slot_of_the_same_name():
    f = FeatureSetV2()
    prev = {"near": _player([0, 0, 10, 100])}
    cur = {"near": _player([10, 0, 20, 100])}   # centroid moved +10px in x
    vec = f.build_feature_vector(cur, prev, None, 0.2)
    assert vec[7] == pytest.approx(50.0)        # 10px / 0.2s
    assert vec[8] == pytest.approx(0.0)


# --- crop slot selection/ordering -------------------------------------------

def _mask_all_court(h=1080, w=1920):
    return np.zeros((h, w), np.uint8)           # 0 == inside court


def test_feet_not_centroid_decides_court_membership():
    mask = np.full((1080, 1920), 255, np.uint8)
    mask[600:1000, :] = 0                       # court band low in the frame
    # Tall box whose FEET are in the band but whose centroid is above it.
    box = np.array([900, 300, 1000, 700], np.float32)
    assert _feet_on_court(box, mask) is True
    cy = int((box[1] + box[3]) / 2)
    assert mask[cy, 950] != 0, "centroid is outside; feet rule must still accept"


def test_crop_slots_order_by_feet_y_ascending_not_confidence():
    """Slot 0 is the higher (farther) player. Ordering by confidence would let
    slots swap between frames and corrupt the velocity features."""
    boxes = np.array([[900, 400, 950, 520],     # nearer (feet y=520), high conf
                      [900, 150, 940, 260]],    # farther (feet y=260), low conf
                     np.float32)
    conf = np.array([0.95, 0.30], np.float32)
    kps = np.zeros((2, K, 2), np.float32)
    kconf = np.ones((2, K), np.float32)
    s0, s1 = _crop_slots(boxes, conf, kps, kconf, _mask_all_court())
    assert s0["box"][3] == 260.0, "slot 0 must be the farther (smaller feet-y)"
    assert s1["box"][3] == 520.0


def test_crop_slots_select_top_two_by_confidence():
    """Selection is by confidence so a weak spurious blob cannot evict a real
    player -- even though ordering is positional."""
    boxes = np.array([[100, 100, 140, 200],
                      [300, 120, 340, 240],
                      [500, 130, 540, 260]], np.float32)
    conf = np.array([0.9, 0.05, 0.8], np.float32)
    kps = np.zeros((3, K, 2), np.float32)
    kconf = np.ones((3, K), np.float32)
    s0, s1 = _crop_slots(boxes, conf, kps, kconf, _mask_all_court())
    kept = sorted(round(float(s["box_conf"]), 3) for s in (s0, s1))
    assert kept == [0.8, 0.9], "the 0.05 blob must be dropped"


def test_crop_slots_are_none_when_nothing_is_on_court():
    boxes = np.array([[10, 10, 20, 40]], np.float32)
    mask = np.full((1080, 1920), 255, np.uint8)   # nothing inside court
    s0, s1 = _crop_slots(boxes, np.array([0.9], np.float32),
                         np.zeros((1, K, 2), np.float32), np.ones((1, K), np.float32), mask)
    assert s0 is None and s1 is None


def test_default_court_mask_is_binary_and_resized():
    """Court-detection failures fall back to the runtime's default mask rather
    than passing every detection through unfiltered (4 of the 26 1080p videos)."""
    m = _default_court_mask(1920, 1080)
    assert m is not None, "default_court_mask.png must ship with the package"
    assert m.shape == (1080, 1920)
    assert set(np.unique(m)).issubset({0, 255}), "resize must not introduce grey"
    on_court = float((m == 0).mean())
    assert 0.05 < on_court < 0.95, f"implausible on-court fraction {on_court:.2f}"


def test_default_mask_rejects_the_stands_and_keeps_mid_court():
    m = _default_court_mask(1920, 1080)
    # Top of frame is stands/backdrop in the golden corpus; mid-frame is court.
    assert _feet_on_court(np.array([950, 0, 990, 5], np.float32), m) is False
    assert _feet_on_court(np.array([930, 500, 990, 700], np.float32), m) is True


def test_single_detection_fills_only_slot_zero():
    boxes = np.array([[900, 200, 940, 300]], np.float32)
    s0, s1 = _crop_slots(boxes, np.array([0.7], np.float32),
                         np.zeros((1, K, 2), np.float32), np.ones((1, K), np.float32),
                         _mask_all_court())
    assert s0 is not None and s1 is None
