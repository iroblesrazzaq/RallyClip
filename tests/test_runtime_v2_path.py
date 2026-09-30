"""Runtime v2 path: far-court crop pass, court line decode, net-line slots, v2 features."""

import numpy as np
import pytest

from extraction.crop_pass import crop_frame, crop_pixels, crop_to_frame, merge_detections
from preprocessing.court_lines import decode
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.netline_slots import CourtGeometry
from rallyclip_engine.models import _iter_v2_features

# Straight-on synthetic court in 1920x1080 reference pixels: x = 960 + 50u,
# y = 600 - 20v (court meters), so the net (v = 0) sits at y = 600.
PX_TO_M = np.linalg.inv(np.array([[50.0, 0.0, 960.0], [0.0, -20.0, 600.0], [0.0, 0.0, 1.0]]))
GEOM = CourtGeometry(np.array([686.0, 600.0]), np.array([1234.0, 600.0]), PX_TO_M, "test")


def box_at(u, v, h=120.0):
    x, y = 960 + 50 * u, 600 - 20 * v
    return np.array([x - 25, y - h, x + 25, y], dtype=np.float32)


def detections(boxes, scale=1.0):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) * scale
    n = len(boxes)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    kps = np.repeat(np.stack([cx, cy], 1)[:, None, :], 17, axis=1)
    return {
        "boxes": boxes,
        "box_conf": np.full(n, 0.9, np.float32),
        "keypoints": kps.astype(np.float32),
        "conf": np.full((n, 17), 0.8, np.float32),
    }


# --- crop pass -------------------------------------------------------------------
def test_crop_window_is_the_training_crop_on_1080p():
    assert crop_pixels(1920, 1080) == (480, 0, 1440, 540)
    assert crop_frame(np.zeros((1080, 1920, 3), np.uint8)).shape == (540, 960, 3)


def test_crop_to_frame_maps_crop_pixels_back():
    boxes, kps = crop_to_frame(np.array([[0, 0, 480, 270]], np.float32), np.array([[[240, 135]]], np.float32),
                               width=3840, height=2160)
    np.testing.assert_allclose(boxes, [[960, 0, 1920, 540]])
    np.testing.assert_allclose(kps, [[[1440, 270]]])


def test_merge_detections_unions_overlapping_fragments():
    boxes = np.array([[0, 0, 10, 20], [0, 1, 10, 21], [50, 50, 60, 70]], np.float32)
    conf = np.array([0.5, 0.9, 0.7], np.float32)
    kps = np.arange(3 * 17 * 2, dtype=np.float32).reshape(3, 17, 2)
    kc = np.ones((3, 17), np.float32)
    ob, oc, ok, _ = merge_detections(boxes, conf, kps, kc)
    np.testing.assert_allclose(ob, [[0, 0, 10, 21], [50, 50, 60, 70]])
    np.testing.assert_allclose(oc, [0.9, 0.7])
    np.testing.assert_allclose(ok[0], kps[1])  # keypoints of the most confident member


# --- court line decode -----------------------------------------------------------
def test_decode_intersects_fitted_lines():
    probs = np.zeros((5, 72, 128), np.float32)
    probs[0, 60, 10:118] = 1.0      # near baseline
    probs[1, 20, 30:98] = 1.0       # far baseline
    probs[2, 15:66, 20] = 1.0       # left sideline
    probs[3, 15:66, 100] = 1.0      # right sideline
    probs[4, 40, 15:113] = 1.0      # net
    line_ok, points, point_ok = decode(probs)
    assert line_ok.all() and point_ok.all()
    # Pixel centers are at +0.5; points are normalized by the map size.
    expected = np.array([[20.5, 60.5], [100.5, 60.5], [20.5, 20.5], [100.5, 20.5], [20.5, 40.5], [100.5, 40.5]])
    np.testing.assert_allclose(points * [128, 72], expected, atol=1e-6)


def test_decode_reports_missing_line():
    probs = np.zeros((5, 72, 128), np.float32)
    probs[2, 10:60, 20] = 1.0
    probs[3, 10:60, 100] = 1.0
    line_ok, _points, point_ok = decode(probs)
    assert line_ok.tolist() == [False, False, True, True, False]
    assert not point_ok.any()


# --- slots + features ------------------------------------------------------------
def _pre():
    return DataPreprocessor(screen_width=1920, screen_height=1080)


def test_netline_slots_take_near_below_net_and_far_from_crop():
    near, far, spectator = box_at(0, -9), box_at(1, 11, h=50), box_at(-1, 3)
    frame = detections([near, spectator])
    frame["crop"] = detections([far])
    mask = np.zeros((1080, 1920), np.uint8)  # 0 = on court
    [(status, players)] = list(_pre().iter_slot_players([frame], mask, 1920, 1080, GEOM))
    assert status == 0
    np.testing.assert_allclose(players["near"]["box"], near)
    np.testing.assert_allclose(players["far"]["box"], far)


def test_slots_run_in_reference_pixels_for_other_resolutions():
    near, far = box_at(0, -9), box_at(1, 11, h=50)
    frame = detections([near], scale=2 / 3)  # 1280x720 source
    frame["crop"] = detections([far], scale=2 / 3)
    mask = np.zeros((720, 1280), np.uint8)
    [(_, players)] = list(_pre().iter_slot_players([frame], mask, 1280, 720, GEOM))
    np.testing.assert_allclose(players["near"]["box"], near, atol=1e-3)
    np.testing.assert_allclose(players["far"]["box"], far, atol=1e-3)


def test_classic_slots_without_geometry_and_skipped_frames():
    frame = detections([box_at(0, -9), box_at(1, 11, h=50)])
    mask = np.zeros((1080, 1920), np.uint8)
    stream = [frame, {"annotation_status": -1}]
    (status, players), skipped = list(_pre().iter_slot_players(stream, mask, 1920, 1080, None))
    assert status == 0 and players["near"] is not None
    assert skipped == (-1, None)


def test_v2_features_are_290_wide_and_drop_skipped_frames():
    frame = detections([box_at(0, -9)])
    frame["crop"] = detections([box_at(1, 11, h=50)])
    mask = np.zeros((1080, 1920), np.uint8)
    stream = _pre().iter_slot_players([frame, {"annotation_status": -1}, frame], mask, 1920, 1080, GEOM)
    out = list(_iter_v2_features(stream, "v2_nearfar", {}, 1920, 1080, 5.0))
    assert len(out) == 2
    assert all(vec.shape == (290,) and np.isfinite(vec).all() for vec, _ in out)


def test_v2_features_honour_manifest_drops():
    frame = detections([box_at(0, -9)])
    mask = np.zeros((1080, 1920), np.uint8)
    stream = _pre().iter_slot_players([frame], mask, 1920, 1080, GEOM)
    [(vec, _)] = list(_iter_v2_features(stream, "v2_nearfar", {"drop_slots": ["far"]}, 1920, 1080, 5.0))
    assert vec.shape == (145,)
