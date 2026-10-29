"""Court 'out' mask from the court line model's doubles sidelines (court_lines.out_mask)."""
from __future__ import annotations

import numpy as np

from preprocessing.court_lines import BASE_HORIZONTAL_SHIFT, CourtDetection, geometry_from_decode, out_mask

W, H = 1280, 720
# A typical broadcast-style court: near baseline low and wide, far baseline high and narrow.
COURT = {"near_left": (240.0, 650.0), "near_right": (1040.0, 652.0),
         "far_left": (470.0, 200.0), "far_right": (810.0, 199.0)}


def _classic_out_mask(left, right, baseline, width, height):
    """The classic detector's rule (estimate_playable_court_area), for sloped sidelines."""
    shift = BASE_HORIZONTAL_SHIFT * abs(baseline[2] - baseline[0]) / width
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    out = np.zeros((height, width), np.uint8)
    for x1, y1, x2, y2 in (left, right):
        slope = (y2 - y1) / (x2 - x1)
        out[ys < slope * xs + (y1 - slope * x1) - shift / np.sqrt(1 + slope**2)] = 255
    return out


def test_out_mask_matches_the_classic_rule_on_doubles_sidelines():
    nl, nr, fl, fr = (COURT[k] for k in ("near_left", "near_right", "far_left", "far_right"))
    want = _classic_out_mask((*nl, *fl), (*nr, *fr), (*nl, *nr), W, H)
    got = out_mask(COURT, W, H)
    assert got.dtype == np.uint8 and got.shape == (H, W)
    np.testing.assert_array_equal(got, want)
    # The padding keeps the doubles sideline itself in, and the corners outside it out.
    assert got[650, 245] == 0 and got[0, 0] == 255 and got[0, W - 1] == 255


def test_out_mask_vertical_sidelines_pad_outward():
    court = {"near_left": (300.0, 700.0), "far_left": (300.0, 100.0),
             "near_right": (940.0, 700.0), "far_right": (940.0, 100.0)}
    shift = BASE_HORIZONTAL_SHIFT * 640 / W  # 50 px
    got = out_mask(court, W, H)
    row = got[400]
    assert row[int(300 - shift) - 1] == 255 and row[int(300 - shift)] == 0
    assert row[int(940 + shift)] == 0 and row[int(940 + shift) + 1] == 255


def test_out_mask_clips_an_off_frame_baseline_to_the_visible_width():
    # Near corners off both frame edges: the visible baseline is the full width.
    court = dict(COURT, near_left=(-300.0, 760.0), near_right=(1580.0, 760.0))
    nl, nr, fl, fr = (court[k] for k in ("near_left", "near_right", "far_left", "far_right"))
    want = _classic_out_mask((*nl, *fl), (*nr, *fr), (0.0, 0.0, float(W), 0.0), W, H)
    np.testing.assert_array_equal(out_mask(court, W, H), want)


def test_out_mask_is_none_without_all_four_corners():
    assert out_mask({k: v for k, v in COURT.items() if k != "far_right"}, W, H) is None
    assert out_mask(dict(COURT, near_left=None), W, H) is None


def test_out_mask_is_none_when_nothing_is_out():
    # Sidelines far outside the frame on both sides: no pixel is out.
    court = {"near_left": (-5000.0, 10000.0), "far_left": (-4000.0, -10000.0),
             "near_right": (6000.0, 10000.0), "far_right": (5000.0, -10000.0)}
    assert out_mask(court, W, H) is None


def _decoded(points_px, size):
    """A decode() tuple: points normalized by the map size, all six points found."""
    names = ("near_left", "near_right", "far_left", "far_right")
    pts = np.array([points_px[n] for n in names] + [(0.0, 0.0), (0.0, 0.0)]) / np.array(size)
    ok = np.array([True] * 4 + [False, False])
    return np.ones(5, bool), pts, ok


def test_court_detection_out_mask_scales_decoded_points_to_the_frame():
    small = {k: (x / 2, y / 2) for k, (x, y) in COURT.items()}  # same court, half-size frame
    court = CourtDetection(_decoded(small, (640, 360)), frame_size=(W, H), frames_used=3)
    got = court.out_mask()
    assert got.shape == (H, W)  # frame_size is (width, height); the mask is (height, width)
    np.testing.assert_array_equal(got, out_mask(COURT, W, H))


def test_geometry_keeps_the_model_net_points_when_the_corners_are_degenerate():
    line_ok, pts, ok = _decoded({k: (0.5, 0.5) for k in COURT}, (1, 1))  # four equal corners
    pts[4:] = [[0.3, 0.4], [0.7, 0.4]]
    ok[4:] = True
    geometry = geometry_from_decode((line_ok, pts, ok), (W, H), frames_used=3)
    assert geometry["net_source"] == ["model", "model"]
    assert geometry["points_px"]["net_left"] == [384.0, 288.0]
