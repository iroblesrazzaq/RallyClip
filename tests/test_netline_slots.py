import numpy as np
import pytest

from training.preprocess.player_assigner import PlayerAssigner
from preprocessing.netline_slots import CourtGeometry, pick_far
from training.preprocess.preprocessor import _netline_players

# A synthetic straight-on court: pixels are an affine map of court meters,
# x = 960 + 50u, y = 600 - 20v, so the net (v = 0) sits at y = 600.
PX_TO_M = np.linalg.inv(np.array([[50.0, 0.0, 960.0], [0.0, -20.0, 600.0], [0.0, 0.0, 1.0]]))
GEOM = CourtGeometry(np.array([686.0, 600.0]), np.array([1234.0, 600.0]), PX_TO_M, "test")


def box_at(u: float, v: float, h: float = 60.0) -> np.ndarray:
    """Box whose feet (bottom-center) stand at court meters (u, v)."""
    x, y = 960 + 50 * u, 600 - 20 * v
    return np.array([x - 15, y - h, x + 15, y], dtype=np.float32)


def test_below_net_uses_box_bottom():
    assert GEOM.below_net(box_at(0, -5))
    assert not GEOM.below_net(box_at(0, 5))


def test_feet_to_court_round_trip():
    u, v = GEOM.feet_to_court(box_at(2.5, 9.0))
    assert u == pytest.approx(2.5) and v == pytest.approx(9.0)


def _perspective_box(u: float, v: float) -> np.ndarray:
    """Feet at court meters (u, v) for a camera 20 m behind the net:
    x = 960 + 800u/z, y = 300 + 6000/z with z = v + 20, so the horizon is y = 300."""
    z = v + 20.0
    x, y = 960 + 800 * u / z, 300 + 6000 / z
    return np.array([x - 10, y - 40, x + 10, y], dtype=np.float32)


def test_horizon_inside_the_frame_keeps_court_points():
    # findHomography output (H[2,2] = 1): w = +1 at the origin, which is above
    # the y = 300 horizon, so w < 0 on the whole court until the sign is fixed.
    m_to_px = np.array([[800.0, 960.0, 960.0 * 20], [0.0, 300.0, 300.0 * 20 + 6000], [0.0, 1.0, 20.0]])
    px_to_m = np.linalg.inv(m_to_px)
    px_to_m /= px_to_m[2, 2]
    half = 800 * 5.485 / 20
    geom = CourtGeometry(np.array([960 - half, 600.0]), np.array([960 + half, 600.0]), px_to_m)
    u, v = geom.feet_to_court(_perspective_box(1.0, 10.0))
    assert u == pytest.approx(1.0) and v == pytest.approx(10.0)
    assert pick_far(np.stack([_perspective_box(1.0, 10.0)]), None, geom) == 0
    assert geom.feet_to_court(np.array([950, 200, 970, 250], dtype=np.float32)) is None  # above the horizon


def test_pick_far_prefers_center_and_gates_the_far_half():
    boxes = np.stack([
        box_at(4.0, 11.0),   # far player, off-center
        box_at(9.0, 10.0),   # beyond the doubles alley + margin -> gated out
        box_at(0.5, 30.0),   # behind the run-back (spectator) -> gated out
        box_at(0.0, -6.0),   # near side of the net -> excluded
    ])
    assert pick_far(boxes, None, GEOM) == 0


def test_pick_far_tie_breaks_toward_the_net():
    boxes = np.stack([box_at(0.8, 12.0), box_at(0.2, 16.0)])
    # |u| within 1 m of each other -> the one closer to the net wins.
    assert pick_far(boxes, None, GEOM) == 0


def test_pick_far_empty():
    assert pick_far(np.empty((0, 4), np.float32), None, GEOM) is None


def test_netline_players_near_below_net_far_from_crop():
    full = np.stack([box_at(0, -10), box_at(1, 10)])  # near player, far player (full frame)
    filtered = {
        "boxes": full,
        "box_conf": np.array([0.9, 0.8], np.float32),
        "keypoints": np.zeros((2, 17, 2), np.float32),
        "keypoint_conf": np.ones((2, 17), np.float32),
    }
    crop = np.stack([box_at(0.5, 11.0)])
    players = _netline_players(
        PlayerAssigner(screen_width=1920, screen_height=1080), filtered,
        crop, np.array([0.7], np.float32), np.full((1, 17, 2), 7.0, np.float32), np.ones((1, 17), np.float32),
        None, GEOM,
    )
    np.testing.assert_allclose(players["near_box"][0], full[0])
    np.testing.assert_allclose(players["far_box"][0], crop[0])
    assert players["far_kps"][0, 0, 0] == 7.0


def test_netline_players_no_far_is_sentinel():
    filtered = {
        "boxes": np.stack([box_at(0, 10)]),  # only a box above the net -> not near
        "box_conf": np.array([0.9], np.float32),
        "keypoints": np.zeros((1, 17, 2), np.float32),
        "keypoint_conf": np.ones((1, 17), np.float32),
    }
    players = _netline_players(
        PlayerAssigner(screen_width=1920, screen_height=1080), filtered,
        np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0, 17, 2), np.float32),
        np.empty((0, 17), np.float32), None, GEOM,
    )
    assert np.all(players["near_box"] == -1) and np.all(players["far_box"] == -1)
