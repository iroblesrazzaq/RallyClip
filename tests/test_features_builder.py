from __future__ import annotations

import numpy as np

from training.features.builder import _pack_player, _player_velocity


def test_pack_player_exists_flag():
    kps = np.full((17, 2), -1.0, dtype=np.float32)
    conf = np.full((17,), -1.0, dtype=np.float32)
    box = np.full((4,), -1.0, dtype=np.float32)
    player = _pack_player(kps, conf, box, np.array(-1.0, dtype=np.float32))
    assert player["exists"] is False


def test_player_velocity_dt_zero():
    curr = {"exists": True, "box": np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)}
    prev = {"exists": True, "box": np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)}
    vel = _player_velocity(curr, prev, dt=0.0)
    assert vel == (0.0, 0.0)
