from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from training.features.builder import FeatureBuildConfig, FeatureBuilder, _pack_player, _player_velocity


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


def test_a_video_with_no_labeled_frames_left_drops_its_old_features(tmp_path):
    import h5py

    # An annotation update moved ignore_before_s past every frame.
    preproc = tmp_path / "m.h5"
    with h5py.File(preproc, "w") as h5f:
        h5f.create_dataset("targets", data=np.full(10, -1, dtype=np.int8))
    old = tmp_path / "m__features__v1.h5"
    old.write_bytes(b"old build")
    assert FeatureBuilder(FeatureBuildConfig(feature_set="v1", target_fps=5.0)).build(preproc, old) is None
    assert not old.exists()
