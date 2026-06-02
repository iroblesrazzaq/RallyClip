from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from features.feature_engineer import FeatureEngineer
from training.features.v1 import FeatureSetV1

from helpers.runtime_fixtures import (
    ACCELERATION,
    ACCELERATION_MAGNITUDE,
    BOX_CONFIDENCE,
    DT,
    FEATURE_DIM,
    FPS,
    KEYPOINT_ACCELERATION,
    KEYPOINT_ACCELERATION_MAGNITUDE,
    PER_PLAYER_DIM,
    SPEED,
    VELOCITY,
    absent_player,
    expected_feature_sequence,
    make_player,
    make_runtime_assigned_players,
    write_preprocessed_runtime_npz,
)


ROOT = Path(__file__).resolve().parents[1]


def _offset(target: slice, amount: int) -> slice:
    return slice(target.start + amount, target.stop + amount)


def _create_features(tmp_path, assigned_frames: list[dict]) -> np.ndarray:
    input_npz = write_preprocessed_runtime_npz(tmp_path / "preprocessed.npz", assigned_frames)
    output_npz = tmp_path / "features.npz"
    engineer = FeatureEngineer()
    engineer.target_fps = FPS

    assert engineer.create_features_from_preprocessed(str(input_npz), str(output_npz), overwrite=True)
    with np.load(output_npz) as data:
        return np.asarray(data["features"], dtype=np.float32)


def test_runtime_feature_dim_matches_v1_and_bundled_scaler(tmp_path):
    assigned_frames = [make_runtime_assigned_players(idx) for idx in range(3)]

    features = _create_features(tmp_path, assigned_frames)

    assert FeatureSetV1.feature_dim() == FEATURE_DIM
    assert features.shape == (3, FeatureSetV1.feature_dim())

    scaler_payload = json.loads((ROOT / "models" / "rallyclip_v0.3.1" / "scaler.json").read_text(encoding="utf-8"))
    mean = np.asarray(scaler_payload["mean"], dtype=np.float32)
    scale = np.asarray(scaler_payload["scale"], dtype=np.float32)
    assert mean.shape == (FEATURE_DIM,)
    assert scale.shape == (FEATURE_DIM,)
    scaled = (features - mean) / np.maximum(scale, 1e-12)
    assert scaled.shape == features.shape


def test_runtime_full_vector_matches_independent_v1_layout_for_two_players(tmp_path):
    assigned_frames = [make_runtime_assigned_players(idx) for idx in range(3)]
    expected = expected_feature_sequence(
        [(make_player("near", idx), make_player("far", idx)) for idx in range(3)],
        dt=DT,
    )

    features = _create_features(tmp_path, assigned_frames)

    np.testing.assert_allclose(features, expected, rtol=1e-6, atol=1e-6)


def test_runtime_motion_features_use_target_fps_dt(tmp_path):
    assigned_frames = [make_runtime_assigned_players(idx) for idx in range(3)]
    expected = expected_feature_sequence(
        [(make_player("near", idx), make_player("far", idx)) for idx in range(3)],
        dt=DT,
    )

    features = _create_features(tmp_path, assigned_frames)

    np.testing.assert_allclose(features[1, VELOCITY], expected[1, VELOCITY], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(features[2, ACCELERATION], expected[2, ACCELERATION], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        features[2, _offset(VELOCITY, PER_PLAYER_DIM)],
        expected[2, _offset(VELOCITY, PER_PLAYER_DIM)],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        features[2, _offset(ACCELERATION, PER_PLAYER_DIM)],
        expected[2, _offset(ACCELERATION, PER_PLAYER_DIM)],
        rtol=1e-6,
        atol=1e-6,
    )


def test_runtime_keypoint_accelerations_are_computed(tmp_path):
    assigned_frames = [make_runtime_assigned_players(idx) for idx in range(3)]
    expected = expected_feature_sequence(
        [(make_player("near", idx), make_player("far", idx)) for idx in range(3)],
        dt=DT,
    )

    features = _create_features(tmp_path, assigned_frames)

    assert np.any(expected[2, KEYPOINT_ACCELERATION] != 0.0)
    np.testing.assert_allclose(
        features[2, KEYPOINT_ACCELERATION],
        expected[2, KEYPOINT_ACCELERATION],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        features[2, KEYPOINT_ACCELERATION_MAGNITUDE],
        expected[2, KEYPOINT_ACCELERATION_MAGNITUDE],
        rtol=1e-6,
        atol=1e-6,
    )


def test_runtime_one_player_absent_uses_v1_missing_player_sentinels(tmp_path):
    assigned_frames = [make_runtime_assigned_players(idx, near=True, far=False) for idx in range(3)]
    expected = expected_feature_sequence(
        [(make_player("near", idx), absent_player()) for idx in range(3)],
        dt=DT,
    )

    features = _create_features(tmp_path, assigned_frames)

    np.testing.assert_allclose(features, expected, rtol=1e-6, atol=1e-6)
    assert np.all(features[:, PER_PLAYER_DIM + 1 :] == -1.0)


def test_runtime_reappearing_player_resets_motion_until_history_is_contiguous(tmp_path):
    assigned_frames = [
        make_runtime_assigned_players(0, near=True, far=True),
        make_runtime_assigned_players(1, near=True, far=False),
        make_runtime_assigned_players(2, near=True, far=True),
    ]
    expected = expected_feature_sequence(
        [
            (make_player("near", 0), make_player("far", 0)),
            (make_player("near", 1), absent_player()),
            (make_player("near", 2), make_player("far", 2)),
        ],
        dt=DT,
    )

    features = _create_features(tmp_path, assigned_frames)
    far_frame_3 = features[2, PER_PLAYER_DIM:]

    np.testing.assert_allclose(features, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(far_frame_3[VELOCITY], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(far_frame_3[ACCELERATION], np.zeros(2, dtype=np.float32))
    assert far_frame_3[SPEED] == -1.0
    assert far_frame_3[ACCELERATION_MAGNITUDE] == -1.0
    assert far_frame_3[BOX_CONFIDENCE] == make_player("far", 2)["box_conf"]
