from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


NUM_KEYPOINTS = 17
FPS = 5.0
DT = 1.0 / FPS
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PER_PLAYER_DIM = 181
FEATURE_DIM = PER_PLAYER_DIM * 2

EXISTS = slice(0, 1)
BOX = slice(1, 5)
CENTROID = slice(5, 7)
VELOCITY = slice(7, 9)
ACCELERATION = slice(9, 11)
SPEED = 11
ACCELERATION_MAGNITUDE = 12
KEYPOINTS = slice(13, 47)
KEYPOINT_CONFIDENCE = slice(47, 64)
KEYPOINT_VELOCITY = slice(64, 98)
KEYPOINT_ACCELERATION = slice(98, 132)
KEYPOINT_SPEED = slice(132, 149)
KEYPOINT_ACCELERATION_MAGNITUDE = slice(149, 166)
LIMB_LENGTHS = slice(166, 180)
BOX_CONFIDENCE = 180

LIMB_CONNECTIONS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
    (6, 5),
    (12, 11),
]

NEAR_BOXES = [
    np.array([100.0, 400.0, 140.0, 500.0], dtype=np.float32),
    np.array([110.0, 400.0, 150.0, 500.0], dtype=np.float32),
    np.array([130.0, 405.0, 170.0, 505.0], dtype=np.float32),
]
FAR_BOXES = [
    np.array([600.0, 200.0, 630.0, 260.0], dtype=np.float32),
    np.array([595.0, 202.0, 625.0, 262.0], dtype=np.float32),
    np.array([585.0, 206.0, 615.0, 266.0], dtype=np.float32),
]
NEAR_KEYPOINT_DELTAS = [
    np.array([0.0, 0.0], dtype=np.float32),
    np.array([2.0, 1.0], dtype=np.float32),
    np.array([5.0, 3.0], dtype=np.float32),
]
FAR_KEYPOINT_DELTAS = [
    np.array([0.0, 0.0], dtype=np.float32),
    np.array([-1.0, 2.0], dtype=np.float32),
    np.array([-3.0, 5.0], dtype=np.float32),
]
NEAR_BOX_CONF = [0.90, 0.91, 0.92]
FAR_BOX_CONF = [0.70, 0.71, 0.72]


class FakeTensor:
    def __init__(self, value: Iterable[float] | np.ndarray) -> None:
        self._value = np.asarray(value, dtype=np.float32)

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


@dataclass
class FakeBoxes:
    xyxy: FakeTensor
    conf: FakeTensor


@dataclass
class FakeKeypoints:
    xy: FakeTensor
    conf: FakeTensor


@dataclass
class FakeYoloResult:
    boxes: FakeBoxes
    keypoints: FakeKeypoints


class RecordingFakeYoloModel:
    def __init__(self, results: list[FakeYoloResult], fail_on_batch: bool = False) -> None:
        self.results = results
        self.fail_on_batch = fail_on_batch
        self.predict_calls: list[dict] = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        if self.fail_on_batch and "batch" in kwargs:
            raise TypeError("batch is not supported by this fake model")
        return self.results


def keypoints(base_x: float, base_y: float, delta: np.ndarray) -> np.ndarray:
    values = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    for idx in range(NUM_KEYPOINTS):
        values[idx] = [base_x + idx + delta[0], base_y + (idx * 2) + delta[1]]
    return values


def confidence(start: float) -> np.ndarray:
    return np.linspace(start, start + 0.16, NUM_KEYPOINTS, dtype=np.float32)


def make_player(side: str, frame_idx: int) -> dict:
    if side == "near":
        return {
            "exists": True,
            "box": NEAR_BOXES[frame_idx].copy(),
            "keypoints": keypoints(100.0, 410.0, NEAR_KEYPOINT_DELTAS[frame_idx]),
            "conf": confidence(0.50),
            "box_conf": float(NEAR_BOX_CONF[frame_idx]),
        }
    if side == "far":
        return {
            "exists": True,
            "box": FAR_BOXES[frame_idx].copy(),
            "keypoints": keypoints(600.0, 210.0, FAR_KEYPOINT_DELTAS[frame_idx]),
            "conf": confidence(0.60),
            "box_conf": float(FAR_BOX_CONF[frame_idx]),
        }
    raise ValueError(f"Unknown side: {side}")


def absent_player() -> dict:
    return {"exists": False}


def make_runtime_assigned_players(frame_idx: int, near: bool = True, far: bool = True) -> dict:
    return {
        "near_player": runtime_player(make_player("near", frame_idx)) if near else None,
        "far_player": runtime_player(make_player("far", frame_idx)) if far else None,
    }


def runtime_player(player: dict) -> dict:
    return {
        "box": player["box"],
        "keypoints": player["keypoints"],
        "conf": player["conf"],
        "box_conf": player["box_conf"],
    }


def fake_yolo_result(frame_idx: int, include_far: bool = True) -> FakeYoloResult:
    players = [make_player("near", frame_idx)]
    if include_far:
        players.append(make_player("far", frame_idx))
    return FakeYoloResult(
        boxes=FakeBoxes(
            xyxy=FakeTensor(np.stack([p["box"] for p in players])),
            conf=FakeTensor(np.array([p["box_conf"] for p in players], dtype=np.float32)),
        ),
        keypoints=FakeKeypoints(
            xy=FakeTensor(np.stack([p["keypoints"] for p in players])),
            conf=FakeTensor(np.stack([p["conf"] for p in players])),
        ),
    )


def expected_feature_sequence(frame_players: list[tuple[dict, dict]], dt: float = DT) -> np.ndarray:
    vectors: list[np.ndarray] = []
    prev_near = None
    prev_far = None
    prev_motion = {
        "near": {"centroid": None, "keypoints": None},
        "far": {"centroid": None, "keypoints": None},
    }
    for near, far in frame_players:
        vector = np.concatenate(
            [
                expected_player_features(near, prev_near, prev_motion["near"], dt),
                expected_player_features(far, prev_far, prev_motion["far"], dt),
            ]
        ).astype(np.float32)
        vectors.append(vector)
        prev_motion = {
            "near": {
                "centroid": player_velocity(near, prev_near, dt),
                "keypoints": keypoint_velocity(near, prev_near, dt),
            },
            "far": {
                "centroid": player_velocity(far, prev_far, dt),
                "keypoints": keypoint_velocity(far, prev_far, dt),
            },
        }
        prev_near = near
        prev_far = far
    return np.asarray(vectors, dtype=np.float32)


def expected_player_features(player: dict, prev_player: dict | None, prev_motion: dict, dt: float) -> np.ndarray:
    vector = np.full(PER_PLAYER_DIM, -1.0, dtype=np.float32)
    if not player.get("exists", False):
        vector[0] = 0.0
        return vector

    box = player["box"]
    kps = player["keypoints"]
    centroid = box_centroid(box)
    velocity = np.zeros(2, dtype=np.float32)
    acceleration = np.zeros(2, dtype=np.float32)
    speed = -1.0
    acceleration_magnitude = -1.0
    kp_velocity = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    kp_acceleration = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    kp_speed = np.full(NUM_KEYPOINTS, -1.0, dtype=np.float32)
    kp_acceleration_magnitude = np.full(NUM_KEYPOINTS, -1.0, dtype=np.float32)

    if prev_player and prev_player.get("exists", False):
        velocity = player_velocity(player, prev_player, dt)
        speed = float(np.linalg.norm(velocity))
        if prev_motion["centroid"] is not None:
            acceleration = (velocity - np.asarray(prev_motion["centroid"], dtype=np.float32)) / dt
            acceleration_magnitude = float(np.linalg.norm(acceleration))

        kp_velocity = keypoint_velocity(player, prev_player, dt)
        kp_speed = np.linalg.norm(kp_velocity, axis=1)
        if prev_motion["keypoints"] is not None:
            kp_acceleration = (kp_velocity - prev_motion["keypoints"]) / dt
        kp_acceleration_magnitude = np.linalg.norm(kp_acceleration, axis=1)

    vector[0] = 1.0
    vector[BOX] = box
    vector[CENTROID] = centroid
    vector[VELOCITY] = velocity
    vector[ACCELERATION] = acceleration
    vector[SPEED] = speed
    vector[ACCELERATION_MAGNITUDE] = acceleration_magnitude
    vector[KEYPOINTS] = kps.flatten()
    vector[KEYPOINT_CONFIDENCE] = player["conf"]
    vector[KEYPOINT_VELOCITY] = kp_velocity.flatten()
    vector[KEYPOINT_ACCELERATION] = kp_acceleration.flatten()
    vector[KEYPOINT_SPEED] = kp_speed
    vector[KEYPOINT_ACCELERATION_MAGNITUDE] = kp_acceleration_magnitude
    vector[LIMB_LENGTHS] = limb_lengths(kps)
    vector[BOX_CONFIDENCE] = player["box_conf"]
    return vector


def box_centroid(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def player_velocity(player: dict, prev_player: dict | None, dt: float) -> np.ndarray | None:
    if not player.get("exists", False) or not prev_player or not prev_player.get("exists", False):
        return None
    return (box_centroid(player["box"]) - box_centroid(prev_player["box"])) / dt


def keypoint_velocity(player: dict, prev_player: dict | None, dt: float) -> np.ndarray | None:
    if not player.get("exists", False) or not prev_player or not prev_player.get("exists", False):
        return None
    return (player["keypoints"] - prev_player["keypoints"]) / dt


def limb_lengths(kps: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.norm(kps[i] - kps[j]) for i, j in LIMB_CONNECTIONS], dtype=np.float32)


def write_preprocessed_runtime_npz(path: Path, assigned_frames: list[dict]) -> Path:
    frames = [{"boxes": np.empty((0, 4)), "keypoints": np.empty((0, 17, 2)), "conf": np.empty((0, 17))} for _ in assigned_frames]
    targets = np.zeros(len(assigned_frames), dtype=np.int8)
    near_players = np.asarray([frame["near_player"] for frame in assigned_frames], dtype=object)
    far_players = np.asarray([frame["far_player"] for frame in assigned_frames], dtype=object)
    np.savez_compressed(
        path,
        frames=np.asarray(frames, dtype=object),
        targets=targets,
        near_players=near_players,
        far_players=far_players,
    )
    return path


def write_raw_pose_npz(path: Path, results: list[FakeYoloResult]) -> Path:
    frames = []
    for result in results:
        frames.append(
            {
                "boxes": result.boxes.xyxy.numpy(),
                "box_conf": result.boxes.conf.numpy(),
                "keypoints": result.keypoints.xy.numpy(),
                "conf": result.keypoints.conf.numpy(),
                "annotation_status": 0,
            }
        )
    np.savez_compressed(path, frames=np.asarray(frames, dtype=object))
    return path


def write_manifest_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.onnx").write_bytes(b"not a real onnx model")
    payload = {
        "feature_pipeline": {
            "feature_dim": FEATURE_DIM,
            "feature_set": "v1",
            "target_fps": FPS,
            "sample_fps": FPS,
            "imgsz": 960,
            "conf": 0.25,
            "num_keypoints": 17,
            "screen_width": 1280,
            "screen_height": 720,
            "yolo_model": "yolov8n-pose.pt",
        },
        "inference": {
            "input_name": "features",
            "input_shape": [1, 100, FEATURE_DIM],
            "seq_len_frames": 100,
            "overlap_frames": 50,
        },
        "postprocess": {
            "params": {
                "high": 0.7,
                "low": 0.45,
                "sigma": 1.0,
                "min_dur_sec": 1.0,
            }
        },
    }
    (path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (path / "scaler.json").write_text(
        json.dumps(
            {
                "feature_dim": FEATURE_DIM,
                "mean": [0.0] * FEATURE_DIM,
                "scale": [1.0] * FEATURE_DIM,
            }
        ),
        encoding="utf-8",
    )
    return path
