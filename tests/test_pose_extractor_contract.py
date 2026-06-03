from __future__ import annotations

import numpy as np

from helpers.module_stubs import import_pose_extractor_with_stubs
from helpers.runtime_fixtures import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    RecordingFakeYoloModel,
    fake_yolo_result,
)


class _FakeStream:
    frames = 1


class _FakeStreams:
    video = [_FakeStream()]


class _FakeContainer:
    streams = _FakeStreams()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _extract_with_fake_model(tmp_path, monkeypatch, *, imgsz: int = 960, fail_on_batch: bool = False):
    pose_module = import_pose_extractor_with_stubs(monkeypatch)
    PoseExtractor = pose_module.PoseExtractor

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pose_module.av, "open", lambda _path: _FakeContainer())

    extractor = PoseExtractor.__new__(PoseExtractor)
    extractor.model_path = "yolov8n-pose.pt"
    extractor.model_dir = None
    extractor.device = "cpu"
    extractor.batch_size = 1
    extractor.imgsz = imgsz
    extractor.model = RecordingFakeYoloModel([fake_yolo_result(0)], fail_on_batch=fail_on_batch)
    extractor.frame_iterator_pyav = lambda _path: iter(
        [(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8), 0.0)]
    )

    output_path = extractor.extract_pose_data(
        video_path=str(tmp_path / "input.mp4"),
        confidence_threshold=0.25,
        start_time_seconds=0,
        duration_seconds=1,
        target_fps=5,
    )
    with np.load(output_path, allow_pickle=True) as data:
        frames = data["frames"]
    return extractor, frames[0]


def test_pose_extractor_preserves_yolo_box_confidence(tmp_path, monkeypatch):
    _extractor, frame = _extract_with_fake_model(tmp_path, monkeypatch)
    expected = fake_yolo_result(0).boxes.conf.numpy()

    np.testing.assert_allclose(frame["box_conf"], expected)


def test_pose_extractor_passes_configured_imgsz_to_yolo_predict(tmp_path, monkeypatch):
    extractor, _frame = _extract_with_fake_model(tmp_path, monkeypatch, imgsz=960)

    assert extractor.model.predict_calls
    assert extractor.model.predict_calls[-1]["imgsz"] == 960


def test_pose_extractor_passes_configured_imgsz_to_fallback_predict(tmp_path, monkeypatch):
    extractor, _frame = _extract_with_fake_model(tmp_path, monkeypatch, imgsz=960, fail_on_batch=True)

    assert len(extractor.model.predict_calls) == 2
    assert "batch" not in extractor.model.predict_calls[-1]
    assert extractor.model.predict_calls[-1]["imgsz"] == 960


def test_pose_extractor_keeps_original_frame_coordinate_scale(tmp_path, monkeypatch):
    _extractor, frame = _extract_with_fake_model(tmp_path, monkeypatch, imgsz=960)

    boxes = frame["boxes"]
    keypoints = frame["keypoints"]

    assert np.all((boxes[:, [0, 2]] >= 0) & (boxes[:, [0, 2]] <= FRAME_WIDTH))
    assert np.all((boxes[:, [1, 3]] >= 0) & (boxes[:, [1, 3]] <= FRAME_HEIGHT))
    assert np.all((keypoints[:, :, 0] >= 0) & (keypoints[:, :, 0] <= FRAME_WIDTH))
    assert np.all((keypoints[:, :, 1] >= 0) & (keypoints[:, :, 1] <= FRAME_HEIGHT))


def test_pose_extractor_preserves_confidence_ranges(tmp_path, monkeypatch):
    _extractor, frame = _extract_with_fake_model(tmp_path, monkeypatch, imgsz=1920)

    assert np.all((frame["box_conf"] >= 0.0) & (frame["box_conf"] <= 1.0))
    assert np.all((frame["conf"] >= 0.0) & (frame["conf"] <= 1.0))
