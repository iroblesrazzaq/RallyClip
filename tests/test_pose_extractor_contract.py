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


def test_pose_extractor_crop_pass_adds_far_court_detections_in_frame_pixels(tmp_path, monkeypatch):
    from extraction.crop_pass import CROP_OUT_SIZE, CROP_WINDOW, crop_to_frame, merge_detections

    pose_module = import_pose_extractor_with_stubs(monkeypatch)
    monkeypatch.setattr(pose_module.av, "open", lambda _path: _FakeContainer())
    extractor = pose_module.PoseExtractor.__new__(pose_module.PoseExtractor)
    extractor.model_path = "yolov8n-pose.pt"
    extractor.model_dir = None
    extractor.device = "cpu"
    extractor.batch_size = 1
    extractor.imgsz = 960
    extractor.model = RecordingFakeYoloModel([fake_yolo_result(0)])
    extractor.frame_iterator_pyav = lambda _path: iter(
        [(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8), 0.0)]
    )

    [frame] = list(extractor.iter_pose_frames(
        video_path=str(tmp_path / "input.mp4"),
        confidence_threshold=0.25,
        duration_seconds=1,
        target_fps=5,
        crop_window=CROP_WINDOW,
    ))

    full_call, crop_call = extractor.model.predict_calls
    assert full_call["source"][0].shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert crop_call["source"][0].shape == (CROP_OUT_SIZE[1], CROP_OUT_SIZE[0], 3)
    res = fake_yolo_result(0)
    boxes, _, kps, _ = merge_detections(
        res.boxes.xyxy.numpy(), res.boxes.conf.numpy(), res.keypoints.xy.numpy(), res.keypoints.conf.numpy()
    )
    want_boxes, want_kps = crop_to_frame(boxes, kps, FRAME_WIDTH, FRAME_HEIGHT, CROP_WINDOW)
    np.testing.assert_allclose(frame["crop"]["boxes"], want_boxes, rtol=1e-6)
    np.testing.assert_allclose(frame["crop"]["keypoints"], want_kps, rtol=1e-6)
    np.testing.assert_allclose(frame["boxes"], res.boxes.xyxy.numpy())


class _PairingFakeModel:
    """Runs frame + crop pairs: answers each image by its shape, like a batch-2 runner."""

    runs_pairs = True

    def __init__(self):
        self.predict_calls: list[dict] = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        # A full frame gets two players, a crop one, so a swapped split shows.
        return [fake_yolo_result(0, include_far=img.shape[0] == FRAME_HEIGHT) for img in kwargs["source"]]


def test_pose_extractor_sends_frame_and_crop_in_one_call_to_a_pair_runner(tmp_path, monkeypatch):
    from extraction.crop_pass import CROP_OUT_SIZE, CROP_WINDOW

    pose_module = import_pose_extractor_with_stubs(monkeypatch)
    monkeypatch.setattr(pose_module.av, "open", lambda _path: _FakeContainer())
    extractor = pose_module.PoseExtractor.__new__(pose_module.PoseExtractor)
    extractor.model_path = "yolov8n-pose-960-dynamic.onnx"
    extractor.model_dir = None
    extractor.device = "coreml"
    extractor.batch_size = 2
    extractor.imgsz = 960
    extractor.model = _PairingFakeModel()
    extractor.frame_iterator_pyav = lambda _path: iter(
        [(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8), t) for t in (0.0, 0.2)]
    )

    frames = list(extractor.iter_pose_frames(
        video_path=str(tmp_path / "input.mp4"),
        confidence_threshold=0.25,
        duration_seconds=1,
        target_fps=5,
        crop_window=CROP_WINDOW,
    ))

    [call] = extractor.model.predict_calls
    crop_shape = (CROP_OUT_SIZE[1], CROP_OUT_SIZE[0], 3)
    assert [img.shape for img in call["source"]] == [(FRAME_HEIGHT, FRAME_WIDTH, 3), crop_shape] * 2
    assert len(frames) == 2
    for frame in frames:
        assert len(frame["boxes"]) == 2 and len(frame["crop"]["boxes"]) == 1
