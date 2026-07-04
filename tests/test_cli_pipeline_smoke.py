from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers.module_stubs import import_cli_main_with_stubs
from helpers.runtime_fixtures import FEATURE_DIM


def test_run_pipeline_streams_runtime_stages(tmp_path, monkeypatch):
    # The release path streams every stage end-to-end: iter_pose_frames ->
    # iter_preprocess_frames -> iter_build_features -> per-row scale -> streaming windowed
    # inference, with no intermediate NPZ round-trips and no np.load of intermediates.
    cli_main = import_cli_main_with_stubs(monkeypatch)
    calls: list[str] = []
    video_path = tmp_path / "match.mp4"
    model_path = tmp_path / "model.onnx"
    scaler_path = tmp_path / "scaler.json"
    video_path.write_bytes(b"fake video")
    model_path.write_bytes(b"fake onnx")
    scaler_path.write_text("{}", encoding="utf-8")

    class FakePoseExtractor:
        def __init__(self, model_path, model_dir):
            calls.append("pose:init")
            assert Path(model_path).name == "yolov8n-pose.pt"
            assert Path(model_dir).name == "models"

        def iter_pose_frames(self, **kwargs):
            calls.append("pose:iter")
            assert kwargs["target_fps"] == 5
            assert kwargs["imgsz"] == 960
            assert kwargs["confidence_threshold"] == 0.25
            return iter(["POSE_STREAM"])  # opaque stream handed to preprocess

    class FakePreprocessor:
        def __init__(self, save_court_masks, screen_width=None, screen_height=None, yolo_model_path=None, conf=None, **_kwargs):
            calls.append("preprocess:init")
            assert save_court_masks is False
            assert screen_width == 1280
            assert screen_height == 720
            assert Path(yolo_model_path).name == "yolov8n-pose.pt"
            assert conf == 0.25

        def compute_court_mask(self, vp):
            calls.append("court:detect")
            assert vp == str(video_path)
            return "COURT_MASK", {"source": "detected", "detected": True, "timestamp_s": 60}

        def _source_frame_shape(self, vp):
            calls.append("preprocess:srcshape")
            assert vp == str(video_path)
            return (720, 1280, 3)

        def iter_preprocess_frames(self, pose_stream, court_mask, src_width, src_height):
            calls.append("preprocess:iter")
            assert list(pose_stream) == ["POSE_STREAM"]  # in-memory hand-off, not an NPZ path
            assert court_mask == "COURT_MASK"
            assert (src_width, src_height) == (1280, 720)
            return iter(["PRE_STREAM"])

    feature_engineer_fps: list[float] = []

    class FakeFeatureEngineer:
        def __init__(self, target_fps=None, **_kwargs):
            calls.append("features:init")
            feature_engineer_fps.append(float(target_fps))

        def iter_build_features(self, preprocessed_stream):
            calls.append("features:iter")
            assert list(preprocessed_stream) == ["PRE_STREAM"]
            for _ in range(100):
                yield np.ones(FEATURE_DIM, dtype=np.float32), 0

    class FakeScaler:
        def transform(self, features):
            return features  # identity; per-row in the streaming path

    def fake_onnx_stream(model_path_arg, feature_rows, sequence_length, overlap, **kwargs):
        calls.append("inference:onnx")
        assert model_path_arg == str(model_path)
        assert sequence_length == 100
        assert overlap == 50
        rows = list(feature_rows)  # consume the streaming chain
        assert len(rows) == 100
        assert all(r.shape == (FEATURE_DIM,) for r in rows)
        probs = np.zeros(len(rows), dtype=np.float32)
        probs[10:25] = 0.95
        return probs

    def fake_np_load(path, *args, **kwargs):
        raise AssertionError(f"release path must not np.load intermediates (got {path})")

    def fake_write_segments_csv(segments, output_csv_path, fps, overwrite):
        calls.append("output:csv")
        assert segments == [(10, 25)]
        assert output_csv_path.endswith("_segments.csv")
        assert fps == 5.0
        assert overwrite is True

    def fake_segment_video(input_video, intervals_sec, output_video):
        calls.append("output:video")
        assert input_video == str(video_path)
        assert intervals_sec == [(2.0, 5.0)]
        assert output_video.endswith("_segmented.mp4")

    monkeypatch.setattr(cli_main, "PoseExtractor", FakePoseExtractor)
    monkeypatch.setattr(cli_main, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(cli_main, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(cli_main.np, "load", fake_np_load)
    monkeypatch.setattr(cli_main, "load_scaler_asset", lambda _path: FakeScaler())
    monkeypatch.setattr(cli_main, "run_windowed_inference_average_onnx_stream", fake_onnx_stream)
    monkeypatch.setattr(cli_main, "gaussian_filter1d", lambda values, sigma: values)
    monkeypatch.setattr(cli_main, "write_segments_csv", fake_write_segments_csv)
    monkeypatch.setattr(cli_main, "segment_video", fake_segment_video)
    # Stub the input preflight; this test stubs the video stages, so the fake
    # video bytes aren't a real decodable file. Validation is covered by
    # tests/test_video_validation.py.
    monkeypatch.setattr(cli_main, "validate_video", lambda *a, **k: None)

    cfg = cli_main.RunConfig(
        video_path=video_path,
        output_dir=tmp_path / "output_videos",
        output_name=None,
        csv_output_dir=tmp_path / "output_csvs",
        write_csv=True,
        segment_video=True,
        yolo_weights="yolov8n-pose.pt",
        yolo_device=None,
        model_path=model_path,
        scaler_path=scaler_path,
        fps=5.0,
        seq_len=100,
        overlap=50,
        sigma=1.0,
        low=0.45,
        high=0.7,
        min_dur_sec=1.0,
        conf=0.25,
        imgsz=960,
        feature_set="v1",
        screen_width=1280,
        screen_height=720,
        start_time=0,
        duration=999999,
    )

    assert cli_main.run_pipeline(cfg) == 0
    assert feature_engineer_fps == [5.0]

    assert calls == [
        "preprocess:init",
        "court:detect",
        "pose:init",
        "preprocess:srcshape",
        "pose:iter",
        "preprocess:iter",
        "features:init",
        "inference:onnx",
        "features:iter",
        "output:csv",
        "output:video",
    ]


def test_unsupported_feature_set_fails_before_pose_extraction(tmp_path, monkeypatch):
    # A v0 (or any non-v1) manifest must fail fast, before the expensive pose extraction runs.
    cli_main = import_cli_main_with_stubs(monkeypatch)
    video_path = tmp_path / "match.mp4"
    model_path = tmp_path / "model.onnx"
    scaler_path = tmp_path / "scaler.json"
    video_path.write_bytes(b"fake video")
    model_path.write_bytes(b"fake onnx")
    scaler_path.write_text("{}", encoding="utf-8")

    calls: list[str] = []

    class ExplodingPoseExtractor:
        def __init__(self, *args, **kwargs):
            calls.append("pose:init")

        def iter_pose_frames(self, *args, **kwargs):
            calls.append("pose:iter")
            raise AssertionError("pose extraction must not run for an unsupported feature_set")

    monkeypatch.setattr(cli_main, "PoseExtractor", ExplodingPoseExtractor)
    # Stub the input preflight (fake video bytes); we're asserting the feature_set
    # guard fires before pose extraction, not video validation.
    monkeypatch.setattr(cli_main, "validate_video", lambda *a, **k: None)

    cfg = cli_main.RunConfig(
        video_path=video_path,
        output_dir=tmp_path / "output_videos",
        output_name=None,
        csv_output_dir=tmp_path / "output_csvs",
        write_csv=False,
        segment_video=False,
        yolo_weights="yolov8n-pose.pt",
        yolo_device=None,
        model_path=model_path,
        scaler_path=scaler_path,
        fps=5.0,
        seq_len=100,
        overlap=50,
        sigma=1.0,
        low=0.45,
        high=0.7,
        min_dur_sec=1.0,
        conf=0.25,
        imgsz=960,
        feature_set="v0",
        screen_width=1280,
        screen_height=720,
        start_time=0,
        duration=999999,
    )

    with pytest.raises(SystemExit) as ei:
        cli_main.run_pipeline(cfg)

    assert "feature_set='v0'" in str(ei.value)
    assert calls == []  # never reached pose extraction


def test_run_pipeline_writes_no_intermediate_npz(tmp_path, monkeypatch):
    # Guard (A6): the streaming release path must never serialize an intermediate .npz. Pinned
    # two ways: (1) the file-writing wrappers (extract_pose_data / preprocess_single_video /
    # create_features_from_preprocessed) are never called and np.savez_compressed is never
    # invoked from the release module; (2) no .npz file (and no persistent pose_data/ dir) is
    # left on disk under a sandboxed cwd.
    cli_main = import_cli_main_with_stubs(monkeypatch)
    video_path = tmp_path / "match.mp4"
    model_path = tmp_path / "model.onnx"
    scaler_path = tmp_path / "scaler.json"
    video_path.write_bytes(b"fake video")
    model_path.write_bytes(b"fake onnx")
    scaler_path.write_text("{}", encoding="utf-8")

    class FakePoseExtractor:
        def __init__(self, model_path, model_dir):
            pass

        def iter_pose_frames(self, **kwargs):
            return iter(["POSE_STREAM"])

        def extract_pose_data(self, *a, **k):
            raise AssertionError("release path must not call extract_pose_data (writes NPZ)")

    class FakePreprocessor:
        def __init__(self, **_kwargs):
            pass

        def compute_court_mask(self, vp):
            return "COURT_MASK", {"source": "default", "detected": False, "timestamp_s": None}

        def _source_frame_shape(self, vp):
            return (720, 1280, 3)

        def iter_preprocess_frames(self, pose_stream, court_mask, src_width, src_height):
            list(pose_stream)
            return iter(["PRE_STREAM"])

        def preprocess_single_video(self, *a, **k):
            raise AssertionError("release path must not call preprocess_single_video (writes NPZ)")

    class FakeFeatureEngineer:
        def __init__(self, **_kwargs):
            pass

        def iter_build_features(self, preprocessed_stream):
            list(preprocessed_stream)
            for _ in range(120):
                yield np.ones(FEATURE_DIM, dtype=np.float32), 0

        def create_features_from_preprocessed(self, *a, **k):
            raise AssertionError("release path must not call create_features_from_preprocessed (writes NPZ)")

    class FakeScaler:
        def transform(self, features):
            return features

    def no_savez(*a, **k):
        raise AssertionError("release path must not np.savez_compressed an intermediate")

    def fake_onnx_stream(model_path_arg, feature_rows, sequence_length, overlap, **kwargs):
        n = len(list(feature_rows))
        return np.zeros(n, dtype=np.float32)  # no segments

    monkeypatch.setattr(cli_main, "PoseExtractor", FakePoseExtractor)
    monkeypatch.setattr(cli_main, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(cli_main, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(cli_main.np, "savez_compressed", no_savez)
    monkeypatch.setattr(cli_main, "load_scaler_asset", lambda _path: FakeScaler())
    monkeypatch.setattr(cli_main, "run_windowed_inference_average_onnx_stream", fake_onnx_stream)
    monkeypatch.setattr(cli_main, "gaussian_filter1d", lambda values, sigma: values)
    monkeypatch.setattr(cli_main, "validate_video", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)

    cfg = cli_main.RunConfig(
        video_path=video_path,
        output_dir=tmp_path / "out",
        output_name=None,
        csv_output_dir=tmp_path / "out_csv",
        write_csv=False,
        segment_video=False,
        yolo_weights="yolov8n-pose.pt",
        yolo_device=None,
        model_path=model_path,
        scaler_path=scaler_path,
        fps=5.0,
        seq_len=100,
        overlap=50,
        sigma=1.0,
        low=0.45,
        high=0.7,
        min_dur_sec=1.0,
        conf=0.25,
        imgsz=960,
        feature_set="v1",
        screen_width=1280,
        screen_height=720,
        start_time=0,
        duration=999999,
    )

    assert cli_main.run_pipeline(cfg) == 0
    # No intermediate .npz anywhere under the sandbox, and no persistent pose_data/ dir.
    assert list(tmp_path.rglob("*.npz")) == []
    assert not (tmp_path / "pose_data").exists()
