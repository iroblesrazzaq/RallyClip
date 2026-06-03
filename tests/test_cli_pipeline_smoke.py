from __future__ import annotations

from pathlib import Path

import numpy as np

from helpers.module_stubs import import_cli_main_with_stubs
from helpers.runtime_fixtures import FEATURE_DIM


def test_run_pipeline_wires_runtime_stages_and_closes_feature_npz(tmp_path, monkeypatch):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    calls: list[str] = []
    opened_npz = []
    video_path = tmp_path / "match.mp4"
    model_path = tmp_path / "model.onnx"
    scaler_path = tmp_path / "scaler.json"
    video_path.write_bytes(b"fake video")
    model_path.write_bytes(b"fake onnx")
    scaler_path.write_text("{}", encoding="utf-8")

    class FakePoseExtractor:
        def __init__(self, model_path, model_dir):
            calls.append("pose:init")
            assert model_path == "yolov8n-pose.pt"
            assert Path(model_dir).name == "models"

        def extract_pose_data(self, **kwargs):
            calls.append("pose:extract")
            assert kwargs["target_fps"] == 5
            assert kwargs["imgsz"] == 960
            assert kwargs["confidence_threshold"] == 0.25
            return str(tmp_path / "raw_pose.npz")

    class FakePreprocessor:
        def __init__(self, save_court_masks, screen_width=None, screen_height=None, yolo_model_path=None, conf=None, **_kwargs):
            calls.append("preprocess:init")
            assert save_court_masks is False
            assert screen_width == 1280
            assert screen_height == 720
            # court detection uses the same manifest-pinned model + conf as pose extraction
            assert yolo_model_path == "yolov8n-pose.pt"
            assert conf == 0.25

        def preprocess_single_video(self, raw_npz, video, output_npz, overwrite):
            calls.append("preprocess:run")
            assert raw_npz.endswith("raw_pose.npz")
            assert video == str(video_path)
            assert overwrite is True
            return True

    feature_engineer_fps: list[float] = []

    class FakeFeatureEngineer:
        def __init__(self, target_fps=None, **_kwargs):
            calls.append("features:init")
            feature_engineer_fps.append(float(target_fps))

        def create_features_from_preprocessed(self, input_npz_path, output_file, overwrite):
            calls.append("features:run")
            assert input_npz_path.endswith("preprocessed.npz")
            assert output_file.endswith("features.npz")
            assert overwrite is True
            return True

    class TrackingNpz:
        def __init__(self):
            self.closed = False
            self.features = np.ones((100, FEATURE_DIM), dtype=np.float32)

        def __getitem__(self, key):
            assert key == "features"
            return self.features

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    class FakeScaler:
        def transform(self, features):
            calls.append("scaler:transform")
            assert features.shape == (100, FEATURE_DIM)
            return features

    def fake_np_load(path, *args, **kwargs):
        calls.append("features:load")
        assert str(path).endswith("features.npz")
        npz = TrackingNpz()
        opened_npz.append(npz)
        return npz

    def fake_onnx(model, features, sequence_length, overlap):
        calls.append("inference:onnx")
        assert model == str(model_path)
        assert features.shape == (100, FEATURE_DIM)
        assert sequence_length == 100
        assert overlap == 50
        probs = np.zeros(features.shape[0], dtype=np.float32)
        probs[10:25] = 0.95
        return probs

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
    monkeypatch.setattr(cli_main, "run_windowed_inference_average_onnx", fake_onnx)
    monkeypatch.setattr(cli_main, "gaussian_filter1d", lambda values, sigma: values)
    monkeypatch.setattr(cli_main, "write_segments_csv", fake_write_segments_csv)
    monkeypatch.setattr(cli_main, "segment_video", fake_segment_video)

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
        "pose:init",
        "pose:extract",
        "preprocess:init",
        "preprocess:run",
        "features:init",
        "features:run",
        "features:load",
        "scaler:transform",
        "inference:onnx",
        "output:csv",
        "output:video",
    ]
    assert opened_npz
    assert all(npz.closed for npz in opened_npz)
