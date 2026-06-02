from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def import_cli_main_with_stubs(monkeypatch):
    _drop_modules("cli.main")

    infer = types.ModuleType("infer")
    infer.extract_segments_from_binary = _extract_segments_from_binary
    infer.gaussian_filter1d = lambda values, sigma: values
    infer.hysteresis_threshold = _hysteresis_threshold
    infer.load_scaler_asset = lambda path: None
    infer.load_model_from_checkpoint = lambda path, return_logits=False: (None, None)
    infer.run_windowed_inference_average_onnx = lambda *args, **kwargs: np.array([], dtype=np.float32)
    infer.run_windowed_inference_average = lambda *args, **kwargs: np.array([], dtype=np.float32)
    infer.write_segments_csv = lambda *args, **kwargs: None

    extraction_pose = types.ModuleType("extraction.pose_extractor")
    extraction_pose.PoseExtractor = object

    features_engineer = types.ModuleType("features.feature_engineer")
    features_engineer.FeatureEngineer = object

    preprocessing_data = types.ModuleType("preprocessing.data_preprocessor")
    preprocessing_data.DataPreprocessor = object

    segmentation_segment = types.ModuleType("segmentation.segment")
    segmentation_segment.segment_video = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "infer", infer)
    monkeypatch.setitem(sys.modules, "extraction.pose_extractor", extraction_pose)
    monkeypatch.setitem(sys.modules, "features.feature_engineer", features_engineer)
    monkeypatch.setitem(sys.modules, "preprocessing.data_preprocessor", preprocessing_data)
    monkeypatch.setitem(sys.modules, "segmentation.segment", segmentation_segment)

    return importlib.import_module("cli.main")


def import_data_preprocessor_with_stubs(monkeypatch):
    _drop_modules("preprocessing.data_preprocessor")
    court_detector = types.ModuleType("preprocessing.court_detector")

    class StubCourtDetector:
        def __init__(self, *args, **kwargs):
            pass

        def process_video(self, *args, **kwargs):
            return None, None, {}

    court_detector.CourtDetector = StubCourtDetector
    monkeypatch.setitem(sys.modules, "preprocessing.court_detector", court_detector)
    return importlib.import_module("preprocessing.data_preprocessor")


def import_pose_extractor_with_stubs(monkeypatch):
    _drop_modules("extraction.pose_extractor")

    av = types.ModuleType("av")
    av.open = lambda path: None

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = lambda *args, **kwargs: None
    ultralytics_utils = types.ModuleType("ultralytics.utils")
    ultralytics_utils.SETTINGS = {}

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", ultralytics_utils)
    monkeypatch.setitem(sys.modules, "torch", torch)

    return importlib.import_module("extraction.pose_extractor")


def _drop_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def _hysteresis_threshold(values: np.ndarray, low: float = 0.3, high: float = 0.7, min_duration: int = 0):
    pred = np.zeros(len(values), dtype=np.int32)
    active = False
    start = None
    for idx, value in enumerate(values):
        if not active and value >= high:
            active = True
            start = idx
        elif active and value < low:
            if start is not None and idx - start >= min_duration:
                pred[start:idx] = 1
            active = False
            start = None
    if active and start is not None and len(values) - start >= min_duration:
        pred[start:] = 1
    return pred


def _extract_segments_from_binary(pred: np.ndarray):
    segments = []
    in_segment = False
    start = None
    for idx, value in enumerate(pred):
        if value and not in_segment:
            in_segment = True
            start = idx
        elif not value and in_segment:
            segments.append((start, idx))
            in_segment = False
            start = None
    if in_segment:
        segments.append((start, len(pred)))
    return segments
