from __future__ import annotations

import numpy as np
import pytest
import torch

from infer.inference import (
    apply_postprocess,
    apply_standard_scaler_json,
    run_windowed_inference_average_onnx,
)
from training.models.lstm import TennisPointLSTM


def test_lstm_output_shape_logits():
    model = TennisPointLSTM(input_size=10, return_logits=True)
    x = torch.randn(2, 5, 10)
    out = model(x)
    assert out.shape == (2, 5)


def test_lstm_output_sigmoid():
    model = TennisPointLSTM(input_size=10, return_logits=False)
    x = torch.randn(2, 5, 10)
    out = model(x)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_apply_standard_scaler_json_scales_and_guards_tiny_scale():
    features = np.array([[2.0, 9.0], [4.0, 9.0]], dtype=np.float32)
    scaler = {
        "feature_dim": 2,
        "mean": [0.0, 9.0],
        "scale": [2.0, 0.0],
    }

    scaled = apply_standard_scaler_json(features, scaler)

    np.testing.assert_allclose(scaled[:, 0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(scaled[:, 1], np.array([0.0, 0.0], dtype=np.float32))


def test_apply_standard_scaler_json_rejects_feature_dim_mismatch():
    features = np.array([[1.0, 2.0]], dtype=np.float32)
    scaler = {"feature_dim": 3, "mean": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}

    with pytest.raises(ValueError, match="feature dimension"):
        apply_standard_scaler_json(features, scaler)


def test_run_windowed_inference_average_onnx_uses_batch_size_one_and_averages_overlap():
    features = np.arange(14, dtype=np.float32).reshape(7, 2)
    outputs = [
        np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32),
        np.array([[3.0, 3.0, 3.0, 3.0]], dtype=np.float32),
    ]
    seen_shapes = []

    class FakeInput:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class FakeSession:
        def get_inputs(self):
            return [FakeInput("features", [1, 4, 2])]

        def run(self, _outputs, feeds):
            tensor = feeds["features"]
            seen_shapes.append((tensor.shape, tensor.dtype))
            return [outputs.pop(0)]

    avg = run_windowed_inference_average_onnx(
        FakeSession(),
        features,
        sequence_length=4,
        overlap=2,
    )

    np.testing.assert_allclose(avg, np.array([1.0, 1.0, 1.5, 2.0, 2.5, 2.5, 3.0], dtype=np.float32))
    assert seen_shapes == [((1, 4, 2), np.float32), ((1, 4, 2), np.float32), ((1, 4, 2), np.float32)]


def test_run_windowed_inference_average_onnx_rejects_feature_dim_mismatch():
    features = np.ones((4, 2), dtype=np.float32)

    class FakeInput:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class FakeSession:
        def get_inputs(self):
            return [FakeInput("features", [1, 4, 3])]

    with pytest.raises(ValueError, match="feature dimension"):
        run_windowed_inference_average_onnx(FakeSession(), features, sequence_length=4, overlap=2)


def test_apply_postprocess_dispatches_hysteresis():
    probs = np.array([0.1, 0.8, 0.9, 0.2], dtype=np.float32)

    pred = apply_postprocess(
        probs,
        method="hysteresis",
        params={"low": 0.4, "high": 0.7, "sigma": 0.0, "min_dur_sec": 0.0},
        fps=5.0,
    )

    np.testing.assert_array_equal(pred, np.array([0, 1, 1, 0], dtype=np.int32))


def test_apply_postprocess_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown postprocess method"):
        apply_postprocess(np.array([0.1, 0.2], dtype=np.float32), method="temporal_cnn", params={}, fps=5.0)
