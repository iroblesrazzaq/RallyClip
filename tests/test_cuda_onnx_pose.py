"""CUDA EP wiring for the torch-free ONNX pose path (issue #42).

No NVIDIA hardware required: cuda_pose_available and OnnxYOLO are mocked.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")

from extraction.pose_extractor import PoseExtractor

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "models" / "rallyclip_v0.3.1"
DYNAMIC_ONNX = ARTIFACT_DIR / "yolov8n-pose-960-dynamic.onnx"

requires_bundled_exports = pytest.mark.skipif(
    not DYNAMIC_ONNX.is_file(),
    reason="bundled pose ONNX export not present",
)


@requires_bundled_exports
def test_resolve_onnx_session_uses_cuda_ep_when_available(tmp_path, monkeypatch):
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: True)
    shutil.copy(DYNAMIC_ONNX, tmp_path / DYNAMIC_ONNX.name)
    extractor = PoseExtractor.__new__(PoseExtractor)
    extractor.device = "cuda"
    path, providers = extractor._resolve_onnx_session(str(tmp_path / DYNAMIC_ONNX.name))
    assert path == str(tmp_path / DYNAMIC_ONNX.name)
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert extractor.device == "cuda"


@requires_bundled_exports
def test_resolve_onnx_session_cuda_degrades_without_ep(tmp_path, monkeypatch):
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: False)
    shutil.copy(DYNAMIC_ONNX, tmp_path / DYNAMIC_ONNX.name)
    extractor = PoseExtractor.__new__(PoseExtractor)
    extractor.device = "cuda"
    path, providers = extractor._resolve_onnx_session(str(tmp_path / DYNAMIC_ONNX.name))
    assert path == str(tmp_path / DYNAMIC_ONNX.name)
    assert providers is None
    assert extractor.device == "cpu"


@requires_bundled_exports
def test_pose_extractor_cuda_falls_back_when_ep_missing(tmp_path, monkeypatch):
    import os

    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: False)
    shutil.copy(DYNAMIC_ONNX, tmp_path / DYNAMIC_ONNX.name)
    monkeypatch.setenv("POSE_DEVICE", "cuda")
    extractor = PoseExtractor(
        model_dir=str(tmp_path),
        model_path=DYNAMIC_ONNX.name,
        device="cuda",
    )
    assert extractor.device == "cpu"
    assert extractor.batch_size == 1
    assert os.environ["POSE_DEVICE"] == "cpu"


@requires_bundled_exports
def test_pose_extractor_cuda_passes_providers_to_runner(tmp_path, monkeypatch):
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: True)
    shutil.copy(DYNAMIC_ONNX, tmp_path / DYNAMIC_ONNX.name)

    captured: dict = {}

    class FakeOnnxYOLO:
        def __init__(self, model_path, *, providers=None, intra_op_threads=None):
            captured["model_path"] = model_path
            captured["providers"] = providers
            self._static_hw = None

    monkeypatch.setattr("extraction.yolo_onnx_runner.YOLO", FakeOnnxYOLO)
    extractor = PoseExtractor(
        model_dir=str(tmp_path),
        model_path=DYNAMIC_ONNX.name,
        device="cuda",
    )
    assert extractor.device == "cuda"
    assert extractor.batch_size == 8
    assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert Path(captured["model_path"]).name == DYNAMIC_ONNX.name
