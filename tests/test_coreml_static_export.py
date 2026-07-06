"""Golden verification for the CoreML-friendly static pose export.

The bundle ships two pose ONNX files: the dynamic-axes export (CPU EP, the
byte-parity default) and a static 544x960 re-export of the same weights —
dynamic axes block the Apple Neural Engine, so the opt-in "coreml" device
runs the static file instead (docs/perf/coreml-spike/: ~8x pose throughput).
These tests lock in that the static export is the same model: on the CPU EP
it must match the bundled dynamic ONNX on the committed fixture clip within
the spike's confident-detection tolerance (1.2e-4), and the exact-canvas
letterbox it requires must be pixel-identical to the shipping stride-32
letterbox for 16:9 sources.

Measured on Apple M-series (2026-07-05, fixture clip, 40 frames): CPU dynamic
18.9 fps vs CoreML static 124.1 fps (6.6x end-to-end predict); max divergence
box 1.3e-4 px, keypoints 2.5e-4 px, confidence 8.4e-7.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from extraction.yolo_onnx_runner import YOLO as OnnxYOLO
from extraction.yolo_onnx_runner import letterbox, letterbox_exact

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIP = REPO_ROOT / "tests" / "fixtures" / "golden_cli" / "clip.mp4"
ARTIFACT_DIR = REPO_ROOT / "models" / "rallyclip_v0.3.1"
DYNAMIC_ONNX = ARTIFACT_DIR / "yolov8n-pose-960-dynamic.onnx"
STATIC_ONNX = ARTIFACT_DIR / "yolov8n-pose-544x960-static.onnx"

# The spike's acceptance bound for confident-detection divergence; the static
# export on CPU actually reproduces the dynamic one bit-exactly on the fixture,
# so this is generous headroom, not an expected error.
TOLERANCE = 1.2e-4

# Only the tests that load the bundled exports need them; the letterbox math
# and sibling-resolution tests are pure functions and must run everywhere.
requires_bundled_exports = pytest.mark.skipif(
    not (DYNAMIC_ONNX.is_file() and STATIC_ONNX.is_file()),
    reason="bundled pose ONNX exports not present",
)


def _rng_frame(height: int, width: int) -> np.ndarray:
    return np.random.default_rng(7).integers(0, 255, (height, width, 3), dtype=np.uint8)


def test_letterbox_exact_is_pixel_identical_for_16x9_at_960():
    img = _rng_frame(720, 1280)
    dyn_tensor, dyn_ratio, dyn_pad = letterbox(img, 960)
    sta_tensor, sta_ratio, sta_pad = letterbox_exact(img, (544, 960))
    assert sta_tensor.shape == (1, 3, 544, 960)
    assert np.array_equal(dyn_tensor, sta_tensor)
    assert dyn_ratio == sta_ratio and dyn_pad == sta_pad


def test_letterbox_exact_pads_any_aspect_to_the_static_canvas():
    for height, width in [(480, 640), (1080, 1440), (960, 544), (2160, 3840)]:
        tensor, ratio, (pad_left, pad_top) = letterbox_exact(_rng_frame(height, width), (544, 960))
        assert tensor.shape == (1, 3, 544, 960), (height, width)
        # The scaled content must fit the canvas with the reported pad.
        assert round(height * ratio) + 2 * pad_top <= 544 + 1
        assert round(width * ratio) + 2 * pad_left <= 960 + 1


def _fixture_frames(count: int = 6, step: int = 120) -> list[np.ndarray]:
    av = pytest.importorskip("av")
    frames = []
    with av.open(str(CLIP)) as container:
        for i, frame in enumerate(container.decode(container.streams.video[0])):
            if i % step == 0:
                frames.append(frame.to_ndarray(format="bgr24"))
            if len(frames) >= count:
                break
    return frames


@requires_bundled_exports
def test_static_export_matches_dynamic_onnx_on_fixture_clip():
    """Golden: static 544x960 export == bundled dynamic ONNX on the CPU EP."""
    if not CLIP.is_file():
        pytest.skip("golden fixture clip not present")
    dynamic = OnnxYOLO(str(DYNAMIC_ONNX))
    static = OnnxYOLO(str(STATIC_ONNX))
    assert dynamic._static_hw is None
    assert static._static_hw == (544, 960)

    detections = 0
    for frame in _fixture_frames():
        res_dyn = dynamic.predict(frame, conf=0.25, imgsz=960)[0]
        res_sta = static.predict(frame, conf=0.25, imgsz=960)[0]
        boxes_dyn = res_dyn.boxes.xyxy.numpy()
        boxes_sta = res_sta.boxes.xyxy.numpy()
        assert boxes_dyn.shape == boxes_sta.shape
        if not boxes_dyn.size:
            continue
        detections += boxes_dyn.shape[0]
        np.testing.assert_allclose(boxes_dyn, boxes_sta, atol=TOLERANCE, rtol=0)
        np.testing.assert_allclose(
            res_dyn.boxes.conf.numpy(), res_sta.boxes.conf.numpy(), atol=TOLERANCE, rtol=0
        )
        np.testing.assert_allclose(
            res_dyn.keypoints.xy.numpy(), res_sta.keypoints.xy.numpy(), atol=TOLERANCE, rtol=0
        )
    assert detections > 0, "fixture produced no confident detections; golden proves nothing"


@requires_bundled_exports
def test_pose_extractor_coreml_falls_back_to_cpu_without_static_sibling(tmp_path, monkeypatch):
    """device='coreml' with no *-static.onnx next to the model degrades to CPU,
    and a POSE_DEVICE=coreml env is corrected so later extractors in the same
    process don't retry (and re-warn about) the unavailable path."""
    from extraction.pose_extractor import PoseExtractor

    shutil.copy(DYNAMIC_ONNX, tmp_path / DYNAMIC_ONNX.name)
    monkeypatch.setenv("POSE_DEVICE", "coreml")
    extractor = PoseExtractor(model_dir=str(tmp_path), model_path=DYNAMIC_ONNX.name)
    assert extractor.device == "cpu"
    assert extractor.model._static_hw is None
    import os

    assert os.environ["POSE_DEVICE"] == "cpu"


def test_static_onnx_sibling_ignores_bare_filenames():
    """A model path with no directory must not glob the CWD for static exports."""
    from extraction.pose_extractor import PoseExtractor

    assert PoseExtractor._static_onnx_sibling("yolov8n-pose-960-dynamic.onnx") is None
    assert PoseExtractor._static_onnx_sibling("x/y-static.onnx") == "x/y-static.onnx"


@requires_bundled_exports
def test_pose_extractor_coreml_uses_static_sibling_when_available():
    """On Apple silicon with the CoreML EP, device='coreml' loads the static export."""
    from runtime.device import coreml_pose_available

    if not coreml_pose_available():
        pytest.skip("CoreML EP not available on this machine")
    from extraction.pose_extractor import PoseExtractor

    extractor = PoseExtractor(
        model_dir=str(ARTIFACT_DIR),
        model_path=DYNAMIC_ONNX.name,
        device="coreml",
    )
    assert extractor.device == "coreml"
    assert extractor.model._static_hw == (544, 960)
