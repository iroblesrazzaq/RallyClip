"""Unit tests for the torch-free ONNX pose runner (extraction.yolo_onnx_runner).

Numerical parity against Ultralytics is validated separately (golden CLI
parity + the YOLO-ONNX parity harness); these tests lock in the contract
surface: the fail-loud output-shape check, the decode path, and the
per-box iteration API the court detector uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from extraction.yolo_onnx_runner import (
    UnsupportedOnnxOutputShapeError,
    _Boxes,
    decode_v8_pose,
)


def _decode(pred: np.ndarray):
    return decode_v8_pose(pred, ratio=1.0, pad=(0, 0), orig_hw=(720, 1280), conf_thr=0.25)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 300, 57),  # YOLO26 end-to-end export (NMS in graph)
        (1, 84, 8400),  # v8 detect head (80 classes, no keypoints)
        (2, 56, 100),  # batched output — runner runs per-frame, batch dim must be 1
        (56,),  # not a detection matrix at all
    ],
)
def test_non_pose_output_shapes_raise_typed_error(shape):
    pred = np.zeros(shape, dtype=np.float32)
    with pytest.raises(UnsupportedOnnxOutputShapeError):
        _decode(pred)


def test_v8_pose_output_decodes():
    # One confident detection at (100..200, 100..300), keypoints at (150, 200).
    pred = np.zeros((1, 56, 2), dtype=np.float32)
    pred[0, :5, 0] = [150.0, 200.0, 100.0, 200.0, 0.9]
    pred[0, 5::3, 0] = 150.0
    pred[0, 6::3, 0] = 200.0
    pred[0, 7::3, 0] = 0.8
    pred[0, 4, 1] = 0.1  # below conf threshold — filtered out

    boxes, conf, kpt_xy, kpt_conf = _decode(pred)
    assert boxes.shape == (1, 4) and conf.shape == (1,)
    assert kpt_xy.shape == (1, 17, 2) and kpt_conf.shape == (1, 17)
    np.testing.assert_allclose(boxes[0], [100.0, 100.0, 200.0, 300.0])
    np.testing.assert_allclose(conf, [0.9])
    np.testing.assert_allclose(kpt_xy[0, 0], [150.0, 200.0])
    np.testing.assert_allclose(kpt_conf[0], np.full(17, 0.8, dtype=np.float32))


def test_boxes_iteration_matches_court_detector_consumption():
    xyxy = np.array([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    conf = np.array([0.9, 0.5], dtype=np.float32)
    boxes = _Boxes(xyxy, conf)

    assert len(boxes) == 2
    seen = []
    for box in boxes:  # exact consumption pattern from CourtDetector
        cls_id = int(box.cls.item())
        c = float(box.conf.item())
        flat = box.xyxy.cpu().numpy().reshape(-1)
        seen.append((cls_id, c, [int(v) for v in flat]))
    assert seen[0] == (0, pytest.approx(0.9), [10, 20, 30, 40])
    assert seen[1] == (0, pytest.approx(0.5), [1, 2, 3, 4])

    # Bulk access used by PoseExtractor.
    assert boxes.xyxy.detach().cpu().numpy().shape == (2, 4)
    assert boxes.conf.detach().cpu().numpy().shape == (2,)


def test_analysis_run_stays_torch_free():
    """Full analysis on the golden clip must never load torch/ultralytics."""
    import json
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    clip = repo_root / "tests" / "fixtures" / "golden_cli" / "clip.mp4"
    artifact = repo_root / "models" / "rallyclip_v0.3.1" / "yolov8n-pose-960-dynamic.onnx"
    if not clip.is_file() or not artifact.is_file():
        pytest.skip("golden clip or pose ONNX artifact absent")

    code = textwrap.dedent(
        """
        import json, sys, tempfile
        sys.argv = [
            "cli.main",
            "--video", %(clip)r,
            "--output-dir", tempfile.mkdtemp(),
            "--csv-output-dir", tempfile.mkdtemp(),
            "--output-name", "probe",
            "--write-csv", "--no-segment-video",
            "--yolo-device", "cpu",
        ]
        from cli.main import main
        rc = main()
        print(json.dumps({
            "rc": rc,
            "torch": "torch" in sys.modules,
            "ultralytics": "ultralytics" in sys.modules,
        }))
        """
        % {"clip": str(clip)}
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "src", "RALLYCLIP_NO_TQDM": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["rc"] in (0, None)
    assert payload["torch"] is False, "analysis run imported torch"
    assert payload["ultralytics"] is False, "analysis run imported ultralytics"
