from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from extraction.pose_backend import (
    PoseBackendMeta,
    load_manifest,
    load_pose_backend,
)
from extraction.yolo_onnx_runner import (
    UnsupportedOnnxOutputShapeError,
    decode_pose,
    decode_yolo26_e2e,
)

ROOT = Path(__file__).resolve().parents[1]
V8_DIR = ROOT / "models" / "pose" / "yolov8n"


def _coreml_available() -> bool:
    import onnxruntime as ort

    return "CoreMLExecutionProvider" in ort.get_available_providers()


def _e2e_row(x1, y1, x2, y2, conf, kpt_xy=(0.0, 0.0), kpt_conf=0.9):
    row = [x1, y1, x2, y2, conf, 0.0]
    row += list(kpt_xy) + [kpt_conf]
    row += [0.0] * (3 * 16)
    return row


def test_decode_yolo26_e2e_filters_and_unletterboxes():
    # letterbox: ratio 0.5, pad (0, 2) — 1080p frame at imgsz 960 -> 544x960
    pred = np.array(
        [
            _e2e_row(100.0, 102.0, 200.0, 202.0, 0.9, kpt_xy=(150.0, 152.0)),
            _e2e_row(10.0, 10.0, 20.0, 20.0, 0.1),
        ],
        dtype=np.float32,
    )[None]
    boxes, conf, kpt_xy, kpt_conf = decode_yolo26_e2e(
        pred, ratio=0.5, pad=(0, 2), orig_hw=(1080, 1920), conf_thr=0.25
    )
    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(boxes[0], [200.0, 200.0, 400.0, 400.0])
    np.testing.assert_allclose(conf, [0.9])
    np.testing.assert_allclose(kpt_xy[0, 0], [300.0, 300.0])
    assert kpt_conf[0, 0] == pytest.approx(0.9)
    assert kpt_xy.shape == (1, 17, 2)


def test_decode_yolo26_e2e_conf_desc_and_max_det():
    rows = [_e2e_row(0, 0, 10, 10, c) for c in (0.3, 0.9, 0.6)]
    pred = np.array(rows, dtype=np.float32)
    _, conf, _, _ = decode_yolo26_e2e(
        pred, ratio=1.0, pad=(0, 0), orig_hw=(720, 1280), conf_thr=0.25, max_det=2
    )
    np.testing.assert_allclose(conf, [0.9, 0.6])


def test_decode_pose_dispatch():
    e2e = np.zeros((1, 300, 57), dtype=np.float32)
    out = decode_pose(e2e, ratio=1.0, pad=(0, 0), orig_hw=(720, 1280), conf_thr=0.25)
    assert out[0].shape == (0, 4)
    v8 = np.zeros((1, 56, 100), dtype=np.float32)
    out = decode_pose(v8, ratio=1.0, pad=(0, 0), orig_hw=(720, 1280), conf_thr=0.25)
    assert out[0].shape == (0, 4)
    with pytest.raises(UnsupportedOnnxOutputShapeError):
        decode_pose(
            np.zeros((1, 84, 100), dtype=np.float32),
            ratio=1.0, pad=(0, 0), orig_hw=(720, 1280), conf_thr=0.25,
        )


def test_load_manifest_real_bundle():
    meta = load_manifest(V8_DIR)
    assert isinstance(meta, PoseBackendMeta)
    assert meta.name == "yolov8n-960"
    assert meta.head_family == "v8-raw-head"
    assert meta.imgsz == 960
    assert meta.tag == f"yolov8n-960@{meta.model_sha256[:8]}"
    assert meta.model_path.exists()
    assert meta.static_model_path is not None and meta.static_model_path.exists()


def test_load_manifest_sha_mismatch_raises(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = json.loads((V8_DIR / "manifest.json").read_text())
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / manifest["model_file"]).write_bytes(b"tampered")
    (bundle / manifest["static_model_file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha mismatch"):
        load_manifest(bundle)


def test_load_manifest_rejects_unknown_head_family(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = json.loads((V8_DIR / "manifest.json").read_text())
    manifest["head_family"] = "segment"
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="head_family"):
        load_manifest(bundle)


def test_coreml_refused_for_e2e_head(tmp_path):
    # A yolo26-e2e manifest must be refused on provider=coreml (fp16 candidate
    # selection is not numerically faithful) before any model file is loaded.
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = json.loads((V8_DIR / "manifest.json").read_text())
    manifest["head_family"] = "yolo26-e2e"
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    shutil.copy(V8_DIR / manifest["model_file"], bundle / manifest["model_file"])
    shutil.copy(
        V8_DIR / manifest["static_model_file"], bundle / manifest["static_model_file"]
    )
    with pytest.raises(ValueError, match="CoreML provider refused"):
        load_pose_backend(bundle, provider="coreml")


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown pose provider"):
        load_pose_backend(V8_DIR, provider="cuda")


def test_yolo_conf_imgsz_tags_manifest():
    from training.artifact_paths import yolo_conf_imgsz_tags

    meta = load_manifest(V8_DIR)
    config = {"yolo": {"model": str(V8_DIR / "manifest.json"), "conf": 0.25}}
    model_tag, conf_tag, imgsz = yolo_conf_imgsz_tags(config)
    assert model_tag == meta.tag
    assert conf_tag == "0p25"
    assert imgsz == 960


def test_extractor_uses_onnx_backend_without_torch():
    from training.pose.yolo_hdf5 import YoloExtractConfig, YoloHdf5Extractor

    extractor = YoloHdf5Extractor(
        YoloExtractConfig(model_path=str(V8_DIR / "manifest.json"), conf=0.25)
    )
    assert extractor.backend_meta is not None
    assert extractor.device == "cpu"
    assert extractor.imgsz == 960
    assert extractor.model_identity == extractor.backend_meta.tag
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    results = extractor.model.predict(source=[frame], conf=0.25, imgsz=960, verbose=False)
    assert len(results) == 1
    assert results[0].boxes.xyxy.detach().cpu().numpy().shape[1] == 4


@pytest.mark.skipif(not _coreml_available(), reason="CoreML EP not available")
def test_coreml_provider_parity_with_cpu():
    """EP is provenance, not identity: coreml output must match cpu sub-pixel."""
    cpu_model, meta = load_pose_backend(V8_DIR, provider="cpu")
    cml_model, meta2 = load_pose_backend(V8_DIR, provider="coreml")
    assert meta.tag == meta2.tag  # identity independent of provider
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    a = cpu_model.predict(source=[frame], conf=0.25, imgsz=960, verbose=False)[0]
    b = cml_model.predict(source=[frame], conf=0.25, imgsz=960, verbose=False)[0]
    xa = a.boxes.xyxy.detach().cpu().numpy()
    xb = b.boxes.xyxy.detach().cpu().numpy()
    assert xa.shape[0] == xb.shape[0]
    if xa.shape[0]:
        assert np.abs(xa - xb).max() < 2.0
        ka = a.keypoints.xy.detach().cpu().numpy()
        kb = b.keypoints.xy.detach().cpu().numpy()
        assert np.abs(ka - kb).max() < 2.0
