"""Postprocess sweep: finding cached datasets and a run's feature root."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from training import pipeline  # noqa: E402
from training.paths import pose_features_dir, pose_raw_dir  # noqa: E402


def _raw(root, tag):
    d = pose_raw_dir(root) / f"yolo={tag}" / "conf=0p25" / "imgsz=960"
    d.mkdir(parents=True)
    (d / "m__start0__durfull.h5").write_bytes(b"")


def test_manifest_tag_dir_maps_back_to_the_configured_manifest(tmp_path, monkeypatch):
    _raw(tmp_path, "yolov8n-960@4a3fe0de")
    _raw(tmp_path, "yolov8n-960@deadbeef")
    _raw(tmp_path, "yolov8n-pose.pt")
    monkeypatch.setattr(pipeline, "yolo_conf_imgsz_tags", lambda cfg: ("yolov8n-960@4a3fe0de", "0p25", 960))
    config = {"yolo": {"model": "models/pose/yolov8n/manifest.json"}}

    specs = pipeline._discover_raw_datasets(tmp_path, {"m"}, True, config=config)
    assert sorted(s["yolo"]["model"] for s in specs) == ["models/pose/yolov8n/manifest.json", "yolov8n-pose.pt"]


def test_old_run_without_saved_yolo_uses_the_run_id_imgsz(tmp_path):
    root = pipeline._run_feature_root(tmp_path, tmp_path / "cv_yolon1440_f0", {}, 5.0)
    assert root == pose_features_dir(tmp_path) / "yolo=yolov8n-pose.pt" / "conf=0p25" / "imgsz=1440" / "fps=5.0"


def test_run_with_saved_yolo_uses_it(tmp_path):
    cfg = {"yolo": {"model": "yolov8n-pose.pt", "conf": 0.25, "imgsz": 960}, "preprocess": {"target_fps": 5.0}}
    root = pipeline._run_feature_root(tmp_path, tmp_path / "cv_yolon1440_f0", cfg, 5.0)
    assert "imgsz=960" in str(root)
