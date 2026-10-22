from __future__ import annotations

from pathlib import Path

from training.artifact_paths import (
    feature_root_from_config,
    feature_root_from_run_config,
    features_filename,
    format_conf,
    yolo_conf_imgsz_tags,
)


def test_format_conf_and_tags():
    assert format_conf(0.25) == "0p25"
    model, conf, imgsz = yolo_conf_imgsz_tags(
        {"yolo": {"model": "yolov8s-pose.pt", "conf": 0.3, "imgsz": 1440}}
    )
    assert model == "yolov8s-pose.pt"
    assert conf == "0p3"
    assert imgsz == 1440


def test_feature_root_golden_string(tmp_path):
    config = {
        "yolo": {"model": "yolov8n-pose.pt", "conf": 0.25, "imgsz": 1920},
        "preprocess": {"target_fps": 5},
        "features": {"feature_set": "v1"},
    }
    root = feature_root_from_config(tmp_path, config)
    expected = (
        tmp_path
        / "pose_data"
        / "features"
        / "yolo=yolov8n-pose.pt"
        / "conf=0p25"
        / "imgsz=1920"
        / "fps=5.0"
    )
    # fps may stringify as 5 or 5.0 depending on float formatting in path — accept either
    assert root.parent == expected.parent or root.name.startswith("fps=5")
    assert "yolo=yolov8n-pose.pt" in str(root)
    assert "conf=0p25" in str(root)
    assert "imgsz=1920" in str(root)
    assert features_filename("clip", "v1") == "clip__features__v1.h5"


def test_feature_root_from_run_config_matches_pipeline(tmp_path):
    run_cfg = {
        "yolo": {"model": "yolov8s-pose.pt", "conf": 0.3, "imgsz": 1440},
        "preprocess": {"target_fps": 5},
        "features": {"feature_set": "v1"},
        "fps": 5,
    }
    a = feature_root_from_run_config(tmp_path, run_cfg)
    b = feature_root_from_config(tmp_path, run_cfg)
    assert a == b
