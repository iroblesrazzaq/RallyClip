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
        / "norm=1280x720@5fps"
        / "features"
        / "yolo=yolov8n-pose.pt"
        / "conf=0p25"
        / "imgsz=1920"
        / "fps=5.0"
    )
    # target_fps is coerced to float, so an int 5 still lands under fps=5.0.
    assert root == expected
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


def test_default_model_matches_extraction():
    # _run_extract writes yolo=yolov8s-pose.pt when the config names no model.
    assert yolo_conf_imgsz_tags({})[0] == "yolov8s-pose.pt"
    assert "yolo=yolov8s-pose.pt" in str(feature_root_from_run_config(Path("d"), {"yolo": {}}))
