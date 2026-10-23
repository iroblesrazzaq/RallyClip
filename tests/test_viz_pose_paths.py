"""The viz stages read pose HDF5s from the same branch the pipeline writes."""
from __future__ import annotations

from pathlib import Path

import pytest

from training.artifact_paths import pose_branch, target_fps_from_config, yolo_conf_imgsz_tags
from training.io.config import load_config
from training.viz.stages import _preproc_h5_path, _raw_h5_path

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(monkeypatch):
    # base.yaml's yolo.model is a repo-relative manifest.json.
    monkeypatch.chdir(_ROOT)
    return load_config(str(_ROOT / "configs" / "train" / "base.yaml"))


def test_manifest_model_paths_use_the_manifest_tag_and_imgsz(tmp_path, config):
    model_tag, conf_tag, imgsz = yolo_conf_imgsz_tags(config)
    assert "@" in model_tag  # name@sha8, not "manifest.json"
    video = Path("m.mp4")

    raw = _raw_h5_path(tmp_path, video, config["yolo"], {"start_time": 0, "duration": None})
    assert raw == pose_branch(
        tmp_path, kind="raw", model_tag=model_tag, conf_tag=conf_tag, imgsz=imgsz
    ) / "m__start0__durfull.h5"

    fps = target_fps_from_config(config)
    preproc = _preproc_h5_path(tmp_path, video, config["yolo"], config["preprocess"])
    assert preproc == pose_branch(
        tmp_path, kind="preprocessed", model_tag=model_tag, conf_tag=conf_tag, imgsz=imgsz, fps=fps
    ) / f"m__fps{fps}.h5"


def test_integer_target_fps_matches_the_pipeline_float_tag(tmp_path):
    preproc = _preproc_h5_path(tmp_path, Path("m.mp4"), {"model": "yolov8n-pose.pt"}, {"target_fps": 5})
    assert preproc.parent.name == "fps=5.0"
    assert preproc.name == "m__fps5.0.h5"
