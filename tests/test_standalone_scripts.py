"""The standalone data scripts read base.yaml the same way the pipeline does."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from training.io.config import court_target_time, load_config
from training.paths import pose_preprocessed_dir, raw_videos_dir

_ROOT = Path(__file__).resolve().parents[1]
_BASE = _ROOT / "configs" / "train" / "base.yaml"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    # base.yaml's yolo.model is repo-relative, as when the scripts run from a checkout.
    monkeypatch.chdir(_ROOT)
    override = tmp_path / "override.yaml"
    override.write_text(f"data_root: {tmp_path}\n")
    video = raw_videos_dir(tmp_path) / "m.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    return f"{_BASE},{override}"


@pytest.mark.parametrize("name, step", [
    ("extract_yolo", "extract"),
    ("preprocess_yolo", "preprocess"),
    ("build_features", "features"),
    ("build_dataset", "dataset"),
])
def test_step_scripts_run_the_pipeline_step(name, step, config_path, monkeypatch):
    pytest.importorskip("torch")
    from training import pipeline

    seen = []
    monkeypatch.setattr(pipeline, "run_pipeline", lambda config, steps_override: seen.append((config, steps_override)))
    assert _load_script(name).main(["--config", config_path, "--mode", "all", "--video", "m.mp4"]) == 0

    (config, steps), = seen
    assert steps == [step]
    assert config["videos"] == ["m.mp4"]
    assert config[step]["mode"] == "all"


def test_build_features_reads_and_writes_the_pipeline_paths(tmp_path, config_path, monkeypatch):
    pytest.importorskip("torch")
    from training import pipeline
    from training.artifact_paths import feature_root_from_config, yolo_conf_imgsz_tags

    config = load_config(config_path)
    model_tag, conf_tag, imgsz = yolo_conf_imgsz_tags(config)
    fps = float(config["preprocess"]["target_fps"])
    preproc = (
        pose_preprocessed_dir(tmp_path) / f"yolo={model_tag}" / f"conf={conf_tag}"
        / f"imgsz={imgsz}" / f"fps={fps}" / f"m__fps{fps}.h5"
    )
    preproc.parent.mkdir(parents=True)
    preproc.write_bytes(b"")

    built = []

    class FakeBuilder:
        def __init__(self, cfg):
            pass

        def build(self, src, dst, overwrite=False):
            built.append((src, dst))
            return dst

    monkeypatch.setattr(pipeline, "FeatureBuilder", FakeBuilder)
    assert _load_script("build_features").main(["--config", config_path, "--video", "m.mp4"]) == 0
    assert built == [(preproc, feature_root_from_config(tmp_path, config) / "m__features__v1.h5")]


def test_cache_court_masks_keeps_the_midpoint_default(config_path, monkeypatch):
    script = _load_script("cache_court_masks")
    seen = {}

    class FakeCache:
        def __init__(self, model_path, target_time):
            seen["target_time"] = target_time

        def get_or_create(self, data_root, video_path, force=False):
            return type("R", (), {"success": True})()

    monkeypatch.setattr(script, "CourtMaskCache", FakeCache)
    monkeypatch.setattr(script, "resolve_court_model_path", lambda config: "court.onnx")
    monkeypatch.setattr("sys.argv", ["cache_court_masks", "--config", config_path, "--video", "m.mp4"])
    assert script.main() == 0
    assert seen["target_time"] is None


@pytest.mark.parametrize("value, expected", [(None, None), (60, 60), ("30", 30)])
def test_court_target_time(value, expected):
    assert court_target_time({"target_time": value}) == expected


@pytest.mark.parametrize("overwrite", [False, True])
def test_normalize_script_refuses_to_replace_a_video_with_other_settings(tmp_path, monkeypatch, overwrite):
    script = _load_script("normalize_videos")
    src = tmp_path / "src" / "m.mp4"
    src.parent.mkdir()
    src.write_bytes(b"")
    dst = raw_videos_dir(tmp_path) / "m.mp4"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"old")
    encoded = []
    monkeypatch.setattr(script, "is_canonical", lambda path, cfg: False)
    monkeypatch.setattr(script, "normalize_video", lambda s, d, cfg: encoded.append(d))
    argv = ["normalize_videos", "--data-root", str(tmp_path), "--inputs", str(src), "--fps", "10"]
    monkeypatch.setattr("sys.argv", argv + (["--overwrite"] if overwrite else []))

    with pytest.raises(SystemExit, match="overwrite_all"):
        script.main()
    assert encoded == []
    assert dst.read_bytes() == b"old"
