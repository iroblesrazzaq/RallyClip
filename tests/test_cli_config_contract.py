from __future__ import annotations

import argparse
import sys

import pytest

from helpers.module_stubs import import_cli_main_with_stubs
from helpers.runtime_fixtures import FPS, write_manifest_model_dir


def _args(**overrides):
    defaults = {
        "config": None,
        "video": None,
        "output_dir": None,
        "output_name": None,
        "csv_output_dir": None,
        "model_path": None,
        "scaler_path": None,
        "artifact_dir": None,
        "manifest_path": None,
        "yolo_size": None,
        "yolo_device": None,
        "fps": None,
        "seq_len": None,
        "overlap": None,
        "sigma": None,
        "low": None,
        "high": None,
        "min_dur_sec": None,
        "conf": None,
        "imgsz": None,
        "start_time": None,
        "duration": None,
        "write_csv": None,
        "segment_video": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _asset_args(tmp_path, **overrides):
    model_dir = write_manifest_model_dir(tmp_path / "models" / "rallyclip_v0.3.1")
    video_path = tmp_path / "match.mp4"
    video_path.write_bytes(b"fake video")
    return _args(
        video=str(video_path),
        model_path=str(model_dir / "model.onnx"),
        scaler_path=str(model_dir / "scaler.json"),
        **overrides,
    )


def test_quick_run_uses_manifest_contract_for_bundled_onnx(tmp_path, monkeypatch):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)

    cfg = cli_main.build_run_config(_asset_args(tmp_path))

    # contract (immutable) comes from the manifest
    assert cfg.fps == FPS
    assert cfg.seq_len == 100
    assert cfg.imgsz == 960
    assert cfg.conf == 0.25
    assert cfg.feature_set == "v1"
    assert cfg.screen_width == 1280
    assert cfg.screen_height == 720
    assert cfg.yolo_weights == "yolov8n-pose.pt"
    # postprocess (mutable) defaults from the manifest
    assert cfg.overlap == 50
    assert cfg.high == 0.7
    assert cfg.low == 0.45
    assert cfg.sigma == 1.0
    assert cfg.min_dur_sec == 1.0


def test_config_overrides_mutable_but_never_contract(tmp_path, monkeypatch):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        """
[run]
# mutable postprocess — should win
sigma = 1.5
high = 0.8
min_dur_sec = 0.5
overlap = 150
# contract — must be ignored (manifest is authoritative)
fps = 15.0
seq_len = 300
imgsz = 1920
""",
        encoding="utf-8",
    )

    cfg = cli_main.build_run_config(_asset_args(tmp_path, config=str(config_path)))

    # mutable overrides applied
    assert cfg.sigma == 1.5
    assert cfg.high == 0.8
    assert cfg.min_dur_sec == 0.5
    assert cfg.overlap == 150
    # contract stays pinned to the manifest despite config trying to override
    assert cfg.fps == FPS
    assert cfg.seq_len == 100
    assert cfg.imgsz == 960


def test_incidental_repo_config_toml_cannot_corrupt_contract(tmp_path, monkeypatch):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        """
[run]
video_path = "stale.mp4"
fps = 15.0
seq_len = 300
imgsz = 1920
conf = 0.5
""",
        encoding="utf-8",
    )

    cfg = cli_main.build_run_config(_asset_args(tmp_path))

    assert cfg.fps == FPS
    assert cfg.seq_len == 100
    assert cfg.imgsz == 960
    assert cfg.conf == 0.25


def test_cli_flag_overrides_contract_with_warning(tmp_path, monkeypatch, caplog):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)

    with caplog.at_level("WARNING"):
        cfg = cli_main.build_run_config(_asset_args(tmp_path, fps=15.0))

    assert cfg.fps == 15.0  # explicit CLI override wins
    assert any("contract field 'fps'" in r.message for r in caplog.records)


def test_write_csv_from_config_honored_even_with_video(tmp_path, monkeypatch):
    # Regression: --video used to skip config.toml entirely, silently dropping write_csv.
    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        """
[run]
write_csv = true
csv_output_dir = "from_config"
""",
        encoding="utf-8",
    )

    cfg = cli_main.build_run_config(_asset_args(tmp_path))

    assert cfg.write_csv is True
    assert cfg.csv_output_dir.name == "from_config"


def test_missing_manifest_crashes_no_phantom_defaults(tmp_path, monkeypatch):
    # A model artifact without a manifest must fail loudly, not fall back to stale literals.
    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "models" / "bare"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"fake onnx")
    (model_dir / "scaler.json").write_text("{}", encoding="utf-8")
    video_path = tmp_path / "match.mp4"
    video_path.write_bytes(b"fake video")
    args = _args(
        video=str(video_path),
        model_path=str(model_dir / "model.onnx"),
        scaler_path=str(model_dir / "scaler.json"),
    )

    with pytest.raises(SystemExit):
        cli_main.build_run_config(args)


def test_readme_documented_artifact_flags_are_supported_by_argparse(monkeypatch, tmp_path):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rallyclip",
            "--video",
            str(tmp_path / "match.mp4"),
            "--artifact-dir",
            str(tmp_path / "models" / "rallyclip_v0.3.1"),
            "--manifest-path",
            str(tmp_path / "models" / "rallyclip_v0.3.1" / "manifest.json"),
            "--no-segment-video",
        ],
    )
    monkeypatch.setattr(cli_main, "build_run_config", lambda args: args)
    monkeypatch.setattr(cli_main, "run_pipeline", lambda cfg: 0)

    assert cli_main.main() == 0
