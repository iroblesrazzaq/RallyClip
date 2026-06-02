from __future__ import annotations

import argparse
import sys

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


def test_quick_run_uses_manifest_defaults_for_bundled_onnx(tmp_path, monkeypatch):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)

    cfg = cli_main.build_run_config(_asset_args(tmp_path))

    assert cfg.fps == FPS
    assert cfg.seq_len == 100
    assert cfg.overlap == 50
    assert cfg.high == 0.7
    assert cfg.low == 0.45
    assert cfg.sigma == 1.0
    assert cfg.min_dur_sec == 1.0


def test_explicit_config_can_override_manifest_defaults(tmp_path, monkeypatch):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        """
[run]
fps = 15.0
seq_len = 300
overlap = 150
high = 0.8
sigma = 1.5
min_dur_sec = 0.5
""",
        encoding="utf-8",
    )

    cfg = cli_main.build_run_config(_asset_args(tmp_path, config=str(config_path)))

    assert cfg.fps == 15.0
    assert cfg.seq_len == 300
    assert cfg.overlap == 150
    assert cfg.high == 0.8
    assert cfg.sigma == 1.5
    assert cfg.min_dur_sec == 0.5


def test_incidental_repo_config_toml_does_not_break_quick_run(tmp_path, monkeypatch):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        """
[run]
video_path = "stale.mp4"
fps = 15.0
seq_len = 300
overlap = 150
high = 0.8
sigma = 1.5
min_dur_sec = 0.5
""",
        encoding="utf-8",
    )

    cfg = cli_main.build_run_config(_asset_args(tmp_path))

    assert cfg.fps == FPS
    assert cfg.seq_len == 100
    assert cfg.overlap == 50
    assert cfg.high == 0.7
    assert cfg.sigma == 1.0
    assert cfg.min_dur_sec == 1.0


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
