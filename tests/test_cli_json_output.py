"""CLI --json output contract.

`rallyclip --json` prints run_result_payload on stdout so scripts and future
clients can consume analysis results without parsing human-oriented text.
"""

from __future__ import annotations

import json
from pathlib import Path

from helpers.module_stubs import import_cli_main_with_stubs


def _run_config(cli_main, tmp_path, *, emit_json: bool) -> "object":
    video = tmp_path / "in.mp4"
    video.write_bytes(b"\x00")
    return cli_main.RunConfig(
        video_path=video,
        output_dir=tmp_path,
        output_name="out",
        csv_output_dir=tmp_path,
        write_csv=False,
        segment_video=False,
        yolo_weights="yolov8n-pose.pt",
        yolo_device=None,
        model_path=tmp_path / "model.onnx",
        scaler_path=tmp_path / "scaler.json",
        fps=5.0,
        seq_len=100,
        imgsz=960,
        conf=0.25,
        feature_set="v1",
        screen_width=1280,
        screen_height=720,
        overlap=50,
        sigma=1.0,
        low=0.45,
        high=0.7,
        min_dur_sec=1.0,
        emit_json=emit_json,
    )


def _fake_services(cli_main, monkeypatch):
    from rallyclip_core.contracts import RunResult

    class FakeServices:
        def run_analysis(self, request, *, deps=None, **kwargs):
            return RunResult(
                frame_segments=[(29, 85)],
                intervals_sec=[(5.8, 17.0)],
                csv_path=Path("/out/out_segments.csv"),
                diagnostics={"pipeline_id": "frame_probability_hysteresis"},
            )

    monkeypatch.setattr(cli_main, "RallyClipServices", FakeServices)
    monkeypatch.setattr(cli_main, "validate_video", lambda *a, **k: None)


def test_json_flag_prints_run_result_payload(tmp_path, monkeypatch, capsys):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    _fake_services(cli_main, monkeypatch)
    assert cli_main.run_pipeline(_run_config(cli_main, tmp_path, emit_json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "pipeline_id": "frame_probability_hysteresis",
        "intervals": [{"start_s": 5.8, "end_s": 17.0}],
        "csv_path": "/out/out_segments.csv",
        "video_path": None,
        "n_segments": 1,
    }


def test_default_output_remains_human_readable(tmp_path, monkeypatch, capsys):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    _fake_services(cli_main, monkeypatch)
    assert cli_main.run_pipeline(_run_config(cli_main, tmp_path, emit_json=False)) == 0
    out = capsys.readouterr().out
    assert "Done. Outputs in" in out


def test_json_flag_parses_from_argv(tmp_path, monkeypatch):
    cli_main = import_cli_main_with_stubs(monkeypatch)
    captured = {}
    monkeypatch.setattr(cli_main, "build_run_config", lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(cli_main, "run_pipeline", lambda cfg: 0)
    monkeypatch.setattr(
        cli_main.sys, "argv", ["rallyclip", "--video", "x.mp4", "--json"]
    )
    assert cli_main.main() == 0
    assert captured["args"].emit_json is True
