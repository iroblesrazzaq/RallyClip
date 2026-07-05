"""Server-side playback proxy/descriptor contract (gui.app).

The Qt native player was removed with the pywebview shell migration — the
system webview plays H.264/HEVC directly, so playback is the frontend's
HTML5 path. These tests cover the Flask-side descriptor and ffmpeg proxy
machinery that remains behind /api/library/<id>/playback.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def test_native_playback_descriptor_shape(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    item_dir.mkdir(parents=True)
    (item_dir / "source.mp4").write_bytes(b"mp4")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,3.0\n5.0,7.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text('{"name": "Match 1", "source_name": "upload.mov"}', encoding="utf-8")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)
    monkeypatch.setattr(gui_app, "_estimate_duration_seconds", lambda path: 9.25)

    descriptor = gui_app.native_playback_descriptor("match-1")

    assert descriptor["id"] == "match-1"
    assert descriptor["name"] == "Match 1"
    assert descriptor["source_path"] == str(item_dir / "source.mp4")
    assert descriptor["source_duration_s"] == 9.25
    assert descriptor["point_intervals"] == [{"start": 1.0, "end": 3.0}, {"start": 5.0, "end": 7.0}]
    assert descriptor["has_csv"] is True
    assert descriptor["csv_url"] == "/api/library/match-1/csv"
    assert descriptor["export_url"] == "/api/library/match-1/video"
    assert descriptor["proxy"]["state"] == "missing"
    assert descriptor["proxy"]["path"] is None


def test_native_proxy_invalidates_when_source_changes(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    item_dir.mkdir(parents=True)
    source = item_dir / "source.mp4"
    proxy = item_dir / gui_app.NATIVE_PLAYBACK_PROXY_FILENAME
    source.write_bytes(b"source")
    proxy.write_bytes(b"proxy")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)

    os.utime(proxy, (100, 100))
    os.utime(source, (200, 200))
    assert gui_app._native_playback_proxy_state(source, proxy)["state"] == "missing"

    os.utime(proxy, (300, 300))
    assert gui_app._native_playback_proxy_state(source, proxy)["state"] == "ready"


def test_native_proxy_command_uses_source_time_preserving_mp4_settings(monkeypatch, tmp_path):
    from gui import app as gui_app

    monkeypatch.setattr(gui_app, "_ffmpeg_executable", lambda: "/usr/local/bin/ffmpeg")
    command = gui_app._native_playback_proxy_command(tmp_path / "source.mp4", tmp_path / "proxy.mp4")

    assert command[:6] == ["/usr/local/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i"]
    assert "-c:v" in command and command[command.index("-c:v") + 1] == "libx264"
    assert "-preset" in command and command[command.index("-preset") + 1] == "veryfast"
    assert "-crf" in command and command[command.index("-crf") + 1] == "23"
    assert "-g" in command and command[command.index("-g") + 1] == "30"
    assert "-movflags" in command and command[command.index("-movflags") + 1] == "+faststart"
    assert "-c:a" in command and command[command.index("-c:a") + 1] == "aac"
    assert "-b:a" in command and command[command.index("-b:a") + 1] == "96k"
    assert "fps=30" in command[command.index("-vf") + 1]
    assert "1280" in command[command.index("-vf") + 1]


def test_native_proxy_missing_ffmpeg_returns_clear_error(monkeypatch, tmp_path):
    from gui import app as gui_app

    monkeypatch.setattr(gui_app, "_ffmpeg_executable", lambda: None)
    with pytest.raises(RuntimeError, match="ffmpeg is not available"):
        gui_app._native_playback_proxy_command(tmp_path / "source.mp4", tmp_path / "proxy.mp4")


def test_native_proxy_generation_runs_command_and_replaces_output(monkeypatch, tmp_path):
    from gui import app as gui_app

    source = tmp_path / "source.mp4"
    proxy = tmp_path / "playback_proxy.mp4"
    source.write_bytes(b"source")
    calls = []

    monkeypatch.setattr(gui_app, "_ffmpeg_executable", lambda: "/usr/local/bin/ffmpeg")

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        Path(command[-1]).write_bytes(b"proxy")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gui_app.subprocess, "run", fake_run)

    assert gui_app._write_native_playback_proxy(source, proxy) == proxy
    assert proxy.read_bytes() == b"proxy"
    assert calls
    assert calls[0][-1].endswith(".tmp.mp4")


