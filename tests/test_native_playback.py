from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gui.native_player import (
    NativePlaybackScheduler,
    native_initial_media_for_descriptor,
    native_overlay_should_show,
    native_watchdog_reload_reason,
)


POINTS = [
    {"start": 10.0, "end": 15.0},
    {"start": 25.0, "end": 30.0},
    {"start": 40.0, "end": 45.0},
]


def test_native_scheduler_starts_at_first_point():
    scheduler = NativePlaybackScheduler(POINTS, 60.0)

    assert scheduler.default_start_ms() == 10_000
    segment = scheduler.seek(scheduler.default_start_ms())
    assert segment.mode == "point"
    assert segment.end_ms == 15_000
    assert scheduler.next_start_after_active() == 25_000


def test_native_scheduler_seek_inside_point():
    scheduler = NativePlaybackScheduler(POINTS, 60.0)
    segment = scheduler.seek(12_500)

    assert segment.mode == "point"
    assert segment.point_index == 0
    assert segment.start_ms == 12_500
    assert segment.end_ms == 15_000
    assert scheduler.next_start_after_active() == 25_000


def test_native_scheduler_seek_before_first_point_bridges_continuously():
    scheduler = NativePlaybackScheduler(POINTS, 60.0)
    segment = scheduler.seek(4_000)

    assert segment.mode == "gap_bridge"
    assert segment.start_ms == 4_000
    assert segment.end_ms == 15_000
    assert scheduler.should_advance(10_000) is False
    assert scheduler.next_start_after_active() == 25_000


def test_native_scheduler_seek_between_points_bridges_gap_and_next_point():
    scheduler = NativePlaybackScheduler(POINTS, 60.0)
    segment = scheduler.seek(20_000)

    assert segment.mode == "gap_bridge"
    assert segment.point_index == 1
    assert segment.start_ms == 20_000
    assert segment.end_ms == 30_000
    assert scheduler.next_start_after_active() == 40_000


def test_native_scheduler_gap_to_point_start_remains_continuous():
    scheduler = NativePlaybackScheduler(POINTS, 60.0)
    segment = scheduler.seek(24_900)

    assert segment.mode == "gap_bridge"
    assert segment.end_ms == 30_000
    assert scheduler.should_advance(25_000) is False
    assert scheduler.should_advance(29_950) is True


def test_native_scheduler_seek_after_last_point_plays_to_source_end():
    scheduler = NativePlaybackScheduler(POINTS, 60.0)
    segment = scheduler.seek(50_000)

    assert segment.mode == "tail"
    assert segment.end_ms == 60_000
    assert scheduler.next_start_after_active() is None


def test_native_scheduler_last_point_continues_into_tail_when_source_remains():
    scheduler = NativePlaybackScheduler(POINTS, 60.0)
    segment = scheduler.seek(42_000)

    assert segment.mode == "point"
    assert segment.end_ms == 45_000
    assert scheduler.next_start_after_active() is None
    assert scheduler.tail_start_after_active() == 45_000

    tail = scheduler.seek(scheduler.tail_start_after_active())
    assert tail.mode == "tail"
    assert tail.start_ms == 45_000
    assert tail.end_ms == 60_000


def test_native_scheduler_last_point_does_not_tail_when_source_ends_at_point():
    scheduler = NativePlaybackScheduler(POINTS, 45.0)
    scheduler.seek(42_000)

    assert scheduler.next_start_after_active() is None
    assert scheduler.tail_start_after_active() is None


def test_native_scheduler_large_forward_and_backward_seeks_are_absolute():
    scheduler = NativePlaybackScheduler(POINTS, 60.0)

    forward = scheduler.seek(42_000)
    backward = scheduler.seek(20_000)

    assert forward.point_index == 2
    assert forward.end_ms == 45_000
    assert backward.mode == "gap_bridge"
    assert backward.point_index == 1
    assert backward.end_ms == 30_000


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


def test_native_initial_media_prefers_ready_proxy():
    descriptor = {
        "source_path": "/tmp/source.mp4",
        "proxy": {"ready": True, "path": "/tmp/playback_proxy.mp4"},
    }

    assert native_initial_media_for_descriptor(descriptor) == ("proxy", "/tmp/playback_proxy.mp4")


def test_native_initial_media_uses_source_when_proxy_missing():
    descriptor = {
        "source_path": "/tmp/source.mp4",
        "proxy": {"ready": False, "path": None},
    }

    assert native_initial_media_for_descriptor(descriptor) == ("source", "/tmp/source.mp4")


def test_native_overlay_does_not_show_when_window_inactive():
    assert native_overlay_should_show(window_active=True) is True
    assert native_overlay_should_show(window_active=False) is False


def test_native_watchdog_detects_frame_stall_and_memory_growth():
    assert native_watchdog_reload_reason(
        playing=True,
        position_ms=8_000,
        last_position_ms=7_000,
        seconds_since_frame=5.5,
        rss_mb=450.0,
        last_rss_mb=440.0,
    ) == "video frames stopped while playback position advanced"

    assert native_watchdog_reload_reason(
        playing=True,
        position_ms=8_000,
        last_position_ms=8_000,
        seconds_since_frame=0.2,
        rss_mb=725.0,
        last_rss_mb=710.0,
    ) == "memory rose to 725.0 MB"

    assert native_watchdog_reload_reason(
        playing=False,
        position_ms=8_000,
        last_position_ms=7_000,
        seconds_since_frame=10.0,
        rss_mb=800.0,
        last_rss_mb=780.0,
    ) is None


def test_native_viewer_bridge_exposes_open_match(monkeypatch):
    pytest.importorskip("PySide6")
    if "gui.native_player" in sys.modules:
        native_player = sys.modules["gui.native_player"]
    else:
        native_player = pytest.importorskip("gui.native_player")
    if not getattr(native_player, "QT_AVAILABLE", False):
        pytest.skip("PySide6 is not available")

    opened = []
    def open_match(item_id):
        opened.append(item_id)
        return True

    bridge = native_player.NativeViewerBridge(open_match)

    assert bridge.openMatch("match-1") is True

    assert opened == ["match-1"]
