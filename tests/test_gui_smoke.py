from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")

from helpers.runtime_fixtures import FEATURE_DIM
from helpers.runtime_fixtures import write_manifest_model_dir
from runtime.paths import resolve_frontend_dir


def test_resolve_frontend_dir_finds_repo_assets():
    frontend = resolve_frontend_dir()
    assert frontend.is_dir()
    assert (frontend / "index.html").exists()


def test_desktop_logo_asset_resolves():
    from gui.desktop import _resource_path

    logo = _resource_path("docs", "rallyclip_logo_cropped.png")
    assert logo is not None
    assert logo.is_file()


def test_gui_health_and_defaults(tmp_path, monkeypatch):
    model_dir = write_manifest_model_dir(tmp_path / "models" / "rallyclip_v0.3.1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RALLYCLIP_MODEL_PATH", str(model_dir / "model.onnx"))
    monkeypatch.setenv("RALLYCLIP_SCALER_PATH", str(model_dir / "scaler.json"))

    from gui import app as gui_app

    monkeypatch.setattr(gui_app, "DEFAULT_CONFIG", gui_app._load_default_config())
    client = gui_app.app.test_client()

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"

    defaults = client.get("/api/config/defaults")
    assert defaults.status_code == 200
    payload = defaults.get_json()
    assert payload["defaults"]["fps"] == 5.0
    assert payload["defaults"]["feature_set"] == "v1"
    assert payload["defaults"]["yolo_size"] == "nano"
    assert payload["yolo_model"] == "yolov8n-pose.pt"
    assert "available_devices" in payload
    assert "auto_device" in payload
    assert "output_dir" not in payload["defaults"]
    assert "csv_output_dir" not in payload["defaults"]


def test_update_version_comparison():
    from gui import app as gui_app

    assert gui_app.is_newer_version("v0.1.1", "0.1.0") is True
    assert gui_app.is_newer_version("0.2.0", "0.10.0") is False
    assert gui_app.is_newer_version("v1.0", "1.0.0") is False


def test_update_status_endpoint_reports_available(monkeypatch):
    from gui import app as gui_app

    monkeypatch.setattr(gui_app, "current_app_version", lambda: "0.1.0")
    monkeypatch.setattr(
        gui_app,
        "_fetch_latest_release",
        lambda: {
            "latest_version": "0.1.1",
            "latest_tag": "v0.1.1",
            "release_url": "https://github.com/iroblesrazzaq/RallyClip/releases/tag/v0.1.1",
            "release_name": "v0.1.1",
        },
    )
    gui_app._UPDATE_STATUS_CACHE.update({"checked_at": 0.0, "payload": None})

    client = gui_app.app.test_client()
    response = client.get("/api/update/status?force=1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["current_version"] == "0.1.0"
    assert payload["latest_version"] == "0.1.1"
    assert payload["update_available"] is True
    assert payload["error"] is None


def test_update_status_endpoint_tolerates_fetch_errors(monkeypatch):
    from gui import app as gui_app

    monkeypatch.setattr(gui_app, "current_app_version", lambda: "0.1.0")

    def fail_fetch():
        raise OSError("offline")

    monkeypatch.setattr(gui_app, "_fetch_latest_release", fail_fetch)
    gui_app._UPDATE_STATUS_CACHE.update({"checked_at": 0.0, "payload": None})

    client = gui_app.app.test_client()
    response = client.get("/api/update/status?force=1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["current_version"] == "0.1.0"
    assert payload["latest_version"] is None
    assert payload["update_available"] is False
    assert "offline" in payload["error"]


def test_update_open_endpoint_opens_releases_page(monkeypatch):
    from gui import app as gui_app

    opened = []
    monkeypatch.setattr(gui_app.webbrowser, "open", lambda url: opened.append(url))

    client = gui_app.app.test_client()
    response = client.post("/api/update/open")

    assert response.status_code == 200
    assert response.get_json()["opened"] is True
    assert opened == [gui_app.GITHUB_RELEASES_URL]


def test_gui_index_served(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    from gui import app as gui_app

    client = gui_app.app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"RallyClip" in response.data


def test_ensure_job_dir_rejects_path_traversal(tmp_path, monkeypatch):
    from gui import app as gui_app

    monkeypatch.setattr(gui_app, "JOBS_DIR", tmp_path / "jobs")
    gui_app.JOBS_DIR.mkdir(parents=True)

    safe = gui_app._ensure_job_dir("abc123")
    assert safe == (gui_app.JOBS_DIR / "abc123").resolve()

    with pytest.raises(ValueError):
        gui_app._ensure_job_dir("../../outside")


def test_default_config_excludes_ui_metadata():
    from gui import app as gui_app

    assert "available_devices" not in gui_app.DEFAULT_CONFIG
    assert "auto_device" not in gui_app.DEFAULT_CONFIG

    normalized = gui_app._normalize_config({})
    assert "available_devices" not in normalized
    assert "auto_device" not in normalized


def test_gui_config_forces_yolo_nano():
    from gui import app as gui_app

    normalized = gui_app._normalize_config({"yolo_size": "large", "yolo_weights": "custom.pt"})

    assert normalized["yolo_size"] == "nano"
    assert normalized["yolo_weights"] == "yolov8n-pose.pt"


def test_persist_library_item_saves_source_without_cutting(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    upload = tmp_path / "source.mp4"
    upload.write_bytes(b"fake video")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)

    def fake_write_segments_csv(segments, output_csv_path, fps, overwrite):
        Path(output_csv_path).write_text("start_time,end_time\n", encoding="utf-8")

    def fail_segment_video(input_video, intervals_sec, output_video):
        raise AssertionError("segment_video should only run during export")

    monkeypatch.setattr(gui_app, "write_segments_csv", fake_write_segments_csv)
    monkeypatch.setattr(gui_app, "segment_video", fail_segment_video)
    monkeypatch.setattr(gui_app, "_write_thumbnail", lambda video_path, thumb_path: True)
    monkeypatch.setattr(gui_app, "_start_preview_window_background", lambda item_id, source_path, start_s, duration_s: "processing")
    job = gui_app._new_job_state("job-1", gui_app._normalize_config({}))

    library_id, source_out, csv_out = gui_app._persist_library_item(
        upload_path=upload,
        base_name="Match",
        segments=[(0, 10)],
        intervals_sec=[(0.0, 2.0)],
        fps=5.0,
        job=job,
    )

    item_dir = library / library_id
    assert source_out == item_dir / "source.mp4"
    assert source_out.read_bytes() == b"fake video"
    assert csv_out == item_dir / "segments.csv"
    assert not (item_dir / "video.mp4").exists()
    assert not (item_dir / "export.mp4").exists()


def test_library_preview_streams_video_inline(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    item_dir.mkdir(parents=True)
    (item_dir / "source.mp4").write_bytes(b"fake mp4 bytes")
    (item_dir / "preview.webm").write_bytes(b"fake webm bytes")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,3.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text('{"name": "Match 1"}', encoding="utf-8")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)
    monkeypatch.setattr(gui_app, "_ensure_web_preview", lambda item_id, source_path: item_dir / "preview.webm")

    client = gui_app.app.test_client()
    response = client.get("/api/library/match-1/preview")

    assert response.status_code == 200
    assert response.mimetype == "video/webm"
    assert "attachment" not in response.headers.get("Content-Disposition", "")
    assert response.data == b"fake webm bytes"

    segments = client.get("/api/library/match-1/segments")
    assert segments.status_code == 200
    assert segments.get_json()["segments"] == [{"start": 1.0, "end": 3.0}]

    playback = client.get("/api/library/match-1/playback")
    assert playback.status_code == 200
    playback_payload = playback.get_json()
    assert playback_payload["chunk_duration_s"] == gui_app.PREVIEW_WINDOW_DURATION_S
    assert playback_payload["segments"] == [{"start": 1.0, "end": 3.0}]
    assert playback_payload["point_intervals"] == [{"start": 1.0, "end": 3.0}]
    assert playback_payload["point_duration_s"] == 2.0


def test_library_preview_supports_range_requests(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    item_dir.mkdir(parents=True)
    (item_dir / "source.mp4").write_bytes(b"fake mp4 bytes")
    (item_dir / "preview.webm").write_bytes(b"fake webm bytes")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,3.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text('{"name": "Match 1"}', encoding="utf-8")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)
    monkeypatch.setattr(gui_app, "_ensure_web_preview", lambda item_id, source_path: item_dir / "preview.webm")

    client = gui_app.app.test_client()
    response = client.get("/api/library/match-1/preview", headers={"Range": "bytes=0-3"})

    assert response.status_code == 206
    assert response.headers["Accept-Ranges"] == "bytes"
    assert response.headers["Content-Range"] == "bytes 0-3/15"
    assert response.data == b"fake"


def test_library_preview_status_starts_background_work(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    item_dir.mkdir(parents=True)
    (item_dir / "source.mp4").write_bytes(b"fake mp4 bytes")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,3.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text('{"name": "Match 1"}', encoding="utf-8")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)
    monkeypatch.setattr(gui_app, "_start_web_preview_background", lambda item_id, source_path: "processing")

    client = gui_app.app.test_client()
    response = client.get("/api/library/match-1/preview/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "processing"
    assert payload["ready"] is False
    assert payload["preview_url"] is None


def test_library_preview_returns_accepted_while_processing(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    item_dir.mkdir(parents=True)
    (item_dir / "source.mp4").write_bytes(b"fake mp4 bytes")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,3.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text('{"name": "Match 1"}', encoding="utf-8")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)
    monkeypatch.setattr(gui_app, "_start_web_preview_background", lambda item_id, source_path: "processing")

    client = gui_app.app.test_client()
    response = client.get("/api/library/match-1/preview")

    assert response.status_code == 202
    assert response.get_json()["status"] == "processing"


def test_library_preview_window_streams_video_inline(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    window_dir = item_dir / "preview_windows"
    window_dir.mkdir(parents=True)
    (item_dir / "source.mp4").write_bytes(b"fake mp4 bytes")
    (window_dir / "000000000000_008000.webm").write_bytes(b"fake window bytes")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,3.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text('{"name": "Match 1"}', encoding="utf-8")
    now = time.time()
    os.utime(item_dir / "source.mp4", (now - 10, now - 10))
    os.utime(item_dir / "segments.csv", (now - 10, now - 10))
    os.utime(window_dir / "000000000000_008000.webm", (now, now))
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)

    client = gui_app.app.test_client()
    response = client.get("/api/library/match-1/preview/window?start=1.0&duration=5.0")

    assert response.status_code == 200
    assert response.mimetype == "video/webm"
    assert response.data == b"fake window bytes"

    status = client.get("/api/library/match-1/preview/window/status?start=1.0&duration=5.0")
    assert status.status_code == 200
    payload = status.get_json()
    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["preview_url"] == "/api/library/match-1/preview/window?start=0.000&duration=8.000"


def test_library_preview_window_returns_accepted_while_processing(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    item_dir.mkdir(parents=True)
    (item_dir / "source.mp4").write_bytes(b"fake mp4 bytes")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,3.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text('{"name": "Match 1"}', encoding="utf-8")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)
    monkeypatch.setattr(gui_app, "_start_preview_window_background", lambda item_id, source_path, start_s, duration_s: "processing")

    client = gui_app.app.test_client()
    response = client.get("/api/library/match-1/preview/window?start=1.0&duration=5.0")

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["status"] == "processing"
    assert payload["ready"] is False
    assert payload["start"] == 0.0
    assert payload["duration"] == 8.0


def test_library_preview_window_status_returns_error_without_polling_forever(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    item_dir.mkdir(parents=True)
    (item_dir / "source.mp4").write_bytes(b"fake mp4 bytes")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,3.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text('{"name": "Match 1"}', encoding="utf-8")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)

    key = gui_app._preview_window_job_key("match-1", 0.0, 8.0)
    monkeypatch.setitem(gui_app.preview_jobs, key, "error")
    monkeypatch.setitem(gui_app.preview_job_errors, key, "Missing VP8 encoder")

    client = gui_app.app.test_client()
    response = client.get("/api/library/match-1/preview/window/status?start=1.0&duration=5.0")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "error"
    assert payload["ready"] is False
    assert payload["preview_url"] is None
    assert payload["error"] == "Missing VP8 encoder"


def test_preview_window_ready_invalidates_when_source_or_csv_changes(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    window_dir = item_dir / "preview_windows"
    window_dir.mkdir(parents=True)
    source = item_dir / "source.mp4"
    csv_path = item_dir / "segments.csv"
    preview = window_dir / "000000000000_008000.webm"
    source.write_bytes(b"source")
    csv_path.write_text("start_time,end_time\n1.0,3.0\n", encoding="utf-8")
    preview.write_bytes(b"preview")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)

    now = time.time()
    os.utime(source, (now - 30, now - 30))
    os.utime(csv_path, (now - 30, now - 30))
    os.utime(preview, (now - 10, now - 10))
    assert gui_app._preview_window_ready("match-1", source, 0.0, 8.0)

    os.utime(csv_path, (now, now))
    assert not gui_app._preview_window_ready("match-1", source, 0.0, 8.0)

    os.utime(preview, (now + 10, now + 10))
    assert gui_app._preview_window_ready("match-1", source, 0.0, 8.0)

    os.utime(source, (now + 20, now + 20))
    assert not gui_app._preview_window_ready("match-1", source, 0.0, 8.0)


def test_preview_window_values_use_aligned_playback_windows(monkeypatch):
    from gui import app as gui_app

    monkeypatch.setattr(gui_app, "_estimate_duration_seconds", lambda _path: 120.0)

    assert gui_app._preview_window_values(Path("source.mp4"), 7.9, None) == (0.0, 8.0)
    assert gui_app._preview_window_values(Path("source.mp4"), 8.0, 5.0) == (8.0, 8.0)
    assert gui_app._preview_window_values(Path("source.mp4"), 10.0, None) == (8.0, 8.0)
    assert gui_app._preview_window_values(Path("source.mp4"), 30.0, 15.0) == (24.0, 24.0)
    assert gui_app._preview_window_values(Path("source.mp4"), 119.9, None) == (112.0, 8.0)


def test_preview_cache_prunes_stale_files_and_inactive_temps(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    active = library / "match-1"
    inactive = library / "match-2"
    active_windows = active / "preview_windows"
    inactive_windows = inactive / "preview_windows"
    active_windows.mkdir(parents=True)
    inactive_windows.mkdir(parents=True)
    stale = active_windows / "000000000000_008000.webm"
    fresh = active_windows / "000000008000_008000.webm"
    inactive_chunk = inactive_windows / "000000000000_008000.webm"
    inactive_tmp = inactive_windows / "000000008000_008000.tmp.webm"
    inactive_preview = inactive / "preview.webm"
    for path in (stale, fresh, inactive_chunk, inactive_tmp, inactive_preview):
        path.write_bytes(b"cache")
    old = time.time() - gui_app.PREVIEW_CACHE_TTL_SECONDS - 10
    stale.touch()
    inactive_chunk.touch()
    inactive_preview.touch()
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)
    monkeypatch.setattr(gui_app, "active_preview_item_id", None)
    monkeypatch.setattr(gui_app, "last_preview_cache_prune", 0.0)
    monkeypatch.setitem(gui_app.preview_jobs, "match-2:window:0:8000", "ready")

    os.utime(stale, (old, old))
    gui_app._activate_preview_cache("match-1", force=True)

    assert not stale.exists()
    assert fresh.exists()
    assert inactive_chunk.exists()
    assert not inactive_tmp.exists()
    assert inactive_preview.exists()


def test_preview_cache_prunes_lru_to_active_and_global_caps(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    active_windows = library / "match-1" / "preview_windows"
    inactive_windows = library / "match-2" / "preview_windows"
    active_windows.mkdir(parents=True)
    inactive_windows.mkdir(parents=True)
    active_old = active_windows / "000000000000_008000.webm"
    active_new = active_windows / "000000008000_008000.webm"
    inactive_old = inactive_windows / "000000000000_008000.webm"
    inactive_new = inactive_windows / "000000008000_008000.webm"
    for path in (active_old, active_new, inactive_old, inactive_new):
        path.write_bytes(b"1234567890")

    now = time.time()
    os.utime(active_old, (now - 40, now - 40))
    os.utime(active_new, (now - 30, now - 30))
    os.utime(inactive_old, (now - 20, now - 20))
    os.utime(inactive_new, (now - 10, now - 10))
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)
    monkeypatch.setattr(gui_app, "PREVIEW_ACTIVE_CACHE_CAP_BYTES", 10)
    monkeypatch.setattr(gui_app, "PREVIEW_GLOBAL_CACHE_CAP_BYTES", 20)

    gui_app._prune_preview_cache("match-1", now=now)

    assert not active_old.exists()
    assert active_new.exists()
    assert not inactive_old.exists()
    assert inactive_new.exists()


def test_web_preview_generation_latency_benchmark(tmp_path):
    from gui import app as gui_app
    from test_segment import _make_clip

    av = pytest.importorskip("av")
    source = tmp_path / "source.mp4"
    preview = tmp_path / "preview.webm"
    try:
        _make_clip(source, seconds=8, fps=10, with_audio=True)
    except Exception as exc:
        pytest.skip(f"cannot encode benchmark clip: {exc}")

    started = time.perf_counter()
    gui_app._write_web_preview(source, preview, max_width=640)
    elapsed = time.perf_counter() - started

    assert preview.exists()
    assert preview.stat().st_size > 0
    with av.open(str(preview)) as container:
        streams = {stream.type for stream in container.streams}

    print(f"viewer_preview_benchmark elapsed_s={elapsed:.3f} input_s=8.000 realtime_factor={elapsed / 8.0:.3f}")
    assert "video" in streams
    assert elapsed < 20.0


def test_preview_window_generation_latency_benchmark(tmp_path, monkeypatch):
    from gui import app as gui_app
    from test_segment import _make_clip

    av = pytest.importorskip("av")
    library = tmp_path / "library"
    item_dir = library / "match-1"
    item_dir.mkdir(parents=True)
    source = item_dir / "source.mp4"
    try:
        _make_clip(source, seconds=20, fps=10, with_audio=True)
    except Exception as exc:
        pytest.skip(f"cannot encode benchmark clip: {exc}")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)
    start_s, duration_s = gui_app._preview_window_values(source, 4.0, 6.0)

    started = time.perf_counter()
    preview = gui_app._ensure_preview_window("match-1", source, start_s, duration_s)
    elapsed = time.perf_counter() - started

    assert preview.exists()
    assert preview.stat().st_size > 0
    with av.open(str(preview)) as container:
        streams = {stream.type for stream in container.streams}
        duration = (container.duration or 0) / av.time_base

    print(f"viewer_window_benchmark elapsed_s={elapsed:.3f} window_s={duration_s:.3f} realtime_factor={elapsed / duration_s:.3f}")
    assert "video" in streams
    assert duration <= duration_s + 1.0
    assert elapsed < 10.0


def test_gui_pipeline_streams_features_into_inference(tmp_path, monkeypatch):
    from gui import app as gui_app

    calls: list[str] = []
    progress_snapshots = []
    upload = tmp_path / "match.mp4"
    model_path = tmp_path / "model.onnx"
    scaler_path = tmp_path / "scaler.json"
    upload.write_bytes(b"fake video")
    model_path.write_bytes(b"fake onnx")
    scaler_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gui_app, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(gui_app, "_estimate_duration_seconds", lambda _path: 20.2)
    monkeypatch.setattr(gui_app, "_resolve_yolo_weights", lambda _cfg: "yolov8n-pose.pt")
    monkeypatch.setattr(gui_app, "_resolve_model_paths", lambda _cfg: (model_path, scaler_path))
    monkeypatch.setattr(gui_app, "candidate_roots", lambda: [tmp_path])
    monkeypatch.setattr(gui_app, "apply_pose_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(gui_app, "_persist_library_item", lambda **_kwargs: ("match-1", upload, tmp_path / "segments.csv"))

    class FakePreprocessor:
        def __init__(self, **_kwargs):
            calls.append("preprocess:init")

        def compute_court_mask(self, video_path):
            calls.append("court:detect")
            assert video_path == str(upload)
            return "COURT_MASK", {}

        def _source_frame_shape(self, video_path):
            calls.append("preprocess:srcshape")
            assert video_path == str(upload)
            return (720, 1280, 3)

        def iter_preprocess_frames(self, pose_stream, court_mask, src_width, src_height):
            calls.append("preprocess:iter")
            assert list(pose_stream) == ["POSE_STREAM"]
            assert court_mask == "COURT_MASK"
            assert (src_width, src_height) == (1280, 720)
            return iter(["PRE_STREAM"])

    class FakePoseExtractor:
        def __init__(self, **_kwargs):
            calls.append("pose:init")

        def iter_pose_frames(self, **kwargs):
            calls.append("pose:iter")
            assert kwargs["target_fps"] == 5
            return iter(["POSE_STREAM"])

    class FakeFeatureEngineer:
        feature_vector_size = FEATURE_DIM

        def __init__(self, **_kwargs):
            calls.append("features:init")

        def iter_build_features(self, preprocessed_stream):
            calls.append("features:iter")
            assert list(preprocessed_stream) == ["PRE_STREAM"]
            for _ in range(100):
                yield np.ones(FEATURE_DIM, dtype=np.float32), 0

    class FakeScaler:
        def transform(self, row):
            assert row.shape == (1, FEATURE_DIM)
            return row

    def fake_onnx_stream(model_path_arg, feature_rows, sequence_length, overlap, **kwargs):
        calls.append("inference:onnx")
        assert model_path_arg == str(model_path)
        assert sequence_length == 100
        assert overlap == 50
        assert kwargs["total_windows"] == 1
        rows = list(feature_rows)
        assert len(rows) == 100
        kwargs["progress_callback"](0.5)
        response = gui_app.app.test_client().get("/api/progress/job-1")
        assert response.status_code == 200
        progress_snapshots.append(response.get_json())
        return np.zeros(len(rows), dtype=np.float32)

    monkeypatch.setattr(gui_app, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(gui_app, "PoseExtractor", FakePoseExtractor)
    monkeypatch.setattr(gui_app, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(gui_app, "load_scaler_asset", lambda _path: FakeScaler())
    monkeypatch.setattr(gui_app, "run_windowed_inference_average_onnx_stream", fake_onnx_stream)
    monkeypatch.setattr(gui_app, "gaussian_filter1d", lambda values, sigma: values)

    cfg = gui_app._normalize_config(
        {
            "fps": 5,
            "seq_len": 100,
            "overlap": 50,
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
        }
    )
    job = gui_app._new_job_state("job-1", cfg)
    job["paths"]["upload"] = str(upload)
    job["paths"]["job_dir"] = str(tmp_path / "jobs" / "job-1")
    gui_app.jobs.clear()
    gui_app.jobs["job-1"] = job

    gui_app._run_pipeline("job-1")

    assert job["status"] == "completed"
    assert calls == [
        "preprocess:init",
        "court:detect",
        "pose:init",
        "preprocess:srcshape",
        "pose:iter",
        "preprocess:iter",
        "features:init",
        "inference:onnx",
        "features:iter",
    ]
    live_steps = progress_snapshots[0]["steps"]
    assert live_steps["preprocess"]["status"] == "in_progress"
    assert live_steps["preprocess"]["progress"] > 1
    assert live_steps["feature"]["status"] == "in_progress"
    assert live_steps["feature"]["progress"] > 1
    assert live_steps["inference"]["status"] == "in_progress"
    assert live_steps["inference"]["progress"] > 5


def test_library_video_export_generates_cut_on_demand(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    item_dir = library / "match-1"
    item_dir.mkdir(parents=True)
    source = item_dir / "source.mp4"
    source.write_bytes(b"full video")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,3.0\n4.0,5.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text('{"name": "Match 1"}', encoding="utf-8")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)
    calls = []

    def fake_segment_video(input_video, intervals_sec, output_video):
        calls.append((input_video, intervals_sec, output_video))
        Path(output_video).write_bytes(b"cut video")

    monkeypatch.setattr(gui_app, "segment_video", fake_segment_video)

    client = gui_app.app.test_client()
    response = client.get("/api/library/match-1/video")

    assert response.status_code == 200
    assert response.data == b"cut video"
    assert calls == [(str(source), [(1.0, 3.0), (4.0, 5.0)], str(item_dir / "export.mp4"))]
    assert (item_dir / "export.mp4").read_bytes() == b"cut video"
