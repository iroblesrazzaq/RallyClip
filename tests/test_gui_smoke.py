from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

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
    monkeypatch.setattr(gui_app, "_start_web_preview_background", lambda item_id, source_path: "processing")
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
