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


def test_persist_library_item_cleans_up_on_failure(tmp_path, monkeypatch):
    from gui import app as gui_app

    library = tmp_path / "library"
    upload = tmp_path / "source.mp4"
    upload.write_bytes(b"fake video")
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", library)

    def fake_write_segments_csv(segments, output_csv_path, fps, overwrite):
        Path(output_csv_path).write_text("start_time,end_time\n", encoding="utf-8")

    def fail_segment_video(input_video, intervals_sec, output_video):
        raise RuntimeError("segment failed")

    monkeypatch.setattr(gui_app, "write_segments_csv", fake_write_segments_csv)
    monkeypatch.setattr(gui_app, "segment_video", fail_segment_video)
    job = gui_app._new_job_state("job-1", gui_app._normalize_config({}))

    with pytest.raises(RuntimeError, match="segment failed"):
        gui_app._persist_library_item(
            upload_path=upload,
            base_name="Match",
            segments=[(0, 10)],
            intervals_sec=[(0.0, 2.0)],
            fps=5.0,
            job=job,
        )

    assert library.exists()
    assert list(library.iterdir()) == []
