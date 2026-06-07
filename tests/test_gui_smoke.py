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
