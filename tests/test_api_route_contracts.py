"""HTTP route contract tests.

Pin the exact JSON shapes the Flask routes emit. The native player and any
future browser/iOS clients parse these payloads; changing a key here is a
breaking API change and must be deliberate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("RALLYCLIP_LIBRARY_DIR", "/tmp/rallyclip-test-library")

from gui import app as gui_app  # noqa: E402


@pytest.fixture()
def client():
    return gui_app.app.test_client()


@pytest.fixture()
def job(monkeypatch):
    monkeypatch.setattr(gui_app, "jobs", {})
    state = gui_app._new_job_state("job-1", {})
    gui_app.jobs["job-1"] = state
    return state


def test_progress_response_contract(client, job):
    resp = client.get("/api/progress/job-1")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert set(payload) == {
        "status",
        "steps",
        "error",
        "weights",
        "eta_seconds",
        "pose_eta_seconds",
        "pose_throughput_fps",
        "library_id",
    }
    assert payload["status"] == "in_progress"
    for step in payload["steps"].values():
        assert {"status", "progress"} <= set(step)


def test_progress_unknown_job_error_contract(client, monkeypatch):
    monkeypatch.setattr(gui_app, "jobs", {})
    resp = client.get("/api/progress/nope")
    assert resp.status_code == 404
    assert resp.get_json() == {"error": "Unknown job id"}


def test_cancel_response_contract(client, job):
    resp = client.post("/api/cancel/job-1")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "cancelled"}


def test_cancel_unknown_job_error_contract(client, monkeypatch):
    monkeypatch.setattr(gui_app, "jobs", {})
    resp = client.post("/api/cancel/nope")
    assert resp.status_code == 404
    assert resp.get_json() == {"error": "Unknown job id"}


def test_library_response_contract(client, monkeypatch, tmp_path):
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", tmp_path)
    item_dir = tmp_path / "item-a"
    item_dir.mkdir()
    (item_dir / "source.mp4").write_bytes(b"\x00")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,2.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text(
        json.dumps(
            {
                "id": "item-a",
                "name": "Match A",
                "source_name": "a.mp4",
                "created": "2026-07-02T09:00:00",
                "created_ts": 100.0,
                "duration_s": 68.09,
                "point_duration_s": 21.6,
                "n_segments": 2,
            }
        ),
        encoding="utf-8",
    )

    resp = client.get("/api/library")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert set(payload) == {"items"}
    assert payload["items"] == [
        {
            "id": "item-a",
            "name": "Match A",
            "source_name": "a.mp4",
            "created": "2026-07-02T09:00:00",
            "created_ts": 100.0,
            "duration_s": 68.09,
            "point_duration_s": 21.6,
            "n_segments": 2,
            "has_csv": True,
            "has_thumbnail": False,
            "has_export": False,
        }
    ]


def test_config_defaults_response_contract(client):
    resp = client.get("/api/config/defaults")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert set(payload) == {
        "defaults",
        "yolo_model",
        "warnings",
        "available_devices",
        "auto_device",
        "runtime_state",
    }
    # Server-internal paths must never reach the browser payload.
    for key in ("model_path", "artifact_dir", "scaler_path", "yolo_weights"):
        assert key not in payload["defaults"]
    assert isinstance(payload["available_devices"], list)
    assert isinstance(payload["defaults"], dict)
