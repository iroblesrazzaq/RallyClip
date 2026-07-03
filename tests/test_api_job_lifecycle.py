"""Facade-level tests for the job lifecycle wired through RallyClipServices.

These exercise gui.app's handler functions through the RallyClipServices
facade directly (no HTTP), pinning the None-for-unknown-job contract and the
lazy export behavior.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("RALLYCLIP_LIBRARY_DIR", "/tmp/rallyclip-test-library")

from gui import app as gui_app  # noqa: E402


@pytest.fixture()
def services():
    return gui_app._api_services()


@pytest.fixture()
def job(monkeypatch):
    monkeypatch.setattr(gui_app, "jobs", {})
    state = gui_app._new_job_state("job-1", {})
    gui_app.jobs["job-1"] = state
    return state


def test_job_status_unknown_job_returns_none(services, monkeypatch):
    monkeypatch.setattr(gui_app, "jobs", {})
    assert services.get_job_status("nope") is None


def test_job_status_payload_shape(services, job):
    payload = services.get_job_status("job-1")
    assert payload is not None
    assert payload["status"] == "in_progress"
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


def test_cancel_unknown_job_returns_none(services, monkeypatch):
    monkeypatch.setattr(gui_app, "jobs", {})
    assert services.cancel_job("nope") is None


def test_cancel_while_running_terminates_worker(services, job):
    class FakeProcess:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None  # still running

        def terminate(self):
            self.terminated = True

    process = FakeProcess()
    job["process"] = process
    payload = services.cancel_job("job-1")
    assert payload == {"status": "cancelled"}
    assert job["cancelled"] is True
    assert process.terminated is True
    # Idempotent: cancelling again reports the same terminal status.
    assert services.cancel_job("job-1") == {"status": "cancelled"}


def test_cancel_finished_job_keeps_status(services, job):
    job["status"] = "finished"
    assert services.cancel_job("job-1") == {"status": "finished"}
    assert job["cancelled"] is False


def test_start_job_rejects_invalid_video(services, monkeypatch, tmp_path):
    class FakeValidation:
        class VideoValidationError(Exception):
            pass

        @staticmethod
        def validate_video(path, *, seq_len, fps):
            raise FakeValidation.VideoValidationError("bad video")

    monkeypatch.setattr(gui_app, "_load_video_validation_runtime", lambda: FakeValidation)
    job_dir = tmp_path / "job-x"
    job_dir.mkdir()
    upload = job_dir / "input.mp4"
    upload.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="bad video"):
        services.start_job(upload, {"seq_len": 30, "fps": 5.0})
    assert not job_dir.exists()  # cleaned up on rejection


def test_start_job_spawns_worker_and_registers_state(services, monkeypatch, tmp_path):
    class FakeValidation:
        class VideoValidationError(Exception):
            pass

        @staticmethod
        def validate_video(path, *, seq_len, fps):
            return None

    ran = []
    monkeypatch.setattr(gui_app, "_load_video_validation_runtime", lambda: FakeValidation)
    monkeypatch.setattr(gui_app, "_run_pipeline_in_worker_process", lambda job_id: ran.append(job_id))
    monkeypatch.setattr(gui_app, "jobs", {})

    job_dir = tmp_path / "job-y"
    job_dir.mkdir()
    upload = job_dir / "input.mp4"
    upload.write_bytes(b"\x00")
    job_id = services.start_job(upload, {"seq_len": 30, "fps": 5.0})
    assert job_id == "job-y"
    state = gui_app.jobs[job_id]
    assert state["paths"]["upload"] == str(upload)
    state["thread"].join(timeout=5)
    assert ran == ["job-y"]


def test_export_missing_item_raises_not_found(services, monkeypatch, tmp_path):
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Video not available"):
        services.export_match("missing")


def test_export_generates_cut_on_demand_then_caches(services, monkeypatch, tmp_path):
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", tmp_path)
    item_dir = tmp_path / "item-a"
    item_dir.mkdir()
    (item_dir / "source.mp4").write_bytes(b"\x00")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,2.0\n", encoding="utf-8")

    calls = []

    def fake_segment_video(source, intervals, out_path):
        calls.append((source, intervals, out_path))
        Path(out_path).write_bytes(b"\x00")

    monkeypatch.setattr(gui_app, "_load_segment_video", lambda: fake_segment_video)

    export_path = services.export_match("item-a")
    assert export_path == item_dir / "export.mp4"
    assert len(calls) == 1
    assert calls[0][1] == [(1.0, 2.0)]

    # Second export reuses the fresh cut without regenerating.
    assert services.export_match("item-a") == export_path
    assert len(calls) == 1


def test_export_legacy_precut_video_served_directly(services, monkeypatch, tmp_path):
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", tmp_path)
    item_dir = tmp_path / "item-legacy"
    item_dir.mkdir()
    (item_dir / "video.mp4").write_bytes(b"\x00")
    assert services.export_match("item-legacy") == item_dir / "video.mp4"


def test_cancel_survives_late_worker_snapshot(services, job):
    """A worker progress snapshot arriving after cancel must not resurrect the
    job into in_progress (which would then be flipped to 'failed' by the
    worker's non-zero exit)."""
    assert services.cancel_job("job-1") == {"status": "cancelled"}
    late_snapshot = {**gui_app._new_job_state("job-1", {}), "status": "in_progress", "cancelled": False}
    gui_app._merge_worker_job("job-1", late_snapshot)
    assert gui_app.jobs["job-1"]["status"] == "cancelled"
    assert gui_app.jobs["job-1"]["cancelled"] is True


def test_concurrent_status_cancel_and_merges_stay_consistent(services, job, monkeypatch):
    """Hammer status/cancel/merge from threads; no exceptions, and the job
    must end cancelled (never failed/in_progress) once cancel has happened."""
    import threading

    errors = []
    stop = threading.Event()

    def poll_status():
        while not stop.is_set():
            try:
                payload = services.get_job_status("job-1")
                assert payload is not None
            except Exception as exc:  # pragma: no cover - failure reporting
                errors.append(exc)
                return

    def merge_snapshots():
        while not stop.is_set():
            try:
                snapshot = {**gui_app._new_job_state("job-1", {}), "status": "in_progress"}
                gui_app._merge_worker_job("job-1", snapshot)
            except Exception as exc:  # pragma: no cover - failure reporting
                errors.append(exc)
                return

    workers = [threading.Thread(target=poll_status) for _ in range(4)]
    workers += [threading.Thread(target=merge_snapshots) for _ in range(2)]
    for t in workers:
        t.start()
    try:
        for _ in range(50):
            assert services.cancel_job("job-1")["status"] == "cancelled"
    finally:
        stop.set()
        for t in workers:
            t.join(timeout=10)

    assert not errors, errors
    assert gui_app.jobs["job-1"]["status"] == "cancelled"


def test_concurrent_exports_generate_once(services, monkeypatch, tmp_path):
    """Simultaneous export requests for the same item must run the slow
    re-encode once; the second caller waits and reuses the fresh cut."""
    import threading
    import time

    monkeypatch.setattr(gui_app, "LIBRARY_DIR", tmp_path)
    monkeypatch.setattr(gui_app, "_export_locks", {})
    item_dir = tmp_path / "item-a"
    item_dir.mkdir()
    (item_dir / "source.mp4").write_bytes(b"\x00")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,2.0\n", encoding="utf-8")

    calls = []

    def slow_segment_video(source, intervals, out_path):
        calls.append(out_path)
        time.sleep(0.2)  # long enough for the second request to pile up
        Path(out_path).write_bytes(b"\x00")

    monkeypatch.setattr(gui_app, "_load_segment_video", lambda: slow_segment_video)

    results, errors = [], []

    def export():
        try:
            results.append(services.export_match("item-a"))
        except Exception as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [threading.Thread(target=export) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    assert len(calls) == 1, f"segment_video ran {len(calls)} times"
    assert results == [item_dir / "export.mp4"] * 4
