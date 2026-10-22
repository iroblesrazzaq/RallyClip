"""Folder-of-clips import: analysis proxy, multi-clip export, and the GUI guards
around them (single-use folder token, cancel race, atomic export)."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from segmentation.proxy import create_analysis_proxy  # noqa: E402
from segmentation.segment import segment_video_sources, timeline_pieces  # noqa: E402
from test_segment import _make_clip  # noqa: E402

RATE = 48000


def _decode(path):
    """(video frame lumas, audio sample count) of a file."""
    with av.open(str(path)) as container:
        lumas = [float(f.to_ndarray(format="gray").mean()) for f in container.decode(video=0)]
    with av.open(str(path)) as container:
        samples = sum(f.samples for f in container.decode(audio=0)) if container.streams.audio else 0
    return lumas, samples


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clips")
    a, b = folder / "GX010001.mp4", folder / "GX020001.mp4"
    _make_clip(a, seconds=3, fps=10)
    _make_clip(b, seconds=2, fps=10, with_audio=False)
    return a, b


def test_proxy_is_constant_rate_and_manifest_comes_from_decoded_frames(clips, tmp_path):
    out = tmp_path / "proxy.mp4"
    manifest = create_analysis_proxy(list(clips), out, fps=10)

    assert [(m["start_s"], m["duration_s"]) for m in manifest] == [(0.0, 3.0), (3.0, 2.0)]
    lumas, samples = _decode(out)
    assert len(lumas) == 50
    # The clip without audio contributes silence, keeping later audio in sync.
    assert abs(samples - 5 * RATE) <= 2048


def test_proxy_holds_frames_across_rate_gaps(tmp_path):
    slow = tmp_path / "slow.mp4"
    _make_clip(slow, seconds=2, fps=5, with_audio=False)
    manifest = create_analysis_proxy([slow, slow], tmp_path / "proxy.mp4", fps=10)

    # A 5fps clip fills 10fps slots and keeps its last frame's full duration,
    # so the second clip starts at exactly 2.0s rather than drifting early.
    assert [(m["start_s"], m["duration_s"]) for m in manifest] == [(0.0, 2.0), (2.0, 2.0)]
    assert len(_decode(tmp_path / "proxy.mp4")[0]) == 40


def test_timeline_pieces_split_an_interval_at_the_clip_boundary():
    sources = [
        {"path": "a.mp4", "start_s": 0.0, "duration_s": 3.0},
        {"path": "b.mp4", "start_s": 3.0, "duration_s": 2.0},
    ]
    assert timeline_pieces(sources, [(1.0, 1.5), (2.0, 4.0)]) == [
        ("a.mp4", [(1.0, 1.5), (2.0, 3.0)]),
        ("b.mp4", [(0.0, 1.0)]),
    ]


def test_export_cuts_across_clip_boundary_from_originals(clips, tmp_path):
    a, b = clips
    sources = [
        {"path": str(a), "start_s": 0.0, "duration_s": 3.0},
        {"path": str(b), "start_s": 3.0, "duration_s": 2.0},
    ]
    out = tmp_path / "export.mp4"
    segment_video_sources(sources, [(2.0, 4.0)], str(out))

    lumas, samples = _decode(out)
    # Frames 2.0-2.9s of clip A (luma ~20-29), then 0-1.0s of clip B (luma ~0-10).
    assert 20 <= len(lumas) <= 21
    assert abs(lumas[0] - 20) < 4 and abs(lumas[10] - 0) < 4
    assert abs(samples - len(lumas) * RATE / 10) <= 2048


# --------------------------------------------------------------------------- GUI


@pytest.fixture
def gui(tmp_path, monkeypatch):
    from gui import app as gui_app

    monkeypatch.setattr(gui_app, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(gui_app, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(gui_app, "pending_folder", None)
    monkeypatch.setattr(gui_app, "_keep_jobs", lambda: False)
    gui_app.export_jobs.clear()
    return gui_app


def test_folder_select_sorts_naturally_and_token_starts_one_job(gui, tmp_path, monkeypatch):
    folder = tmp_path / "card"
    folder.mkdir()
    for name in ("clip10.MP4", "clip2.mp4", "._clip1.mp4", "notes.txt"):
        (folder / name).write_bytes(b"x")
    monkeypatch.setattr(gui, "_choose_match_folder", lambda: folder)
    started = []
    monkeypatch.setattr(gui, "_start_folder_analysis_job", lambda files, cfg: started.append(files) or "job-1")
    client = gui.app.test_client()

    picked = client.post("/api/folder/select").get_json()
    assert picked["files"] == ["clip2.mp4", "clip10.MP4"]

    assert client.post("/api/folder/start", json={"token": picked["token"]}).status_code == 200
    # A double-click (or replayed token) must not start the folder twice.
    assert client.post("/api/folder/start", json={"token": picked["token"]}).status_code == 400
    assert started == [[folder / "clip2.mp4", folder / "clip10.MP4"]]


def test_folder_token_is_spent_by_a_failed_start(gui, tmp_path, monkeypatch):
    folder = tmp_path / "card"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"x")
    monkeypatch.setattr(gui, "_choose_match_folder", lambda: folder)
    calls = []

    def reject(files, cfg):
        calls.append(files)
        raise ValueError("too short")

    monkeypatch.setattr(gui, "_start_folder_analysis_job", reject)
    client = gui.app.test_client()
    token = client.post("/api/folder/select").get_json()["token"]

    assert client.post("/api/folder/start", json={"token": token}).get_json() == {"error": "too short"}
    retry = client.post("/api/folder/start", json={"token": token})
    assert retry.status_code == 400 and retry.get_json()["error"] == "Choose the folder again."
    assert len(calls) == 1


def test_folder_preflight_rejects_mismatched_clips(gui, monkeypatch):
    infos = {
        "a.mp4": SimpleNamespace(width=1920, height=1080, fps=30.0, duration_s=600.0),
        "b.mp4": SimpleNamespace(width=1920, height=1080, fps=60.0, duration_s=600.0),
    }
    validation = gui._load_video_validation_runtime()
    monkeypatch.setattr(
        gui,
        "_load_video_validation_runtime",
        lambda: SimpleNamespace(
            VideoValidationError=validation.VideoValidationError,
            MIN_HEIGHT=720,
            probe_video=lambda path: infos[Path(path).name],
        ),
    )
    cfg = {"seq_len": 300, "fps": 30.0}
    with pytest.raises(ValueError, match="single recording"):
        gui._start_folder_analysis_job([Path("a.mp4"), Path("b.mp4")], cfg)


def test_cancel_before_worker_handle_is_recorded_still_terminates(gui, monkeypatch):
    job_id = "job-race"
    state = gui._new_job_state(job_id, {"fps": 30.0})
    with gui.jobs_lock:
        gui.jobs[job_id] = state

    class FakeProcess:
        terminated = False
        stdout = io.StringIO("")

        def __init__(self, *args, **kwargs):
            # The user cancels while Popen is still returning: no handle yet.
            assert gui._cancel_analysis_job(job_id) == {"status": "cancelled"}

        def terminate(self):
            FakeProcess.terminated = True

        def poll(self):
            return None

        def wait(self):
            return -15

    monkeypatch.setattr(gui.subprocess, "Popen", FakeProcess)
    try:
        gui._run_pipeline_in_worker_process(job_id)
        assert FakeProcess.terminated
        assert gui.jobs[job_id]["status"] == "cancelled"
        assert not Path(state["paths"]["job_dir"]).exists()
    finally:
        gui.jobs.pop(job_id, None)


def _library_item(gui, item_id="m1", sources=None):
    item_dir = gui.LIBRARY_DIR / item_id
    item_dir.mkdir(parents=True)
    (item_dir / "source.mp4").write_bytes(b"proxy")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,2.0\n", encoding="utf-8")
    (item_dir / "meta.json").write_text('{"name": "M"}', encoding="utf-8")
    if sources is not None:
        (item_dir / "sources.json").write_text(json.dumps({"version": 1, "sources": sources}), encoding="utf-8")
    return item_dir


def test_failed_export_leaves_no_stale_export_file(gui, monkeypatch):
    item_dir = _library_item(gui)

    def crash(input_video, intervals, output_video):
        Path(output_video).write_bytes(b"truncated")
        raise RuntimeError("encoder died")

    monkeypatch.setattr(gui, "segment_video", crash)
    assert gui.app.test_client().get("/api/library/m1/video").status_code == 500
    assert not (item_dir / "export.mp4").exists()
    assert not (item_dir / "export.partial.mp4").exists()


def test_folder_item_exports_from_original_clips(gui, tmp_path, monkeypatch):
    original = tmp_path / "GX010001.mp4"
    original.write_bytes(b"4k")
    sources = [{"path": str(original), "name": original.name, "start_s": 0.0, "duration_s": 600.0}]
    item_dir = _library_item(gui, sources=sources)
    calls = []

    def fake_sources(srcs, intervals, output_path, **kwargs):
        calls.append((srcs, intervals))
        Path(output_path).write_bytes(b"cut")

    import segmentation.segment as segment

    monkeypatch.setattr(segment, "segment_video_sources", fake_sources)
    response = gui.app.test_client().get("/api/library/m1/video")
    assert response.status_code == 200 and response.data == b"cut"
    assert calls == [(sources, [(1.0, 2.0)])]

    (item_dir / "export.mp4").unlink()
    original.unlink()  # drive unplugged
    missing = gui.app.test_client().get("/api/library/m1/video")
    assert missing.status_code == 404 and "Original clips not found" in missing.get_json()["error"]


def test_background_export_reports_progress_then_ready(gui, monkeypatch):
    _library_item(gui)

    def fake_segment(input_video, intervals, output_video, progress_callback=None):
        progress_callback(50)
        Path(output_video).write_bytes(b"cut")

    monkeypatch.setattr(gui, "segment_video", fake_segment)
    client = gui.app.test_client()
    assert client.post("/api/library/m1/export").status_code == 202
    deadline = time.time() + 5
    while (state := client.get("/api/library/m1/export/status").get_json())["status"] == "processing":
        assert time.time() < deadline
        time.sleep(0.01)
    assert state == {"status": "ready", "progress": 100, "error": None}
    assert client.get("/api/library/m1/video").data == b"cut"


def test_persist_writes_sources_manifest(gui, tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "_start_preview_window_background", lambda *args: None)
    proxy = tmp_path / "analysis_proxy.mp4"
    proxy.write_bytes(b"proxy")
    manifest = [
        {"path": "/cam/a.mp4", "name": "a.mp4", "start_s": 0.0, "duration_s": 3.0},
        {"path": "/cam/b.mp4", "name": "b.mp4", "start_s": 3.0, "duration_s": 2.0},
    ]
    job = gui._new_job_state("j", {"fps": 30.0})
    item_id, _, _ = gui._persist_library_item(
        upload_path=proxy,
        base_name="Card",
        segments=[(30, 60)],
        intervals_sec=[(1.0, 2.0)],
        fps=30.0,
        job=job,
        source_manifest=manifest,
    )
    item_dir = gui._library_item_dir(item_id)
    assert json.loads((item_dir / "sources.json").read_text())["sources"] == manifest
    assert json.loads((item_dir / "meta.json").read_text())["source_name"] == "2 consecutive clips"
