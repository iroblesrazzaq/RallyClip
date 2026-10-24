"""Folder-of-clips import: analysis proxy, multi-clip export, and the GUI guards
around them (single-use folder token, cancel race, atomic export)."""

from __future__ import annotations

import io
import json
import threading
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


def test_proxy_spans_ignore_container_duration(clips, tmp_path, monkeypatch):
    # A truncated camera clip can claim more time than it decodes; spans must
    # come from the frames, so a wrong container duration changes nothing.
    import segmentation.proxy as proxy

    monkeypatch.setattr(proxy, "_duration_s", lambda container, stream: 99.0)
    manifest = create_analysis_proxy(list(clips), tmp_path / "proxy.mp4", fps=10)
    assert [(m["start_s"], m["duration_s"]) for m in manifest] == [(0.0, 3.0), (3.0, 2.0)]


def test_proxy_holds_frames_across_rate_gaps(tmp_path):
    slow = tmp_path / "slow.mp4"
    _make_clip(slow, seconds=2, fps=5, with_audio=False)
    manifest = create_analysis_proxy([slow, slow], tmp_path / "proxy.mp4", fps=10)

    # A 5fps clip fills 10fps slots and keeps its last frame's full duration,
    # so the second clip starts at exactly 2.0s rather than drifting early.
    assert [(m["start_s"], m["duration_s"]) for m in manifest] == [(0.0, 2.0), (2.0, 2.0)]
    assert len(_decode(tmp_path / "proxy.mp4")[0]) == 40


def test_proxy_gap_slots_repeat_the_previous_picture(tmp_path):
    steps = tmp_path / "steps.mp4"
    values = [40, 120, 200, 80]
    container = av.open(str(steps), "w")
    try:
        v = container.add_stream("libx264", rate=5)
        v.width, v.height, v.pix_fmt = 320, 240, "yuv420p"
        for i, value in enumerate(values):
            frame = av.VideoFrame.from_ndarray(np.full((240, 320, 3), value, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for pkt in v.encode(frame):
                container.mux(pkt)
        for pkt in v.encode():
            container.mux(pkt)
    finally:
        container.close()

    create_analysis_proxy([steps], tmp_path / "proxy.mp4", fps=10)
    lumas, _ = _decode(tmp_path / "proxy.mp4")
    # Each 5fps picture fills two 10fps slots; a gap slot shows the held picture, not black.
    expected = [_decode(steps)[0][i // 2] for i in range(2 * len(values))]
    assert len(lumas) == len(expected)
    assert all(abs(a - b) < 4 for a, b in zip(lumas, expected)), (lumas, expected)


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


def test_audio_stays_locked_to_video_across_many_short_cuts(tmp_path, monkeypatch):
    import segmentation.segment as segment

    errors = []
    pad_to = segment.AudioTrack.pad_to

    def checked_pad_to(track, total):
        pad_to(track, total)
        errors.append(track.written - total)

    monkeypatch.setattr(segment.AudioTrack, "pad_to", checked_pad_to)
    clip = tmp_path / "sixty.mp4"
    _make_clip(clip, seconds=40, fps=60)
    # A cut's last AAC block runs past its last video frame. That overrun must be
    # trimmed before it is encoded, or it piles up cut after cut (~60 ms here).
    segment_video_sources(
        [{"path": str(clip), "start_s": 0.0, "duration_s": 40.0}],
        [(2.003 * i, 2.003 * i + 1.01) for i in range(15)],
        str(tmp_path / "cuts.mp4"),
    )
    assert len(errors) == 15 and set(errors) == {0}


def test_proxy_downscales_and_drops_frames_to_the_model_rate(clips, tmp_path):
    out = tmp_path / "proxy.mp4"
    manifest = create_analysis_proxy(list(clips), out, fps=5, max_height=120)

    assert [(m["start_s"], m["duration_s"]) for m in manifest] == [(0.0, 3.0), (3.0, 2.0)]
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        assert (stream.codec_context.width, stream.codec_context.height) == (160, 120)
        lumas = [float(f.to_ndarray(format="gray").mean()) for f in container.decode(stream)]
    # Every other 10fps frame: clip A's frame 2k (luma ~2k) lands at proxy frame k.
    assert len(lumas) == 25 and abs(lumas[5] - 10) < 4


def _make_ts(path, gop, seconds=3, fps=10):
    """Video-only MPEG-TS clip (non-zero start_time) whose frame i has luma ~i."""
    with av.open(str(path), "w", format="mpegts") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        stream.options = {"g": str(gop)}
        for i in range(seconds * fps):
            frame = av.VideoFrame.from_ndarray(np.full((240, 320, 3), i, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("gop", [10, 250])  # seek lands late / seek lands at EOF
def test_mpegts_clips_are_timed_from_their_first_frame(tmp_path, gop):
    clip = tmp_path / "00001.MTS"
    _make_ts(clip, gop)
    with av.open(str(clip), format="mpegts") as container:
        assert container.streams.video[0].start_time > 0  # the case under test
    manifest = create_analysis_proxy([clip], tmp_path / "proxy.mp4", fps=10)
    assert [(m["start_s"], m["duration_s"]) for m in manifest] == [(0.0, 3.0)]

    out = tmp_path / "export.mp4"
    segment_video_sources([{"path": str(clip), "start_s": 0.0, "duration_s": 3.0}], [(1.2, 2.2)], str(out))
    lumas, _ = _decode(out)
    assert 10 <= len(lumas) <= 11 and abs(lumas[0] - 12) < 4


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


def _fake_probe(gui, monkeypatch, infos):
    validation = gui._load_video_validation_runtime()
    monkeypatch.setattr(
        gui,
        "_load_video_validation_runtime",
        lambda: SimpleNamespace(
            VideoValidationError=validation.VideoValidationError,
            MIN_HEIGHT=720,
            probe_video=lambda path: SimpleNamespace(**dict(zip(("width", "height", "fps", "duration_s"), infos[Path(path).name]))),
        ),
    )


CFG = {"seq_len": 300, "fps": 30.0}  # needs 10s of footage


@pytest.mark.parametrize(
    "second, error",
    [
        ((1920, 1080, 60.0, 600.0), "single recording"),
        ((1280, 720, 30.0, 600.0), "single recording"),
        ((1280, 640, 30.0, 600.0), "at least 720p"),
    ],
)
def test_folder_preflight_rejects_mismatched_clips(gui, monkeypatch, second, error):
    _fake_probe(gui, monkeypatch, {"a.mp4": (1920, 1080, 30.0, 600.0), "b.mp4": second})
    with pytest.raises(ValueError, match=error):
        gui._start_folder_analysis_job([Path("a.mp4"), Path("b.mp4")], CFG)


@pytest.mark.parametrize(
    "lengths, ok",
    [((4.0, 4.0), False), ((6.0, 6.0), True), ((0.0, 4.0), True)],  # 0 = unknown (MPEG-TS)
)
def test_folder_preflight_sums_clip_lengths(gui, monkeypatch, lengths, ok):
    _fake_probe(gui, monkeypatch, {"a.mp4": (1920, 1080, 30.0, lengths[0]), "b.mp4": (1920, 1080, 30.0, lengths[1])})
    monkeypatch.setattr(gui, "_launch_job", lambda state: state)
    files = [Path("/cam/card/a.mp4"), Path("/cam/card/b.mp4")]
    if not ok:
        with pytest.raises(ValueError, match="too short"):
            gui._start_folder_analysis_job(files, CFG)
        return
    state = gui._start_folder_analysis_job(files, CFG)
    assert state["sources"] == [str(f) for f in files]
    assert state["config"]["output_name"] == "card"
    assert "proxy" in state["steps"] and state["paths"]["upload"].endswith("analysis_proxy.mp4")


def test_upload_size_is_not_capped(gui):
    assert gui.app.config.get("MAX_CONTENT_LENGTH") is None


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
    response.close()  # send_file holds export.mp4 open until then; Windows can't unlink an open file

    (item_dir / "export.mp4").unlink()
    original.unlink()  # drive unplugged
    missing = gui.app.test_client().get("/api/library/m1/video")
    assert missing.status_code == 404 and "Original clips not found" in missing.get_json()["error"]


def test_background_export_reports_progress_then_ready(gui, monkeypatch):
    _library_item(gui)

    release = threading.Event()

    def fake_segment(input_video, intervals, output_video, progress_callback=None):
        progress_callback(50)
        release.wait(5)
        Path(output_video).write_bytes(b"cut")

    monkeypatch.setattr(gui, "segment_video", fake_segment)
    client = gui.app.test_client()
    assert client.post("/api/library/m1/export").status_code == 202
    deadline = time.time() + 5
    while (state := client.get("/api/library/m1/export/status").get_json())["progress"] != 50:
        assert time.time() < deadline and state["status"] == "processing", state
        time.sleep(0.01)
    assert state == {"status": "processing", "progress": 50, "error": None}  # what the button shows
    release.set()
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


def test_points_edited_mid_export_are_not_lost(gui, monkeypatch):
    _library_item(gui)
    client = gui.app.test_client()
    calls = []

    def fake_segment(input_video, intervals, output_video):
        calls.append(intervals)
        if len(calls) == 1:
            # The user saves new points while this encode runs. The save must not
            # wait for the export lock (this would deadlock if it did).
            body = {"segments": [{"start": 3.0, "end": 4.0}]}
            assert client.put("/api/library/m1/segments", json=body).status_code == 200
        Path(output_video).write_bytes(f"cut{len(calls)}".encode())

    monkeypatch.setattr(gui, "segment_video", fake_segment)
    assert client.get("/api/library/m1/video").data == b"cut2"
    assert calls == [[(1.0, 2.0)], [(3.0, 4.0)]]
