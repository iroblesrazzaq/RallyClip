"""Contract tests for the per-point export routes.

Covers the two export flows added alongside the single "all points" clip:

  * ``GET /api/library/<id>/highlight?points=…`` — concatenate a selected
    subset of points into one highlight clip.
  * ``GET /api/library/<id>/points.zip`` — each point as its own clip, zipped.

``segment_video`` is stubbed so the tests stay fast and assert exactly which
intervals each route hands the cutter (the numeric selection is the contract).
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("RALLYCLIP_LIBRARY_DIR", "/tmp/rallyclip-test-library")

from gui import app as gui_app  # noqa: E402


@pytest.fixture()
def client():
    return gui_app.app.test_client()


@pytest.fixture()
def match(monkeypatch, tmp_path):
    """A saved match with three points, and a stubbed cutter recording calls."""
    monkeypatch.setattr(gui_app, "LIBRARY_DIR", tmp_path)
    item_dir = tmp_path / "item-a"
    item_dir.mkdir()
    (item_dir / "source.mp4").write_bytes(b"\x00")
    (item_dir / "segments.csv").write_text(
        "start_time,end_time\n1.0,2.0\n5.0,6.0\n9.0,10.0\n", encoding="utf-8"
    )

    calls: list[tuple[str, list[tuple[float, float]], str]] = []

    def fake_segment_video(source, intervals, output, *args, **kwargs):
        calls.append((source, list(intervals), output))
        # Write a stand-in file so send_file has something to return.
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"CLIP:" + str(list(intervals)).encode())

    monkeypatch.setattr(gui_app, "_load_segment_video", lambda: fake_segment_video)
    return {"id": "item-a", "dir": item_dir, "calls": calls}


# ----- highlight (selected points) ----- #


def test_highlight_cuts_only_selected_points(client, match):
    resp = client.get("/api/library/item-a/highlight?points=0,2")
    assert resp.status_code == 200
    assert resp.headers["Content-Disposition"].endswith("filename=item-a_highlight.mp4")
    # One cut call, given exactly the selected intervals in chronological order.
    assert len(match["calls"]) == 1
    _source, intervals, _out = match["calls"][0]
    assert intervals == [(1.0, 2.0), (9.0, 10.0)]


def test_highlight_dedupes_and_sorts_indices(client, match):
    resp = client.get("/api/library/item-a/highlight?points=2,0,2")
    assert resp.status_code == 200
    assert match["calls"][0][1] == [(1.0, 2.0), (9.0, 10.0)]


def test_highlight_rejects_empty_selection(client, match):
    resp = client.get("/api/library/item-a/highlight?points=")
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "No points selected"}
    assert match["calls"] == []


def test_highlight_rejects_out_of_range(client, match):
    resp = client.get("/api/library/item-a/highlight?points=0,7")
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "Points selection out of range"}
    assert match["calls"] == []


def test_highlight_rejects_non_integer(client, match):
    resp = client.get("/api/library/item-a/highlight?points=0,abc")
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "Invalid points selection"}


def test_highlight_unknown_item_404(client, match):
    resp = client.get("/api/library/nope/highlight?points=0")
    assert resp.status_code == 404


# ----- points.zip (each point individually) ----- #


def test_points_zip_has_one_clip_per_point(client, match):
    resp = client.get("/api/library/item-a/points.zip")
    assert resp.status_code == 200
    assert resp.headers["Content-Disposition"].endswith("filename=item-a_points.zip")

    # Each point cut on its own, in order.
    assert [intervals for _s, intervals, _o in match["calls"]] == [
        [(1.0, 2.0)],
        [(5.0, 6.0)],
        [(9.0, 10.0)],
    ]

    with zipfile.ZipFile(io.BytesIO(resp.data)) as archive:
        assert archive.namelist() == ["point_01.mp4", "point_02.mp4", "point_03.mp4"]


def test_points_zip_is_cached_until_segments_change(client, match):
    first = client.get("/api/library/item-a/points.zip")
    assert first.status_code == 200
    calls_after_first = len(match["calls"])

    # Second request with no changes reuses the built zip: no new cutting.
    second = client.get("/api/library/item-a/points.zip")
    assert second.status_code == 200
    assert len(match["calls"]) == calls_after_first

    # A stale zip (older than the segments/source) is rebuilt on next request.
    os.utime(match["dir"] / "points.zip", (0, 0))  # make the zip "old"
    third = client.get("/api/library/item-a/points.zip")
    assert third.status_code == 200
    assert len(match["calls"]) > calls_after_first


def test_points_zip_unknown_item_404(client, match):
    resp = client.get("/api/library/nope/points.zip")
    assert resp.status_code == 404
