"""JSON serialization contract tests for RunResult and SavedMatch payloads.

These pin the exact shapes clients receive. Changing a key or value type here
is a breaking API change and must be deliberate.
"""

from __future__ import annotations

import json
from pathlib import Path

from rallyclip_api import run_result_payload, saved_match_payload
from rallyclip_core.contracts import RunResult, SavedMatch


def test_run_result_payload_contract():
    result = RunResult(
        frame_segments=[(29, 217), (230, 282)],
        intervals_sec=[(5.8, 43.4), (46.0, 56.4)],
        csv_path=Path("/out/match_segments.csv"),
        video_path=Path("/out/match_segmented.mp4"),
        diagnostics={"pipeline_id": "frame_probability_hysteresis"},
    )
    payload = run_result_payload(result)
    assert payload == {
        "pipeline_id": "frame_probability_hysteresis",
        "intervals": [
            {"start_s": 5.8, "end_s": 43.4},
            {"start_s": 46.0, "end_s": 56.4},
        ],
        "csv_path": str(Path("/out/match_segments.csv")),
        "video_path": str(Path("/out/match_segmented.mp4")),
        "n_segments": 2,
    }
    json.dumps(payload)  # must be JSON-serializable as-is


def test_run_result_payload_optional_outputs_are_null():
    result = RunResult(frame_segments=[], intervals_sec=[])
    payload = run_result_payload(result)
    assert payload == {
        "pipeline_id": None,
        "intervals": [],
        "csv_path": None,
        "video_path": None,
        "n_segments": 0,
    }
    json.dumps(payload)


def test_saved_match_payload_contract():
    match = SavedMatch(
        id="20260701-120000-abc123",
        title="Practice set",
        source_path=Path("/library/20260701-120000-abc123/source.mp4"),
        csv_path=Path("/library/20260701-120000-abc123/segments.csv"),
        thumbnail_path=Path("/library/20260701-120000-abc123/thumb.jpg"),
        metadata={"duration_s": 68.09},
    )
    payload = saved_match_payload(match)
    assert payload == {
        "id": "20260701-120000-abc123",
        "title": "Practice set",
        "source_path": str(Path("/library/20260701-120000-abc123/source.mp4")),
        "csv_path": str(Path("/library/20260701-120000-abc123/segments.csv")),
        "thumbnail_path": str(Path("/library/20260701-120000-abc123/thumb.jpg")),
        "metadata": {"duration_s": 68.09},
    }
    json.dumps(payload)


def test_saved_match_payload_without_thumbnail():
    match = SavedMatch(
        id="x",
        title="x",
        source_path=Path("/library/x/source.mp4"),
        csv_path=Path("/library/x/segments.csv"),
        thumbnail_path=None,
    )
    payload = saved_match_payload(match)
    assert payload["thumbnail_path"] is None
    assert payload["metadata"] == {}
    json.dumps(payload)
