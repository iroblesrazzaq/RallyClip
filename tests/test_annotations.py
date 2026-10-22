from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from training.io.annotations import csv_to_json, load_annotations_json, normalize_annotations


def test_csv_to_json_case_insensitive_headers(tmp_path):
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Start_Time", "End_Time"])
        writer.writerow(["0.5", "1.25"])

    video_path = tmp_path / "sample.mp4"
    data = csv_to_json(csv_path, video_path)
    assert data["video_path"] == str(video_path)
    assert len(data["segments"]) == 1
    assert data["segments"][0]["start_time"] == 0.5
    assert data["segments"][0]["end_time"] == 1.25
    assert data["segments"][0]["label"] == "in_play"


def test_csv_to_json_hard_fails_on_hhmmss(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_time", "end_time"])
        writer.writerow(["00:01:02.500", "00:01:05.000"])

    with pytest.raises(ValueError, match="Unparseable time"):
        csv_to_json(csv_path, tmp_path / "bad.mp4")


def test_csv_to_json_hard_fails_on_start_ge_end(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_time", "end_time"])
        writer.writerow(["2.0", "1.0"])

    with pytest.raises(ValueError, match="Invalid segment"):
        csv_to_json(csv_path, tmp_path / "bad.mp4")


def test_normalize_annotations_sorts_and_rejects_overlap():
    data = {
        "video_path": "x.mp4",
        "segments": [
            {"start_time": 2.0, "end_time": 3.0},
            {"start_time": 0.5, "end_time": 1.0},
        ],
        "metadata": {},
    }
    out = normalize_annotations(data, video_path=Path("x.mp4"))
    assert [s["start_time"] for s in out["segments"]] == [0.5, 2.0]

    with pytest.raises(ValueError, match="Overlapping"):
        normalize_annotations(
            {
                "video_path": "x.mp4",
                "segments": [
                    {"start_time": 0.0, "end_time": 2.0},
                    {"start_time": 1.5, "end_time": 3.0},
                ],
            },
            video_path=Path("x.mp4"),
        )


def test_ignore_before_s_roundtrip(tmp_path):
    path = tmp_path / "clip.mp4.json"
    path.write_text(
        json.dumps(
            {
                "video_path": "clip.mp4",
                "segments": [{"start_time": 10.0, "end_time": 12.0, "label": "in_play"}],
                "metadata": {"ignore_before_s": 5.0},
            }
        ),
        encoding="utf-8",
    )
    data = load_annotations_json(path)
    assert data["metadata"]["ignore_before_s"] == 5.0
