from __future__ import annotations

import csv

from training.io.annotations import csv_to_json


def test_csv_to_json_case_insensitive_headers(tmp_path):
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Start_Time", "End_Time"])
        writer.writerow(["0.5", "1.25"])
        writer.writerow(["bad", "row"])

    video_path = tmp_path / "sample.mp4"
    data = csv_to_json(csv_path, video_path)
    assert data["video_path"] == str(video_path)
    assert len(data["segments"]) == 1
    assert data["segments"][0]["start_time"] == 0.5
    assert data["segments"][0]["end_time"] == 1.25
    assert data["segments"][0]["label"] == "in_play"
