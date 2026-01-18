from __future__ import annotations

from pathlib import Path

from training.io.videos import match_video_from_annotation, resolve_videos


def test_match_video_from_annotation(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    video = raw_dir / "match1.mp4"
    video.write_text("", encoding="utf-8")

    ann = tmp_path / "match1.mp4.csv"
    ann.write_text("start_time,end_time\n", encoding="utf-8")

    matched = match_video_from_annotation(ann, raw_dir)
    assert matched == "match1.mp4"


def test_resolve_videos_modes(tmp_path):
    raw_dir = tmp_path / "raw"
    ann_dir = tmp_path / "annotations"
    raw_dir.mkdir()
    ann_dir.mkdir()

    (raw_dir / "a.mp4").write_text("", encoding="utf-8")
    (raw_dir / "b.mov").write_text("", encoding="utf-8")

    (ann_dir / "a.csv").write_text("start_time,end_time\n", encoding="utf-8")

    annotated = resolve_videos("annotated", raw_dir, ann_dir, None)
    assert annotated == ["a.mp4"]

    all_videos = resolve_videos("all", raw_dir, ann_dir, None)
    assert set(all_videos) == {"a.mp4", "b.mov"}

    explicit = resolve_videos("list", raw_dir, ann_dir, ["b.mov"])
    assert explicit == ["b.mov"]
