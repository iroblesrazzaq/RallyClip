from __future__ import annotations

from pathlib import Path

from training.io.videos import (
    flipped_video_name,
    flipped_video_output_path,
    match_video_from_annotation,
    resolve_videos,
)


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


def test_flipped_video_name():
    assert flipped_video_name("match1.mp4") == "match1__flip_h.mp4"
    assert flipped_video_name("nested/match2.mov", suffix="__mirror") == "match2__mirror.mov"


def test_flipped_video_output_path_preserves_relative_structure(tmp_path):
    source_root = tmp_path / "raw_videos"
    output_root = tmp_path / "raw_videos_flip_h"
    source_root.mkdir()
    output_root.mkdir()

    video_path = source_root / "group_a" / "match1.mp4"
    video_path.parent.mkdir()
    video_path.write_text("", encoding="utf-8")

    output_path = flipped_video_output_path(
        video_path,
        source_root=source_root,
        output_root=output_root,
    )

    assert output_path == output_root / "group_a" / "match1__flip_h.mp4"
