from __future__ import annotations

import pytest

from training.dataset.splits import (
    SplitConfig,
    assert_no_holdout_overlap,
    split_videos,
    temporal_split_indices,
)
from training.io.videos import is_flipped_video


def test_split_videos_explicit():
    videos = ["a.mp4", "b.mp4", "c.mp4"]
    cfg = SplitConfig(strategy="by_video", test_videos=["c.mp4"], val_videos=["b.mp4"])
    split = split_videos(videos, cfg)
    assert split.test == ["c.mp4"]
    assert split.val == ["b.mp4"]
    assert split.train == ["a.mp4"]


def test_split_videos_ratio_repro():
    videos = ["a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4"]
    cfg = SplitConfig(strategy="by_video", seed=42, val_ratio=0.2, test_ratio=0.2)
    split1 = split_videos(videos, cfg)
    split2 = split_videos(videos, cfg)
    assert split1 == split2


def test_temporal_split_indices():
    splits = temporal_split_indices(100, val_ratio=0.1, test_ratio=0.2)
    assert splits["train"] == (0, 70)
    assert splits["val"] == (70, 80)
    assert splits["test"] == (80, 100)


def test_flips_never_in_val_or_test_by_video():
    videos = [
        "a.mp4",
        "a__flip_h.mp4",
        "b.mp4",
        "b__flip_h.mp4",
        "c.mp4",
        "c__flip_h.mp4",
        "d.mp4",
        "e.mp4",
    ]
    for strategy in ("by_video", "hybrid"):
        cfg = SplitConfig(strategy=strategy, seed=7, val_ratio=0.25, test_ratio=0.25)
        split = split_videos(videos, cfg)
        for name in split.val + split.test:
            assert not is_flipped_video(name), f"{name} leaked into {strategy} eval"
        # Flips of train originals should appear in train.
        assert any(is_flipped_video(v) for v in split.train)


def test_holdout_overlap_guard():
    assert_no_holdout_overlap(["a.mp4", "b.mp4"], ["c.mp4"], context="test")
    with pytest.raises(ValueError, match="intersects frozen holdout"):
        assert_no_holdout_overlap(["a.mp4", "b.mp4"], ["b.mp4"], context="test")
