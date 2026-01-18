from __future__ import annotations

from training.dataset.splits import SplitConfig, split_videos, temporal_split_indices


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
