from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from training.io.videos import is_flipped_video, original_video_name


@dataclass
class SplitConfig:
    strategy: str
    seed: int = 1337
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    test_videos: List[str] = field(default_factory=list)
    val_videos: List[str] = field(default_factory=list)


@dataclass
class VideoSplit:
    train: List[str]
    val: List[str]
    test: List[str]


def split_videos(videos: List[str], cfg: SplitConfig) -> VideoSplit:
    """Split videos into train/val/test.

    Flipped (`__flip_h`) variants never enter val/test — they only augment train.
    For by_video/hybrid, originals are split first; flips of train originals are
    then appended to train.
    """
    rng = random.Random(cfg.seed)
    originals = [v for v in videos if not is_flipped_video(v)]
    flips = [v for v in videos if is_flipped_video(v)]

    test_set = {v for v in cfg.test_videos if not is_flipped_video(v)}
    val_set = {v for v in cfg.val_videos if not is_flipped_video(v)}

    remaining = [v for v in originals if v not in test_set and v not in val_set]

    if cfg.strategy in {"by_video", "hybrid"}:
        if not test_set and cfg.test_ratio > 0:
            rng.shuffle(remaining)
            test_count = max(1, int(len(remaining) * cfg.test_ratio))
            test_set = set(remaining[:test_count])
            remaining = remaining[test_count:]
        if not val_set and cfg.val_ratio > 0:
            rng.shuffle(remaining)
            val_count = max(1, int(len(remaining) * cfg.val_ratio))
            val_set = set(remaining[:val_count])
            remaining = remaining[val_count:]

    train = [v for v in originals if v not in test_set and v not in val_set]
    val = [v for v in originals if v in val_set]
    test = [v for v in originals if v in test_set]

    # Flips augment train only (same original must exist in the corpus).
    train_stems = {Path(v).stem for v in train}
    for flip in flips:
        orig = original_video_name(flip)
        if Path(orig).stem in train_stems or orig in train:
            train.append(flip)

    return VideoSplit(train=train, val=val, test=test)


def temporal_split_indices(n: int, val_ratio: float, test_ratio: float) -> Dict[str, Tuple[int, int]]:
    if n <= 0:
        return {"train": (0, 0), "val": (0, 0), "test": (0, 0)}
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - test_ratio - val_ratio))
    return {
        "train": (0, max(val_start, 0)),
        "val": (max(val_start, 0), max(test_start, 0)),
        "test": (max(test_start, 0), n),
    }


def assert_no_holdout_overlap(pool: List[str], holdout: List[str], *, context: str) -> None:
    """Hard-fail if a sweep/LOSO pool intersects the frozen holdout test set."""
    holdout_set = {Path(v).name for v in holdout}
    holdout_stems = {Path(v).stem for v in holdout}
    overlap = []
    for video in pool:
        name = Path(video).name
        stem = Path(video).stem
        if name in holdout_set or stem in holdout_stems or original_video_name(name) in holdout_set:
            overlap.append(name)
    if overlap:
        raise ValueError(
            f"{context}: sweep/LOSO pool intersects frozen holdout test set: {sorted(set(overlap))}"
        )
