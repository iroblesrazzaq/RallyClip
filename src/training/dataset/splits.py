from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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
    rng = random.Random(cfg.seed)
    test_set = set(cfg.test_videos)
    val_set = set(cfg.val_videos)

    remaining = [v for v in videos if v not in test_set and v not in val_set]

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

    train = [v for v in videos if v not in test_set and v not in val_set]
    val = [v for v in videos if v in val_set]
    test = [v for v in videos if v in test_set]
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
