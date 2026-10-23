"""Flipped (__flip_h) variants of val/test videos must not reach train."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from training.dataset.builder import DatasetBuilder, DatasetConfig
from training.dataset.splits import SplitConfig

VIDEOS = ["a.mp4", "a__flip_h.mp4", "b.mp4", "b__flip_h.mp4", "c.mp4", "c__flip_h.mp4"]


def _write_features(root: Path, n_frames: int = 60) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for video in VIDEOS:
        with h5py.File(root / f"{Path(video).stem}__features__v1.h5", "w") as h5f:
            h5f.create_dataset("features", data=rng.standard_normal((n_frames, 4)).astype(np.float32))
            h5f.create_dataset("targets", data=(rng.random(n_frames) > 0.5).astype(np.float32))


def _build(tmp_path: Path, split: SplitConfig) -> dict:
    _write_features(tmp_path / "features")
    cfg = DatasetConfig(seq_len_seconds=2.0, overlap_seconds=1.0, target_fps=5.0, split=split)
    DatasetBuilder(cfg).build(tmp_path / "features", tmp_path / "out", VIDEOS, "v1")
    return json.loads((tmp_path / "out" / "dataset_manifest.json").read_text())


def test_by_video_drops_flips_of_val_and_test_videos(tmp_path):
    manifest = _build(tmp_path, SplitConfig(strategy="by_video", test_videos=["b.mp4"], val_videos=["c.mp4"]))
    assert manifest["splits"] == {
        "train": ["a.mp4", "a__flip_h.mp4"],
        "val": ["c.mp4"],
        "test": ["b.mp4"],
    }


def test_loso_drops_the_flip_of_the_held_out_video(tmp_path):
    manifest = _build(tmp_path, SplitConfig(strategy="loso_temporal_val", test_videos=["b.mp4"], val_ratio=0.2))
    assert manifest["splits"]["test"] == ["b.mp4"]
    assert "b__flip_h.mp4" not in manifest["videos"]
    assert {"a__flip_h.mp4", "c__flip_h.mp4"} <= set(manifest["splits"]["train"])
    assert all("flip" not in v for v in manifest["splits"]["val"])


def test_ratio_split_never_trains_on_a_held_out_mirror(tmp_path):
    manifest = _build(tmp_path, SplitConfig(strategy="hybrid", seed=3, val_ratio=0.34, test_ratio=0.34))
    held_out = {Path(v).stem for v in manifest["splits"]["val"] + manifest["splits"]["test"]}
    for video in manifest["splits"]["train"]:
        assert Path(video).stem.removesuffix("__flip_h") not in held_out


def test_loso_refuses_a_flipped_holdout(tmp_path):
    with pytest.raises(ValueError, match="not its flip"):
        _build(tmp_path, SplitConfig(strategy="loso_temporal_val", test_videos=["b__flip_h.mp4"], val_ratio=0.2))
