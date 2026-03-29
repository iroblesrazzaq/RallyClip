from __future__ import annotations

import h5py
import numpy as np

from training.dataset.builder import DatasetBuilder, DatasetConfig
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


def test_within_video_mirror_variants_only_expand_train_split(tmp_path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()

    def write_feature_file(name: str) -> None:
        path = feature_root / f"{name}__features__v1.h5"
        features = np.arange(32, dtype=np.float32).reshape(16, 2)
        targets = np.zeros(16, dtype=np.int8)
        frame_index = np.arange(16, dtype=np.int64)
        timestamps = np.arange(16, dtype=np.float64)
        with h5py.File(path, "w") as h5f:
            h5f.create_dataset("features", data=features)
            h5f.create_dataset("targets", data=targets)
            h5f.create_dataset("frame_index", data=frame_index)
            h5f.create_dataset("timestamps", data=timestamps)

    write_feature_file("match1")
    write_feature_file("match1__flip_h")

    builder = DatasetBuilder(
        DatasetConfig(
            seq_len_seconds=4.0,
            overlap_seconds=0.0,
            target_fps=1.0,
            split=SplitConfig(strategy="within_video", val_ratio=0.25, test_ratio=0.0),
            mirror_train=True,
        )
    )

    out_dir = tmp_path / "dataset"
    built = builder.build(feature_root, out_dir, ["match1.mp4"], "v1")
    assert built == out_dir

    with h5py.File(out_dir / "train.h5", "r") as h5f:
        train_names = {name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in h5f["video_index_to_name"][:]}
    with h5py.File(out_dir / "val.h5", "r") as h5f:
        val_names = {name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in h5f["video_index_to_name"][:]}

    assert train_names == {"match1.mp4", "match1__flip_h.mp4"}
    assert val_names == {"match1.mp4"}
