"""Synthetic 1-epoch smoke for classic + e2e_seg heads (no YOLO / no real video)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from training.train.loop import train as train_classic
from training.train.seg_loop import train_seg


def _write_split(path: Path, *, n_seq: int = 4, seq_len: int = 8, feat_dim: int = 16) -> None:
    features = np.random.randn(n_seq, seq_len, feat_dim).astype(np.float32)
    targets = (np.random.rand(n_seq, seq_len) > 0.7).astype(np.float32)
    video_idx = np.zeros((n_seq,), dtype=np.int32)
    frame_idx = np.tile(np.arange(seq_len, dtype=np.int64), (n_seq, 1))
    timestamps = frame_idx.astype(np.float64) / 5.0
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("features", data=features)
        h5f.create_dataset("targets", data=targets)
        h5f.create_dataset("sequence_video_index", data=video_idx)
        h5f.create_dataset("sequence_frame_index", data=frame_idx)
        h5f.create_dataset("sequence_timestamps", data=timestamps)
        h5f.create_dataset(
            "video_index_to_name",
            data=np.asarray(["synth.mp4"], dtype=h5py.string_dtype(encoding="utf-8")),
        )


def _dataset_dir(tmp_path: Path) -> Path:
    ds = tmp_path / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    _write_split(ds / "train.h5")
    _write_split(ds / "val.h5")
    return ds


def test_classic_train_smoke(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    dataset_dir = _dataset_dir(tmp_path / "classic")
    run_dir = tmp_path / "classic_run"
    train_classic(
        dataset_dir,
        run_dir,
        {
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-3,
            "pos_weight": 3.0,
            "seed": 0,
            "fps": 5.0,
            "early_stopping_patience": 0,
            "segment_eval": {"low": 0.45, "high": 0.8, "sigma": 1.5, "min_dur_sec": 0.5},
        },
    )
    assert (run_dir / "checkpoints" / "best.pth").exists()
    assert (run_dir / "metrics.jsonl").exists()
    ckpt = torch.load(run_dir / "checkpoints" / "best.pth", map_location="cpu")
    assert "arch" in ckpt
    assert ckpt["arch"]["bidirectional"] is True


def test_e2e_seg_train_smoke(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    dataset_dir = _dataset_dir(tmp_path / "e2e")
    run_dir = tmp_path / "e2e_run"
    train_seg(
        dataset_dir,
        run_dir,
        {
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-3,
            "pos_weight": 3.0,
            "fps": 5.0,
            "selection_metric": "val_loss",
            "early_stopping_patience": 0,
            "threshold": 0.5,
        },
    )
    assert (run_dir / "checkpoints" / "best.pth").exists()
    assert (run_dir / "metrics.jsonl").exists()
