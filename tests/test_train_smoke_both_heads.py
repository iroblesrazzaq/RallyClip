"""Synthetic 1-epoch smoke for the classic and e2e_heatmap heads (no YOLO / no real video)."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from infer.heatmap_decode import HeatmapDecodeConfig as RuntimeDecodeConfig  # noqa: E402
from training.train.heatmap_loop import load_heatmap_model, train_heatmap  # noqa: E402
from training.train.loop import train as train_classic  # noqa: E402


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


def test_heatmap_train_smoke(tmp_path):
    np.random.seed(0)
    dataset_dir = _dataset_dir(tmp_path / "heatmap")
    run_dir = tmp_path / "heatmap_run"
    config = {
        "device": "cpu",
        "epochs": 2,
        "batch_size": 2,
        "lr": 1e-3,
        "pos_weight": 3.0,
        "seed": 0,
        "fps": 5.0,
        "hidden_size": 16,
        "heatmap_tcn_levels": 2,
        "selection_metric": "val_loss",
        "lr_schedule": "cosine",
        "threshold": 0.5,
    }
    train_heatmap(dataset_dir, run_dir, config)
    assert (run_dir / "metrics.jsonl").exists()
    decode = json.loads((run_dir / "manifest.json").read_text())["decode"]
    runtime = RuntimeDecodeConfig(**decode["runtime_params"])   # the runtime can rebuild it
    assert runtime.mode == decode["mode"] and runtime.peak_threshold == decode["peak_threshold"]
    ckpt = torch.load(run_dir / "checkpoints" / "best.pth", map_location="cpu")
    assert ckpt["arch"]["levels"] == 2
    load_heatmap_model(run_dir / "checkpoints" / "best.pth", config)

    # Same seed -> same weights.
    rerun_dir = tmp_path / "heatmap_rerun"
    train_heatmap(dataset_dir, rerun_dir, config)
    again = torch.load(rerun_dir / "checkpoints" / "last.pth", map_location="cpu")["model_state_dict"]
    first = torch.load(run_dir / "checkpoints" / "last.pth", map_location="cpu")["model_state_dict"]
    assert all(torch.equal(first[k], again[k]) for k in first)


def test_heatmap_trains_without_a_val_split(tmp_path):
    # The final model trains on every video: no val.h5, a fixed save_every_n checkpoint.
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_split(dataset_dir / "train.h5")
    run_dir = tmp_path / "run"
    # A best.pth left by an earlier run of this run_id must not survive.
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best.pth").write_bytes(b"stale")
    train_heatmap(dataset_dir, run_dir, {
        "device": "cpu", "epochs": 4, "batch_size": 2, "pos_weight": 3.0, "seed": 0, "fps": 5.0,
        "hidden_size": 16, "heatmap_tcn_levels": 2, "save_every_n": 3, "early_stopping_patience": 1,
    })

    ckpts = sorted(p.name for p in (run_dir / "checkpoints").iterdir())
    assert ckpts == ["epoch_3.pth", "last.pth"]
    assert torch.load(run_dir / "checkpoints" / "last.pth", map_location="cpu")["epoch"] == 4
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["training"]["selection_metric"] == "none"
    assert manifest["source_run"]["checkpoint_path"].endswith("last.pth")


def test_heatmap_honors_train_batch_size_and_refuses_grad_accum(tmp_path, monkeypatch):
    from training.train import heatmap_loop

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_split(dataset_dir / "train.h5")
    cfg = {
        "device": "cpu", "epochs": 1, "batch_size": 8, "train_batch_size": 2, "pos_weight": 3.0, "seed": 0,
        "fps": 5.0, "hidden_size": 16, "heatmap_tcn_levels": 2,
    }
    with pytest.raises(ValueError, match="grad_accum_steps"):
        train_heatmap(dataset_dir, tmp_path / "run", {**cfg, "grad_accum_steps": 4})

    sizes = []
    real_loader = heatmap_loop.DataLoader
    monkeypatch.setattr(heatmap_loop, "DataLoader", lambda ds, batch_size, **kw: sizes.append(batch_size) or real_loader(ds, batch_size=batch_size, **kw))
    train_heatmap(dataset_dir, tmp_path / "run", cfg)
    assert sizes == [2]
