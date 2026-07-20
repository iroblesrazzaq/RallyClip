from __future__ import annotations

import pytest

from training.pipeline import _run_train


def test_unknown_head_errors(monkeypatch, tmp_path):
    calls = []

    def fake_classic(*args, **kwargs):
        calls.append("classic")

    def fake_seg(*args, **kwargs):
        calls.append("e2e_seg")

    monkeypatch.setattr("training.pipeline.train_loop", fake_classic)
    monkeypatch.setattr("training.pipeline.train_seg", fake_seg)

    config = {
        "data_root": str(tmp_path),
        "run_id": "t",
        "train": {"head": "nope"},
        "preprocess": {"target_fps": 5},
        "dataset": {},
        "yolo": {},
        "features": {},
    }
    with pytest.raises(ValueError, match="Unknown train.head"):
        _run_train(config)


def test_classic_and_e2e_dispatch(monkeypatch, tmp_path):
    calls = []

    def fake_classic(dataset_dir, run_dir, train_cfg):
        calls.append(("classic", train_cfg["fps"]))

    def fake_seg(dataset_dir, run_dir, train_cfg):
        calls.append(("e2e_seg", train_cfg["fps"]))

    monkeypatch.setattr("training.pipeline.train_loop", fake_classic)
    monkeypatch.setattr("training.pipeline.train_seg", fake_seg)

    base = {
        "data_root": str(tmp_path),
        "run_id": "t",
        "preprocess": {"target_fps": 5},
        "dataset": {"seq_len_seconds": 10, "overlap_seconds": 5, "split": {}},
        "yolo": {"model": "yolov8n-pose.pt", "conf": 0.25, "imgsz": 960},
        "features": {"feature_set": "v1"},
    }
    _run_train({**base, "train": {"head": "classic", "segment_eval": {}}})
    _run_train({**base, "train": {"head": "e2e_seg", "segment_eval": {}}})
    assert calls == [("classic", 5.0), ("e2e_seg", 5.0)]


def test_eval_skips_without_test_h5(monkeypatch, tmp_path, caplog):
    from training.pipeline import _run_eval

    data_root = tmp_path
    run_id = "r1"
    (data_root / "runs" / run_id / "checkpoints").mkdir(parents=True)
    (data_root / "datasets" / run_id).mkdir(parents=True)
    # no test.h5

    config = {
        "data_root": str(data_root),
        "run_id": run_id,
        "train": {"head": "classic", "segment_eval": {}},
        "preprocess": {"target_fps": 5},
    }
    _run_eval(config)  # should not raise
