from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from cli.main import RunConfig, build_run_config, run_pipeline
from training.io.config import load_config
from training.models.lstm import TennisPointLSTM
from training.pipeline import export_model_artifact

cli_main = importlib.import_module("cli.main")


def test_load_config_deep_merge(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_root": "data/custom",
                "wandb": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))
    assert cfg["data_root"] == "data/custom"
    assert cfg["wandb"]["enabled"] is True
    assert cfg["wandb"]["project"] is None
    assert cfg["wandb"]["entity"] is None
    assert cfg["videos"] == []
    assert cfg["config_path"] == str(cfg_path)


def test_export_model_artifact_writes_manifest_model_and_scaler_json(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    run_id = "prod_yolon960_fps5_seq20_mirror_v1"
    run_dir = data_root / "runs" / run_id
    dataset_dir = data_root / "datasets" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    run_config = {
        "run_id": run_id,
        "fps": 5.0,
        "yolo": {"model": "yolov8n-pose.pt", "conf": 0.25, "imgsz": 960},
        "extract": {"sampling_mode": "downsample_then_extract", "sample_fps": 5},
        "preprocess": {"target_fps": 5},
        "features": {"feature_set": "v1"},
        "augmentation": {"enabled": True, "mirror_train": True, "flip_suffix": "__flip_h"},
        "dataset": {
            "seq_len_seconds": 20,
            "overlap_seconds": 10,
            "split": {"strategy": "within_video", "val_ratio": 0.1, "test_ratio": 0.0},
        },
        "train": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": True,
            "lr": 0.001,
            "weight_decay": 0.01,
            "pos_weight": 3.0,
            "batch_size": 8,
            "grad_accum_steps": 1,
            "epochs": 30,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.001,
            "device": "mps",
            "threshold": 0.5,
            "segment_eval": {"low": 0.45, "high": 0.7, "sigma": 1.0, "min_dur_sec": 1.0},
        },
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    metrics_rows = [
        {"epoch": 6, "balanced_accuracy": 0.9579276059, "val_loss": 0.1764368806},
        {"epoch": 10, "balanced_accuracy": 0.9582694099, "val_loss": 0.2017740325},
    ]
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in metrics_rows:
            handle.write(json.dumps(row) + "\n")

    dataset_manifest = {
        "feature_set": "v1",
        "splits": {
            "train": ["match1.mp4", "match1__flip_h.mp4"],
            "val": ["match1.mp4"],
            "test": [],
        },
        "videos": {
            "match1.mp4": {"total_frames": 1000, "split": "train_val_temporal"},
            "match1__flip_h.mp4": {"total_frames": 1000, "split": "train", "variant": "flip_h", "augmented_from": "match1.mp4"},
        },
        "config": {
            "seq_len_seconds": 20,
            "overlap_seconds": 10,
            "target_fps": 5,
            "split": {"strategy": "within_video", "val_ratio": 0.1, "test_ratio": 0.0},
        },
    }
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

    scaler = StandardScaler()
    scaler.fit(np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    joblib.dump(scaler, run_dir / "scaler.joblib")

    model = TennisPointLSTM(
        input_size=4,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        return_logits=True,
    )
    torch.save(
        {
            "epoch": 10,
            "model_state_dict": model.state_dict(),
            "metrics": metrics_rows[-1],
        },
        run_dir / "checkpoints" / "best.pth",
    )

    def fake_onnx_export(model, args, f, **kwargs):
        Path(f).write_bytes(b"fake-onnx")

    monkeypatch.setattr("training.pipeline.torch.onnx.export", fake_onnx_export)

    artifact_dir = export_model_artifact(
        data_root=data_root,
        run_id=run_id,
        version="rallyclip_v0.2.0",
        output_root=tmp_path / "models",
    )

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    scaler_json = json.loads((artifact_dir / "scaler.json").read_text(encoding="utf-8"))

    assert artifact_dir.name == "rallyclip_v0.2.0"
    assert (artifact_dir / "model.onnx").read_bytes() == b"fake-onnx"
    assert manifest["manifest_version"] == 1
    assert manifest["artifact"]["model_version"] == "rallyclip_v0.2.0"
    assert manifest["source_run"]["run_id"] == run_id
    assert manifest["source_run"]["selected_checkpoint"] == "best"
    assert manifest["source_run"]["selected_epoch"] == 10
    assert manifest["feature_pipeline"]["feature_set"] == "v1"
    assert manifest["feature_pipeline"]["feature_dim"] == 4
    assert manifest["inference"]["seq_len_frames"] == 100
    assert manifest["postprocess"]["method"] == "hysteresis"
    assert manifest["postprocess"]["params"]["high"] == 0.7
    assert scaler_json["type"] == "standard_scaler"
    assert scaler_json["feature_dim"] == 4
    assert len(scaler_json["mean"]) == 4
    assert len(scaler_json["scale"]) == 4


def test_export_model_artifact_requires_overwrite_for_existing_version(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    run_id = "prod_yolon960_fps5_seq20_mirror_v1"
    run_dir = data_root / "runs" / run_id
    dataset_dir = data_root / "datasets" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    (run_dir / "config.json").write_text(json.dumps({"run_id": run_id, "fps": 5.0, "dataset": {"seq_len_seconds": 20, "overlap_seconds": 10}, "train": {"segment_eval": {}}}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"epoch": 1, "balanced_accuracy": 0.9, "val_loss": 0.2}) + "\n", encoding="utf-8")
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps({"feature_set": "v1", "splits": {"train": [], "val": [], "test": []}, "videos": {}, "config": {"target_fps": 5}}), encoding="utf-8")

    scaler = StandardScaler()
    scaler.fit(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32))
    joblib.dump(scaler, run_dir / "scaler.joblib")

    model = TennisPointLSTM(input_size=2, return_logits=True)
    torch.save({"epoch": 1, "model_state_dict": model.state_dict(), "metrics": {"balanced_accuracy": 0.9, "val_loss": 0.2}}, run_dir / "checkpoints" / "best.pth")

    monkeypatch.setattr("training.pipeline.torch.onnx.export", lambda model, args, f, **kwargs: Path(f).write_bytes(b"fake-onnx"))

    export_model_artifact(
        data_root=data_root,
        run_id=run_id,
        version="rallyclip_v0.2.0",
        output_root=tmp_path / "models",
    )
    with pytest.raises(FileExistsError):
        export_model_artifact(
            data_root=data_root,
            run_id=run_id,
            version="rallyclip_v0.2.0",
            output_root=tmp_path / "models",
        )


def test_export_model_artifact_rejects_scaler_feature_dim_mismatch(tmp_path):
    data_root = tmp_path / "data"
    run_id = "prod_yolon960_fps5_seq20_mirror_v1"
    run_dir = data_root / "runs" / run_id
    dataset_dir = data_root / "datasets" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    (run_dir / "config.json").write_text(json.dumps({"run_id": run_id, "fps": 5.0, "dataset": {"seq_len_seconds": 20, "overlap_seconds": 10}, "train": {"segment_eval": {}}}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"epoch": 1, "balanced_accuracy": 0.9, "val_loss": 0.2}) + "\n", encoding="utf-8")
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps({"feature_set": "v1", "splits": {"train": [], "val": [], "test": []}, "videos": {}, "config": {"target_fps": 5}}), encoding="utf-8")

    scaler = StandardScaler()
    scaler.fit(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32))
    joblib.dump(scaler, run_dir / "scaler.joblib")

    model = TennisPointLSTM(input_size=3, return_logits=True)
    torch.save({"epoch": 1, "model_state_dict": model.state_dict(), "metrics": {"balanced_accuracy": 0.9, "val_loss": 0.2}}, run_dir / "checkpoints" / "best.pth")

    with pytest.raises(ValueError, match="feature dimension"):
        export_model_artifact(
            data_root=data_root,
            run_id=run_id,
            version="rallyclip_v0.2.0",
            output_root=tmp_path / "models",
        )


def _build_cli_args(**overrides) -> argparse.Namespace:
    defaults = {
        "config": None,
        "video": "tests/fixtures/input.mp4",
        "output_dir": None,
        "output_name": None,
        "csv_output_dir": None,
        "artifact_dir": None,
        "manifest_path": None,
        "model_path": None,
        "scaler_path": None,
        "yolo_size": None,
        "yolo_device": None,
        "fps": None,
        "seq_len": None,
        "overlap": None,
        "sigma": None,
        "low": None,
        "high": None,
        "min_dur_sec": None,
        "conf": None,
        "start_time": None,
        "duration": None,
        "write_csv": False,
        "segment_video": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_artifact_dir(base_dir: Path, *, manifest_overrides: dict | None = None) -> Path:
    artifact_dir = base_dir / "models" / "rallyclip_v0.3.0"
    artifact_dir.mkdir(parents=True)
    manifest = {
        "manifest_version": 1,
        "artifact": {"model_version": "rallyclip_v0.3.0"},
        "files": {"model_file": "model.onnx", "scaler_file": "scaler.json"},
        "feature_pipeline": {"feature_set": "v1", "feature_dim": 2},
        "inference": {
            "fps": 5.0,
            "seq_len_seconds": 20,
            "overlap_seconds": 10,
            "seq_len_frames": 100,
            "overlap_frames": 50,
        },
        "postprocess": {
            "method": "hysteresis",
            "params": {"low": 0.45, "high": 0.7, "sigma": 1.0, "min_dur_sec": 1.0},
        },
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (artifact_dir / "model.onnx").write_bytes(b"fake-onnx")
    (artifact_dir / "scaler.json").write_text(
        json.dumps(
            {
                "type": "standard_scaler",
                "feature_dim": 2,
                "mean": [0.0, 1.0],
                "scale": [2.0, 4.0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_dir


def test_build_run_config_loads_artifact_directory_defaults(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path)
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"")

    cfg = build_run_config(_build_cli_args(video=str(video_path), artifact_dir=str(artifact_dir)))

    assert cfg.inference_backend == "onnx"
    assert cfg.model_path == artifact_dir / "model.onnx"
    assert cfg.scaler_json_path == artifact_dir / "scaler.json"
    assert cfg.manifest_path == artifact_dir / "manifest.json"
    assert cfg.feature_set == "v1"
    assert cfg.feature_dim == 2
    assert cfg.fps == 5.0
    assert cfg.seq_len == 100
    assert cfg.overlap == 50
    assert cfg.postprocess_method == "hysteresis"
    assert cfg.high == 0.7


def test_build_run_config_loads_manifest_path_and_cli_overrides(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path)
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"")

    cfg = build_run_config(
        _build_cli_args(
            video=str(video_path),
            manifest_path=str(artifact_dir / "manifest.json"),
            fps=10.0,
            seq_len=40,
            overlap=20,
            sigma=2.0,
            low=0.3,
            high=0.8,
            min_dur_sec=0.5,
        )
    )

    assert cfg.inference_backend == "onnx"
    assert cfg.fps == 10.0
    assert cfg.seq_len == 40
    assert cfg.overlap == 20
    assert cfg.sigma == 2.0
    assert cfg.low == 0.3
    assert cfg.high == 0.8
    assert cfg.min_dur_sec == 0.5


def test_build_run_config_artifact_path_from_config_toml(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path)
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[run]
video_path = "{video_path}"
artifact_dir = "{artifact_dir}"
write_csv = false
segment_video = false
""".strip(),
        encoding="utf-8",
    )

    cfg = build_run_config(_build_cli_args(config=str(config_path), video=None))

    assert cfg.inference_backend == "onnx"
    assert cfg.model_path == artifact_dir / "model.onnx"


def test_build_run_config_prefers_artifact_over_legacy_paths(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path)
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"")
    legacy_model = tmp_path / "legacy.pth"
    legacy_scaler = tmp_path / "legacy.joblib"
    legacy_model.write_bytes(b"legacy")
    legacy_scaler.write_bytes(b"legacy")

    cfg = build_run_config(
        _build_cli_args(
            video=str(video_path),
            artifact_dir=str(artifact_dir),
            model_path=str(legacy_model),
            scaler_path=str(legacy_scaler),
        )
    )

    assert cfg.inference_backend == "onnx"
    assert cfg.model_path == artifact_dir / "model.onnx"


def test_build_run_config_legacy_paths_still_work(tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"")
    model_path = tmp_path / "model.pth"
    scaler_path = tmp_path / "scaler.joblib"
    model_path.write_bytes(b"legacy")
    scaler_path.write_bytes(b"legacy")

    cfg = build_run_config(
        _build_cli_args(
            video=str(video_path),
            model_path=str(model_path),
            scaler_path=str(scaler_path),
        )
    )

    assert cfg.inference_backend == "pytorch"
    assert cfg.model_path == model_path
    assert cfg.scaler_path == scaler_path
    assert cfg.postprocess_method == "hysteresis"


def test_build_run_config_rejects_missing_artifact_manifest(tmp_path):
    artifact_dir = tmp_path / "models" / "rallyclip_v0.3.0"
    artifact_dir.mkdir(parents=True)
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        build_run_config(_build_cli_args(video=str(video_path), artifact_dir=str(artifact_dir)))


def test_build_run_config_rejects_unsupported_manifest_version(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, manifest_overrides={"manifest_version": 2})
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported manifest_version"):
        build_run_config(_build_cli_args(video=str(video_path), artifact_dir=str(artifact_dir)))


def test_build_run_config_rejects_missing_artifact_files(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path)
    (artifact_dir / "model.onnx").unlink()
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="model.onnx"):
        build_run_config(_build_cli_args(video=str(video_path), artifact_dir=str(artifact_dir)))


def test_run_pipeline_uses_artifact_inference_path(tmp_path, monkeypatch):
    artifact_dir = _write_artifact_dir(tmp_path)
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"")
    output_dir = tmp_path / "out"
    csv_dir = tmp_path / "csv"
    features = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)
    captured = {"segments": None, "onnx_called": 0, "torch_called": 0}

    class FakePoseExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def extract_pose_data(self, **kwargs):
            raw_npz = tmp_path / "raw.npz"
            raw_npz.write_bytes(b"raw")
            return str(raw_npz)

    class FakePreprocessor:
        def __init__(self, *args, **kwargs):
            pass

        def preprocess_single_video(self, *args, **kwargs):
            return None

    class FakeFeatureEngineer:
        def create_features_from_preprocessed(self, _src, dest, overwrite=True):
            np.savez(dest, features=features)

    monkeypatch.setattr(cli_main, "PoseExtractor", FakePoseExtractor)
    monkeypatch.setattr(cli_main, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(cli_main, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(cli_main, "create_onnx_session", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli_main,
        "run_windowed_inference_average_onnx",
        lambda *args, **kwargs: captured.__setitem__("onnx_called", captured["onnx_called"] + 1) or np.array([0.1, 0.9, 0.9, 0.1], dtype=np.float32),
    )
    monkeypatch.setattr(
        cli_main,
        "load_model_from_checkpoint",
        lambda *args, **kwargs: captured.__setitem__("torch_called", captured["torch_called"] + 1) or (object(), "cpu"),
    )
    monkeypatch.setattr(cli_main, "write_segments_csv", lambda segments, *_args, **_kwargs: captured.__setitem__("segments", segments))
    monkeypatch.setattr(cli_main, "segment_video", lambda *_args, **_kwargs: None)

    cfg = build_run_config(
        _build_cli_args(
            video=str(video_path),
            output_dir=str(output_dir),
            csv_output_dir=str(csv_dir),
            artifact_dir=str(artifact_dir),
                write_csv=True,
                segment_video=False,
                seq_len=4,
                overlap=2,
                sigma=0.0,
                min_dur_sec=0.0,
            )
        )

    assert run_pipeline(cfg) == 0
    assert captured["onnx_called"] == 1
    assert captured["torch_called"] == 0
    assert captured["segments"] == [(1, 3)]


def test_run_pipeline_uses_legacy_pytorch_inference_path(tmp_path, monkeypatch):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"")
    model_path = tmp_path / "model.pth"
    scaler_path = tmp_path / "scaler.joblib"
    model_path.write_bytes(b"legacy")
    features = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)
    scaler = StandardScaler().fit(features)
    joblib.dump(scaler, scaler_path)
    captured = {"segments": None, "onnx_called": 0, "torch_called": 0}

    class FakePoseExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def extract_pose_data(self, **kwargs):
            raw_npz = tmp_path / "raw.npz"
            raw_npz.write_bytes(b"raw")
            return str(raw_npz)

    class FakePreprocessor:
        def __init__(self, *args, **kwargs):
            pass

        def preprocess_single_video(self, *args, **kwargs):
            return None

    class FakeFeatureEngineer:
        def create_features_from_preprocessed(self, _src, dest, overwrite=True):
            np.savez(dest, features=features)

    monkeypatch.setattr(cli_main, "PoseExtractor", FakePoseExtractor)
    monkeypatch.setattr(cli_main, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(cli_main, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(
        cli_main,
        "create_onnx_session",
        lambda *args, **kwargs: captured.__setitem__("onnx_called", captured["onnx_called"] + 1) or object(),
    )
    monkeypatch.setattr(
        cli_main,
        "load_model_from_checkpoint",
        lambda *args, **kwargs: captured.__setitem__("torch_called", captured["torch_called"] + 1) or (object(), "cpu"),
    )
    monkeypatch.setattr(
        cli_main,
        "run_windowed_inference_average",
        lambda *args, **kwargs: np.array([0.1, 0.9, 0.9, 0.1], dtype=np.float32),
    )
    monkeypatch.setattr(cli_main, "write_segments_csv", lambda segments, *_args, **_kwargs: captured.__setitem__("segments", segments))
    monkeypatch.setattr(cli_main, "segment_video", lambda *_args, **_kwargs: None)

    cfg = build_run_config(
        _build_cli_args(
            video=str(video_path),
            model_path=str(model_path),
            scaler_path=str(scaler_path),
                write_csv=True,
                segment_video=False,
                seq_len=4,
                overlap=2,
                sigma=0.0,
                min_dur_sec=0.0,
            )
        )

    assert run_pipeline(cfg) == 0
    assert captured["onnx_called"] == 0
    assert captured["torch_called"] == 1
    assert captured["segments"] == [(1, 3)]
