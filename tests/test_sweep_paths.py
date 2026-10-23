"""Postprocess sweep: finding cached datasets and a run's feature root."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from training import pipeline  # noqa: E402
from training.paths import pose_features_dir, pose_raw_dir  # noqa: E402


def _raw(root, tag):
    d = pose_raw_dir(root) / f"yolo={tag}" / "conf=0p25" / "imgsz=960"
    d.mkdir(parents=True)
    (d / "m__start0__durfull.h5").write_bytes(b"")


def test_manifest_tag_dir_maps_back_to_the_configured_manifest(tmp_path, monkeypatch):
    _raw(tmp_path, "yolov8n-960@4a3fe0de")
    _raw(tmp_path, "yolov8n-960@deadbeef")
    _raw(tmp_path, "yolov8n-pose.pt")
    monkeypatch.setattr(pipeline, "yolo_conf_imgsz_tags", lambda cfg: ("yolov8n-960@4a3fe0de", "0p25", 960))
    config = {"yolo": {"model": "models/pose/yolov8n/manifest.json"}}

    specs = pipeline._discover_raw_datasets(tmp_path, {"m"}, True, config=config)
    assert sorted(s["yolo"]["model"] for s in specs) == ["models/pose/yolov8n/manifest.json", "yolov8n-pose.pt"]


def test_old_run_without_saved_yolo_uses_the_run_id_imgsz(tmp_path):
    root = pipeline._run_feature_root(tmp_path, tmp_path / "cv_yolon1440_f0", {}, 5.0)
    assert root == pose_features_dir(tmp_path) / "yolo=yolov8n-pose.pt" / "conf=0p25" / "imgsz=1440" / "fps=5.0"


def test_run_with_saved_yolo_uses_it(tmp_path):
    cfg = {"yolo": {"model": "yolov8n-pose.pt", "conf": 0.25, "imgsz": 960}, "preprocess": {"target_fps": 5.0}}
    root = pipeline._run_feature_root(tmp_path, tmp_path / "cv_yolon1440_f0", cfg, 5.0)
    assert "imgsz=960" in str(root)


def test_unset_yolo_fields_are_not_saved_as_null(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, "train_loop", lambda dataset_dir, run_dir, cfg: seen.update(cfg))
    pipeline._run_train({"data_root": str(tmp_path), "yolo": {"model": "yolov8n-pose.pt"}})
    assert seen["yolo"] == {"model": "yolov8n-pose.pt"}
    # The sweep reads it back with extraction's default imgsz instead of int(None).
    root = pipeline._run_feature_root(tmp_path, tmp_path / "run", seen, 5.0)
    assert "imgsz=1920" in str(root)


class _Captured(Exception):
    pass


@pytest.mark.parametrize("config, force", [({}, False), ({"overwrite_all": True}, True), ({"court": {"force": True}}, True)])
def test_overwrite_all_rebuilds_court_masks(tmp_path, monkeypatch, config, force):
    def capture(**cfg):
        raise _Captured(cfg["court_force"])

    monkeypatch.setattr(pipeline, "resolve_videos", lambda *a: ["m.mp4"])
    monkeypatch.setattr(pipeline, "PreprocessConfig", capture)
    with pytest.raises(_Captured) as info:
        pipeline._run_preprocess({"data_root": str(tmp_path), "yolo": {"model": "yolov8n-pose.pt"}, **config})
    assert info.value.args[0] is force


def test_loso_folds_build_only_from_the_checked_pool(tmp_path, monkeypatch):
    runs = []
    monkeypatch.setattr(pipeline, "run_pipeline", lambda cfg, steps_override: runs.append((cfg, steps_override)))
    monkeypatch.setattr(pipeline, "_resolve_sweep_datasets", lambda *a: [{"name": "d"}])
    monkeypatch.setattr(pipeline, "_is_fold_complete", lambda **kw: False)
    pipeline.run_sweep({
        "data_root": str(tmp_path),
        "dataset": {"mode": "annotated", "split": {"test_videos": ["frozen.mp4"]}},
        "sweep": {"enabled": True, "fps_values": [5], "seq_len_seconds": [20],
                  "loso": {"videos": ["a.mp4", "b.mp4"]}},
    })

    folds = [cfg for cfg, steps in runs if "dataset" in steps]
    assert [f["dataset"]["split"]["test_videos"] for f in folds] == [["a.mp4"], ["b.mp4"]]
    assert all(f["videos"] == ["a.mp4", "b.mp4"] for f in folds)


def test_flips_train_in_every_fold_but_are_never_held_out(tmp_path, monkeypatch):
    runs = []
    monkeypatch.setattr(pipeline, "run_pipeline", lambda cfg, steps_override: runs.append((cfg, steps_override)))
    monkeypatch.setattr(pipeline, "_resolve_sweep_datasets", lambda *a: [{"name": "d"}])
    monkeypatch.setattr(pipeline, "_is_fold_complete", lambda **kw: False)
    pool = ["a.mp4", "a__flip_h.mp4", "b.mp4", "b__flip_h.mp4"]
    pipeline.run_sweep({
        "data_root": str(tmp_path),
        "sweep": {"enabled": True, "fps_values": [5], "seq_len_seconds": [20], "loso": {"videos": pool}},
    })

    folds = [cfg for cfg, steps in runs if "dataset" in steps]
    assert [f["dataset"]["split"]["test_videos"] for f in folds] == [["a.mp4"], ["b.mp4"]]
    assert [f["run_id"].rsplit("_", 2)[-2:] for f in folds] == [["fold01", "a"], ["fold02", "b"]]
    assert all(f["videos"] == pool for f in folds)


@pytest.mark.parametrize("head, stopped", [("classic", False), ("e2e_heatmap", True)])
def test_restart_replays_the_stopping_rule_of_the_head(tmp_path, head, stopped):
    import json

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"head": head, "early_stopping_patience": 2}))
    # balanced_accuracy keeps rising, the heatmap's acceptable share is flat.
    records = [
        {"epoch": e, "balanced_accuracy": 0.5 + 0.1 * e, "val_loss": 1.0, "share_good": 0.4, "share_decent": 0.1}
        for e in range(1, 4)
    ]
    assert pipeline._stopped_by_early_stopping(run_dir, records) is stopped
