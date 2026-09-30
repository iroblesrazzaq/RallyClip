from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from training.io.annotations import csv_to_json
from training.preprocess.preprocessor import _build_targets

REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_DATA_LABELS = REPO_ROOT.parent / "new_data" / "labels"
D5 = "d5ac3da3cd8a8f525a58bbfe375aebc735d4887ac6de1477cc33546e092cb135"


def test_d5_csv_roundtrip_targets():
    csv_path = NEW_DATA_LABELS / f"{D5}.mp4.csv"
    meta_path = NEW_DATA_LABELS / f"{D5}.labels_meta.json"
    if not csv_path.exists():
        pytest.skip("new_data/labels fixture not present")

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    metadata = {}
    if meta.get("ignore_before_s") is not None:
        metadata["ignore_before_s"] = float(meta["ignore_before_s"])

    data = csv_to_json(csv_path, Path(f"{D5}.mp4"), metadata=metadata)
    assert len(data["segments"]) == 38

    duration = float(meta.get("video_duration_s", 1405.0))
    timestamps = np.arange(0.0, duration, 0.2, dtype=np.float64)
    targets = _build_targets(timestamps, data)
    in_point = float(np.mean(targets == 1))
    # LABELS_REPORT: ~39.9% in-point
    assert 0.30 < in_point < 0.50


def test_convert_new_data_labels_script(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    csv_path = labels_dir / "abc.mp4.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_time", "end_time"])
        writer.writerow(["1.0", "2.0"])
    meta_path = labels_dir / "abc.labels_meta.json"
    meta_path.write_text('{"ignore_before_s": 0.5, "formula_version": "test"}', encoding="utf-8")

    out_dir = tmp_path / "annotations"
    script = REPO_ROOT / "scripts" / "convert_new_data_labels.py"
    spec = importlib.util.spec_from_file_location("convert_new_data_labels", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    data = mod.convert_one(csv_path, meta_path, out_dir / "abc.mp4.json")
    assert data["metadata"]["ignore_before_s"] == 0.5
    assert (out_dir / "abc.mp4.json").exists()
