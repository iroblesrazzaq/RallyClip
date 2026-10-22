from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from training.io.fingerprint import (
    build_features_fingerprint,
    fingerprint_dict,
    read_h5_fingerprint,
    tmp_path_for,
)


def test_fingerprint_changes_with_annotation(tmp_path):
    ann_a = tmp_path / "a.json"
    ann_b = tmp_path / "b.json"
    raw = tmp_path / "raw.h5"
    ann_a.write_text('{"segments":[]}', encoding="utf-8")
    ann_b.write_text('{"segments":[{"start_time":0,"end_time":1}]}', encoding="utf-8")
    with h5py.File(raw, "w") as h5f:
        h5f.create_dataset("x", data=np.arange(3))

    from training.io.fingerprint import build_preprocess_fingerprint

    fa = fingerprint_dict(
        build_preprocess_fingerprint(
            annotations_path=ann_a,
            raw_h5_path=raw,
            target_fps=5,
            court_model_path="m",
            court_target_time=60,
        )
    )
    fb = fingerprint_dict(
        build_preprocess_fingerprint(
            annotations_path=ann_b,
            raw_h5_path=raw,
            target_fps=5,
            court_model_path="m",
            court_target_time=60,
        )
    )
    assert fa != fb


def test_atomic_tmp_path_and_fingerprint_attr(tmp_path):
    out = tmp_path / "feat.h5"
    tmp = tmp_path_for(out)
    assert tmp != out
    with h5py.File(tmp, "w") as h5f:
        h5f.create_dataset("features", data=np.zeros((2, 3), dtype=np.float32))
        h5f.attrs["fingerprint"] = "abc"
    tmp.replace(out)
    assert read_h5_fingerprint(out) == "abc"

    pre = tmp_path / "pre.h5"
    with h5py.File(pre, "w") as h5f:
        h5f.create_dataset("x", data=[1, 2])
    payload = build_features_fingerprint(preproc_h5_path=pre, feature_set="v1", target_fps=5)
    assert fingerprint_dict(payload)
