"""GUI defaults follow the shipped manifest (fast; the e2e contract test is deselected by default)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.defaults import DEFAULT_ARTIFACT_DIR, build_gui_defaults

MANIFEST = Path(__file__).resolve().parents[1] / DEFAULT_ARTIFACT_DIR / "manifest.json"


def test_gui_defaults_match_the_shipped_manifest():
    try:
        defaults = build_gui_defaults()
    except FileNotFoundError as exc:  # model.onnx comes from scripts/fetch_artifact.py
        pytest.skip(f"artifacts not fetched: {exc}")
    # The tracked manifest, read raw, so a mapping bug in manifest_values shows too.
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    features, inference = manifest["feature_pipeline"], manifest["inference"]
    assert defaults["feature_set"] == features["feature_set"]
    assert defaults["fps"] == float(features["target_fps"])
    assert defaults["imgsz"] == int(float(features["imgsz"]))
    assert defaults["conf"] == float(features["conf"])
    assert (defaults["screen_width"], defaults["screen_height"]) == (features["screen_width"], features["screen_height"])
    assert defaults["seq_len"] == int(inference["seq_len_frames"])
    assert defaults["pipeline_id"] == manifest["pipeline"]["id"]
