"""GUI defaults follow the shipped manifest (fast; the e2e contract test is deselected by default)."""
from __future__ import annotations

import json

import pytest

from runtime.defaults import build_gui_defaults, resolve_default_artifacts

_MODEL, _SCALER, MANIFEST = resolve_default_artifacts(None)

pytestmark = pytest.mark.skipif(not MANIFEST.is_file(), reason=f"shipped manifest absent ({MANIFEST})")


def test_gui_defaults_feature_set_matches_the_shipped_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert build_gui_defaults()["feature_set"] == manifest["feature_pipeline"]["feature_set"]
