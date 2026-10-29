"""The shipped court line model on a real clip (the golden CLI fixture).

detect_court hides every failure behind the default mask, so a broken or
mismatched court_lines.onnx would otherwise pass the gates.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("av")

from preprocessing.court_lines import detect_court, netline_geometry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIP = REPO_ROOT / "tests" / "fixtures" / "golden_cli" / "clip.mp4"
MODEL = REPO_ROOT / "models" / "rallyclip_v0.6.0" / "court_lines.onnx"

pytestmark = pytest.mark.skipif(not MODEL.is_file(), reason=f"court model absent ({MODEL})")


def test_court_model_finds_the_court_on_the_golden_clip():
    court = detect_court(str(CLIP), str(MODEL), ["CPUExecutionProvider"])
    assert court is not None, "court model failed (see the warning log)"
    assert court.frame_size == (1280, 720) and court.frames_used == 7
    geometry = court.geometry()
    assert geometry["ok"] and all(geometry["lines_ok"])
    mask = court.out_mask()
    assert mask.shape == (720, 1280)
    assert 0.1 < float((mask > 0).mean()) < 0.7
    assert netline_geometry(court) is not None
