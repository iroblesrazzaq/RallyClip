"""Golden CLI parity test.

Runs the real shipped pipeline (CPU, no fakes) on a small committed fixture
clip and asserts the segments CSV matches the committed golden byte-for-byte.
This locks in the analysis output of the frame_probability_hysteresis
pipeline across refactors.

Regenerate the golden (only after a deliberate model/pipeline change):

    PYTHONPATH=src python3 -m cli.main \
      --video tests/fixtures/golden_cli/clip.mp4 \
      --output-dir /tmp/golden --csv-output-dir /tmp/golden \
      --output-name golden --write-csv --no-segment-video --yolo-device cpu
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")
pytest.importorskip("onnxruntime")
pytest.importorskip("av")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "golden_cli"
CLIP = FIXTURE_DIR / "clip.mp4"
GOLDEN_CSV = FIXTURE_DIR / "golden_segments.csv"
ARTIFACT_DIR = REPO_ROOT / "models" / "rallyclip_v0.3.1"

pytestmark = pytest.mark.skipif(
    not (ARTIFACT_DIR / "model.onnx").is_file(),
    reason=f"model artifacts absent ({ARTIFACT_DIR})",
)


def test_cli_segments_csv_matches_golden(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.main",
            "--video",
            str(CLIP),
            "--output-dir",
            str(tmp_path),
            "--csv-output-dir",
            str(tmp_path),
            "--output-name",
            "golden",
            "--write-csv",
            "--no-segment-video",
            "--yolo-device",
            "cpu",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"
    produced = tmp_path / "golden_segments.csv"
    assert produced.is_file(), "CLI did not write the segments CSV"
    assert produced.read_text(encoding="utf-8") == GOLDEN_CSV.read_text(encoding="utf-8")
