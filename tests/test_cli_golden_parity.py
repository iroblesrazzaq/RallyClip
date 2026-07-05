"""Golden CLI parity test.

Runs the real shipped pipeline (CPU, no fakes) on a small committed fixture
clip and asserts the segments CSV matches the committed golden: identical
segment count, boundaries within one 0.2s hysteresis hop (CPU inference
differs across platforms/BLAS backends by up to one frame-probability hop;
byte-exact only holds on the platform the golden was generated on). This
locks in the analysis output of the frame_probability_hysteresis pipeline
across refactors.

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

# No torch/ultralytics gate: the shipped pipeline is torch-free (pose runs on
# the bundled ONNX via extraction.yolo_onnx_runner).
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

    def rows(path: Path) -> tuple[str, list[tuple[float, float]]]:
        header, *body = path.read_text(encoding="utf-8").strip().splitlines()
        return header, [tuple(float(v) for v in line.split(",")) for line in body]

    got_header, got = rows(produced)
    want_header, want = rows(GOLDEN_CSV)
    assert got_header == want_header
    assert len(got) == len(want), f"segment count differs: {got} vs golden {want}"
    for (g_start, g_end), (w_start, w_end) in zip(got, want):
        assert abs(g_start - w_start) <= 0.25, f"{got} vs golden {want}"
        assert abs(g_end - w_end) <= 0.25, f"{got} vs golden {want}"
