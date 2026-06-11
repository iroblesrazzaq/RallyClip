"""End-to-end quality checks for the RallyClip inference pipeline.

These tests use a small, fixed subset of the training/validation annotations. That is
intentional: this suite gates pipeline health and packaging regressions, not ML
generalization. Real model selection should use the separate training eval workflow.

Examples:
  RALLYCLIP_EVAL_VIDEO_DIR=/path/to/raw_videos pytest tests/test_quality_e2e.py -q
  RALLYCLIP_EVAL_VIDEO_DIR=/path/to/raw_videos RALLYCLIP_EVAL_ARTIFACT_DIR=models/new_model pytest tests/test_quality_e2e.py -q
  RALLYCLIP_EVAL_VIDEO_DIR=/path/to/raw_videos RALLYCLIP_RELEASE_BIN=/path/to/RallyClip pytest tests/test_quality_e2e.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers.quality_fixtures import (
    FIX_DIR,
    load_manifest,
    resolve_artifact_dir,
    resolve_release_bin,
    resolve_video_dir,
)
from training.eval.quality_harness import (
    QualityEntry,
    RunnerConfig,
    run_pipeline_to_csv,
    run_quality_eval,
)

VIDEO_DIR = resolve_video_dir()
RELEASE_BIN = resolve_release_bin()
ARTIFACT_DIR = resolve_artifact_dir()
MANIFEST = load_manifest()
BASELINE_PATH = FIX_DIR / "baseline.json"
ARTIFACT_ARGS = ("--artifact-dir", str(ARTIFACT_DIR)) if ARTIFACT_DIR is not None else ()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(VIDEO_DIR is None, reason="set RALLYCLIP_EVAL_VIDEO_DIR to the source videos"),
]


def test_quality_smoke_writes_csv(tmp_path):
    entry_data = dict(MANIFEST[0])
    entry_data["duration_s"] = 60
    entry = QualityEntry(**entry_data)

    csv_path = run_pipeline_to_csv(
        video_path=VIDEO_DIR / entry.video,
        entry=entry,
        csv_output_dir=tmp_path,
        runner=RunnerConfig(mode="dev", extra_args=ARTIFACT_ARGS),
    )

    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "start_time,end_time"
    assert len(rows) > 1, f"{entry.id}: smoke run wrote no predicted segments"


@pytest.fixture(scope="module", params=["dev", "release"])
def quality_report(request, tmp_path_factory):
    mode = request.param
    if mode == "release" and RELEASE_BIN is None:
        pytest.skip("set RALLYCLIP_RELEASE_BIN to run release-binary quality checks")
    runner = (
        RunnerConfig(mode="release", release_bin=RELEASE_BIN, extra_args=ARTIFACT_ARGS)
        if mode == "release"
        else RunnerConfig(mode="dev", extra_args=ARTIFACT_ARGS)
    )
    output_path = tmp_path_factory.mktemp(f"quality_{mode}") / "report.json"
    return run_quality_eval(
        manifest_path=FIX_DIR / "manifest.json",
        video_dir=VIDEO_DIR,
        gt_dir=FIX_DIR / "gt",
        output_path=output_path,
        runner=runner,
    )


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE_PATH.is_file():
        pytest.skip("baseline fixture missing; run scripts/eval_quality.py and commit baseline.json")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_quality_good_rate_floor(quality_report, baseline):
    floor = max(0.0, float(baseline["pooled"]["well_classified_rate"]) - 0.15)
    assert quality_report["pooled"]["well_classified_rate"] >= floor


def test_quality_missed_ceiling(quality_report, baseline):
    ceiling = int(baseline["pooled"]["missed_points"]) + 2
    assert quality_report["pooled"]["missed_points"] <= ceiling


def test_quality_false_detected_ceiling(quality_report, baseline):
    ceiling = int(baseline["pooled"]["false_detected_points"]) + 2
    assert quality_report["pooled"]["false_detected_points"] <= ceiling


def test_quality_every_video_alive(quality_report):
    dead = [
        report["id"]
        for report in quality_report["per_video"]
        if int(report["point_metrics"]["well_classified_points"]) < 1
    ]
    assert dead == []
