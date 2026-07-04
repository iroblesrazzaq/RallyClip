from __future__ import annotations

import json
from pathlib import Path

import pytest

from rallyclip_core.contracts import UnsupportedPipelineError
from rallyclip_core.playback import (
    SourceTimelineScheduler,
    build_playback_manifest,
    playback_manifest_payload,
)
from rallyclip_core.pipelines import FRAME_PROBABILITY_HYSTERESIS, resolve_pipeline_spec
from rallyclip_engine.models import decode_start_end_votes


def test_current_manifest_resolves_to_hysteresis_pipeline():
    spec = resolve_pipeline_spec(Path("models/rallyclip_v0.3.1/model.onnx"))
    assert spec.pipeline_id == FRAME_PROBABILITY_HYSTERESIS
    assert spec.model_output == "frame_probability"
    assert spec.decode_method == "gaussian_hysteresis"


def test_unknown_pipeline_fails_before_model_execution(tmp_path):
    model = tmp_path / "model.onnx"
    manifest = tmp_path / "manifest.json"
    model.write_bytes(b"fake")
    manifest.write_text(json.dumps({"pipeline": {"id": "unknown_pipeline"}}), encoding="utf-8")

    with pytest.raises(UnsupportedPipelineError, match="unknown_pipeline"):
        resolve_pipeline_spec(model, manifest)


def test_start_end_pipeline_override_requires_compatible_artifact(tmp_path):
    model = tmp_path / "model.onnx"
    manifest = tmp_path / "manifest.json"
    model.write_bytes(b"fake")
    manifest.write_text(json.dumps({"postprocess": {"method": "hysteresis", "params": {}}}), encoding="utf-8")

    with pytest.raises(UnsupportedPipelineError, match="incompatible"):
        resolve_pipeline_spec(model, manifest, override_pipeline_id="start_end_attention_voting")


def test_start_end_vote_decoder_pairs_starts_and_ends_without_hysteresis():
    intervals = decode_start_end_votes(
        start_scores=[0.1, 0.9, 0.2, 0.8, 0.1, 0.1],
        end_scores=[0.1, 0.2, 0.95, 0.1, 0.8, 0.1],
        threshold=0.5,
        min_duration_frames=1,
    )
    assert intervals == [(1, 2), (3, 4)]


def test_playback_manifest_payload_is_source_time_contract():
    manifest = build_playback_manifest(
        source_duration_s=60.0,
        chunk_duration_s=8.0,
        point_intervals=[(25.0, 30.0), (10.0, 15.0)],
    )

    assert playback_manifest_payload(manifest) == {
        "source_duration_s": 60.0,
        "chunk_duration_s": 8.0,
        "segments": [{"start": 10.0, "end": 15.0}, {"start": 25.0, "end": 30.0}],
        "point_intervals": [{"start": 10.0, "end": 15.0}, {"start": 25.0, "end": 30.0}],
        "point_duration_s": 10.0,
    }


def test_source_timeline_scheduler_is_absolute_and_gap_aware():
    scheduler = SourceTimelineScheduler(
        [{"start": 10.0, "end": 15.0}, {"start": 25.0, "end": 30.0}, {"start": 40.0, "end": 45.0}],
        60.0,
    )

    assert scheduler.default_start_ms() == 10_000
    assert scheduler.seek(12_500).mode == "point"
    assert scheduler.next_start_after_active() == 25_000

    gap = scheduler.seek(20_000)
    assert gap.mode == "gap_bridge"
    assert gap.start_ms == 20_000
    assert gap.end_ms == 30_000
    assert scheduler.should_advance(25_000) is False
    assert scheduler.next_start_after_active() == 40_000

    forward = scheduler.seek(42_000)
    backward = scheduler.seek(20_000)
    assert forward.point_index == 2
    assert backward.point_index == 1


def test_source_timeline_scheduler_allows_tail_after_last_point():
    scheduler = SourceTimelineScheduler([(40.0, 45.0)], 60.0)
    point = scheduler.seek(42_000)

    assert point.mode == "point"
    assert scheduler.next_start_after_active() is None
    assert scheduler.tail_start_after_active() == 45_000
    assert scheduler.seek(45_000).mode == "tail"
