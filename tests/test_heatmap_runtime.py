"""HeatmapHybridModel wiring + heatmap pipeline resolution.

Covers the engine-side glue that the pure-decode unit tests (test_heatmap_decode)
do not: manifest -> PipelineSpec resolution, dispatch to HeatmapHybridModel, and
the full model.run() producing float-second intervals + a segments CSV. The pose
feature stream and the ONNX runner are faked so this needs no video or model
artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rallyclip_core.contracts import RunRequest, RuntimeDeps
from rallyclip_core.pipelines import (
    FRAME_STARTEND_HEATMAP,
    resolve_pipeline_spec,
)
from rallyclip_engine import models as engine_models
from rallyclip_engine.models import HeatmapHybridModel, build_analysis_model

FPS = 5.0


# --------------------------------------------------------- pipeline resolution


def _write_heatmap_manifest(tmp_path: Path) -> tuple[Path, Path]:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"not a real onnx")  # only the suffix matters here
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "pipeline": {"id": "frame_startend_heatmap"},
                "feature_pipeline": {"feature_set": "v1", "target_fps": FPS},
                "inference": {"seq_len_frames": 100, "overlap_frames": 50},
                "model": {"outputs": ["pointness_logit", "start_heatmap_logit", "end_heatmap_logit"]},
                "postprocess": {
                    "method": "heatmap_hybrid",
                    "params": {
                        "mode": "hybrid",
                        "threshold": 0.5,
                        "sigma_frames": 2.5,
                        "min_duration_sec": 0.3,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return model_path, manifest_path


def test_resolve_pipeline_spec_from_heatmap_manifest(tmp_path):
    model_path, manifest_path = _write_heatmap_manifest(tmp_path)
    spec = resolve_pipeline_spec(model_path, manifest_path)
    assert spec.pipeline_id == FRAME_STARTEND_HEATMAP
    assert spec.model_output == "pointness_start_end_heatmap"
    assert spec.decode_method == "heatmap_hybrid"
    assert spec.feature_set == "v1"
    # decode knobs flow through from postprocess.params
    assert spec.params["mode"] == "hybrid"
    assert spec.params["sigma_frames"] == 2.5
    assert spec.params["min_duration_sec"] == 0.3


def test_resolve_pipeline_spec_via_method_only(tmp_path):
    # No explicit pipeline.id -> the postprocess.method selects the heatmap pipeline.
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "feature_pipeline": {"feature_set": "v1", "target_fps": FPS},
                "inference": {"seq_len_frames": 100, "overlap_frames": 50},
                "postprocess": {"method": "heatmap_hybrid", "params": {"threshold": 0.4}},
            }
        ),
        encoding="utf-8",
    )
    spec = resolve_pipeline_spec(model_path, manifest_path)
    assert spec.pipeline_id == FRAME_STARTEND_HEATMAP
    assert spec.params["threshold"] == 0.4


def test_heatmap_override_incompatible_with_classic_manifest(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "feature_pipeline": {"feature_set": "v1", "target_fps": FPS},
                "inference": {"seq_len_frames": 100, "overlap_frames": 50},
                "postprocess": {"method": "hysteresis", "params": {"sigma": 1.5, "low": 0.5, "high": 0.65}},
            }
        ),
        encoding="utf-8",
    )
    from rallyclip_core.contracts import UnsupportedPipelineError

    with pytest.raises(UnsupportedPipelineError):
        resolve_pipeline_spec(model_path, manifest_path, override_pipeline_id=FRAME_STARTEND_HEATMAP)


# ----------------------------------------------------------- full model.run()


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=np.float32)


def _make_request(tmp_path: Path, *, write_csv=True, segment_video=False) -> RunRequest:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    return RunRequest(
        video_path=tmp_path / "match.mp4",
        output_dir=tmp_path / "out",
        output_name="match",
        csv_output_dir=tmp_path / "out",
        write_csv=write_csv,
        segment_video=segment_video,
        yolo_weights="yolov8n-pose-960-dynamic.onnx",
        yolo_device=None,
        model_path=model_path,
        scaler_path=tmp_path / "scaler.json",
        fps=FPS,
        seq_len=10,
        imgsz=960,
        conf=0.25,
        feature_set="v1",
        screen_width=1280,
        screen_height=720,
        overlap=5,
        sigma=0.0,
        low=0.0,
        high=0.0,
        min_dur_sec=0.0,
        pipeline_id=FRAME_STARTEND_HEATMAP,
    )


def _fake_deps(segment_calls=None) -> RuntimeDeps:
    def fake_segment_video(video, intervals, out):
        if segment_calls is not None:
            segment_calls.append((video, intervals, out))

    return RuntimeDeps(
        np=np,
        PoseExtractor=None,
        DataPreprocessor=None,
        FeatureEngineer=None,
        load_scaler_asset=lambda path: _IdentityScaler(),
        load_model_from_checkpoint=None,
        run_windowed_inference_average_onnx_stream=None,
        run_windowed_inference_average_torch_stream=None,
        gaussian_filter1d=None,
        hysteresis_threshold=None,
        extract_segments_from_binary=None,
        write_segments_csv=lambda *a, **k: None,
        segment_video=fake_segment_video,
    )


def _spec(tmp_path, params=None):
    _model, manifest = _write_heatmap_manifest(tmp_path)
    spec = resolve_pipeline_spec(tmp_path / "model.onnx", manifest)
    if params is not None:
        # rebuild with overridden decode params
        from dataclasses import replace

        spec = replace(spec, params=params)
    return spec


def _install_fakes(monkeypatch, tracks):
    # bypass the heavy pose->feature stream
    monkeypatch.setattr(
        engine_models, "build_feature_stream",
        lambda request, deps, pcb, ccb: [(np.zeros(4, dtype=np.float32), 0) for _ in range(len(tracks[0]))],
    )
    # the ONNX runner returns our synthetic (3, N) tracks regardless of inputs
    import infer

    monkeypatch.setattr(
        infer, "run_multitrack_windowed_inference_onnx_stream",
        lambda *a, **k: np.asarray(tracks, dtype=np.float32),
    )


def test_heatmap_model_run_produces_intervals_and_csv(tmp_path, monkeypatch):
    n = 20
    point = np.full(n, 0.1, dtype=np.float32)
    point[5:11] = 0.9  # run frames 5..10
    start = np.zeros(n, dtype=np.float32)
    start[5] = 0.9
    end = np.zeros(n, dtype=np.float32)
    end[10] = 0.9
    _install_fakes(monkeypatch, np.stack([point, start, end], axis=0))

    request = _make_request(tmp_path)
    spec = _spec(tmp_path)
    model = build_analysis_model(request, spec, _fake_deps())
    assert isinstance(model, HeatmapHybridModel)

    result = model.run()

    assert result.intervals_sec == [pytest.approx((5 / FPS, 10 / FPS))]
    assert result.frame_segments == [(5, 10)]  # rounded, for RunResult consumers
    assert result.diagnostics["pipeline_id"] == FRAME_STARTEND_HEATMAP
    # CSV written in the segments.csv contract, sub-frame precision preserved
    csv_text = (tmp_path / "out" / "match_segments.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == "start_time,end_time"
    assert csv_text.splitlines()[1] == "1.000,2.000"


def test_heatmap_model_segments_video_from_intervals(tmp_path, monkeypatch):
    n = 16
    point = np.full(n, 0.1, dtype=np.float32)
    point[4:9] = 0.9
    start = np.zeros(n, dtype=np.float32)
    start[4] = 0.9
    end = np.zeros(n, dtype=np.float32)
    end[8] = 0.9
    _install_fakes(monkeypatch, np.stack([point, start, end], axis=0))

    calls = []
    request = _make_request(tmp_path, write_csv=False, segment_video=True)
    model = build_analysis_model(request, _spec(tmp_path), _fake_deps(segment_calls=calls))
    result = model.run()

    assert len(calls) == 1
    _video, intervals, _out = calls[0]
    assert intervals == result.intervals_sec == [pytest.approx((4 / FPS, 8 / FPS))]


def test_heatmap_model_rejects_non_onnx(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, np.zeros((3, 12), dtype=np.float32))
    from rallyclip_core.contracts import UnsupportedPipelineError

    request = _make_request(tmp_path)
    object.__setattr__(request, "model_path", tmp_path / "model.pth")  # frozen dataclass
    model = build_analysis_model(request, _spec(tmp_path), _fake_deps())
    with pytest.raises(UnsupportedPipelineError, match="ONNX models only"):
        model.run()
