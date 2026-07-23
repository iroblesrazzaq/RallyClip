from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from runtime.assets import manifest_values

from .contracts import PipelineSpec, UnsupportedPipelineError

FRAME_PROBABILITY_HYSTERESIS = "frame_probability_hysteresis"
START_END_ATTENTION_VOTING = "start_end_attention_voting"
FRAME_STARTEND_HEATMAP = "frame_startend_heatmap"

# postprocess.method strings a manifest may use to select the heatmap pipeline
# without spelling out pipeline.id explicitly.
_HEATMAP_METHODS = {"heatmap", "heatmap_hybrid", "heatmap_peakpair", "gaussian_heatmap"}


def pipeline_id_from_manifest_values(values: Dict[str, Any]) -> str:
    explicit = values.get("pipeline_id") or values.get("pipeline")
    if isinstance(explicit, dict):
        explicit = explicit.get("id")
    if explicit:
        return str(explicit)

    method = str(values.get("postprocess_method") or values.get("postprocess") or "").strip().lower()
    if method in _HEATMAP_METHODS:
        return FRAME_STARTEND_HEATMAP
    if method == "hysteresis" or not method:
        return FRAME_PROBABILITY_HYSTERESIS
    return method


def resolve_pipeline_spec(
    model_path: Path,
    manifest_path: Optional[Path] = None,
    *,
    override_pipeline_id: Optional[str] = None,
) -> PipelineSpec:
    values = manifest_values(model_path, manifest_path)
    default_id = pipeline_id_from_manifest_values(values)
    pipeline_id = override_pipeline_id or default_id
    if pipeline_id not in {FRAME_PROBABILITY_HYSTERESIS, START_END_ATTENTION_VOTING, FRAME_STARTEND_HEATMAP}:
        raise UnsupportedPipelineError(f"Unsupported pipeline_id '{pipeline_id}'.")
    if pipeline_id == START_END_ATTENTION_VOTING and default_id != START_END_ATTENTION_VOTING:
        raise UnsupportedPipelineError(
            "Pipeline override 'start_end_attention_voting' is incompatible with this artifact. "
            "Use an artifact whose manifest declares start/end attention outputs."
        )
    if pipeline_id == FRAME_STARTEND_HEATMAP and default_id != FRAME_STARTEND_HEATMAP:
        raise UnsupportedPipelineError(
            "Pipeline override 'frame_startend_heatmap' is incompatible with this artifact. "
            "Use an artifact whose manifest declares pointness/start/end heatmap outputs."
        )
    if pipeline_id == FRAME_PROBABILITY_HYSTERESIS:
        return PipelineSpec(
            pipeline_id=pipeline_id,
            feature_set=str(values.get("feature_set") or "v1"),
            model_output="frame_probability",
            decode_method="gaussian_hysteresis",
            params={
                "sigma": values.get("sigma"),
                "low": values.get("low"),
                "high": values.get("high"),
                "min_dur_sec": values.get("min_dur_sec"),
            },
        )
    if pipeline_id == FRAME_STARTEND_HEATMAP:
        # Decode knobs come straight from the manifest's postprocess.params; the
        # runtime HeatmapDecodeConfig fills any it omits with its defaults.
        return PipelineSpec(
            pipeline_id=pipeline_id,
            feature_set=str(values.get("feature_set") or "v1"),
            model_output="pointness_start_end_heatmap",
            decode_method="heatmap_hybrid",
            params=dict(values.get("postprocess_params") or {}),
        )
    return PipelineSpec(
        pipeline_id=pipeline_id,
        feature_set=str(values.get("feature_set") or "v1"),
        model_output="start_end_attention",
        decode_method="start_end_voting",
        params={},
    )

