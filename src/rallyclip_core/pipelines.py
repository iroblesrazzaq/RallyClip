from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from runtime.assets import manifest_values

from .contracts import PipelineSpec, UnsupportedPipelineError

FRAME_PROBABILITY_HYSTERESIS = "frame_probability_hysteresis"
START_END_ATTENTION_VOTING = "start_end_attention_voting"


def pipeline_id_from_manifest_values(values: Dict[str, Any]) -> str:
    explicit = values.get("pipeline_id") or values.get("pipeline")
    if isinstance(explicit, dict):
        explicit = explicit.get("id")
    if explicit:
        return str(explicit)

    method = str(values.get("postprocess_method") or values.get("postprocess") or "").strip().lower()
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
    if pipeline_id not in {FRAME_PROBABILITY_HYSTERESIS, START_END_ATTENTION_VOTING}:
        raise UnsupportedPipelineError(f"Unsupported pipeline_id '{pipeline_id}'.")
    if pipeline_id == START_END_ATTENTION_VOTING and default_id != START_END_ATTENTION_VOTING:
        raise UnsupportedPipelineError(
            "Pipeline override 'start_end_attention_voting' is incompatible with this artifact. "
            "Use an artifact whose manifest declares start/end attention outputs."
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
    return PipelineSpec(
        pipeline_id=pipeline_id,
        feature_set=str(values.get("feature_set") or "v1"),
        model_output="start_end_attention",
        decode_method="start_end_voting",
        params={},
    )

