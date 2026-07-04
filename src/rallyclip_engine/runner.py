from __future__ import annotations

from typing import Optional

from rallyclip_core.contracts import CancelCheck, ProgressCallback, RunRequest, RunResult, RuntimeDeps
from rallyclip_core.pipelines import resolve_pipeline_spec

from .models import build_analysis_model
from .runtime import load_runtime_deps


def run_analysis(
    request: RunRequest,
    *,
    deps: Optional[RuntimeDeps] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> RunResult:
    deps = deps or load_runtime_deps()
    spec = resolve_pipeline_spec(
        request.model_path,
        request.manifest_path,
        override_pipeline_id=request.pipeline_id,
    )
    model = build_analysis_model(request, spec, deps)
    return model.run(progress_callback=progress_callback, cancel_check=cancel_check)

