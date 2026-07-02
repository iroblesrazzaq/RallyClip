from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from rallyclip_core.library import SavedMatchStore


@dataclass
class RallyClipServices:
    """Thin local API service facade.

    The current Flask app can inject its existing stateful handlers while the
    runtime/engine code moves underneath this stable application boundary.
    """

    defaults_provider: Optional[Callable[[], Dict[str, Any]]] = None
    runtime_status_provider: Optional[Callable[[], Dict[str, Any]]] = None
    runtime_warmup: Optional[Callable[[], None]] = None
    start_job_handler: Optional[Callable[..., Any]] = None
    job_status_provider: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None
    cancel_job_handler: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None
    library_provider: Optional[Callable[[], Dict[str, Any]]] = None
    playback_manifest_provider: Optional[Callable[[str], Dict[str, Any]]] = None
    export_handler: Optional[Callable[[str], Any]] = None
    saved_match_store: Optional[SavedMatchStore] = None
    analysis_runner: Optional[Callable[..., Any]] = None

    def get_defaults(self) -> Dict[str, Any]:
        if self.defaults_provider is None:
            raise NotImplementedError("get_defaults is not wired")
        return self.defaults_provider()

    def get_runtime_status(self) -> Dict[str, Any]:
        if self.runtime_status_provider is None:
            raise NotImplementedError("get_runtime_status is not wired")
        return self.runtime_status_provider()

    def warmup_runtime(self) -> Dict[str, Any]:
        if self.runtime_warmup is None:
            raise NotImplementedError("warmup_runtime is not wired")
        self.runtime_warmup()
        return self.get_runtime_status()

    def start_job(self, *args, **kwargs):
        if self.start_job_handler is None:
            raise NotImplementedError("start_job is not wired")
        return self.start_job_handler(*args, **kwargs)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Job progress payload, or None for an unknown job."""
        if self.job_status_provider is None:
            raise NotImplementedError("get_job_status is not wired")
        return self.job_status_provider(job_id)

    def cancel_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Cancel a job (idempotent), or None for an unknown job."""
        if self.cancel_job_handler is None:
            raise NotImplementedError("cancel_job is not wired")
        return self.cancel_job_handler(job_id)

    def list_library(self) -> Dict[str, Any]:
        if self.library_provider is not None:
            return self.library_provider()
        if self.saved_match_store is not None:
            return {"items": self.saved_match_store.list_items()}
        raise NotImplementedError("list_library is not wired")

    def get_playback_manifest(self, item_id: str) -> Dict[str, Any]:
        if self.playback_manifest_provider is None:
            raise NotImplementedError("get_playback_manifest is not wired")
        return self.playback_manifest_provider(item_id)

    def export_match(self, item_id: str):
        if self.export_handler is None:
            raise NotImplementedError("export_match is not wired")
        return self.export_handler(item_id)

    def run_analysis(self, request, *, deps=None, progress_callback=None, cancel_check=None):
        """Run an analysis pipeline for a RunRequest and return the RunResult.

        The engine is imported lazily so this facade stays import-light for
        replay/library startup paths.
        """
        runner = self.analysis_runner
        if runner is None:
            from rallyclip_engine import run_analysis as runner  # noqa: WPS433
        return runner(
            request,
            deps=deps,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

