from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class RallyClipServices:
    """Thin local API service facade.

    The current Flask app can inject its existing stateful handlers while the
    runtime/engine code moves underneath this stable application boundary.
    """

    defaults_provider: Callable[[], Dict[str, Any]]
    runtime_status_provider: Callable[[], Dict[str, Any]]
    runtime_warmup: Callable[[], None]
    start_job_handler: Optional[Callable[..., Any]] = None
    job_status_provider: Optional[Callable[[str], Dict[str, Any]]] = None
    cancel_job_handler: Optional[Callable[[str], Dict[str, Any]]] = None
    library_provider: Optional[Callable[[], Dict[str, Any]]] = None
    playback_manifest_provider: Optional[Callable[[str], Dict[str, Any]]] = None
    export_handler: Optional[Callable[[str], Any]] = None

    def get_defaults(self) -> Dict[str, Any]:
        return self.defaults_provider()

    def get_runtime_status(self) -> Dict[str, Any]:
        return self.runtime_status_provider()

    def warmup_runtime(self) -> Dict[str, Any]:
        self.runtime_warmup()
        return self.get_runtime_status()

    def start_job(self, *args, **kwargs):
        if self.start_job_handler is None:
            raise NotImplementedError("start_job is not wired")
        return self.start_job_handler(*args, **kwargs)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        if self.job_status_provider is None:
            raise NotImplementedError("get_job_status is not wired")
        return self.job_status_provider(job_id)

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        if self.cancel_job_handler is None:
            raise NotImplementedError("cancel_job is not wired")
        return self.cancel_job_handler(job_id)

    def list_library(self) -> Dict[str, Any]:
        if self.library_provider is None:
            raise NotImplementedError("list_library is not wired")
        return self.library_provider()

    def get_playback_manifest(self, item_id: str) -> Dict[str, Any]:
        if self.playback_manifest_provider is None:
            raise NotImplementedError("get_playback_manifest is not wired")
        return self.playback_manifest_provider(item_id)

    def export_match(self, item_id: str):
        if self.export_handler is None:
            raise NotImplementedError("export_match is not wired")
        return self.export_handler(item_id)

