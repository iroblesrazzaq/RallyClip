from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO


def _json_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"thread", "process"}}


def _emit(stream: TextIO, event: dict[str, Any]) -> None:
    stream.write(json.dumps(event, default=str) + "\n")
    stream.flush()


def _monitor_job(gui_app, job_id: str, stream: TextIO, stop: threading.Event) -> None:
    last_payload = ""
    while not stop.wait(0.5):
        job = gui_app.jobs.get(job_id)
        if job is None:
            continue
        payload = json.dumps(_json_job(job), sort_keys=True, default=str)
        if payload == last_payload:
            continue
        last_payload = payload
        _emit(stream, {"type": "snapshot", "job": _json_job(job)})


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m gui.analysis_worker [--warmup|<job-json>]", file=sys.stderr)
        return 2

    event_stream = sys.stdout
    # Keep third-party logs/progress chatter away from the JSON event stream.
    sys.stdout = sys.stderr

    if argv[0] == "--warmup":
        from gui import app as gui_app

        try:
            gui_app._get_analysis_runtime()
            _emit(event_stream, {"type": "runtime_status", "status": dict(gui_app._RUNTIME_STATUS)})
            return 0
        except Exception as exc:
            status = dict(gui_app._RUNTIME_STATUS)
            status["state"] = "error"
            status["error"] = str(exc)
            _emit(event_stream, {"type": "runtime_status", "status": status})
            return 1

    job_path = Path(argv[0]).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_id = str(job["id"])

    from gui import app as gui_app

    gui_app.jobs.clear()
    gui_app.jobs[job_id] = job
    stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor_job,
        args=(gui_app, job_id, event_stream, stop),
        daemon=True,
    )
    monitor.start()
    try:
        _emit(event_stream, {"type": "snapshot", "job": _json_job(job)})
        gui_app._run_pipeline(job_id)
        final_job = gui_app.jobs.get(job_id, job)
        _emit(event_stream, {"type": "final", "job": _json_job(final_job)})
        return 0 if final_job.get("status") in {"completed", "cancelled"} else 1
    finally:
        stop.set()
        monitor.join(timeout=1.0)


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(main())
