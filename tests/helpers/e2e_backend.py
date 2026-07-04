"""Helpers for L1 backend-journey e2e tests.

Spins up the *real* Flask backend (the one the desktop webview talks to) on a
free localhost port with isolated jobs/output dirs, and a thin HTTP client that
mirrors what the frontend does: upload-and-start, poll progress, download.

This is deliberately the real socket server (via ``start_backend_thread``), not
Flask's in-process ``test_client`` — that's the distinguishing value of an e2e:
it exercises the actual ``app.run`` serving path, the background worker threads,
and the cross-origin/host guards the way the shipped desktop app does.
"""
from __future__ import annotations

import json
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class BackendClient:
    """Minimal HTTP client over the running backend, matching the frontend's calls."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get(self, path: str, **kw: Any) -> requests.Response:
        return self.session.get(self.base_url + path, **kw)

    def post(self, path: str, **kw: Any) -> requests.Response:
        return self.session.post(self.base_url + path, **kw)

    def delete(self, path: str, **kw: Any) -> requests.Response:
        return self.session.delete(self.base_url + path, **kw)

    def start_job(self, video_path: Path, config: Optional[Dict[str, Any]] = None) -> requests.Response:
        with open(video_path, "rb") as fh:
            files = {"video": (Path(video_path).name, fh, "video/mp4")}
            data = {"config": json.dumps(config or {})}
            return self.post("/api/upload-and-start", files=files, data=data)

    def progress(self, job_id: str) -> requests.Response:
        return self.get(f"/api/progress/{job_id}")

    def wait_for(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll: float = 0.5,
        terminal: Tuple[str, ...] = ("completed", "failed", "cancelled"),
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Poll progress until a terminal state; return (final_body, all_snapshots)."""
        deadline = time.time() + timeout
        snapshots: List[Dict[str, Any]] = []
        while time.time() < deadline:
            resp = self.progress(job_id)
            resp.raise_for_status()
            body = resp.json()
            snapshots.append(body)
            if body.get("status") in terminal:
                return body, snapshots
            time.sleep(poll)
        last = snapshots[-1] if snapshots else None
        raise TimeoutError(f"job {job_id} did not finish within {timeout}s; last={last}")


@contextmanager
def running_backend(
    jobs_dir: Path, output_dir: Path, csv_dir: Path, library_dir: Path
) -> Iterator[BackendClient]:
    """Start the real backend on a free port with isolated dirs; yield a client.

    Redirects the module-level jobs/output/csv/library globals to the given temp
    dirs and rebuilds ``DEFAULT_CONFIG`` so jobs never touch the repo. Restores
    them on exit. The server runs in a daemon thread (cannot be force-stopped), so
    use one backend per test module.
    """
    from gui import app as gui_app

    for directory in (jobs_dir, output_dir, csv_dir, library_dir):
        directory.mkdir(parents=True, exist_ok=True)

    saved = (
        gui_app.JOBS_DIR,
        gui_app.DEFAULT_OUTPUT_DIR,
        gui_app.DEFAULT_CSV_DIR,
        gui_app.LIBRARY_DIR,
        gui_app.DEFAULT_CONFIG,
    )
    gui_app.JOBS_DIR = jobs_dir.resolve()
    gui_app.DEFAULT_OUTPUT_DIR = output_dir.resolve()
    gui_app.DEFAULT_CSV_DIR = csv_dir.resolve()
    gui_app.LIBRARY_DIR = library_dir.resolve()
    gui_app.DEFAULT_CONFIG = gui_app._load_default_config()

    port = find_free_port()
    gui_app.start_backend_thread(port=port)
    client = BackendClient(f"http://127.0.0.1:{port}")

    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            if client.get("/api/health", timeout=1.0).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        raise RuntimeError(f"backend did not come up on port {port}")

    try:
        yield client
    finally:
        (
            gui_app.JOBS_DIR,
            gui_app.DEFAULT_OUTPUT_DIR,
            gui_app.DEFAULT_CSV_DIR,
            gui_app.LIBRARY_DIR,
            gui_app.DEFAULT_CONFIG,
        ) = saved
