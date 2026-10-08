"""Private loopback service for the Swift desktop client.

The native player reads original files directly; analysis retains the same
manifest-driven, torch-free engine as the CLI. Every request requires a random
per-launch bearer token, including reads that disclose local source paths.
"""
from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
import signal
import sys


def register_native_routes(gui_app, token: str) -> None:
    from flask import jsonify, request

    if len(token) < 32:
        raise ValueError("Native backend requires a per-launch token of at least 32 characters")

    @gui_app.app.before_request
    def authenticate_native_client():
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied.encode("utf-8"), f"Bearer {token}".encode("utf-8")):
            return jsonify({"error": "Unauthorized"}), 401
        if request.headers.get("Origin"):
            return jsonify({"error": "Browser requests are not accepted"}), 403
        if request.host.split(":")[0] != "127.0.0.1":
            return jsonify({"error": "Forbidden host"}), 403
        return None

    @gui_app.app.post("/api/native/import")
    def import_match():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object."}), 400
        paths = payload.get("paths")
        if not isinstance(paths, list) or not paths or len(paths) > 1000:
            return jsonify({"error": "Choose one video or a folder of consecutive clips."}), 400
        try:
            if any(not isinstance(p, str) for p in paths):
                raise ValueError("Each source must be a file path.")
            sources = [Path(p).expanduser().resolve(strict=True) for p in paths]
            if any(not p.is_file() for p in sources):
                raise ValueError("A selected video is no longer available.")
            if len(set(sources)) != len(sources):
                raise ValueError("A video was selected more than once.")
            sources.sort(key=gui_app._natural_filename_key)
            cfg = gui_app._normalize_config({"output_name": str(payload.get("name") or sources[0].stem)})
            job_id = gui_app._start_folder_analysis_job(sources, cfg)
            return jsonify({"job_id": job_id}), 202
        except (ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400

    @gui_app.app.get("/api/native/library/<item_id>/sources")
    def sources_for_match(item_id):
        try:
            item_dir = gui_app._library_item_dir(item_id)
            source = gui_app._resolve_library_source(item_id)
            if source is None:
                return jsonify({"error": "Match not found"}), 404
            manifest = item_dir / "sources.json"
            if manifest.exists():
                sources = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
            else:
                sources = [{"path": str(source), "start_s": 0.0,
                            "duration_s": gui_app._estimate_duration_seconds(source)}]
            missing = [Path(s["path"]).name for s in sources if not Path(s["path"]).is_file()]
            if missing:
                return jsonify({"error": "Reconnect the original footage: " + ", ".join(missing)}), 409
            return jsonify({"sources": sources})
        except (ValueError, KeyError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--analysis-worker":
        from gui.analysis_worker import main as worker_main
        return worker_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from cli.main import main as cli_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return cli_main(force_cli=True)

    token = os.environ.pop("RALLYCLIP_NATIVE_TOKEN", "")
    ready_path = Path(os.environ.pop("RALLYCLIP_NATIVE_READY_FILE"))
    from gui import app as gui_app
    from werkzeug.serving import make_server

    register_native_routes(gui_app, token)
    server = make_server("127.0.0.1", 0, gui_app.app, threaded=True)
    server.timeout = 0.5
    stopping = False
    parent = os.getppid()

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ready_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"port": server.server_port}, stream)
    try:
        while not stopping and os.getppid() == parent:
            server.handle_request()
    finally:
        with gui_app.jobs_lock:
            ids = list(gui_app.jobs)
        for job_id in ids:
            gui_app._cancel_analysis_job(job_id)
        server.server_close()
        ready_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
