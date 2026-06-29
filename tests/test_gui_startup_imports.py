from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run_import_probe(code: str) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_gui_app_import_does_not_load_analysis_or_transcode_stack():
    payload = _run_import_probe(
        textwrap.dedent(
            """
            import json
            import sys
            import time

            start = time.perf_counter()
            import gui.app  # noqa: F401
            elapsed = time.perf_counter() - start

            print(json.dumps({
                "elapsed_s": elapsed,
                "torch": "torch" in sys.modules,
                "ultralytics": "ultralytics" in sys.modules,
                "av": "av" in sys.modules,
                "numpy": "numpy" in sys.modules,
            }))
            """
        )
    )

    assert payload["elapsed_s"] < 3.0
    assert payload["torch"] is False
    assert payload["ultralytics"] is False
    assert payload["av"] is False
    assert payload["numpy"] is False


def test_defaults_endpoint_does_not_probe_analysis_runtime():
    payload = _run_import_probe(
        textwrap.dedent(
            """
            import json
            import sys

            from gui import app as gui_app

            response = gui_app.app.test_client().get("/api/config/defaults")
            body = response.get_json()
            print(json.dumps({
                "status": response.status_code,
                "runtime_state": body.get("runtime_state"),
                "available_devices": body.get("available_devices"),
                "auto_device": body.get("auto_device"),
                "torch": "torch" in sys.modules,
                "ultralytics": "ultralytics" in sys.modules,
                "av": "av" in sys.modules,
                "numpy": "numpy" in sys.modules,
            }))
            """
        )
    )

    assert payload["status"] == 200
    assert payload["runtime_state"] == "cold"
    assert payload["available_devices"] == ["cpu"]
    assert payload["auto_device"] == "cpu"
    assert payload["torch"] is False
    assert payload["ultralytics"] is False
    assert payload["av"] is False
    assert payload["numpy"] is False
