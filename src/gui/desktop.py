"""RallyClip desktop shell: a system webview over the local Flask backend.

The window is a pywebview WKWebView (macOS) / WebView2 (Windows) pointed at
the same localhost Flask app the browser e2e suite exercises — every behavior
is an /api/* HTTP call. The system webview plays H.264/HEVC natively via the
OS media stack, so the frontend's plain HTML5 <video> path is used (the old
QtWebEngine shell needed a separate Qt-Multimedia native player because
Chromium ships without proprietary codecs).

The frozen binary also dispatches two headless personalities:
    RallyClip --cli ...              # full analysis CLI (cli.main)
    RallyClip --analysis-worker ...  # GUI job subprocess
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request


def _wait_for_backend(port: int, timeout_sec: float = 30.0) -> bool:
    deadline = time.time() + timeout_sec
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    return False


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--analysis-worker":
        from gui.analysis_worker import main as analysis_worker_main

        return analysis_worker_main(sys.argv[2:])

    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Lazy import keeps GUI startup free of the analysis import chain.
        from cli.main import main as cli_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return cli_main()

    try:
        import webview
    except ImportError as exc:
        print(
            "rallyclip-desktop requires pywebview. Install with: pip install '.[desktop]'",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 1

    from gui.app import start_backend_thread

    port, _thread = start_backend_thread()
    if not _wait_for_backend(port):
        print("RallyClip backend failed to start.", file=sys.stderr)
        return 1

    # "Export video" / "Download CSV" navigate to Content-Disposition
    # responses; this makes the webview surface a native Save dialog for them
    # (the QtWebEngine shell needed an explicit downloadRequested handler).
    webview.settings["ALLOW_DOWNLOADS"] = True

    webview.create_window(
        "RallyClip",
        f"http://127.0.0.1:{port}/",
        width=1280,
        height=840,
        min_size=(960, 600),
    )
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
