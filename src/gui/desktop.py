from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request


def _wait_for_backend(port: int, timeout_sec: float = 15.0) -> bool:
    deadline = time.time() + timeout_sec
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    return False


def main() -> int:
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtWidgets import QApplication, QMainWindow
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError as exc:
        print(
            "rallyclip-desktop requires PySide6. Install with: pip install '.[desktop]'",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 1

    from gui.app import start_backend_thread

    port, _thread = start_backend_thread()
    if not _wait_for_backend(port):
        print("RallyClip backend failed to start.", file=sys.stderr)
        return 1

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("RallyClip")

    window = QMainWindow()
    window.setWindowTitle("RallyClip")
    window.resize(1280, 840)

    view = QWebEngineView()
    view.load(QUrl(f"http://127.0.0.1:{port}/"))
    window.setCentralWidget(view)
    window.show()

    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
