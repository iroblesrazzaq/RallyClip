from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request


def _wait_for_backend(port: int, timeout_sec: float = 15.0, pump=None) -> bool:
    deadline = time.time() + timeout_sec
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        if pump is not None:
            pump()
        try:
            # Short timeout so the splash-screen event pump isn't starved.
            with urllib.request.urlopen(url, timeout=0.3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    return False


def main() -> int:
    try:
        from PySide6.QtCore import Qt, QUrl
        from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
        from PySide6.QtWidgets import QApplication, QMainWindow, QSplashScreen
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError as exc:
        print(
            "rallyclip-desktop requires PySide6. Install with: pip install '.[desktop]'",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 1

    from gui.app import start_backend_thread

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("RallyClip")

    splash_pix = QPixmap(420, 160)
    splash_pix.fill(QColor("#f5f0e8"))
    painter = QPainter(splash_pix)
    painter.setPen(QColor("#1a2744"))
    font = QFont()
    font.setPointSize(14)
    painter.setFont(font)
    painter.drawText(splash_pix.rect(), Qt.AlignmentFlag.AlignCenter, "Starting RallyClip…")
    painter.end()

    splash = QSplashScreen(splash_pix)
    splash.show()
    qt_app.processEvents()

    port, _thread = start_backend_thread()
    if not _wait_for_backend(port, pump=qt_app.processEvents):
        splash.close()
        print("RallyClip backend failed to start.", file=sys.stderr)
        return 1

    window = QMainWindow()
    window.setWindowTitle("RallyClip")
    window.resize(1280, 840)

    view = QWebEngineView()
    view.load(QUrl(f"http://127.0.0.1:{port}/"))
    window.setCentralWidget(view)
    window.show()
    splash.finish(window)

    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
