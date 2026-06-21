from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _fix_frozen_webengine_paths() -> None:
    """Point Qt at the relocated QtWebEngineCore.framework in macOS bundles.

    PyInstaller moves the framework to the bundle root (codesign rules) but
    Qt 6.11 still searches the PySide6/Qt/lib copy, which only carries the
    main binary — without these overrides the app aborts at QWebEngineView.
    """
    if not getattr(sys, "frozen", False) or sys.platform != "darwin":
        return
    framework = Path(getattr(sys, "_MEIPASS", "")) / "QtWebEngineCore.framework" / "Versions" / "A"
    helper = framework / "Helpers" / "QtWebEngineProcess.app" / "Contents" / "MacOS" / "QtWebEngineProcess"
    resources = framework / "Resources"
    if helper.exists():
        os.environ.setdefault("QTWEBENGINEPROCESS_PATH", str(helper))
    if resources.is_dir():
        os.environ.setdefault("QTWEBENGINE_RESOURCES_PATH", str(resources))


def _wait_for_backend(port: int, timeout_sec: float = 30.0, pump=None) -> bool:
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
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Lazy import is deliberate: cli.main pulls torch/ultralytics, and
        # importing it at module top would delay GUI startup before the
        # splash screen can appear.
        from cli.main import main as cli_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return cli_main()

    _fix_frozen_webengine_paths()
    try:
        from PySide6.QtCore import Qt, QUrl
        from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
        from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QSplashScreen
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

    def _on_download_requested(download) -> None:
        # QWebEngineView silently drops downloads unless one is accepted here.
        # This is what makes "Export video" / "Download CSV" actually save a file
        # in the desktop app (in a browser the page's download just works).
        suggested = download.suggestedFileName() or "rallyclip_export"
        downloads_dir = Path.home() / "Downloads"
        base_dir = downloads_dir if downloads_dir.is_dir() else Path.home()
        target, _ = QFileDialog.getSaveFileName(window, "Save", str(base_dir / suggested))
        if not target:
            download.cancel()
            return
        target_path = Path(target)
        download.setDownloadDirectory(str(target_path.parent))
        download.setDownloadFileName(target_path.name)
        download.accept()

    view.page().profile().downloadRequested.connect(_on_download_requested)
    view.load(QUrl(f"http://127.0.0.1:{port}/"))
    window.setCentralWidget(view)
    window.show()
    splash.finish(window)

    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
