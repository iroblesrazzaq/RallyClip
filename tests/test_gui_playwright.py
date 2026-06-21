"""L2 browser-UI e2e — drives the real frontend with a headless browser.

Where L1 (test_gui_e2e.py) hits the JSON API directly, this runs the actual
frontend JavaScript the desktop webview renders: the welcome screen, file
picker, Start button enablement, progress polling, the results view, and the
download buttons. A bug in script.js (stuck poller, disabled download, welcome
not dismissing) passes L1 but fails here.

The shipped desktop app is a QWebEngineView (Chromium) pointed at the same
localhost Flask, so headless Chromium against the same URL is a faithful proxy
for the desktop UI — only the native window chrome differs.

Marked ``e2e``+``slow``. Needs the e2e-ui extra + a browser:
    pip install ".[dev,e2e-ui]" && playwright install chromium
Self-skips cleanly when Playwright or the v0.3.1 ONNX artifact is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODEL_ONNX = REPO_ROOT / "models" / "rallyclip_v0.3.1" / "model.onnx"

pytest.importorskip("playwright.sync_api")
pytest.importorskip("requests")
pytest.importorskip("cv2")
pytest.importorskip("ultralytics")
pytest.importorskip("onnxruntime")
pytest.importorskip("av")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from playwright.sync_api import Page, expect  # noqa: E402

from helpers.e2e_backend import BackendClient, running_backend  # noqa: E402

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        not MODEL_ONNX.exists(),
        reason="shipped v0.3.1 ONNX artifact missing (models/rallyclip_v0.3.1/model.onnx)",
    ),
]

CLIP_SECONDS = 30.0
RESULTS_TIMEOUT = 180_000  # ms; real pipeline on the synthetic clip


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        yield mp
    finally:
        mp.undo()


@pytest.fixture(scope="session")
def ui_clip(tmp_path_factory) -> Path:
    from make_smoke_clip import make_clip

    clip = tmp_path_factory.mktemp("ui_clip") / "smoke.mp4"
    make_clip(clip, duration_s=CLIP_SECONDS)
    return clip


@pytest.fixture(scope="module")
def ui_backend(tmp_path_factory, monkeypatch_module) -> BackendClient:
    root = tmp_path_factory.mktemp("ui_backend")
    monkeypatch_module.setenv("RALLYCLIP_KEEP_JOBS", "1")
    with running_backend(root / "jobs", root / "output", root / "csv") as client:
        yield client


def _open_app(page: Page, base_url: str) -> None:
    page.goto(base_url)
    # Fresh browser context -> no localStorage -> welcome screen is shown.
    expect(page.locator("#welcomeScreen")).to_be_visible()
    page.locator("#welcomeStartBtn").click()
    expect(page.locator("#uploadView")).to_be_visible()


def test_ui_welcome_dismisses_to_upload(page: Page, ui_backend: BackendClient):
    """Welcome screen renders, dismisses to the upload view, Start gated on a file."""
    _open_app(page, ui_backend.base_url)
    expect(page.locator("#welcomeScreen")).to_be_hidden()
    # No file selected yet -> Start disabled.
    expect(page.locator("#startBtn")).to_be_disabled()


def test_ui_happy_path_runs_to_results(page: Page, ui_backend: BackendClient, ui_clip: Path):
    """Full UI journey: welcome -> pick file -> Start -> progress -> results +
    a real CSV download, all driven through the frontend JavaScript."""
    _open_app(page, ui_backend.base_url)

    page.locator("#fileInput").set_input_files(str(ui_clip))
    expect(page.locator("#selectedFile")).to_be_visible()
    expect(page.locator("#startBtn")).to_be_enabled()

    page.locator("#startBtn").click()
    expect(page.locator("#progress")).to_be_visible()

    # Pipeline runs for real; the results view appears on completion.
    expect(page.locator("#results")).to_be_visible(timeout=RESULTS_TIMEOUT)
    expect(page.locator("#downloadCsvBtn")).to_be_enabled()
    expect(page.locator("#downloadVideoBtn")).to_be_visible()

    # The CSV download button round-trips through the JS (fetch -> blob -> save).
    with page.expect_download() as download_info:
        page.locator("#downloadCsvBtn").click()
    download = download_info.value
    assert download.suggested_filename.endswith(".csv")


def test_ui_cancel_returns_to_idle(page: Page, ui_backend: BackendClient, ui_clip: Path):
    """Starting then cancelling brings the UI back to a non-results state."""
    _open_app(page, ui_backend.base_url)
    page.locator("#fileInput").set_input_files(str(ui_clip))
    page.locator("#startBtn").click()
    expect(page.locator("#progress")).to_be_visible()

    cancel = page.locator("#cancelBtn")
    expect(cancel).to_be_enabled(timeout=30_000)
    cancel.click()
    # Cancellation must not land on the results/download view.
    expect(page.locator("#results")).to_be_hidden()
