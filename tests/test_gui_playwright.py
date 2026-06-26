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
    with running_backend(root / "jobs", root / "output", root / "csv", root / "library") as client:
        yield client


def _fabricate_ui_item() -> str:
    """Drop a fake library item on disk (into the running backend's library dir)
    so the library UI can be tested without running the pipeline."""
    import json
    import time
    import uuid

    from gui import app as gui_app

    item_id = f"20990101-000000-{uuid.uuid4().hex[:6]}"
    item_dir = Path(gui_app.LIBRARY_DIR) / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "source.mp4").write_bytes(b"fake-video")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,3.0\n", encoding="utf-8")
    (item_dir / "thumb.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")
    (item_dir / "meta.json").write_text(
        json.dumps({
            "id": item_id, "name": "UI Test Match", "created": "2099-01-01T00:00:00",
            "created_ts": time.time(), "duration_s": 12.0, "n_segments": 3,
        }),
        encoding="utf-8",
    )
    return item_id


def _fabricate_viewer_item() -> str:
    """Create a real, short source clip so the saved-match viewer can exercise
    preview-window generation instead of reading a prebuilt fake."""
    import json
    import time
    import uuid

    from gui import app as gui_app
    from make_smoke_clip import make_clip

    item_id = f"20990101-000000-{uuid.uuid4().hex[:6]}"
    item_dir = Path(gui_app.LIBRARY_DIR) / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    make_clip(item_dir / "source.mp4", duration_s=12.0)
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,2.0\n4.0,5.0\n", encoding="utf-8")
    (item_dir / "thumb.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")
    (item_dir / "meta.json").write_text(
        json.dumps({
            "id": item_id, "name": "Viewer Window Match", "created": "2099-01-01T00:00:00",
            "created_ts": time.time(), "duration_s": 12.0, "point_duration_s": 2.0, "n_segments": 2,
        }),
        encoding="utf-8",
    )
    return item_id


def _open_to_library(page: Page, base_url: str) -> None:
    page.goto(base_url)
    # Fresh browser context -> no localStorage -> welcome screen is shown.
    expect(page.locator("#welcomeScreen")).to_be_visible()
    page.locator("#welcomeStartBtn").click()
    expect(page.locator("#libraryView")).to_be_visible()


def test_ui_welcome_dismisses_to_library(page: Page, ui_backend: BackendClient):
    """Welcome dismisses to the saved-matches library (the default view)."""
    _open_to_library(page, ui_backend.base_url)
    expect(page.locator("#welcomeScreen")).to_be_hidden()
    expect(page.locator("#newMatchBtn")).to_be_visible()


def test_ui_library_renders_and_deletes_item(page: Page, ui_backend: BackendClient):
    """A saved match renders as a card with actions; delete removes it."""
    item_id = _fabricate_ui_item()
    _open_to_library(page, ui_backend.base_url)

    card = page.locator(f'.lib-card[data-id="{item_id}"]')
    expect(card).to_be_visible()
    expect(card).to_contain_text("UI Test Match")
    expect(card.locator('button[data-action="export"]')).to_be_visible()
    expect(card.locator('button[data-action="csv"]')).to_be_visible()

    page.on("dialog", lambda dialog: dialog.accept())  # auto-confirm the delete prompt
    card.locator('button[data-action="delete"]').click()
    expect(page.locator(f'.lib-card[data-id="{item_id}"]')).to_have_count(0)


def test_ui_viewer_uses_preview_windows_and_point_crossing(page: Page, ui_backend: BackendClient):
    """Saved-match viewer loads bounded preview windows and detects point-end
    crossings in source-video time."""
    item_id = _fabricate_viewer_item()
    _open_to_library(page, ui_backend.base_url)

    page.locator(f'.lib-card[data-id="{item_id}"]').click()
    expect(page.locator("#viewerView")).to_be_visible()
    expect(page.locator("#viewerTimeline")).to_be_visible(timeout=30_000)
    page.wait_for_function(
        "() => window.rallyClipApp?.matchVideo?.src.includes('/preview/window')",
        timeout=30_000,
    )
    expect(page.locator("#viewerBackBtn")).to_be_visible()
    expect(page.locator("#viewerPlayPauseBtn")).to_be_visible()
    expect(page.locator("#viewerForwardBtn")).to_be_visible()
    result = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            app.pointIntervals = [{ start: 1, end: 2 }, { start: 4, end: 5 }];
            return {
                crosses: app.findCrossedPointIndex(1.95, 2.10),
                gap: app.findCrossedPointIndex(2.10, 3.90),
                later: app.findCrossedPointIndex(4.90, 5.05),
            };
        }"""
    )
    assert result == {"crosses": 0, "gap": -1, "later": 1}

    prefetch_plan = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            app.previewWindowDuration = 8;
            app.previewLookaheadChunks = 12;
            app.pointBufferSeconds = 5;
            app.sourceDuration = 200;
            app.pointIntervals = [
                { start: 30, end: 40 },
                { start: 70, end: 74 },
                { start: 100, end: 104 },
            ];
            return {
                canonical: [
                    app.canonicalPreviewChunkStart(30),
                    app.canonicalPreviewChunkStart(41.9),
                    app.canonicalPreviewChunkStart(42),
                ],
                starts: app.getPointAwarePrefetchStarts(30),
                gapStarts: app.getPointAwarePrefetchStarts(55),
            };
        }"""
    )
    assert prefetch_plan == {
        "canonical": [24, 40, 40],
        "starts": [32, 40, 64, 72],
        "gapStarts": [56, 64, 72, 88, 96, 104],
    }

    loading_state = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            app.hidePreviewLoading();
            app.showPreviewLoading();
            const state = {
                hidden: app.previewStatus.hidden,
                slow: app.previewStatus.classList.contains("is-slow"),
                text: app.previewStatus.textContent,
            };
            app.hidePreviewLoading();
            return state;
        }"""
    )
    assert loading_state == {"hidden": False, "slow": False, "text": ""}

    click_toggle = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            const rect = app.matchVideo.getBoundingClientRect();
            const originalPlay = app.matchVideo.play;
            const originalPause = app.matchVideo.pause;
            const pausedDescriptor = Object.getOwnPropertyDescriptor(app.matchVideo, "paused");
            let paused = true;
            let plays = 0;
            let pauses = 0;
            app.matchVideo.play = () => {
                plays += 1;
                paused = false;
                return Promise.resolve();
            };
            Object.defineProperty(app.matchVideo, "pause", {
                value: () => {
                    pauses += 1;
                    paused = true;
                },
                configurable: true,
            });
            Object.defineProperty(app.matchVideo, "paused", {
                get: () => paused,
                configurable: true,
            });
            app.toggleViewerPlayback({ clientY: rect.top + rect.height / 2 });
            app.toggleViewerPlayback({ clientY: rect.top + rect.height / 2 });
            app.matchVideo.play = originalPlay;
            app.matchVideo.pause = originalPause;
            if (pausedDescriptor) Object.defineProperty(app.matchVideo, "paused", pausedDescriptor);
            else delete app.matchVideo.paused;
            return { plays, pauses };
        }"""
    )
    assert click_toggle == {"plays": 1, "pauses": 1}

    controls = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            const pausedDescriptor = Object.getOwnPropertyDescriptor(app.matchVideo, "paused");
            let paused = true;
            Object.defineProperty(app.matchVideo, "paused", {
                get: () => paused,
                configurable: true,
            });
            app.updateViewerControls();
            const pausedLabel = app.viewerPlayPauseBtn.textContent;
            paused = false;
            app.updateViewerControls();
            const playingLabel = app.viewerPlayPauseBtn.textContent;

            app.currentPreviewWindowStart = 20;
            app.currentPreviewWindowDuration = 20;
            app.sourceDuration = 120;
            app.matchVideo.currentTime = 10;
            const calls = [];
            const originalSeek = app.seekViewerToSourceTime.bind(app);
            app.seekViewerToSourceTime = (time, autoplay) => calls.push({ time, autoplay });
            let prevented = 0;
            app.handleViewerKeyboardShortcuts({
                key: "ArrowLeft",
                target: app.viewerSeek,
                defaultPrevented: false,
                preventDefault: () => { prevented += 1; },
            });
            app.handleViewerKeyboardShortcuts({
                key: "ArrowRight",
                target: app.viewerSeek,
                defaultPrevented: false,
                preventDefault: () => { prevented += 1; },
            });
            app.seekViewerToSourceTime = originalSeek;

            if (pausedDescriptor) Object.defineProperty(app.matchVideo, "paused", pausedDescriptor);
            else delete app.matchVideo.paused;
            return { pausedLabel, playingLabel, calls, prevented };
        }"""
    )
    assert controls == {
        "pausedLabel": "Play",
        "playingLabel": "Pause",
        "calls": [{"time": 23, "autoplay": True}, {"time": 33, "autoplay": True}],
        "prevented": 2,
    }

    timeline_seek = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            const originalSeek = app.seekViewerToSourceTime.bind(app);
            const originalRect = app.viewerSeekWrap.getBoundingClientRect.bind(app.viewerSeekWrap);
            const pausedDescriptor = Object.getOwnPropertyDescriptor(app.matchVideo, "paused");
            const calls = [];
            app.sourceDuration = 100;
            app.viewerSeekWrap.getBoundingClientRect = () => ({ left: 100, width: 200 });
            Object.defineProperty(app.matchVideo, "paused", { value: true, configurable: true });
            app.seekViewerToSourceTime = (time, autoplay) => calls.push({ time, autoplay });
            app.seekViewerFromTimelinePointer({
                target: app.viewerBufferTrack,
                clientX: 200,
            });
            app.seekViewerToSourceTime = originalSeek;
            app.viewerSeekWrap.getBoundingClientRect = originalRect;
            if (pausedDescriptor) Object.defineProperty(app.matchVideo, "paused", pausedDescriptor);
            else delete app.matchVideo.paused;
            return calls;
        }"""
    )
    assert timeline_seek == [{"time": 50, "autoplay": False}]

    jump = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            app.pointIntervals = [{ start: 1, end: 2 }, { start: 4, end: 5 }];
            app.currentPreviewWindowStart = 1;
            app.currentPreviewWindowDuration = 11;
            app.viewerSeekInProgress = false;
            app.matchVideo.play = () => Promise.resolve();
            Object.defineProperty(app.matchVideo, "paused", { value: false, configurable: true });
            app.matchVideo.currentTime = 1.1;
            app.lastViewerTime = 1.95;
            app.handleViewerTimeUpdate();
            return { videoTime: app.matchVideo.currentTime, sourceTime: app.lastViewerTime };
        }"""
    )
    assert jump["videoTime"] == pytest.approx(3.0, abs=0.1)
    assert jump["sourceTime"] == pytest.approx(4.0, abs=0.1)

    delayed_jump = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            app.pointIntervals = [{ start: 1, end: 2 }, { start: 4, end: 5 }];
            app.currentPreviewWindowStart = 1;
            app.currentPreviewWindowDuration = 11;
            app.viewerSeekInProgress = false;
            app.matchVideo.play = () => Promise.resolve();
            Object.defineProperty(app.matchVideo, "paused", { value: false, configurable: true });
            app.matchVideo.currentTime = 3.2;
            app.lastViewerTime = 1.0;
            app.handleViewerTimeUpdate();
            return { videoTime: app.matchVideo.currentTime, sourceTime: app.lastViewerTime };
        }"""
    )
    assert delayed_jump["videoTime"] == pytest.approx(3.0, abs=0.1)
    assert delayed_jump["sourceTime"] == pytest.approx(4.0, abs=0.1)

    ended_jump = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            app.pointIntervals = [{ start: 1, end: 2 }, { start: 4, end: 5 }];
            app.currentPreviewWindowStart = 1;
            app.currentPreviewWindowDuration = 3.2;
            app.viewerSeekInProgress = false;
            app.previewLoadInProgress = false;
            app.matchVideo.play = () => Promise.resolve();
            app.matchVideo.currentTime = 3.2;
            app.lastViewerTime = 1.5;
            app.handleViewerWindowEnded();
            return { videoTime: app.matchVideo.currentTime, sourceTime: app.lastViewerTime };
        }"""
    )
    assert ended_jump["videoTime"] == pytest.approx(3.0, abs=0.1)
    assert ended_jump["sourceTime"] == pytest.approx(4.0, abs=0.1)

    chunk_continue = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            const calls = [];
            const originalLoad = app.loadPreviewWindowAt.bind(app);
            app.loadPreviewWindowAt = (time, autoplay) => calls.push({ time, autoplay });
            app.pointIntervals = [{ start: 24, end: 44 }, { start: 70, end: 74 }];
            app.currentPreviewWindowStart = 24;
            app.currentPreviewWindowDuration = 8;
            app.sourceDuration = 120;
            app.lastViewerTime = 24.5;
            app.previewLoadInProgress = false;
            app.handleViewerWindowEnded();
            app.loadPreviewWindowAt = originalLoad;
            return calls;
        }"""
    )
    assert chunk_continue == [{"time": 32, "autoplay": True}]


def test_ui_new_match_runs_to_library(page: Page, ui_backend: BackendClient, ui_clip: Path):
    """Full UI journey: library -> New match -> pick file -> Start -> progress ->
    back to the library on completion. The synthetic clip finds no points, so no
    item is saved, but the whole frontend flow is driven for real."""
    _open_to_library(page, ui_backend.base_url)
    page.locator("#newMatchBtn").click()
    expect(page.locator("#uploadView")).to_be_visible()
    expect(page.locator("#startBtn")).to_be_disabled()

    page.locator("#fileInput").set_input_files(str(ui_clip))
    expect(page.locator("#selectedFile")).to_be_visible()
    expect(page.locator("#startBtn")).to_be_enabled()

    page.locator("#startBtn").click()
    expect(page.locator("#progress")).to_be_visible()
    expect(page.locator("#libraryView")).to_be_visible(timeout=RESULTS_TIMEOUT)


def test_ui_cancel_returns_to_library(page: Page, ui_backend: BackendClient, ui_clip: Path):
    """Starting then cancelling brings the UI back to the library."""
    _open_to_library(page, ui_backend.base_url)
    page.locator("#newMatchBtn").click()
    page.locator("#fileInput").set_input_files(str(ui_clip))
    page.locator("#startBtn").click()
    expect(page.locator("#progress")).to_be_visible()

    cancel = page.locator("#cancelBtn")
    expect(cancel).to_be_enabled(timeout=30_000)
    cancel.click()
    expect(page.locator("#libraryView")).to_be_visible(timeout=30_000)
