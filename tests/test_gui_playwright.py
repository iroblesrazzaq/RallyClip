"""L2 browser-UI e2e — drives the real frontend with a headless browser.

Where L1 (test_gui_e2e.py) hits the JSON API directly, this runs the actual
frontend JavaScript the desktop webview renders: the welcome screen, file
picker, Start button enablement, progress polling, the results view, and the
download buttons. A bug in script.js (stuck poller, disabled download, welcome
not dismissing) passes L1 but fails here.

The shipped desktop app is a system webview (WKWebView/WebView2) over the same
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
pytest.importorskip("onnxruntime")
pytest.importorskip("av")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, expect  # noqa: E402

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


@pytest.fixture(autouse=True)
def _fresh_welcome_state(ui_backend):
    """Every test starts from a never-seen-welcome install.

    Welcome-seen is persisted server-side (one shared backend per module), so
    without this reset the first test's dismissal skips the welcome screen for
    every test after it.
    """
    from gui import app as gui_app

    Path(gui_app.PREFERENCES_PATH).unlink(missing_ok=True)


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


def _fabricate_direct_playback_item() -> str:
    """Library item whose source headless Chromium can decode natively.

    The mpeg4 smoke fixture never passes the direct-playback probe in
    Chromium, so the /source path only ever runs as a fallback-skipped branch
    here. VP9 in an MP4 container is a legal combination Chromium plays,
    letting this suite exercise direct playback the way WKWebView exercises
    H.264 in the shipped app. Small frame keeps the VP9 encode fast; the
    viewer has no resolution minimum (only the analysis pipeline does)."""
    import json
    import time
    import uuid

    from gui import app as gui_app
    from make_smoke_clip import make_clip

    item_id = f"20990101-000000-{uuid.uuid4().hex[:6]}"
    item_dir = Path(gui_app.LIBRARY_DIR) / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    make_clip(item_dir / "source.mp4", duration_s=8.0, width=320, height=180, codec="libvpx-vp9")
    (item_dir / "segments.csv").write_text("start_time,end_time\n1.0,2.0\n4.0,5.0\n", encoding="utf-8")
    (item_dir / "thumb.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")
    (item_dir / "meta.json").write_text(
        json.dumps({
            "id": item_id, "name": "Direct Playback Match", "created": "2099-01-01T00:00:00",
            "created_ts": time.time(), "duration_s": 8.0, "point_duration_s": 2.0, "n_segments": 2,
        }),
        encoding="utf-8",
    )
    return item_id


def _open_to_library(page: Page, base_url: str) -> None:
    """Land on the library view, dismissing the welcome screen if it shows.

    The per-test prefs reset usually brings the welcome back, but a previous
    test's fire-and-forget welcome-seen POST can land after the reset — so
    treat the welcome as optional here. test_ui_welcome_dismisses_to_library
    (first in the module, nothing racing it) asserts the strict welcome flow.
    """
    page.goto(base_url)
    start = page.locator("#welcomeStartBtn")
    try:
        start.wait_for(state="visible", timeout=3_000)
        start.click()
    except PlaywrightTimeoutError:
        pass
    expect(page.locator("#libraryView")).to_be_visible()


def test_ui_welcome_dismisses_to_library(page: Page, ui_backend: BackendClient):
    """Welcome dismisses to the saved-matches library (the default view)."""
    page.goto(ui_backend.base_url)
    # Fresh install (prefs reset) + fresh context (no localStorage): the
    # welcome must show, and dismissing it must land on the library.
    expect(page.locator("#welcomeScreen")).to_be_visible()
    page.locator("#welcomeStartBtn").click()
    expect(page.locator("#libraryView")).to_be_visible()
    expect(page.locator("#welcomeScreen")).to_be_hidden()
    expect(page.locator("#newMatchBtn")).to_be_visible()


def test_ui_shows_update_action_when_release_is_newer(page: Page, ui_backend: BackendClient):
    """Update check stays out of the way unless the backend reports a newer release."""
    opened = {"called": False}

    def fulfill_status(route):
        route.fulfill(
            json={
                "current_version": "0.1.0",
                "latest_version": "0.1.1",
                "latest_tag": "v0.1.1",
                "update_available": True,
                "release_url": "https://github.com/iroblesrazzaq/RallyClip/releases/tag/v0.1.1",
                "release_name": "v0.1.1",
                "error": None,
            }
        )

    def fulfill_open(route):
        opened["called"] = True
        route.fulfill(json={"opened": True, "release_url": "https://github.com/iroblesrazzaq/RallyClip/releases"})

    page.route("**/api/update/status", fulfill_status)
    page.route("**/api/update/open", fulfill_open)

    _open_to_library(page, ui_backend.base_url)
    expect(page.locator("#updateBtn")).to_be_visible()
    expect(page.locator("#updateBtn")).to_have_text("Update v0.1.1")

    with page.expect_response(lambda response: "/api/update/open" in response.url) as response_info:
        page.locator("#updateBtn").click()
    assert response_info.value.ok
    assert opened["called"] is True


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


def test_ui_viewer_uses_source_timeline_scheduler(page: Page, ui_backend: BackendClient):
    """Saved-match viewer loads source chunks and schedules point skips from
    explicit source-time segments."""
    item_id = _fabricate_viewer_item()
    _open_to_library(page, ui_backend.base_url)

    page.locator(f'.lib-card[data-id="{item_id}"]').click()
    expect(page.locator("#viewerView")).to_be_visible()
    expect(page.locator("#viewerTimeline")).to_be_visible(timeout=30_000)
    # Direct source playback engages when the engine can decode the clip;
    # otherwise the viewer falls back to WebM preview windows.
    page.wait_for_function(
        "() => { const src = window.rallyClipApp?.matchVideo?.src || ''; return src.includes('/preview/window') || src.includes('/source'); }",
        timeout=30_000,
    )
    page.wait_for_function(
        "() => Boolean(window.rallyClipApp?.viewerHasVideo?.())",
        timeout=30_000,
    )
    # The initial seek to the first point lands asynchronously after the video
    # element appears; on slow runners seekValue is still 0 at this point.
    page.wait_for_function(
        "() => Number(window.rallyClipApp?.viewerSeek?.value) > 0",
        timeout=30_000,
    )
    expect(page.locator("#viewerBackBtn")).to_be_visible()
    expect(page.locator("#viewerPlayPauseBtn")).to_be_visible()
    expect(page.locator("#viewerForwardBtn")).to_be_visible()

    startup = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            return {
                sourceDuration: app.sourceDuration,
                chunkDuration: app.previewWindowDuration,
                points: app.pointIntervals,
                seekValue: Number(app.viewerSeek.value),
                pointMarkers: app.viewerPointTrack.querySelectorAll(".viewer-point-segment").length,
            };
        }"""
    )
    assert startup["sourceDuration"] == 12
    assert startup["chunkDuration"] == 8
    assert startup["points"] == [{"start": 1, "end": 2}, {"start": 4, "end": 5}]
    assert startup["seekValue"] == pytest.approx(1.0, abs=0.2)
    assert startup["pointMarkers"] == 2

    scheduler = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            app.previewWindowDuration = 8;
            app.previewLookaheadChunks = 12;
            app.sourceDuration = 120;
            app.pointIntervals = [
                { start: 30, end: 40 },
                { start: 70, end: 74 },
                { start: 100, end: 104 },
            ];
            const inside = app.playbackSegmentForSourceTime(35);
            const beforeFirst = app.playbackSegmentForSourceTime(20);
            const gap = app.playbackSegmentForSourceTime(55);
            const tail = app.playbackSegmentForSourceTime(110);
            app.activePlaybackSegment = app.playbackSegmentForSourceTime(35);
            const insidePrefetch = app.getSchedulerPrefetchStarts(35);
            app.activePlaybackSegment = app.playbackSegmentForSourceTime(55);
            const gapPrefetch = app.getSchedulerPrefetchStarts(55);
            app.activePlaybackSegment = app.playbackSegmentForSourceTime(110);
            app.setPlaybackSegmentForSourceTime(55);
            return {
                canonical: [
                    app.canonicalPreviewChunkStart(30),
                    app.canonicalPreviewChunkStart(41.9),
                    app.canonicalPreviewChunkStart(42),
                ],
                currentWindow: app.previewWindowRequestForSourceTime(30),
                nextWindow: app.previewWindowRequestForSourceTime(64),
                inside,
                beforeFirst,
                gap,
                tail,
                insidePrefetch,
                gapPrefetch,
                backwardSeekSegment: app.activePlaybackSegment,
            };
        }"""
    )
    assert scheduler == {
        "canonical": [24, 40, 40],
        "currentWindow": {"start": 24, "duration": 8, "end": 32},
        "nextWindow": {"start": 64, "duration": 8, "end": 72},
        "inside": {"kind": "point", "start": 35, "end": 40, "pointIndex": 0, "nextPointIndex": 1},
        "beforeFirst": {"kind": "gap", "start": 20, "end": 40, "pointIndex": 0, "nextPointIndex": 1},
        "gap": {"kind": "gap", "start": 55, "end": 74, "pointIndex": 1, "nextPointIndex": 2},
        "tail": {"kind": "tail", "start": 110, "end": 120, "pointIndex": None, "nextPointIndex": None},
        "insidePrefetch": [32, 64],
        "gapPrefetch": [48, 56, 64, 72, 96],
        "backwardSeekSegment": {"kind": "gap", "start": 55, "end": 74, "pointIndex": 1, "nextPointIndex": 2},
    }

    default_skip = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            const originalSeek = app.seekViewerToSourceTime.bind(app);
            const originalGetTime = app.getViewerSourceTime.bind(app);
            const originalPrefetch = app.prefetchForPlaybackSchedule.bind(app);
            const pausedDescriptor = Object.getOwnPropertyDescriptor(app.matchVideo, "paused");
            const calls = [];
            app.pointIntervals = [{ start: 1, end: 2 }, { start: 4, end: 5 }, { start: 10, end: 12 }];
            app.activePlaybackSegment = { kind: "point", start: 1.8, end: 2, pointIndex: 0, nextPointIndex: 1 };
            app.getViewerSourceTime = () => 2.03;
            app.prefetchForPlaybackSchedule = () => {};
            app.seekViewerToSourceTime = (time, autoplay) => calls.push({ time, autoplay });
            Object.defineProperty(app.matchVideo, "paused", { value: false, configurable: true });
            app.handleViewerTimeUpdate();
            app.seekViewerToSourceTime = originalSeek;
            app.getViewerSourceTime = originalGetTime;
            app.prefetchForPlaybackSchedule = originalPrefetch;
            if (pausedDescriptor) Object.defineProperty(app.matchVideo, "paused", pausedDescriptor);
            else delete app.matchVideo.paused;
            return calls;
        }"""
    )
    assert default_skip == [{"time": 4, "autoplay": True}]

    gap_to_point_start_is_continuous = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            const originalSeek = app.seekViewerToSourceTime.bind(app);
            const originalGetTime = app.getViewerSourceTime.bind(app);
            const originalPrefetch = app.prefetchForPlaybackSchedule.bind(app);
            const originalPause = app.matchVideo.pause;
            const pausedDescriptor = Object.getOwnPropertyDescriptor(app.matchVideo, "paused");
            const calls = [];
            let pauses = 0;
            app.pointIntervals = [{ start: 1, end: 2 }, { start: 4, end: 5 }, { start: 10, end: 12 }];
            app.activePlaybackSegment = { kind: "gap", start: 2.5, end: 5, pointIndex: 1, nextPointIndex: 2 };
            app.lastViewerTime = 3.95;
            app.getViewerSourceTime = () => 4.03;
            app.prefetchForPlaybackSchedule = () => {};
            app.seekViewerToSourceTime = (time, autoplay) => calls.push({ time, autoplay });
            Object.defineProperty(app.matchVideo, "paused", { value: false, configurable: true });
            Object.defineProperty(app.matchVideo, "pause", {
                value: () => { pauses += 1; },
                configurable: true,
            });
            app.handleViewerTimeUpdate();
            const result = {
                calls,
                pauses,
                lastViewerTime: app.lastViewerTime,
                activePlaybackSegment: app.activePlaybackSegment,
            };
            app.seekViewerToSourceTime = originalSeek;
            app.getViewerSourceTime = originalGetTime;
            app.prefetchForPlaybackSchedule = originalPrefetch;
            app.matchVideo.pause = originalPause;
            if (pausedDescriptor) Object.defineProperty(app.matchVideo, "paused", pausedDescriptor);
            else delete app.matchVideo.paused;
            return result;
        }"""
    )
    assert gap_to_point_start_is_continuous == {
        "calls": [],
        "pauses": 0,
        "lastViewerTime": 4.03,
        "activePlaybackSegment": {"kind": "gap", "start": 2.5, "end": 5, "pointIndex": 1, "nextPointIndex": 2},
    }

    manual_gap_bridge = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            const originalSeek = app.seekViewerToSourceTime.bind(app);
            const originalGetTime = app.getViewerSourceTime.bind(app);
            const originalPrefetch = app.prefetchForPlaybackSchedule.bind(app);
            const pausedDescriptor = Object.getOwnPropertyDescriptor(app.matchVideo, "paused");
            const calls = [];
            app.pointIntervals = [{ start: 1, end: 2 }, { start: 4, end: 5 }, { start: 10, end: 12 }];
            app.activePlaybackSegment = { kind: "gap", start: 2.5, end: 5, pointIndex: 1, nextPointIndex: 2 };
            app.getViewerSourceTime = () => 5.04;
            app.prefetchForPlaybackSchedule = () => {};
            app.seekViewerToSourceTime = (time, autoplay) => calls.push({ time, autoplay });
            Object.defineProperty(app.matchVideo, "paused", { value: false, configurable: true });
            app.handleViewerTimeUpdate();
            app.seekViewerToSourceTime = originalSeek;
            app.getViewerSourceTime = originalGetTime;
            app.prefetchForPlaybackSchedule = originalPrefetch;
            if (pausedDescriptor) Object.defineProperty(app.matchVideo, "paused", pausedDescriptor);
            else delete app.matchVideo.paused;
            return calls;
        }"""
    )
    assert manual_gap_bridge == [{"time": 10, "autoplay": True}]

    bridge_continues_across_chunks = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            const originalLoad = app.loadPreviewWindowAt.bind(app);
            const calls = [];
            app.loadPreviewWindowAt = (time, autoplay, options) => calls.push({
                time,
                autoplay,
                preserveSegment: Boolean(options?.preserveSegment),
            });
            app.viewingItemId = "match-ready";
            app.directPlayback = false;
            app.sourceDuration = 120;
            app.pointIntervals = [{ start: 1, end: 2 }, { start: 4, end: 5 }, { start: 10, end: 12 }];
            app.activePlaybackSegment = { kind: "gap", start: 2.5, end: 5, pointIndex: 1, nextPointIndex: 2 };
            app.currentPreviewWindowStart = 2;
            app.currentPreviewWindowDuration = 1;
            app.matchVideo.dataset.windowStart = "2";
            app.matchVideo.dataset.windowDuration = "1";
            app.readyPreviewWindows.clear();
            app.previewLoadInProgress = false;
            app.handleViewerWindowEnded();
            app.loadPreviewWindowAt = originalLoad;
            return calls;
        }"""
    )
    assert bridge_continues_across_chunks == [{"time": 3, "autoplay": True, "preserveSegment": True}]

    chunk_error = page.evaluate(
        """() => new Promise((resolve) => {
            const app = window.rallyClipApp;
            const originalFetch = window.fetch;
            const originalToast = app.showToast.bind(app);
            const toasts = [];
            window.fetch = () => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ status: "error", ready: false, error: "Missing VP8 encoder" }),
            });
            app.showToast = (message, kind) => toasts.push({ message, kind });
            app.viewingItemId = "match-error";
            app.previewRequestSeq = 0;
            app.previewLoadInProgress = false;
            app.previewPollTimeout = null;
            app.hidePreviewLoading();
            app.loadPreviewWindowAt(12, true);
            setTimeout(() => {
                const result = {
                    loading: app.previewLoadInProgress,
                    pollActive: Boolean(app.previewPollTimeout),
                    errorText: app.previewStatus.textContent,
                    hidden: app.previewStatus.hidden,
                    isError: app.previewStatus.classList.contains("is-error"),
                    toasts,
                };
                window.fetch = originalFetch;
                app.showToast = originalToast;
                resolve(result);
            }, 0);
        })"""
    )
    assert chunk_error == {
        "loading": False,
        "pollActive": False,
        "errorText": "Missing VP8 encoder",
        "hidden": False,
        "isError": True,
        "toasts": [{"message": "Missing VP8 encoder", "kind": "error"}],
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
            const defaultFullscreenLabel = app.viewerFullscreenBtn.textContent;
            const defaultFullscreenAria = app.viewerFullscreenBtn.getAttribute("aria-label");
            const originalFullscreenElement = app.viewerFullscreenElement.bind(app);
            app.viewerFullscreenElement = () => app.viewerVideoWrap;
            app.updateViewerFullscreenState();
            const activeFullscreenLabel = app.viewerFullscreenBtn.textContent;
            const activeFullscreenAria = app.viewerFullscreenBtn.getAttribute("aria-label");
            const activeFullscreenClass = app.viewerVideoWrap.classList.contains("is-fullscreen");
            app.viewerFullscreenElement = () => null;
            app.updateViewerFullscreenState();
            const resetFullscreenLabel = app.viewerFullscreenBtn.textContent;
            const resetFullscreenClass = app.viewerVideoWrap.classList.contains("is-fullscreen");
            app.viewerFullscreenElement = originalFullscreenElement;

            app.currentPreviewWindowStart = 20;
            app.currentPreviewWindowDuration = 20;
            app.matchVideo.dataset.windowStart = "20";
            app.matchVideo.dataset.windowDuration = "20";
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
            return {
                pausedLabel,
                playingLabel,
                defaultFullscreenLabel,
                defaultFullscreenAria,
                activeFullscreenLabel,
                activeFullscreenAria,
                activeFullscreenClass,
                resetFullscreenLabel,
                resetFullscreenClass,
                calls,
                prevented,
            };
        }"""
    )
    assert controls == {
        "pausedLabel": "▶",
        "playingLabel": "❚❚",
        "defaultFullscreenLabel": "⛶",
        "defaultFullscreenAria": "Enter fullscreen",
        "activeFullscreenLabel": "×",
        "activeFullscreenAria": "Exit fullscreen",
        "activeFullscreenClass": True,
        "resetFullscreenLabel": "⛶",
        "resetFullscreenClass": False,
        "calls": [{"time": 23, "autoplay": True}, {"time": 33, "autoplay": True}],
        "prevented": 2,
    }

    skip_buttons = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            const calls = [];
            const originalSeek = app.seekViewerToSourceTime.bind(app);
            const originalGetTime = app.getViewerSourceTime.bind(app);
            app.sourceDuration = 120;
            app.getViewerSourceTime = () => 30;
            app.seekViewerToSourceTime = (time, autoplay) => calls.push({ time, autoplay });
            app.viewerBackBtn.click();
            app.viewerForwardBtn.click();
            app.seekViewerToSourceTime = originalSeek;
            app.getViewerSourceTime = originalGetTime;
            return calls;
        }"""
    )
    assert skip_buttons == [{"time": 25, "autoplay": True}, {"time": 35, "autoplay": True}]

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

    timeline_config_preserves_time = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            app.configureViewerTimeline(120, 42.5);
            return {
                value: Number(app.viewerSeek.value),
                current: app.viewerCurrentTime.textContent,
                duration: app.viewerDuration.textContent,
            };
        }"""
    )
    assert timeline_config_preserves_time == {"value": 42.5, "current": "0:42", "duration": "2:00"}


def test_ui_direct_playback_swaps_to_preseeked_standby_between_points(page: Page, ui_backend: BackendClient):
    """Continuous playback: crossing a point boundary in direct playback must
    swap to the pre-seeked standby <video>, not cold-seek the playing element
    (a cold seek stalls audio while the browser range-fetches + decodes)."""
    item_id = _fabricate_direct_playback_item()
    _open_to_library(page, ui_backend.base_url)

    page.locator(f'.lib-card[data-id="{item_id}"]').click()
    expect(page.locator("#viewerView")).to_be_visible()
    try:
        page.wait_for_function("() => window.rallyClipApp?.directPlayback === true", timeout=15_000)
    except PlaywrightTimeoutError:
        pytest.skip("browser cannot decode the VP9 source; direct playback unavailable")

    initial_video_id = page.evaluate("() => window.rallyClipApp.matchVideo.id")
    # Muted so headless Chromium's autoplay policy cannot block play(); the
    # swap copies muted/volume to the standby element.
    page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            app.matchVideo.muted = true;
            app.matchVideoBuffer.muted = true;
            app.seekViewerToSourceTime(1.4, true);
        }"""
    )
    # Playback must roll out of point 1 (1-2s) and land in point 2 (4-5s) on
    # its own — the watcher prepares the standby and swaps at the boundary.
    page.wait_for_function(
        "() => window.rallyClipApp.getViewerSourceTime() >= 3.9",
        timeout=20_000,
    )
    after = page.evaluate(
        """() => {
            const app = window.rallyClipApp;
            return {
                videoId: app.matchVideo.id,
                time: app.getViewerSourceTime(),
                paused: app.matchVideo.paused,
                direct: app.directPlayback,
                windowStart: Number(app.matchVideo.dataset.windowStart),
            };
        }"""
    )
    assert after["videoId"] != initial_video_id, "boundary should swap video elements, not seek in place"
    assert after["direct"] is True
    assert after["paused"] is False
    assert after["windowStart"] == 0
    assert 3.9 <= after["time"] <= 5.6


def test_ui_segment_edit_mode_trims_adds_deletes_and_resets(page: Page, ui_backend: BackendClient):
    """Edit mode: drag-trim a point, add and delete points, then reset. Edits
    land in segments_edited.csv; segments.csv is never touched."""
    from gui import app as gui_app

    item_id = _fabricate_viewer_item()
    item_dir = Path(gui_app.LIBRARY_DIR) / item_id
    original_csv = (item_dir / "segments.csv").read_text(encoding="utf-8")
    _open_to_library(page, ui_backend.base_url)

    page.locator(f'.lib-card[data-id="{item_id}"]').click()
    expect(page.locator("#viewerView")).to_be_visible()
    expect(page.locator("#viewerTimeline")).to_be_visible(timeout=30_000)
    page.wait_for_function(
        "() => Boolean(window.rallyClipApp?.viewerHasVideo?.())",
        timeout=30_000,
    )

    page.locator("#viewerEditBtn").click()
    expect(page.locator("#viewerEditBar")).to_be_visible()
    expect(page.locator(".viewer-edit-segment")).to_have_count(2)
    # First point is auto-selected with trim handles.
    selected = page.locator(".viewer-edit-segment.is-selected")
    expect(selected).to_have_count(1)
    expect(selected.locator(".viewer-edit-handle")).to_have_count(2)

    # Drag the end handle right: point 1 (1.0-2.0 of a 12s clip) stretches
    # toward 3s. The track spans the seek wrap, so 1s = 1/12 of its width.
    wrap_box = page.locator(".viewer-seek-wrap").bounding_box()
    handle_box = selected.locator('.viewer-edit-handle[data-edge="end"]').bounding_box()
    start_x = handle_box["x"] + handle_box["width"] / 2
    start_y = handle_box["y"] + handle_box["height"] / 2
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(start_x + wrap_box["width"] / 12, start_y, steps=8)
    page.mouse.up()

    page.wait_for_function(
        f"""() => fetch('/api/library/{item_id}/segments')
            .then((r) => r.json())
            .then((p) => p.edited && p.segments[0].end > 2.4 && p.segments[0].end < 3.6)"""
    )
    assert (item_dir / "segments_edited.csv").exists()
    assert (item_dir / "segments.csv").read_text(encoding="utf-8") == original_csv

    # Add a point at ~8s (a gap), then delete it again. In preview-window
    # fallback mode the seek lands asynchronously, so wait for it to settle
    # before adding (add uses the current playback time).
    page.evaluate("() => window.rallyClipApp.seekViewerToSourceTime(8, false)")
    page.wait_for_function(
        "() => Math.abs(window.rallyClipApp.getViewerSourceTime() - 8) < 0.5",
        timeout=30_000,
    )
    page.locator("#editAddPointBtn").click()
    expect(page.locator(".viewer-edit-segment")).to_have_count(3)
    page.wait_for_function(
        f"""() => fetch('/api/library/{item_id}/segments')
            .then((r) => r.json())
            .then((p) => p.segments.length === 3)"""
    )
    page.locator("#editDeletePointBtn").click()
    expect(page.locator(".viewer-edit-segment")).to_have_count(2)

    # Reset discards the edited copy and restores the original times.
    page.on("dialog", lambda dialog: dialog.accept())
    expect(page.locator("#editResetBtn")).to_be_enabled()
    page.locator("#editResetBtn").click()
    page.wait_for_function(
        f"""() => fetch('/api/library/{item_id}/segments')
            .then((r) => r.json())
            .then((p) => !p.edited && p.segments.length === 2 && p.segments[0].end === 2)"""
    )
    assert not (item_dir / "segments_edited.csv").exists()
    edit_state = page.evaluate(
        "() => ({ points: window.rallyClipApp.pointIntervals, edited: window.rallyClipApp.segmentsEdited })"
    )
    assert edit_state["points"] == [{"start": 1, "end": 2}, {"start": 4, "end": 5}]
    assert edit_state["edited"] is False

    page.locator("#editDoneBtn").click()
    expect(page.locator("#viewerEditBar")).to_be_hidden()


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
