const WELCOME_SEEN_KEY = "rallyclip_welcome_seen";
const JOB_ID_KEY = "rallyclip_job_id";
const WELCOME_TYPE_START_DELAY_MS = 504;
const WELCOME_TYPE_LINE_DELAY_MS = WELCOME_TYPE_START_DELAY_MS;
const WELCOME_TYPE_FIRST_CHAR_MS = 110;
const WELCOME_TYPE_CHAR_MS = 89;
const SYSTEM_DARK_QUERY = "(prefers-color-scheme: dark)";
const MSE_PREVIEW_MIME = 'video/webm; codecs="vp8,opus"';
const WELCOME_LAYOUT = [
    [{ text: "RallyClip.", immediate: true }],
    [{ text: "AI Match Segmentation." }],
    [{ text: "Free, Forever.", className: "welcome-free" }],
];

function applySystemTheme() {
    const isDark = window.matchMedia && window.matchMedia(SYSTEM_DARK_QUERY).matches;
    document.documentElement.dataset.systemTheme = isDark ? "dark" : "light";
}

applySystemTheme();

class RallyClipApp {
    constructor() {
        this.selectedFile = null;
        this.isProcessing = false;
        this.currentJobId = null;
        this.progressInterval = null;
        this.defaults = {};
        this.warnings = {};
        this.availableDevices = [];
        this.autoDevice = "cpu";
        this.runtimeState = "cold";
        this.runtimeWarmupPoll = null;
        this.weights = null;
        this.etaSeconds = null;
        this.libraryId = null;
        this.viewingItemId = null;
        this.previewPollTimeout = null;
        this.pointIntervals = [];
        this.lastViewerTime = null;
        this.viewerSeekInProgress = false;
        this.viewerSkipSeconds = 5;
        this.previewWindowDuration = 8;
        this.previewMaxWindowDuration = 90;
        this.previewLookaheadChunks = 12;
        this.pointBufferSeconds = 5;
        this.currentPreviewWindowStart = 0;
        this.currentPreviewWindowDuration = 0;
        this.sourceDuration = 0;
        this.previewRequestSeq = 0;
        this.previewLoadInProgress = false;
        this.directPlayback = false;
        this.directProbeInProgress = false;
        this.previewSpinnerTimeout = null;
        this.viewerControlsHideTimeout = null;
        this.welcomeTypeTimers = [];
        this.welcomeCursor = null;
        this.systemThemeQuery = window.matchMedia ? window.matchMedia(SYSTEM_DARK_QUERY) : null;
        this.prefetchedWindowKeys = new Set();
        this.prefetchWindowTimers = new Map();
        this.readyPreviewWindows = new Map();
        this.previewTransitionInProgress = false;
        this.pendingPreviewTransition = null;
        this.pendingOutgoingVideo = null;
        this.viewerSeekDragging = false;
        this.activePlaybackSegment = null;
        this.viewerFullscreenFallback = false;
        this.mseActive = false;
        this.mseDisabled = false;
        this.mseObjectUrl = null;
        this.mseMediaSource = null;
        this.mseSourceBuffer = null;
        this.mseSessionId = 0;
        this.mseAppendQueue = [];
        this.mseCurrentAppend = null;
        this.mseQueuedWindowKeys = new Set();
        this.mseAppendedWindowKeys = new Set();
        this.msePendingSeek = null;
        this.msePrunePending = false;
        this.nativeViewer = null;
        this.nativeBridgeReady = false;
        this.updateStatus = null;
        this.steps = ["pose", "preprocess", "feature", "inference", "output"];
        this.stepLabels = {
            pose: "Extracting pose",
            preprocess: "Preprocessing",
            feature: "Building features",
            inference: "Finding points",
            output: "Saving match",
        };

        this.cacheElements();
        this.bindEvents();
        this.bindSystemTheme();
        this.initNativeViewerBridge();
        this.loadDefaults();
        this.checkForUpdates();
        this.restoreJobIfAny().then((restored) => {
            if (restored) return;
            this.showInitialView();
        });
    }

    cacheElements() {
        this.welcomeScreen = document.getElementById("welcomeScreen");
        this.welcomeHeadline = document.getElementById("welcomeHeadline");
        this.welcomeStartBtn = document.getElementById("welcomeStartBtn");
        this.appShell = document.getElementById("appShell");

        this.libraryView = document.getElementById("libraryView");
        this.libraryGrid = document.getElementById("libraryGrid");
        this.libraryEmpty = document.getElementById("libraryEmpty");
        this.newMatchBtn = document.getElementById("newMatchBtn");
        this.updateBtn = document.getElementById("updateBtn");

        this.viewerView = document.getElementById("viewerView");
        this.backFromViewer = document.getElementById("backFromViewer");
        this.viewerTitle = document.getElementById("viewerTitle");
        this.viewerMeta = document.getElementById("viewerMeta");
        this.viewerVideoWrap = document.querySelector(".viewer-video-wrap");
        this.primaryMatchVideo = document.getElementById("matchVideo");
        this.secondaryMatchVideo = document.getElementById("matchVideoBuffer");
        this.matchVideo = this.primaryMatchVideo;
        this.matchVideoBuffer = this.secondaryMatchVideo;
        this.previewStatus = document.getElementById("previewStatus");
        this.viewerOverlay = document.getElementById("viewerOverlay");
        this.viewerControls = document.getElementById("viewerControls");
        this.viewerBackBtn = document.getElementById("viewerBackBtn");
        this.viewerPlayPauseBtn = document.getElementById("viewerPlayPauseBtn");
        this.viewerForwardBtn = document.getElementById("viewerForwardBtn");
        this.viewerFullscreenBtn = document.getElementById("viewerFullscreenBtn");
        this.viewerTimeline = document.getElementById("viewerTimeline");
        this.viewerSeek = document.getElementById("viewerSeek");
        this.viewerSeekWrap = document.querySelector(".viewer-seek-wrap");
        this.viewerBufferTrack = document.getElementById("viewerBufferTrack");
        this.viewerPointTrack = document.getElementById("viewerPointTrack");
        this.viewerCurrentTime = document.getElementById("viewerCurrentTime");
        this.viewerDuration = document.getElementById("viewerDuration");
        this.viewerExportBtn = document.getElementById("viewerExportBtn");
        this.viewerCsvBtn = document.getElementById("viewerCsvBtn");

        this.uploadView = document.getElementById("uploadView");
        this.backToLibrary = document.getElementById("backToLibrary");
        this.dropZone = document.getElementById("dropZone");
        this.fileInput = document.getElementById("fileInput");
        this.browseBtn = document.getElementById("browseBtn");
        this.selectedFileDiv = document.getElementById("selectedFile");
        this.fileName = document.getElementById("fileName");
        this.fileSize = document.getElementById("fileSize");
        this.removeFileBtn = document.getElementById("removeFileBtn");
        this.startBtn = document.getElementById("startBtn");
        this.cancelBtn = document.getElementById("cancelBtn");
        this.advancedToggle = document.getElementById("advancedToggle");
        this.advancedPanel = document.getElementById("advancedPanel");
        this.resetAdvanced = document.getElementById("resetAdvanced");
        this.outputName = document.getElementById("outputName");
        this.yoloDevice = document.getElementById("yoloDevice");
        this.deviceNote = document.getElementById("deviceNote");
        this.low = document.getElementById("low");
        this.high = document.getElementById("high");
        this.minDurSec = document.getElementById("minDurSec");
        this.lowWarning = document.getElementById("lowWarning");
        this.highWarning = document.getElementById("highWarning");
        this.minDurWarning = document.getElementById("minDurWarning");

        this.progressCard = document.getElementById("progress");
        this.stageText = document.getElementById("stageText");
        this.overallFill = document.getElementById("overallFill");
        this.overallPercentage = document.getElementById("overallPercentage");
        this.etaText = document.getElementById("etaText");
        this.progressDetails = document.querySelector(".progress-details");
        this.toastStack = document.getElementById("toastStack");

        this.progressItems = {
            pose: { status: document.getElementById("poseStatus"), fill: document.getElementById("poseFill") },
            preprocess: { status: document.getElementById("preprocessStatus"), fill: document.getElementById("preprocessFill") },
            feature: { status: document.getElementById("featureStatus"), fill: document.getElementById("featureFill") },
            inference: { status: document.getElementById("inferenceStatus"), fill: document.getElementById("inferenceFill") },
            output: { status: document.getElementById("outputStatus"), fill: document.getElementById("outputFill") },
        };
    }

    bindEvents() {
        this.welcomeStartBtn.addEventListener("click", () => this.dismissWelcome());
        this.newMatchBtn.addEventListener("click", () => this.showUpload());
        this.updateBtn.addEventListener("click", () => this.openUpdatePage());
        this.backToLibrary.addEventListener("click", () => this.showLibrary());
        this.backFromViewer.addEventListener("click", () => this.showLibrary());
        this.viewerExportBtn.addEventListener("click", () => {
            if (this.viewingItemId) this.triggerDownload(`/api/library/${this.viewingItemId}/video`);
        });
        this.viewerCsvBtn.addEventListener("click", () => {
            if (this.viewingItemId) this.triggerDownload(`/api/library/${this.viewingItemId}/csv`);
        });
        [this.primaryMatchVideo, this.secondaryMatchVideo].forEach((video) => this.bindViewerVideoEvents(video));
        this.viewerVideoWrap.addEventListener("pointermove", () => this.showViewerControls());
        this.viewerVideoWrap.addEventListener("pointerenter", () => this.showViewerControls());
        this.viewerVideoWrap.addEventListener("focusin", () => this.showViewerControls());
        this.viewerVideoWrap.addEventListener("click", (e) => this.handleViewerSurfaceClick(e));
        this.viewerBackBtn.addEventListener("click", () => this.skipViewerBy(-this.viewerSkipSeconds));
        this.viewerPlayPauseBtn.addEventListener("click", (e) => this.toggleViewerPlayback(e));
        this.viewerForwardBtn.addEventListener("click", () => this.skipViewerBy(this.viewerSkipSeconds));
        this.viewerFullscreenBtn.addEventListener("click", (e) => this.toggleViewerFullscreen(e));
        this.viewerSeek.addEventListener("input", () => {
            this.viewerSeekDragging = true;
            this.updateViewerTimeline(Number(this.viewerSeek.value));
        });
        this.viewerSeek.addEventListener("change", () => {
            this.viewerSeekDragging = false;
            this.seekViewerToSourceTime(Number(this.viewerSeek.value), this.viewerHasVideo() && !this.matchVideo.paused);
        });
        this.viewerSeekWrap.addEventListener("pointerdown", (e) => this.seekViewerFromTimelinePointer(e));
        document.addEventListener("keydown", (e) => this.handleViewerKeyboardShortcuts(e), true);
        document.addEventListener("fullscreenchange", () => this.updateViewerFullscreenState());
        document.addEventListener("webkitfullscreenchange", () => this.updateViewerFullscreenState());
        this.libraryGrid.addEventListener("click", (e) => this.onLibraryClick(e));

        this.dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            this.dropZone.classList.add("dragover");
        });
        this.dropZone.addEventListener("dragleave", () => this.dropZone.classList.remove("dragover"));
        this.dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            this.dropZone.classList.remove("dragover");
            if (e.dataTransfer.files.length) this.processFile(e.dataTransfer.files[0]);
        });
        this.dropZone.addEventListener("click", () => this.fileInput.click());
        this.browseBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.fileInput.click();
        });
        this.fileInput.addEventListener("change", (e) => {
            if (e.target.files.length) this.processFile(e.target.files[0]);
        });
        this.removeFileBtn.addEventListener("click", () => this.removeFile());
        this.startBtn.addEventListener("click", () => this.startAnalysis());
        this.cancelBtn.addEventListener("click", () => this.cancelAnalysis());
        this.advancedToggle.addEventListener("click", () => {
            this.advancedPanel.hidden = !this.advancedPanel.hidden;
        });
        this.resetAdvanced.addEventListener("click", () => {
            this.applyDefaults();
            this.showToast("Advanced settings reset", "success");
        });
        this.yoloDevice.addEventListener("change", () => this.updateDeviceNote());
    }

    bindViewerVideoEvents(video) {
        if (!video) return;
        video.addEventListener("timeupdate", () => {
            if (video === this.matchVideo) this.handleViewerTimeUpdate();
        });
        video.addEventListener("play", () => {
            if (video === this.matchVideo) this.updateViewerControls();
        });
        video.addEventListener("pause", () => {
            if (video === this.matchVideo) this.updateViewerControls();
        });
        video.addEventListener("loadedmetadata", () => {
            if (video === this.matchVideo) this.updateViewerControls();
        });
        video.addEventListener("seeking", () => {
            if (video === this.matchVideo) this.viewerSeekInProgress = true;
        });
        video.addEventListener("seeked", () => {
            if (video !== this.matchVideo) return;
            this.viewerSeekInProgress = false;
            this.lastViewerTime = this.getViewerSourceTime();
        });
        video.addEventListener("ended", () => {
            if (video === this.matchVideo) this.handleViewerWindowEnded();
        });
        video.addEventListener("waiting", () => {
            if (video === this.matchVideo && this.mseActive) this.ensureMsePlaybackRange(this.getViewerSourceTime());
        });
        video.addEventListener("error", () => {
            if (video === this.matchVideo) this.handleViewerVideoError();
        });
        video.addEventListener("click", (e) => {
            e.stopPropagation();
            if (video === this.matchVideo) this.toggleViewerPlayback(e);
        });
    }

    bindSystemTheme() {
        if (!this.systemThemeQuery) return;
        const updateTheme = () => applySystemTheme();
        if (typeof this.systemThemeQuery.addEventListener === "function") {
            this.systemThemeQuery.addEventListener("change", updateTheme);
        } else if (typeof this.systemThemeQuery.addListener === "function") {
            this.systemThemeQuery.addListener(updateTheme);
        }
    }

    hasSeenWelcomeLocal() {
        try {
            return localStorage.getItem(WELCOME_SEEN_KEY) === "1";
        } catch (_) {
            return true;
        }
    }

    async hasSeenWelcome() {
        try {
            const resp = await fetch("/api/preferences/welcome");
            if (resp.ok) {
                const payload = await resp.json();
                if (payload.welcome_seen) {
                    this.markWelcomeSeenLocal();
                    return true;
                }
                return this.hasSeenWelcomeLocal();
            }
        } catch (_) {}
        return this.hasSeenWelcomeLocal();
    }

    async showInitialView() {
        if (await this.hasSeenWelcome()) this.showLibrary();
        else this.showWelcome();
    }

    async checkForUpdates() {
        try {
            const resp = await fetch("/api/update/status");
            if (!resp.ok) return;
            const payload = await resp.json();
            this.updateStatus = payload;
            this.renderUpdateStatus(payload);
        } catch (err) {
            console.debug("Could not check for updates", err);
        }
    }

    renderUpdateStatus(payload) {
        if (!this.updateBtn) return;
        const available = Boolean(payload && payload.update_available);
        this.updateBtn.hidden = !available;
        if (!available) return;
        const latest = payload.latest_tag || payload.latest_version || "latest";
        this.updateBtn.textContent = `Update ${latest}`;
        this.updateBtn.title = `RallyClip ${latest} is available`;
        if (payload.release_url) this.updateBtn.dataset.releaseUrl = payload.release_url;
    }

    async openUpdatePage() {
        const fallbackUrl = this.updateBtn?.dataset.releaseUrl || this.updateStatus?.release_url;
        try {
            const resp = await fetch("/api/update/open", { method: "POST" });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            return;
        } catch (err) {
            console.debug("Backend update opener failed", err);
        }
        if (fallbackUrl) window.open(fallbackUrl, "_blank", "noopener");
    }

    markWelcomeSeenLocal() {
        try {
            localStorage.setItem(WELCOME_SEEN_KEY, "1");
        } catch (_) {}
    }

    markWelcomeSeen() {
        this.markWelcomeSeenLocal();
        try {
            fetch("/api/preferences/welcome", { method: "POST" }).catch(() => {});
        } catch (_) {}
    }

    dismissWelcome() {
        this.markWelcomeSeen();
        this.clearWelcomeTypewriter();
        this.showLibrary();
    }

    showWelcome() {
        this.welcomeScreen.hidden = false;
        this.appShell.hidden = true;
        this.runWelcomeTypewriter();
    }

    clearWelcomeTypewriter() {
        this.welcomeTypeTimers.forEach((timer) => clearTimeout(timer));
        this.welcomeTypeTimers = [];
        this.welcomeHeadline?.querySelectorAll(".is-typing").forEach((item) => item.classList.remove("is-typing"));
        this.welcomeCursor?.remove();
        this.welcomeCursor = null;
    }

    welcomeTimeout(callback, delay) {
        const timer = setTimeout(() => {
            this.welcomeTypeTimers = this.welcomeTypeTimers.filter((item) => item !== timer);
            callback();
        }, delay);
        this.welcomeTypeTimers.push(timer);
    }

    runWelcomeTypewriter() {
        this.clearWelcomeTypewriter();
        this.welcomeScreen.classList.remove("is-ready");
        const pieces = this.renderWelcomeLayout();

        const prefersReducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (prefersReducedMotion) {
            pieces.forEach((piece) => {
                piece.element.textContent = piece.text;
            });
            this.welcomeCursor?.remove();
            this.welcomeCursor = null;
            this.welcomeScreen.classList.add("is-ready");
            return;
        }

        this.welcomeTimeout(() => {
            this.typeWelcomePieces(pieces, 0);
        }, WELCOME_TYPE_START_DELAY_MS);
    }

    renderWelcomeLayout() {
        const piecesToType = [];
        this.welcomeHeadline.replaceChildren();
        this.welcomeCursor = document.createElement("span");
        this.welcomeCursor.className = "typing-cursor";
        this.welcomeCursor.setAttribute("aria-hidden", "true");
        WELCOME_LAYOUT.forEach((line) => {
            const lineEl = document.createElement("span");
            lineEl.className = "welcome-line";
            line.forEach((piece) => {
                const pieceEl = document.createElement("span");
                if (piece.className) pieceEl.className = piece.className;
                if (piece.immediate) {
                    pieceEl.textContent = piece.text;
                }
                else piecesToType.push({ element: pieceEl, text: piece.text });
                lineEl.appendChild(pieceEl);
            });
            this.welcomeHeadline.appendChild(lineEl);
        });
        if (piecesToType.length) this.moveWelcomeCursor(piecesToType[0].element);
        return piecesToType;
    }

    typeWelcomePieces(pieces, index) {
        if (index >= pieces.length) {
            this.welcomeCursor?.remove();
            this.welcomeCursor = null;
            this.welcomeScreen.classList.add("is-ready");
            return;
        }
        this.typeWelcomeText(pieces[index].element, pieces[index].text, () => {
            const nextPiece = pieces[index + 1];
            if (nextPiece) this.moveWelcomeCursor(nextPiece.element);
            this.welcomeTimeout(() => this.typeWelcomePieces(pieces, index + 1), WELCOME_TYPE_LINE_DELAY_MS);
        });
    }

    typeWelcomeText(element, text, done, index = 0) {
        this.setActiveWelcomeTypingElement(element);
        if (index >= text.length) {
            element.classList.remove("is-typing");
            done();
            return;
        }
        if (this.welcomeCursor?.parentElement === element) this.welcomeCursor.remove();
        element.textContent += text[index];
        this.moveWelcomeCursor(element);
        this.welcomeTimeout(
            () => this.typeWelcomeText(element, text, done, index + 1),
            index < 1 ? WELCOME_TYPE_FIRST_CHAR_MS : WELCOME_TYPE_CHAR_MS,
        );
    }

    setActiveWelcomeTypingElement(element) {
        this.welcomeHeadline.querySelectorAll(".is-typing").forEach((item) => {
            if (item !== element) item.classList.remove("is-typing");
        });
        element.classList.add("is-typing");
        this.moveWelcomeCursor(element);
    }

    moveWelcomeCursor(element) {
        if (!this.welcomeCursor) return;
        element.appendChild(this.welcomeCursor);
    }

    showView(viewName) {
        this.clearWelcomeTypewriter();
        this.welcomeScreen.hidden = true;
        this.appShell.hidden = false;
        this.libraryView.hidden = viewName !== "library";
        this.viewerView.hidden = viewName !== "viewer";
        this.uploadView.hidden = viewName !== "upload";
        this.progressCard.hidden = viewName !== "processing";
        if (viewName !== "viewer" && this.matchVideo) {
            this.clearPreviewPoll();
            this.resetMsePreview();
            this.resetViewerVideos();
            this.directPlayback = false;
            this.viewingItemId = null;
            this.pointIntervals = [];
            this.lastViewerTime = null;
            this.viewerSeekInProgress = false;
            this.previewLoadInProgress = false;
            this.previewTransitionInProgress = false;
            this.pendingPreviewTransition = null;
            this.pendingOutgoingVideo = null;
            this.activePlaybackSegment = null;
            this.currentPreviewWindowStart = 0;
            this.currentPreviewWindowDuration = 0;
            this.sourceDuration = 0;
            this.clearPrefetchTimers();
            this.prefetchedWindowKeys.clear();
            this.readyPreviewWindows.clear();
            this.renderViewerBufferedRanges();
            this.renderViewerPointRanges();
            this.viewerSeekDragging = false;
            this.viewerTimeline.hidden = true;
            this.hideViewerControls();
            this.updateViewerControls();
            this.hidePreviewLoading();
        }
    }

    clearVideoElement(video) {
        if (!video) return;
        video.pause();
        video.removeAttribute("src");
        video.removeAttribute("data-preview-url");
        video.removeAttribute("data-window-start");
        video.removeAttribute("data-window-duration");
        video.load();
    }

    resetViewerVideos() {
        this.resetMsePreview();
        this.clearVideoElement(this.primaryMatchVideo);
        this.clearVideoElement(this.secondaryMatchVideo);
        this.matchVideo = this.primaryMatchVideo;
        this.matchVideoBuffer = this.secondaryMatchVideo;
        this.pendingPreviewTransition = null;
        this.primaryMatchVideo.classList.add("is-active");
        this.primaryMatchVideo.classList.remove("is-outgoing");
        this.secondaryMatchVideo.classList.remove("is-active", "is-outgoing");
    }

    showViewerControls() {
        if (!this.viewerVideoWrap || !this.isViewerActive()) return;
        if (this.viewerControlsHideTimeout) clearTimeout(this.viewerControlsHideTimeout);
        this.viewerVideoWrap.classList.add("is-controls-visible");
        this.viewerVideoWrap.classList.remove("is-controls-idle");
        this.viewerControlsHideTimeout = setTimeout(() => this.hideViewerControls(), 2600);
    }

    hideViewerControls() {
        if (this.viewerControlsHideTimeout) {
            clearTimeout(this.viewerControlsHideTimeout);
            this.viewerControlsHideTimeout = null;
        }
        if (!this.viewerVideoWrap) return;
        this.viewerVideoWrap.classList.remove("is-controls-visible");
        this.viewerVideoWrap.classList.add("is-controls-idle");
    }

    showLibrary() {
        this.showView("library");
        this.loadLibrary();
    }

    showUpload() {
        this.removeFile();
        this.showView("upload");
        this.warmAnalysisRuntime();
    }

    // ----- Library --------------------------------------------------------- //
    async loadLibrary() {
        try {
            const resp = await fetch("/api/library");
            if (!resp.ok) throw new Error("Failed to load library");
            const { items } = await resp.json();
            this.renderLibrary(items || []);
        } catch (err) {
            console.error(err);
            this.renderLibrary([]);
            this.showToast("Could not load saved matches", "error");
        }
    }

    renderLibrary(items) {
        const hasItems = items.length > 0;
        this.libraryGrid.innerHTML = "";
        this.libraryEmpty.hidden = hasItems;
        this.newMatchBtn.textContent = hasItems ? "New Match" : "Process a Match";
        items.forEach((item) => this.libraryGrid.appendChild(this.buildCard(item)));
    }

    buildCard(item) {
        const card = document.createElement("div");
        card.className = "lib-card";
        card.dataset.id = item.id;
        if (typeof item.duration_s === "number") card.dataset.duration = String(item.duration_s);
        card.tabIndex = 0;
        card.setAttribute("role", "button");
        card.setAttribute("aria-label", `View ${item.name || "saved match"}`);

        const thumb = document.createElement("div");
        thumb.className = "lib-thumb";
        if (item.has_thumbnail) {
            const img = document.createElement("img");
            img.src = `/api/library/${item.id}/thumbnail`;
            img.alt = item.name || "match";
            img.loading = "lazy";
            thumb.appendChild(img);
        }
        card.appendChild(thumb);

        const info = document.createElement("div");
        info.className = "lib-info";
        const name = document.createElement("div");
        name.className = "lib-name";
        name.textContent = item.name || item.id;
        const meta = document.createElement("div");
        meta.className = "lib-meta";
        meta.textContent = this.cardMeta(item);
        info.append(name, meta);
        card.appendChild(info);

        const actions = document.createElement("div");
        actions.className = "lib-actions";
        actions.appendChild(this.actionButton("Export video", "btn-primary", "export"));
        if (item.has_csv) actions.appendChild(this.actionButton("CSV", "btn-secondary", "csv"));
        actions.appendChild(this.actionButton("Delete", "btn-ghost lib-delete", "delete"));
        card.appendChild(actions);
        card.addEventListener("keydown", (e) => {
            if (e.target.closest("button")) return;
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                this.openLibraryItem(item);
            }
        });

        return card;
    }

    actionButton(label, extraClass, action) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `btn ${extraClass}`;
        btn.textContent = label;
        btn.dataset.action = action;
        return btn;
    }

    cardMeta(item) {
        const parts = [];
        if (typeof item.n_segments === "number") {
            parts.push(`${item.n_segments} point${item.n_segments === 1 ? "" : "s"}`);
        }
        if (typeof item.point_duration_s === "number") parts.push(`${Math.round(item.point_duration_s)}s points`);
        if (typeof item.duration_s === "number") parts.push(`${Math.round(item.duration_s)}s video`);
        if (item.created) parts.push(this.formatDate(item.created));
        return parts.join(" · ");
    }

    formatDate(iso) {
        try {
            return new Date(iso).toLocaleString(undefined, {
                year: "numeric",
                month: "numeric",
                day: "numeric",
                hour: "numeric",
                minute: "2-digit",
            });
        } catch (_) {
            return iso;
        }
    }

    onLibraryClick(e) {
        const btn = e.target.closest("button[data-action]");
        const card = e.target.closest(".lib-card");
        const id = card && card.dataset.id;
        if (!id) return;
        if (!btn) {
            this.showViewerFromCard(card);
            return;
        }
        const action = btn.dataset.action;
        if (action === "export") this.triggerDownload(`/api/library/${id}/video`);
        else if (action === "csv") this.triggerDownload(`/api/library/${id}/csv`);
        else if (action === "delete") this.deleteItem(id);
    }

    showViewerFromCard(card) {
        const item = {
            id: card.dataset.id,
            name: card.querySelector(".lib-name")?.textContent || card.dataset.id,
            metaText: card.querySelector(".lib-meta")?.textContent || "",
            duration_s: Number(card.dataset.duration) || 0,
            has_csv: Boolean(card.querySelector("button[data-action='csv']")),
        };
        this.openLibraryItem(item);
    }

    openLibraryItem(item) {
        if (!item || !item.id) return false;
        if (this.nativeBridgeReady && this.nativeViewer && typeof this.nativeViewer.openMatch === "function") {
            try {
                const fallback = () => {
                    this.showToast("Could not open native player; using browser preview", "error");
                    this.showViewer(item);
                };
                const result = this.nativeViewer.openMatch(String(item.id), (opened) => {
                    if (!opened) fallback();
                });
                if (result === false) fallback();
                return true;
            } catch (err) {
                console.error("Native viewer bridge failed", err);
                this.showToast("Could not open native player; using browser preview", "error");
            }
        }
        this.showViewer(item);
        return false;
    }

    initNativeViewerBridge() {
        if (!window.qt || !window.qt.webChannelTransport) return;
        const attach = () => {
            if (typeof window.QWebChannel !== "function") return;
            new window.QWebChannel(window.qt.webChannelTransport, (channel) => {
                this.nativeViewer = channel.objects.nativeViewer || null;
                this.nativeBridgeReady = Boolean(this.nativeViewer && this.nativeViewer.openMatch);
                if (this.nativeBridgeReady) {
                    document.documentElement.dataset.nativeViewer = "available";
                }
            });
        };
        if (typeof window.QWebChannel === "function") {
            attach();
            return;
        }
        const script = document.createElement("script");
        script.src = "qrc:///qtwebchannel/qwebchannel.js";
        script.onload = attach;
        script.onerror = () => console.warn("Could not load Qt WebChannel bridge");
        document.head.appendChild(script);
    }

    async showViewer(item) {
        if (!item || !item.id) return;
        this.viewingItemId = item.id;
        this.pointIntervals = [];
        this.lastViewerTime = null;
        this.viewerSeekInProgress = false;
        this.previewLoadInProgress = false;
        this.previewTransitionInProgress = false;
        this.pendingPreviewTransition = null;
        this.pendingOutgoingVideo = null;
        this.activePlaybackSegment = null;
        this.currentPreviewWindowStart = 0;
        this.currentPreviewWindowDuration = 0;
        this.sourceDuration = Number(item.duration_s) || 0;
        this.clearPrefetchTimers();
        this.prefetchedWindowKeys.clear();
        this.readyPreviewWindows.clear();
        this.renderViewerBufferedRanges();
        this.renderViewerPointRanges();
        this.configureViewerTimeline(this.sourceDuration);
        this.viewerTitle.textContent = item.name || "Match";
        this.viewerMeta.textContent = item.metaText || this.cardMeta(item);
        this.viewerCsvBtn.hidden = !item.has_csv;
        this.directPlayback = false;
        this.resetViewerVideos();
        this.updateViewerControls();
        this.showPreviewLoading();
        this.showView("viewer");
        this.showViewerControls();
        await this.loadPlaybackManifest(item.id);
        const start = this.pointIntervals.length ? this.pointIntervals[0].start : 0;
        if (await this.tryDirectSourcePlayback(start, true)) return;
        this.seekViewerToSourceTime(start, true);
    }

    tryDirectSourcePlayback(sourceTime, autoplay = true) {
        // Stream the original file (Range requests, native codec decode) and
        // model it as one full-length window so the existing source-time math
        // (seeks, point skips, timeline) applies unchanged. Resolves false when
        // the engine cannot play the codec; callers then use preview windows.
        if (!this.viewingItemId) return Promise.resolve(false);
        const itemId = this.viewingItemId;
        const video = this.matchVideo;
        const sourceUrl = `/api/library/${itemId}/source`;
        this.directPlayback = false;
        this.directProbeInProgress = true;
        return new Promise((resolve) => {
            let settled = false;
            let timer = null;
            const settle = (ok) => {
                if (settled) return;
                settled = true;
                clearTimeout(timer);
                video.removeEventListener("canplay", onReady);
                video.removeEventListener("error", onError);
                this.directProbeInProgress = false;
                resolve(ok);
            };
            const onReady = () => {
                if (this.viewingItemId !== itemId || video !== this.matchVideo) {
                    settle(false);
                    return;
                }
                if (!(this.sourceDuration > 0) && Number.isFinite(video.duration) && video.duration > 0) {
                    this.sourceDuration = video.duration;
                    this.configureViewerTimeline(this.sourceDuration);
                }
                video.dataset.windowStart = "0";
                video.dataset.windowDuration = String(this.sourceDuration > 0 ? this.sourceDuration : video.duration || 0);
                this.directPlayback = true;
                const target = this.clampViewerSourceTime(sourceTime);
                video.currentTime = target;
                this.lastViewerTime = target;
                this.updateViewerTimeline(target);
                this.hidePreviewLoading();
                this.updateViewerControls();
                if (autoplay) this.startViewerPlayback();
                settle(true);
            };
            const onError = () => {
                console.warn("Direct source playback unavailable; falling back to preview windows", video.error);
                this.clearVideoElement(video);
                settle(false);
            };
            timer = setTimeout(onError, 10000);
            video.addEventListener("canplay", onReady);
            video.addEventListener("error", onError);
            video.src = sourceUrl;
            video.dataset.previewUrl = sourceUrl;
            video.load();
        });
    }

    clearPreviewPoll() {
        if (this.previewPollTimeout) {
            clearTimeout(this.previewPollTimeout);
            this.previewPollTimeout = null;
        }
    }

    clearPrefetchTimers() {
        this.prefetchWindowTimers.forEach((timer) => clearTimeout(timer));
        this.prefetchWindowTimers.clear();
    }

    clearPreviewSpinner() {
        if (this.previewSpinnerTimeout) {
            clearTimeout(this.previewSpinnerTimeout);
            this.previewSpinnerTimeout = null;
        }
    }

    showPreviewLoading() {
        this.previewStatus.textContent = "";
        if (!this.previewStatus.hidden && !this.previewStatus.classList.contains("is-error")) return;
        this.clearPreviewSpinner();
        this.previewStatus.hidden = false;
        this.previewStatus.classList.remove("is-slow", "is-error");
        this.previewSpinnerTimeout = setTimeout(() => {
            if (!this.previewStatus.hidden) this.previewStatus.classList.add("is-slow");
        }, 2000);
    }

    showPreviewError(message) {
        this.clearPreviewSpinner();
        this.previewStatus.textContent = message || "Could not prepare this video preview.";
        this.previewStatus.hidden = false;
        this.previewStatus.classList.remove("is-slow");
        this.previewStatus.classList.add("is-error");
    }

    hidePreviewLoading() {
        this.clearPreviewSpinner();
        this.previewStatus.hidden = true;
        this.previewStatus.classList.remove("is-slow", "is-error");
        this.previewStatus.textContent = "";
    }

    buildPreviewWindowUrl(itemId, start, duration, status = false) {
        const suffix = status ? "/status" : "";
        return `/api/library/${itemId}/preview/window${suffix}?start=${start.toFixed(3)}&duration=${duration.toFixed(3)}`;
    }

    mediaSourceCtor() {
        return window.MediaSource || window.WebKitMediaSource || null;
    }

    canUseMsePreview() {
        // The source-time scheduler is authoritative for v1; keep MSE dormant
        // until it applies the same point-skip boundaries.
        return false;
    }

    resetMsePreview() {
        this.mseSessionId += 1;
        const objectUrl = this.mseObjectUrl;
        const sourceBuffer = this.mseSourceBuffer;
        const mediaSource = this.mseMediaSource;
        this.mseActive = false;
        this.mseObjectUrl = null;
        this.mseMediaSource = null;
        this.mseSourceBuffer = null;
        this.mseAppendQueue = [];
        this.mseCurrentAppend = null;
        this.mseQueuedWindowKeys.clear();
        this.mseAppendedWindowKeys.clear();
        this.msePendingSeek = null;
        this.msePrunePending = false;
        try {
            if (sourceBuffer && sourceBuffer.updating) sourceBuffer.abort();
        } catch (_) {
            // SourceBuffer abort is best-effort during teardown.
        }
        try {
            if (mediaSource?.readyState === "open") mediaSource.endOfStream();
        } catch (_) {
            // The media source may already be closing.
        }
        if (objectUrl) URL.revokeObjectURL(objectUrl);
    }

    startMsePreviewAt(sourceTime, autoplay = true) {
        if (!this.viewingItemId || !this.canUseMsePreview()) {
            this.loadPreviewWindowAt(sourceTime, autoplay);
            return;
        }

        this.resetMsePreview();
        this.clearPreviewPoll();
        this.clearVideoElement(this.primaryMatchVideo);
        this.clearVideoElement(this.secondaryMatchVideo);
        this.matchVideo = this.primaryMatchVideo;
        this.matchVideoBuffer = this.secondaryMatchVideo;
        this.primaryMatchVideo.classList.add("is-active");
        this.secondaryMatchVideo.classList.remove("is-active", "is-outgoing");

        const MediaSourceCtor = this.mediaSourceCtor();
        const mediaSource = new MediaSourceCtor();
        const sessionId = this.mseSessionId + 1;
        const target = this.clampViewerSourceTime(sourceTime);
        this.mseSessionId = sessionId;
        this.mseActive = true;
        this.previewLoadInProgress = true;
        this.currentPreviewWindowStart = 0;
        this.currentPreviewWindowDuration = this.sourceDuration || Number.MAX_SAFE_INTEGER;
        this.lastViewerTime = target;
        this.msePendingSeek = { target, autoplay };
        this.mseMediaSource = mediaSource;
        this.mseObjectUrl = URL.createObjectURL(mediaSource);
        this.matchVideo.src = this.mseObjectUrl;
        this.matchVideo.dataset.previewUrl = "mse";
        this.updateViewerTimeline(target);
        this.updateViewerControls();
        this.showPreviewLoading();

        mediaSource.addEventListener("sourceopen", () => {
            if (!this.isCurrentMseSession(sessionId)) return;
            try {
                const sourceBuffer = mediaSource.addSourceBuffer(MSE_PREVIEW_MIME);
                sourceBuffer.mode = "segments";
                sourceBuffer.addEventListener("updateend", () => this.handleMseUpdateEnd(sessionId));
                sourceBuffer.addEventListener("error", () => this.fallbackFromMse(target, autoplay, new Error("MSE append failed")));
                this.mseSourceBuffer = sourceBuffer;
                if (this.sourceDuration > 0) mediaSource.duration = this.sourceDuration;
                this.ensureMsePlaybackRange(target);
            } catch (err) {
                this.fallbackFromMse(target, autoplay, err);
            }
        }, { once: true });
        this.matchVideo.load();
    }

    isCurrentMseSession(sessionId) {
        return this.mseActive && sessionId === this.mseSessionId && Boolean(this.mseSourceBuffer || this.mseMediaSource);
    }

    fallbackFromMse(sourceTime, autoplay = true, err = null) {
        if (err) console.warn("Falling back from MediaSource preview", err);
        const target = this.clampViewerSourceTime(sourceTime);
        this.mseDisabled = true;
        this.resetMsePreview();
        this.resetViewerVideos();
        this.loadPreviewWindowAt(target, autoplay);
    }

    mseDesiredChunkStarts(sourceTime) {
        const chunk = Math.max(1, Number(this.previewWindowDuration) || 8);
        const limit = Math.max(1, Number(this.previewLookaheadChunks) || 12);
        const starts = [];
        const seen = new Set();
        const addStart = (start) => {
            if (starts.length >= limit) return false;
            const canonicalStart = this.canonicalPreviewChunkStart(start);
            if (this.sourceDuration > 0 && canonicalStart >= this.sourceDuration - 0.1) return false;
            const key = this.previewWindowKeyForStart(canonicalStart, this.previewWindowDuration);
            if (seen.has(key)) return true;
            seen.add(key);
            starts.push(canonicalStart);
            return starts.length < limit;
        };
        const addRange = (start, end) => {
            for (const candidate of this.previewChunkStartsForRange(start, end)) {
                if (!addStart(candidate)) return false;
            }
            return true;
        };

        const target = Math.max(0, Number(sourceTime) || 0);
        const pointIdx = this.findPlaybackTargetPointIndex(target);
        if (pointIdx >= 0) {
            const point = this.pointIntervals[pointIdx];
            const desiredEnd = Math.min(
                this.sourceDuration > 0 ? this.sourceDuration : target + (chunk * limit),
                Math.max(target + chunk, point.end + this.pointBufferSeconds),
            );
            addRange(target, desiredEnd);
            const next = this.pointIntervals[pointIdx + 1];
            if (next && starts.length < limit) {
                addRange(Math.max(0, next.start - this.pointBufferSeconds), next.end + this.pointBufferSeconds);
            }
        } else {
            addRange(target, target + (chunk * limit));
        }
        return starts;
    }

    ensureMsePlaybackRange(sourceTime) {
        if (!this.mseActive || !this.mseSourceBuffer || !this.viewingItemId) return;
        this.mseDesiredChunkStarts(sourceTime).forEach((start) => this.queueMseChunk(start));
    }

    queueMseChunk(sourceTime) {
        if (!this.mseActive || !this.viewingItemId) return;
        const start = this.canonicalPreviewChunkStart(sourceTime);
        if (this.sourceDuration > 0 && start >= this.sourceDuration - 0.1) return;
        const duration = this.sourceDuration > 0
            ? Math.min(this.previewWindowDuration, Math.max(0.1, this.sourceDuration - start))
            : this.previewWindowDuration;
        const key = this.previewWindowKeyForStart(start, duration);
        if (this.mseAppendedWindowKeys.has(key) || this.mseQueuedWindowKeys.has(key)) return;
        this.mseQueuedWindowKeys.add(key);
        const readyWindow = this.readyPreviewWindows.get(key);
        if (readyWindow?.previewUrl) {
            this.fetchMsePreviewChunk(readyWindow.previewUrl, start, duration, this.mseSessionId);
            return;
        }
        this.pollMsePreviewChunk(start, duration, this.mseSessionId);
    }

    async fetchMsePreviewChunk(previewUrl, start, duration, sessionId) {
        try {
            const videoResp = await fetch(previewUrl);
            if (!videoResp.ok) throw new Error(`HTTP ${videoResp.status}`);
            const bytes = await videoResp.arrayBuffer();
            if (!this.isCurrentMseSession(sessionId)) return;
            this.enqueueMseAppend({
                start,
                duration,
                previewUrl,
                bytes,
                sessionId,
            });
        } catch (_) {
            if (!this.isCurrentMseSession(sessionId)) return;
            const key = this.previewWindowKeyForStart(start, duration);
            this.readyPreviewWindows.delete(key);
            this.mseQueuedWindowKeys.delete(key);
            this.pollMsePreviewChunk(start, duration, sessionId, 1200);
        }
    }

    async pollMsePreviewChunk(start, duration, sessionId, delayMs = 0) {
        if (!this.isCurrentMseSession(sessionId) || !this.viewingItemId) return;
        if (delayMs > 0) {
            await new Promise((resolve) => setTimeout(resolve, delayMs));
            if (!this.isCurrentMseSession(sessionId)) return;
        }
        try {
            const statusResp = await fetch(this.buildPreviewWindowUrl(this.viewingItemId, start, duration, true));
            if (!statusResp.ok) throw new Error(`HTTP ${statusResp.status}`);
            const payload = await statusResp.json();
            if (!this.isCurrentMseSession(sessionId)) return;
            if (payload.status === "error") {
                this.fallbackFromMse(start, true, new Error(payload.error || "Preview chunk failed"));
                return;
            }
            if (!payload.ready || !payload.preview_url) {
                this.pollMsePreviewChunk(start, duration, sessionId, 700);
                return;
            }
            const previewUrl = payload.preview_url;
            this.markPreviewWindowReady(Number(payload.start) || start, Number(payload.duration) || duration, previewUrl);
            this.fetchMsePreviewChunk(previewUrl, Number(payload.start) || start, Number(payload.duration) || duration, sessionId);
        } catch (err) {
            if (!this.isCurrentMseSession(sessionId)) return;
            this.pollMsePreviewChunk(start, duration, sessionId, 1200);
        }
    }

    enqueueMseAppend(segment) {
        if (!this.isCurrentMseSession(segment.sessionId)) return;
        this.mseAppendQueue.push(segment);
        this.mseAppendQueue.sort((a, b) => a.start - b.start);
        this.pumpMseAppendQueue(segment.sessionId);
    }

    pumpMseAppendQueue(sessionId = this.mseSessionId) {
        if (!this.isCurrentMseSession(sessionId) || !this.mseSourceBuffer) return;
        if (this.mseSourceBuffer.updating || this.mseCurrentAppend || !this.mseAppendQueue.length) return;
        const segment = this.mseAppendQueue.shift();
        this.mseCurrentAppend = segment;
        try {
            this.mseSourceBuffer.timestampOffset = segment.start;
            this.mseSourceBuffer.appendBuffer(segment.bytes);
        } catch (err) {
            this.mseCurrentAppend = null;
            this.fallbackFromMse(segment.start, true, err);
        }
    }

    handleMseUpdateEnd(sessionId) {
        if (!this.isCurrentMseSession(sessionId)) return;
        const appended = this.mseCurrentAppend;
        this.mseCurrentAppend = null;
        if (appended) {
            this.markPreviewWindowReady(appended.start, appended.duration, appended.previewUrl);
            const key = this.previewWindowKeyForStart(appended.start, appended.duration);
            this.mseQueuedWindowKeys.delete(key);
            this.mseAppendedWindowKeys.add(key);
        }
        this.resolveMsePendingSeek();
        if (this.msePrunePending) {
            this.msePrunePending = false;
            if (this.pruneMseBuffer()) return;
        }
        if (this.pruneMseBuffer()) return;
        this.pumpMseAppendQueue(sessionId);
    }

    isMseSourceTimeBuffered(sourceTime) {
        if (!this.mseActive || !this.matchVideo?.buffered) return false;
        const target = Number(sourceTime);
        if (!Number.isFinite(target)) return false;
        const ranges = this.matchVideo.buffered;
        for (let idx = 0; idx < ranges.length; idx += 1) {
            if (target >= ranges.start(idx) - 0.05 && target < ranges.end(idx) - 0.1) return true;
        }
        return false;
    }

    resolveMsePendingSeek() {
        if (!this.msePendingSeek) return;
        const { target, autoplay } = this.msePendingSeek;
        if (!this.isMseSourceTimeBuffered(target)) return;
        try {
            this.matchVideo.currentTime = target;
        } catch (_) {
            return;
        }
        this.msePendingSeek = null;
        this.previewLoadInProgress = false;
        this.lastViewerTime = target;
        this.updateViewerTimeline(target);
        this.hidePreviewLoading();
        this.updateViewerControls();
        this.ensureMsePlaybackRange(target);
        if (autoplay) this.startViewerPlayback();
    }

    seekMseViewerToSourceTime(sourceTime, autoplay = true) {
        const target = this.clampViewerSourceTime(sourceTime);
        this.ensureMsePlaybackRange(target);
        if (this.isMseSourceTimeBuffered(target)) {
            this.msePendingSeek = null;
            this.matchVideo.currentTime = target;
            this.lastViewerTime = target;
            this.updateViewerTimeline(target);
            this.hidePreviewLoading();
            if (autoplay) this.startViewerPlayback();
            return;
        }
        this.previewLoadInProgress = true;
        this.msePendingSeek = { target, autoplay };
        this.lastViewerTime = target;
        this.updateViewerTimeline(target);
        if (autoplay && !this.matchVideo.paused) this.matchVideo.pause();
        this.showPreviewLoading();
    }

    pruneMseBuffer() {
        if (!this.mseActive || !this.mseSourceBuffer || !this.matchVideo?.buffered) return false;
        if (this.mseSourceBuffer.updating) {
            this.msePrunePending = true;
            return false;
        }
        const keepBehindSeconds = 16;
        const current = Number(this.matchVideo.currentTime) || 0;
        const removeEnd = current - keepBehindSeconds;
        if (removeEnd <= this.previewWindowDuration) return false;
        try {
            this.mseSourceBuffer.remove(0, removeEnd);
            for (const [key, windowInfo] of Array.from(this.readyPreviewWindows.entries())) {
                if (windowInfo.start + windowInfo.duration < removeEnd - 0.5) {
                    this.readyPreviewWindows.delete(key);
                    this.mseAppendedWindowKeys.delete(key);
                }
            }
            this.renderViewerBufferedRanges();
            return true;
        } catch (_) {
            return false;
        }
    }

    setPreviewVideoSource(video, previewUrl, start, duration) {
        if (!video || !previewUrl) return;
        if (video.dataset.previewUrl !== previewUrl) {
            video.src = previewUrl;
            video.dataset.previewUrl = previewUrl;
            video.load();
        }
        video.dataset.windowStart = String(start);
        video.dataset.windowDuration = String(duration);
    }

    videoWindowStart(video = this.matchVideo) {
        const value = Number(video?.dataset.windowStart);
        return Number.isFinite(value) ? value : this.currentPreviewWindowStart;
    }

    videoWindowDuration(video = this.matchVideo) {
        const value = Number(video?.dataset.windowDuration);
        return Number.isFinite(value) && value > 0 ? value : this.currentPreviewWindowDuration;
    }

    videoWindowEnd(video = this.matchVideo) {
        return this.videoWindowStart(video) + this.videoWindowDuration(video);
    }

    preloadNearestReadyWindow() {
        if (!this.matchVideoBuffer || !this.viewingItemId) return;
        if (this.pendingOutgoingVideo === this.matchVideoBuffer) return;
        const starts = this.getSchedulerPrefetchStarts(this.lastViewerTime ?? this.getViewerSourceTime());
        const nextStart = starts.find((start) => start > this.videoWindowStart(this.matchVideo) + 0.001);
        if (!Number.isFinite(nextStart)) return;
        const requestWindow = this.previewWindowRequestForSourceTime(nextStart);
        const next = this.getReadyPreviewWindow(nextStart, requestWindow.end);
        if (next?.previewUrl) this.setPreviewVideoSource(this.matchVideoBuffer, next.previewUrl, next.start, next.duration);
    }

    activatePreviewWindow(itemId, requestId, payload, targetSourceTime, autoplay, options = {}) {
        if (this.viewingItemId !== itemId) return;
        if (!options.allowCurrent && requestId !== this.previewRequestSeq) return;
        const transition = this.pendingPreviewTransition;
        if (!transition || transition.id !== requestId) return;
        const previewUrl = payload.preview_url || payload.previewUrl;
        if (!previewUrl) return;
        const windowStart = Number(payload.start) || 0;
        const windowDuration = Number(payload.duration) || this.previewWindowDuration;
        this.markPreviewWindowReady(windowStart, windowDuration, previewUrl);

        const hasCurrentVideo = this.viewerHasVideo();
        const alreadyBuffered = [this.matchVideo, this.matchVideoBuffer].find((video) => (
            video?.dataset.previewUrl === previewUrl && video.readyState >= 2
        ));
        const targetVideo = alreadyBuffered || (hasCurrentVideo ? this.matchVideoBuffer : this.matchVideo);
        this.setPreviewVideoSource(targetVideo, previewUrl, windowStart, windowDuration);

        const finishActivation = () => {
            if (this.viewingItemId !== itemId) return;
            if (!options.allowCurrent && requestId !== this.previewRequestSeq) return;
            if (!this.pendingPreviewTransition || this.pendingPreviewTransition.id !== requestId) return;
            this.currentPreviewWindowStart = windowStart;
            this.currentPreviewWindowDuration = windowDuration;
            const offset = Math.max(0, Math.min(targetSourceTime - windowStart, targetVideo.duration || windowDuration));
            targetVideo.currentTime = offset;
            const switchingVideos = targetVideo !== this.matchVideo;
            const previousVideo = switchingVideos ? this.matchVideo : null;
            if (switchingVideos) {
                previousVideo.classList.add("is-outgoing");
                this.pendingOutgoingVideo = previousVideo;
                this.matchVideo = targetVideo;
                this.matchVideoBuffer = previousVideo;
            }
            this.hidePreviewLoading();
            this.lastViewerTime = windowStart + offset;
            this.updateViewerTimeline(this.lastViewerTime);
            this.updateViewerControls();
            this.prefetchForPlaybackSchedule(this.lastViewerTime);
            this.preloadNearestReadyWindow();
            this.previewLoadInProgress = false;
            this.previewTransitionInProgress = false;
            this.pendingPreviewTransition = null;
            if (autoplay) this.startViewerPlayback();
            if (switchingVideos && previousVideo) this.showActivePreviewVideo(targetVideo, previousVideo);
        };

        if (targetVideo.readyState >= 2) finishActivation();
        else targetVideo.addEventListener("canplay", finishActivation, { once: true });
    }

    showActivePreviewVideo(targetVideo, previousVideo) {
        targetVideo.tabIndex = 0;
        previousVideo.tabIndex = -1;
        const reveal = () => {
            if (this.matchVideo !== targetVideo) return;
            targetVideo.classList.add("is-active");
            previousVideo.classList.remove("is-active");
            window.setTimeout(() => this.finishOutgoingPreviewVideo(previousVideo), 80);
        };
        if (!targetVideo.paused && targetVideo.readyState >= 3) {
            requestAnimationFrame(reveal);
            return;
        }
        targetVideo.addEventListener("playing", reveal, { once: true });
        targetVideo.addEventListener("timeupdate", reveal, { once: true });
        window.setTimeout(reveal, targetVideo.readyState >= 3 ? 140 : 240);
    }

    finishOutgoingPreviewVideo(video = this.pendingOutgoingVideo) {
        if (!video || video === this.matchVideo) return;
        video.classList.remove("is-active", "is-outgoing");
        this.clearVideoElement(video);
        if (this.pendingOutgoingVideo === video) this.pendingOutgoingVideo = null;
        this.preloadNearestReadyWindow();
    }

    loadPreviewWindowAt(sourceTime, autoplay = true, options = {}) {
        if (!this.viewingItemId) return;
        const itemId = this.viewingItemId;
        const requestId = ++this.previewRequestSeq;
        const targetSourceTime = this.clampViewerSourceTime(sourceTime);
        if (!options.preserveSegment) this.setPlaybackSegmentForSourceTime(targetSourceTime);
        const requestWindow = this.previewWindowRequestForSourceTime(targetSourceTime);
        this.clearPreviewPoll();
        this.previewLoadInProgress = true;
        this.previewTransitionInProgress = true;
        this.pendingPreviewTransition = {
            id: requestId,
            itemId,
            targetSourceTime,
            targetWindowStart: requestWindow.start,
            targetWindowEnd: requestWindow.end,
            autoplay: Boolean(autoplay),
            preserveSegment: Boolean(options.preserveSegment),
        };
        if (this.viewerHasVideo()) this.hidePreviewLoading();
        else this.showPreviewLoading();
        const readyWindow = this.getReadyPreviewWindow(targetSourceTime, requestWindow.end);
        if (readyWindow?.previewUrl) {
            this.activatePreviewWindow(itemId, requestId, {
                start: readyWindow.start,
                duration: readyWindow.duration,
                preview_url: readyWindow.previewUrl,
            }, targetSourceTime, autoplay);
            return;
        }
        const poll = async () => {
            if (this.viewingItemId !== itemId || requestId !== this.previewRequestSeq) return;
            try {
                const resp = await fetch(this.buildPreviewWindowUrl(itemId, requestWindow.start, requestWindow.duration, true));
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const payload = await resp.json();
                if (this.viewingItemId !== itemId || requestId !== this.previewRequestSeq) return;
                if (typeof payload.source_duration === "number" && payload.source_duration > 0) {
                    this.sourceDuration = payload.source_duration;
                    this.configureViewerTimeline(this.sourceDuration, this.lastViewerTime ?? targetSourceTime);
                }
                if (payload.ready && payload.preview_url) {
                    this.activatePreviewWindow(itemId, requestId, payload, targetSourceTime, autoplay);
                    return;
                }
                if (payload.status === "error") {
                    const message = payload.error || "Could not prepare this video preview.";
                    this.previewLoadInProgress = false;
                    this.previewTransitionInProgress = false;
                    this.pendingPreviewTransition = null;
                    this.showPreviewError(message);
                    this.showToast(message, "error");
                    return;
                }
                this.previewPollTimeout = setTimeout(poll, 750);
            } catch (err) {
                console.error(err);
                if (this.viewingItemId !== itemId || requestId !== this.previewRequestSeq) return;
                this.previewPollTimeout = setTimeout(poll, 1500);
            }
        };
        poll();
    }

    canonicalPreviewChunkStart(sourceTime) {
        const chunk = Math.max(1, Number(this.previewWindowDuration) || 8);
        return Math.max(0, Math.floor((Number(sourceTime) || 0) / chunk) * chunk);
    }

    pointIndexAtSourceTime(sourceTime) {
        const t = Number(sourceTime) || 0;
        return this.pointIntervals.findIndex((seg) => t >= seg.start && t < seg.end);
    }

    nextPointIndexAfterSourceTime(sourceTime) {
        const t = Number(sourceTime) || 0;
        return this.pointIntervals.findIndex((seg) => t < seg.start);
    }

    playbackSegmentForSourceTime(sourceTime) {
        const target = this.clampViewerSourceTime(sourceTime);
        const sourceEnd = this.sourceDuration > 0 ? this.sourceDuration : Number.POSITIVE_INFINITY;
        if (!this.pointIntervals.length) {
            return {
                kind: "continuous",
                start: target,
                end: sourceEnd,
                pointIndex: null,
                nextPointIndex: null,
            };
        }

        const activePointIndex = this.pointIndexAtSourceTime(target);
        if (activePointIndex >= 0) {
            const point = this.pointIntervals[activePointIndex];
            const nextPointIndex = activePointIndex + 1 < this.pointIntervals.length ? activePointIndex + 1 : null;
            return {
                kind: "point",
                start: target,
                end: point.end,
                pointIndex: activePointIndex,
                nextPointIndex,
            };
        }

        const nextPointIndex = this.nextPointIndexAfterSourceTime(target);
        if (nextPointIndex >= 0) {
            const point = this.pointIntervals[nextPointIndex];
            const followingPointIndex = nextPointIndex + 1 < this.pointIntervals.length ? nextPointIndex + 1 : null;
            return {
                kind: "gap",
                start: target,
                end: point.end,
                pointIndex: nextPointIndex,
                nextPointIndex: followingPointIndex,
            };
        }

        return {
            kind: "tail",
            start: target,
            end: sourceEnd,
            pointIndex: null,
            nextPointIndex: null,
        };
    }

    setPlaybackSegmentForSourceTime(sourceTime) {
        this.activePlaybackSegment = this.playbackSegmentForSourceTime(sourceTime);
        return this.activePlaybackSegment;
    }

    setManualPlaybackSegmentForSourceTime(sourceTime) {
        const target = this.clampViewerSourceTime(sourceTime);
        this.activePlaybackSegment = {
            kind: "manual",
            start: target,
            end: this.sourceDuration > 0 ? this.sourceDuration : Number.POSITIVE_INFINITY,
            pointIndex: null,
            nextPointIndex: null,
        };
        return this.activePlaybackSegment;
    }

    previewWindowRequestForSourceTime(sourceTime) {
        const chunk = Math.max(1, Number(this.previewWindowDuration) || 8);
        const target = Math.max(0, Number(sourceTime) || 0);
        const start = this.canonicalPreviewChunkStart(target);
        const duration = chunk;
        const cappedDuration = this.sourceDuration > 0 ? Math.min(duration, Math.max(0.1, this.sourceDuration - start)) : duration;
        return {
            start: Number(start.toFixed(3)),
            duration: Number(cappedDuration.toFixed(3)),
            end: Number((start + cappedDuration).toFixed(3)),
        };
    }

    previewWindowKeyForStart(start, duration = this.previewWindowDuration) {
        return `${this.viewingItemId}:${Number(start).toFixed(1)}:${Number(duration).toFixed(1)}`;
    }

    previewChunkKey(sourceTime) {
        const requestWindow = this.previewWindowRequestForSourceTime(sourceTime);
        return this.previewWindowKeyForStart(requestWindow.start, requestWindow.duration);
    }

    markPreviewWindowReady(start, duration = this.previewWindowDuration, previewUrl = null) {
        if (!this.viewingItemId) return;
        const canonicalStart = this.canonicalPreviewChunkStart(start);
        const safeDuration = Math.max(0.1, Number(duration) || this.previewWindowDuration);
        const key = this.previewWindowKeyForStart(canonicalStart, safeDuration);
        this.readyPreviewWindows.set(key, { start: canonicalStart, duration: safeDuration, previewUrl });
        this.prefetchedWindowKeys.add(key);
        if (this.prefetchWindowTimers.has(key)) {
            clearTimeout(this.prefetchWindowTimers.get(key));
            this.prefetchWindowTimers.delete(key);
        }
        this.renderViewerBufferedRanges();
        if (!this.mseActive) this.preloadNearestReadyWindow();
    }

    getReadyPreviewWindow(sourceTime, requiredEnd = null) {
        const target = Math.max(0, Number(sourceTime) || 0);
        const minEnd = Number(requiredEnd);
        return Array.from(this.readyPreviewWindows.values())
            .filter((windowInfo) => {
                if (!windowInfo.previewUrl) return false;
                const end = windowInfo.start + windowInfo.duration;
                if (target < windowInfo.start - 0.001 || target >= end - 0.05) return false;
                return !Number.isFinite(minEnd) || end >= minEnd - 0.05;
            })
            .sort((a, b) => b.start - a.start || b.duration - a.duration)[0] || null;
    }

    renderViewerBufferedRanges() {
        if (!this.viewerBufferTrack) return;
        this.viewerBufferTrack.innerHTML = "";
        const duration = Math.max(0, Number(this.sourceDuration) || 0);
        if (duration <= 0) return;

        const mergedRanges = [];
        Array.from(this.readyPreviewWindows.values())
            .sort((a, b) => a.start - b.start)
            .forEach((windowInfo) => {
                const start = Math.max(0, Math.min(duration, windowInfo.start));
                const end = Math.max(start, Math.min(duration, windowInfo.start + windowInfo.duration));
                if (end <= start) return;
                const last = mergedRanges[mergedRanges.length - 1];
                if (last && start <= last.end + 0.05) last.end = Math.max(last.end, end);
                else mergedRanges.push({ start, end });
            });

        mergedRanges.forEach(({ start, end }) => {
            const segment = document.createElement("span");
            segment.className = "viewer-buffer-segment";
            segment.style.left = `${(start / duration) * 100}%`;
            segment.style.width = `${((end - start) / duration) * 100}%`;
            this.viewerBufferTrack.appendChild(segment);
        });
    }

    previewChunkStartsForRange(start, end) {
        const chunk = Math.max(1, Number(this.previewWindowDuration) || 8);
        const rangeStart = Math.max(0, Number(start) || 0);
        const rangeEnd = Math.max(rangeStart, Number(end) || rangeStart);
        const first = this.canonicalPreviewChunkStart(rangeStart);
        const last = this.canonicalPreviewChunkStart(Math.max(rangeStart, rangeEnd - 0.001));
        const starts = [];
        for (let value = first; value <= last + 0.001; value += chunk) {
            if (this.sourceDuration > 0 && value >= this.sourceDuration - 0.1) break;
            starts.push(Number(value.toFixed(3)));
        }
        return starts;
    }

    findPlaybackTargetPointIndex(sourceTime) {
        const segment = this.playbackSegmentForSourceTime(sourceTime);
        return Number.isInteger(segment.pointIndex) ? segment.pointIndex : -1;
    }

    getSchedulerPrefetchStarts(sourceTime) {
        const limit = Math.max(0, Number(this.previewLookaheadChunks) || 0);
        if (!limit) return [];

        const chunk = Math.max(1, Number(this.previewWindowDuration) || 8);
        const selected = [];
        const seen = new Set();

        const addStart = (start) => {
            const canonicalStart = this.canonicalPreviewChunkStart(start);
            if (this.sourceDuration > 0 && canonicalStart >= this.sourceDuration - 0.1) return true;
            const requestWindow = this.previewWindowRequestForSourceTime(canonicalStart);
            const key = this.previewWindowKeyForStart(requestWindow.start, requestWindow.duration);
            if (seen.has(key)) return true;
            seen.add(key);
            selected.push(canonicalStart);
            return selected.length < limit;
        };

        const segment = this.activePlaybackSegment || this.playbackSegmentForSourceTime(sourceTime);
        const segmentEnd = Number(segment?.end);
        if (Number.isFinite(segmentEnd)) {
            const rangeEnd = Math.max(Number(sourceTime) || 0, segmentEnd);
            for (const start of this.previewChunkStartsForRange(sourceTime, rangeEnd)) {
                if (!addStart(start)) break;
            }
            const nextPoint = Number.isInteger(segment.nextPointIndex) ? this.pointIntervals[segment.nextPointIndex] : null;
            if (nextPoint && selected.length < limit) addStart(nextPoint.start);
        } else {
            const currentStart = this.canonicalPreviewChunkStart(sourceTime);
            for (let start = currentStart; selected.length < limit; start += chunk) {
                if (!addStart(start)) break;
            }
        }

        return selected;
    }

    getPointAwarePrefetchStarts(sourceTime) {
        return this.getSchedulerPrefetchStarts(sourceTime);
    }

    trimPrefetchedWindowKeys(sourceTime) {
        const minStart = this.canonicalPreviewChunkStart(sourceTime) - (this.previewWindowDuration * 2);
        for (const key of Array.from(this.prefetchedWindowKeys)) {
            const parts = key.split(":");
            const start = Number(parts[parts.length - 2]);
            if (Number.isFinite(start) && start < minStart) this.prefetchedWindowKeys.delete(key);
        }
    }

    prefetchPointAwareWindows(sourceTime) {
        this.prefetchForPlaybackSchedule(sourceTime);
    }

    prefetchForPlaybackSchedule(sourceTime) {
        if (!this.viewingItemId || this.directPlayback) return;
        this.trimPrefetchedWindowKeys(sourceTime);
        this.getSchedulerPrefetchStarts(sourceTime).forEach((start) => this.prefetchPreviewWindow(start));
    }

    prefetchPreviewWindow(sourceTime) {
        if (!this.viewingItemId || !Number.isFinite(sourceTime)) return;
        if (this.sourceDuration > 0 && sourceTime >= this.sourceDuration - 0.1) return;
        const requestWindow = this.previewWindowRequestForSourceTime(sourceTime);
        const key = this.previewWindowKeyForStart(requestWindow.start, requestWindow.duration);
        if (this.getReadyPreviewWindow(sourceTime, requestWindow.end) || this.prefetchWindowTimers.has(key)) return;
        if (this.prefetchedWindowKeys.has(key)) return;
        this.prefetchedWindowKeys.add(key);
        this.pollPrefetchPreviewWindow(requestWindow.start, requestWindow.duration);
    }

    async pollPrefetchPreviewWindow(start, duration = this.previewWindowDuration, delayMs = 0) {
        if (!this.viewingItemId) return;
        const canonicalStart = this.canonicalPreviewChunkStart(start);
        const safeDuration = Math.max(0.1, Number(duration) || this.previewWindowDuration);
        const key = this.previewWindowKeyForStart(canonicalStart, safeDuration);
        if (this.readyPreviewWindows.has(key)) return;
        if (delayMs > 0) {
            const timer = setTimeout(() => {
                this.prefetchWindowTimers.delete(key);
                this.pollPrefetchPreviewWindow(canonicalStart, safeDuration);
            }, delayMs);
            this.prefetchWindowTimers.set(key, timer);
            return;
        }
        try {
            const resp = await fetch(this.buildPreviewWindowUrl(this.viewingItemId, canonicalStart, safeDuration, true));
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const payload = await resp.json();
            if (!this.viewingItemId) return;
            if (payload.ready && payload.preview_url) {
                this.markPreviewWindowReady(Number(payload.start) || canonicalStart, Number(payload.duration) || this.previewWindowDuration, payload.preview_url);
                return;
            }
            if (payload.status === "error") {
                this.prefetchedWindowKeys.delete(key);
                return;
            }
            this.pollPrefetchPreviewWindow(canonicalStart, safeDuration, 900);
        } catch (_) {
            this.pollPrefetchPreviewWindow(canonicalStart, safeDuration, 1500);
        }
    }

    startViewerPlayback() {
        const playPromise = this.matchVideo.play();
        if (playPromise && typeof playPromise.catch === "function") {
            playPromise.catch(() => {
                // Browser autoplay policy can still block this if the card click
                // is not treated as the media gesture.
            });
        }
    }

    viewerHasVideo() {
        return Boolean(this.matchVideo && this.matchVideo.src);
    }

    isViewerActive() {
        return this.viewerView && !this.viewerView.hidden && Boolean(this.viewingItemId);
    }

    clampViewerSourceTime(sourceTime) {
        const target = Math.max(0, Number(sourceTime) || 0);
        if (this.sourceDuration > 0) return Math.min(target, Math.max(0, this.sourceDuration - 0.05));
        return target;
    }

    updateViewerControls() {
        const hasVideo = this.viewerHasVideo();
        const isPaused = !hasVideo || this.matchVideo.paused;
        const disabled = !hasVideo;
        [this.viewerBackBtn, this.viewerPlayPauseBtn, this.viewerForwardBtn, this.viewerFullscreenBtn].forEach((btn) => {
            if (btn) btn.disabled = disabled;
        });
        if (this.viewerPlayPauseBtn) {
            this.viewerPlayPauseBtn.textContent = isPaused ? "▶" : "❚❚";
            this.viewerPlayPauseBtn.setAttribute("aria-label", isPaused ? "Play video" : "Pause video");
        }
        this.updateViewerFullscreenState();
    }

    toggleViewerPlayback(event) {
        if (event && typeof event.preventDefault === "function") event.preventDefault();
        if (!this.matchVideo.src) return;
        this.showViewerControls();
        if (this.mseActive && this.msePendingSeek) {
            this.msePendingSeek.autoplay = true;
            this.ensureMsePlaybackRange(this.msePendingSeek.target);
            return;
        }
        if (this.matchVideo.paused) this.startViewerPlayback();
        else this.matchVideo.pause();
    }

    handleViewerSurfaceClick(event) {
        if (!this.isViewerActive() || !this.viewerHasVideo()) return;
        if (event.target.closest(".viewer-timeline, button, input, select, textarea, a")) return;
        this.toggleViewerPlayback(event);
    }

    viewerFullscreenElement() {
        return (
            document.fullscreenElement ||
            document.webkitFullscreenElement ||
            document.mozFullScreenElement ||
            document.msFullscreenElement ||
            null
        );
    }

    viewerIsFullscreen() {
        return this.viewerFullscreenElement() === this.viewerVideoWrap || this.viewerFullscreenFallback;
    }

    updateViewerFullscreenState() {
        if (!this.viewerFullscreenBtn || !this.viewerVideoWrap) return;
        const isFullscreen = this.viewerIsFullscreen();
        this.viewerVideoWrap.classList.toggle("is-fullscreen", isFullscreen);
        this.viewerFullscreenBtn.textContent = isFullscreen ? "×" : "⛶";
        this.viewerFullscreenBtn.setAttribute("aria-label", isFullscreen ? "Exit fullscreen" : "Enter fullscreen");
    }

    async toggleViewerFullscreen(event = null) {
        if (event && typeof event.preventDefault === "function") event.preventDefault();
        if (event && typeof event.stopPropagation === "function") event.stopPropagation();
        if (!this.viewerVideoWrap || this.viewerFullscreenBtn.disabled) return;
        this.showViewerControls();
        try {
            const fullscreenElement = this.viewerFullscreenElement();
            if (fullscreenElement || this.viewerFullscreenFallback) {
                this.viewerFullscreenFallback = false;
                if (document.exitFullscreen && fullscreenElement) await document.exitFullscreen();
                else if (document.webkitExitFullscreen && fullscreenElement) document.webkitExitFullscreen();
                else if (document.webkitCancelFullScreen && fullscreenElement) document.webkitCancelFullScreen();
                else if (document.mozCancelFullScreen && fullscreenElement) document.mozCancelFullScreen();
                else if (document.msExitFullscreen && fullscreenElement) document.msExitFullscreen();
            } else {
                await this.requestViewerFullscreen();
            }
        } catch (err) {
            console.warn("Could not toggle fullscreen", err);
            this.viewerFullscreenFallback = !this.viewerFullscreenFallback;
        }
        this.updateViewerFullscreenState();
    }

    async requestViewerFullscreen() {
        const target = this.viewerVideoWrap;
        const video = this.matchVideo;
        try {
            if (target.requestFullscreen) {
                await target.requestFullscreen();
                this.viewerFullscreenFallback = false;
                return;
            }
            if (target.webkitRequestFullscreen) {
                target.webkitRequestFullscreen();
                this.viewerFullscreenFallback = false;
                return;
            }
            if (target.webkitRequestFullScreen) {
                target.webkitRequestFullScreen();
                this.viewerFullscreenFallback = false;
                return;
            }
            if (target.mozRequestFullScreen) {
                target.mozRequestFullScreen();
                this.viewerFullscreenFallback = false;
                return;
            }
            if (target.msRequestFullscreen) {
                target.msRequestFullscreen();
                this.viewerFullscreenFallback = false;
                return;
            }
            if (video?.webkitEnterFullscreen) {
                video.webkitEnterFullscreen();
                this.viewerFullscreenFallback = false;
                return;
            }
        } catch (err) {
            console.warn("Native fullscreen request failed; using in-page fullscreen", err);
        }
        this.viewerFullscreenFallback = true;
    }

    skipViewerBy(deltaSeconds) {
        if (!this.viewerHasVideo()) return;
        this.showViewerControls();
        const wasPlaying = !this.matchVideo.paused;
        const current = this.getViewerSourceTime();
        const target = this.clampViewerSourceTime(current + deltaSeconds);
        this.setManualPlaybackSegmentForSourceTime(target);
        this.seekViewerToSourceTime(target, wasPlaying, { preserveSegment: true });
    }

    seekViewerFromTimelinePointer(event) {
        if (!this.isViewerActive() || !this.sourceDuration || event.target === this.viewerSeek) return;
        const rect = this.viewerSeekWrap.getBoundingClientRect();
        if (!rect.width) return;
        const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        const target = this.clampViewerSourceTime(ratio * this.sourceDuration);
        this.updateViewerTimeline(target);
        this.seekViewerToSourceTime(target, this.viewerHasVideo() && !this.matchVideo.paused);
    }

    handleViewerKeyboardShortcuts(event) {
        if (!this.isViewerActive() || event.defaultPrevented) return;
        const target = event.target;
        const tagName = target?.tagName;
        const isTextEntry = target?.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(tagName);
        if (isTextEntry && target !== this.viewerSeek) return;
        if (event.key === "ArrowLeft") {
            event.preventDefault();
            this.showViewerControls();
            this.skipViewerBy(-this.viewerSkipSeconds);
        } else if (event.key === "ArrowRight") {
            event.preventDefault();
            this.showViewerControls();
            this.skipViewerBy(this.viewerSkipSeconds);
        } else if (event.key === " " || event.code === "Space" || event.key === "Spacebar") {
            event.preventDefault();
            if (typeof event.stopPropagation === "function") event.stopPropagation();
            this.toggleViewerPlayback(event);
        } else if (event.key && event.key.toLowerCase() === "f") {
            event.preventDefault();
            if (typeof event.stopPropagation === "function") event.stopPropagation();
            this.toggleViewerFullscreen(event);
        } else if (event.key === "Escape" && this.viewerFullscreenFallback) {
            event.preventDefault();
            this.viewerFullscreenFallback = false;
            this.updateViewerFullscreenState();
        }
    }

    normalizePointIntervals(segments = []) {
        return (segments || [])
            .map((seg) => ({ start: Number(seg.start), end: Number(seg.end) }))
            .filter((seg) => Number.isFinite(seg.start) && Number.isFinite(seg.end) && seg.end > seg.start)
            .sort((a, b) => a.start - b.start || a.end - b.end);
    }

    async loadPlaybackManifest(itemId) {
        try {
            const resp = await fetch(`/api/library/${itemId}/playback`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const payload = await resp.json();
            if (this.viewingItemId !== itemId) return;
            if (Number(payload.chunk_duration_s) > 0) {
                this.previewWindowDuration = Number(payload.chunk_duration_s);
            }
            if (Number(payload.source_duration_s) > 0) {
                this.sourceDuration = Number(payload.source_duration_s);
            }
            this.pointIntervals = this.normalizePointIntervals(payload.point_intervals || payload.segments);
            this.lastViewerTime = null;
            this.configureViewerTimeline(this.sourceDuration);
            this.renderViewerPointRanges();
        } catch (err) {
            console.error(err);
            await this.loadPointIntervals(itemId);
        }
    }

    async loadPointIntervals(itemId) {
        try {
            const resp = await fetch(`/api/library/${itemId}/segments`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const payload = await resp.json();
            if (this.viewingItemId !== itemId) return;
            this.pointIntervals = this.normalizePointIntervals(payload.segments);
            this.lastViewerTime = null;
            this.renderViewerPointRanges();
        } catch (err) {
            console.error(err);
            this.pointIntervals = [];
            this.renderViewerPointRanges();
            this.showToast("Could not load point times.", "error");
        }
    }

    configureViewerTimeline(duration, sourceTime = null) {
        const safeDuration = Math.max(0, Number(duration) || 0);
        this.viewerTimeline.hidden = safeDuration <= 0;
        this.viewerSeek.min = "0";
        this.viewerSeek.max = safeDuration > 0 ? String(safeDuration) : "0";
        this.viewerSeek.step = "0.1";
        this.viewerDuration.textContent = this.formatClock(safeDuration);
        const current = Number.isFinite(sourceTime) ? sourceTime : Number(this.viewerSeek.value) || 0;
        this.updateViewerTimeline(current);
        this.renderViewerPointRanges();
    }

    renderViewerPointRanges() {
        if (!this.viewerPointTrack) return;
        this.viewerPointTrack.innerHTML = "";
        const duration = Math.max(0, Number(this.sourceDuration) || 0);
        if (duration <= 0) return;
        this.pointIntervals.forEach((seg) => {
            const start = Math.max(0, Math.min(duration, seg.start));
            const end = Math.max(start, Math.min(duration, seg.end));
            if (end <= start) return;
            const marker = document.createElement("span");
            marker.className = "viewer-point-segment";
            marker.style.left = `${(start / duration) * 100}%`;
            marker.style.width = `${Math.max(0.12, ((end - start) / duration) * 100)}%`;
            this.viewerPointTrack.appendChild(marker);
        });
    }

    formatClock(seconds) {
        const safe = Math.max(0, Math.floor(Number(seconds) || 0));
        const hours = Math.floor(safe / 3600);
        const minutes = Math.floor((safe % 3600) / 60);
        const secs = safe % 60;
        if (hours > 0) return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
        return `${minutes}:${String(secs).padStart(2, "0")}`;
    }

    updateViewerTimeline(sourceTime) {
        const safeTime = Math.max(0, Number(sourceTime) || 0);
        this.viewerCurrentTime.textContent = this.formatClock(safeTime);
        const max = Number(this.viewerSeek.max) || 0;
        const progress = max > 0 ? Math.max(0, Math.min(100, (safeTime / max) * 100)) : 0;
        this.viewerSeek.style.setProperty("--viewer-progress", `${progress}%`);
        if (!this.viewerSeekDragging) {
            this.viewerSeek.value = String(Math.min(safeTime, max || safeTime));
        }
    }

    getViewerSourceTime() {
        if (this.mseActive) {
            const videoTime = Number(this.matchVideo?.currentTime);
            if (Number.isFinite(videoTime) && (!this.msePendingSeek || this.isMseSourceTimeBuffered(videoTime))) return videoTime;
            if (Number.isFinite(this.lastViewerTime)) return this.lastViewerTime;
            return Number.isFinite(videoTime) ? videoTime : 0;
        }
        const videoTime = Number(this.matchVideo?.currentTime);
        if (Number.isFinite(videoTime)) return this.videoWindowStart(this.matchVideo) + videoTime;
        if (Number.isFinite(this.lastViewerTime)) return this.lastViewerTime;
        return this.videoWindowStart(this.matchVideo);
    }

    seekViewerToSourceTime(sourceTime, autoplay = true, options = {}) {
        const target = this.clampViewerSourceTime(sourceTime);
        if (!options.preserveSegment) this.setPlaybackSegmentForSourceTime(target);
        if (this.mseActive) {
            this.seekMseViewerToSourceTime(target, autoplay);
            return;
        }
        if (this.directPlayback && this.matchVideo.src) {
            this.matchVideo.currentTime = target;
            this.lastViewerTime = target;
            this.updateViewerTimeline(target);
            if (autoplay) this.startViewerPlayback();
            return;
        }
        const windowStart = this.videoWindowStart(this.matchVideo);
        const windowEnd = this.videoWindowEnd(this.matchVideo);
        if (this.matchVideo.src && target >= windowStart && target < windowEnd - 0.15) {
            this.matchVideo.currentTime = Math.max(0, target - windowStart);
            this.lastViewerTime = target;
            this.updateViewerTimeline(target);
            this.prefetchForPlaybackSchedule(target);
            if (autoplay) this.startViewerPlayback();
            return;
        }
        const requestWindow = this.previewWindowRequestForSourceTime(target);
        const readyWindow = this.getReadyPreviewWindow(target, requestWindow.end);
        if (readyWindow?.previewUrl) {
            this.clearPreviewPoll();
            const requestId = ++this.previewRequestSeq;
            this.previewLoadInProgress = true;
            this.previewTransitionInProgress = true;
            this.pendingPreviewTransition = {
                id: requestId,
                itemId: this.viewingItemId,
                targetSourceTime: target,
                targetWindowStart: readyWindow.start,
                targetWindowEnd: readyWindow.start + readyWindow.duration,
                autoplay: Boolean(autoplay),
                preserveSegment: Boolean(options.preserveSegment),
            };
            this.activatePreviewWindow(this.viewingItemId, requestId, {
                start: readyWindow.start,
                duration: readyWindow.duration,
                preview_url: readyWindow.previewUrl,
            }, target, autoplay, { allowCurrent: true });
            return;
        }
        this.loadPreviewWindowAt(target, autoplay, { preserveSegment: true });
    }

    handleViewerVideoError() {
        if (this.directProbeInProgress) return;
        if (!this.matchVideo.src || !this.matchVideo.error) return;
        if (this.mseActive) {
            this.fallbackFromMse(this.getViewerSourceTime(), !this.matchVideo.paused, this.matchVideo.error);
            return;
        }
        if (this.directPlayback) {
            console.warn("Direct source playback failed mid-stream; falling back to preview windows", this.matchVideo.error);
            const target = this.lastViewerTime ?? this.getViewerSourceTime();
            const wasPlaying = !this.matchVideo.paused;
            this.directPlayback = false;
            this.clearVideoElement(this.matchVideo);
            this.loadPreviewWindowAt(target, wasPlaying);
            return;
        }
        this.showToast("Could not play this video in the viewer.", "error");
    }

    handleViewerTimeUpdate() {
        if (!this.matchVideo.src) return;
        const t = this.getViewerSourceTime();
        this.updateViewerTimeline(t);
        if (this.mseActive) this.ensureMsePlaybackRange(t);
        const remaining = this.videoWindowEnd(this.matchVideo) - t;
        if (!this.mseActive && remaining < 8) this.prefetchForPlaybackSchedule(t);
        if (this.matchVideo.paused) {
            this.lastViewerTime = t;
            return;
        }
        if (!this.activePlaybackSegment) this.setPlaybackSegmentForSourceTime(t);
        this.lastViewerTime = t;
        if (this.pendingPreviewTransition) return;
        const segmentEnd = Number(this.activePlaybackSegment?.end);
        if (Number.isFinite(segmentEnd) && t >= segmentEnd - 0.08) {
            this.advanceAfterActivePlaybackSegment();
            return;
        }
        if (!this.mseActive && !this.previewLoadInProgress && remaining <= 0.18 && remaining > -0.5) {
            this.continueToNextPreviewWindow({ preserveSegment: true });
        }
    }

    handleViewerWindowEnded() {
        if (!this.matchVideo.src || !this.viewingItemId) return;
        if (this.mseActive) return;
        const sourceTime = this.videoWindowEnd(this.matchVideo);
        this.updateViewerTimeline(sourceTime);
        this.lastViewerTime = sourceTime;
        if (this.pendingPreviewTransition || this.previewLoadInProgress) {
            this.showPreviewLoading();
            return;
        }
        if (!this.activePlaybackSegment) this.setPlaybackSegmentForSourceTime(sourceTime);
        const segmentEnd = Number(this.activePlaybackSegment?.end);
        if (Number.isFinite(segmentEnd) && sourceTime >= segmentEnd - 0.08) {
            this.advanceAfterActivePlaybackSegment();
            return;
        }
        if (this.sourceDuration > 0 && sourceTime >= this.sourceDuration - 0.25) return;
        this.continueToNextPreviewWindow({ preserveSegment: true });
    }

    advanceAfterActivePlaybackSegment() {
        if (!this.activePlaybackSegment || this.previewLoadInProgress) return;
        const nextPointIndex = this.activePlaybackSegment.nextPointIndex;
        if (Number.isInteger(nextPointIndex) && this.pointIntervals[nextPointIndex]) {
            this.seekViewerToSourceTime(this.pointIntervals[nextPointIndex].start, true);
            return;
        }
        const stopTime = (
            this.sourceDuration > 0 && this.activePlaybackSegment.end >= this.sourceDuration - 0.1
        )
            ? this.sourceDuration
            : this.clampViewerSourceTime(this.activePlaybackSegment.end);
        this.matchVideo.pause();
        this.lastViewerTime = stopTime;
        this.updateViewerTimeline(stopTime);
        this.updateViewerControls();
    }

    continueToNextPreviewWindow(options = {}) {
        if (!this.viewingItemId || this.pendingPreviewTransition || this.directPlayback) return;
        const sourceTime = this.videoWindowEnd(this.matchVideo);
        if (!this.activePlaybackSegment) this.setPlaybackSegmentForSourceTime(sourceTime);
        const segmentEnd = Number(this.activePlaybackSegment?.end);
        if (Number.isFinite(segmentEnd) && sourceTime >= segmentEnd - 0.08) {
            this.advanceAfterActivePlaybackSegment();
            return;
        }
        const requestWindow = this.previewWindowRequestForSourceTime(sourceTime);
        const readyWindow = this.getReadyPreviewWindow(sourceTime, requestWindow.end);
        if (readyWindow?.previewUrl) {
            const requestId = ++this.previewRequestSeq;
            this.previewLoadInProgress = true;
            this.previewTransitionInProgress = true;
            this.pendingPreviewTransition = {
                id: requestId,
                itemId: this.viewingItemId,
                targetSourceTime: sourceTime,
                targetWindowStart: readyWindow.start,
                targetWindowEnd: readyWindow.start + readyWindow.duration,
                autoplay: true,
                preserveSegment: Boolean(options.preserveSegment),
            };
            this.activatePreviewWindow(this.viewingItemId, requestId, {
                start: readyWindow.start,
                duration: readyWindow.duration,
                preview_url: readyWindow.previewUrl,
            }, sourceTime, true, { allowCurrent: true });
            return;
        }
        this.loadPreviewWindowAt(sourceTime, true, options);
    }

    findPointIndexAt(sourceTime) {
        return this.pointIndexAtSourceTime(sourceTime);
    }

    prefetchUpcomingPointWindow(sourceTime) {
        this.prefetchPointAwareWindows(sourceTime);
    }

    // A real navigation to an attachment URL: the browser downloads it, and in
    // the desktop webview it fires QWebEngineProfile.downloadRequested (handled
    // in desktop.py with a native Save dialog).
    triggerDownload(url) {
        const anchor = document.createElement("a");
        anchor.href = url;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
    }

    async deleteItem(id) {
        if (!window.confirm("Delete this match? This also deletes its CSV.")) return;
        try {
            const resp = await fetch(`/api/library/${id}`, { method: "DELETE" });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            this.showToast("Match deleted.", "success");
            this.loadLibrary();
        } catch (err) {
            console.error(err);
            this.showToast("Could not delete match.", "error");
        }
    }

    // ----- Defaults / form ------------------------------------------------- //
    async loadDefaults() {
        try {
            const resp = await fetch("/api/config/defaults");
            if (!resp.ok) throw new Error("Failed to load defaults");
            const payload = await resp.json();
            this.defaults = payload.defaults || {};
            this.warnings = payload.warnings || {};
            this.availableDevices = payload.available_devices || ["cpu"];
            this.autoDevice = payload.auto_device || "cpu";
            this.runtimeState = payload.runtime_state || "cold";
            this.applyDefaults();
        } catch (err) {
            console.error(err);
            this.showToast("Could not load server defaults", "error");
        }
    }

    applyDefaults() {
        this.populateDeviceSelect();
        this.outputName.value = "";
        this.low.value = this.defaults.low ?? 0.45;
        this.high.value = this.defaults.high ?? 0.7;
        this.minDurSec.value = this.defaults.min_dur_sec ?? 1.0;
        this.lowWarning.textContent = this.warnings.low || "";
        this.highWarning.textContent = this.warnings.high || "";
        this.minDurWarning.textContent = this.warnings.min_dur_sec || "";
        this.updateDeviceNote();
    }

    populateSelect(selectEl, options, selected) {
        selectEl.innerHTML = "";
        options.forEach((opt) => {
            const option = document.createElement("option");
            option.value = opt;
            option.textContent = opt;
            option.selected = opt === selected;
            selectEl.appendChild(option);
        });
    }

    populateDeviceSelect() {
        const options = [{ value: "", label: `Auto (${this.autoDevice})` }];
        ["cuda", "mps", "cpu"].forEach((device) => {
            const available = this.availableDevices.includes(device);
            options.push({
                value: device,
                label: available ? device.toUpperCase() : `${device.toUpperCase()} (unavailable)`,
                disabled: !available,
            });
        });
        this.yoloDevice.innerHTML = "";
        options.forEach(({ value, label, disabled }) => {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = label;
            option.disabled = Boolean(disabled);
            this.yoloDevice.appendChild(option);
        });
    }

    updateDeviceNote() {
        if (this.runtimeState === "warming") {
            this.deviceNote.textContent = "Preparing analysis runtime...";
            return;
        }
        if (this.runtimeState === "error") {
            this.deviceNote.textContent = "Could not prepare analysis runtime; processing may fail.";
            return;
        }
        const selected = this.yoloDevice.value;
        if (!selected) {
            this.deviceNote.textContent = `Auto picks ${this.autoDevice.toUpperCase()} on this machine (CUDA > MPS > CPU).`;
            return;
        }
        this.deviceNote.textContent = `Using ${selected.toUpperCase()} for pose extraction.`;
    }

    async warmAnalysisRuntime() {
        if (this.runtimeState === "ready" || this.runtimeState === "warming") return;
        this.runtimeState = "warming";
        this.updateDeviceNote();
        try {
            await fetch("/api/runtime/warmup", { method: "POST" });
            this.pollAnalysisRuntime();
        } catch (err) {
            console.error(err);
            this.runtimeState = "error";
            this.updateDeviceNote();
        }
    }

    async pollAnalysisRuntime() {
        if (this.runtimeWarmupPoll) clearTimeout(this.runtimeWarmupPoll);
        try {
            const resp = await fetch("/api/runtime/status");
            if (!resp.ok) throw new Error("Failed to load runtime status");
            const payload = await resp.json();
            this.runtimeState = payload.state || "cold";
            this.availableDevices = payload.available_devices || this.availableDevices || ["cpu"];
            this.autoDevice = payload.auto_device || this.autoDevice || "cpu";
            this.populateDeviceSelect();
            this.updateDeviceNote();
            if (this.runtimeState === "warming" || this.runtimeState === "cold") {
                this.runtimeWarmupPoll = setTimeout(() => this.pollAnalysisRuntime(), 750);
            }
        } catch (err) {
            console.error(err);
            this.runtimeState = "error";
            this.updateDeviceNote();
        }
    }

    processFile(file) {
        // Accept any video; the backend validates by content (codec/resolution).
        const maxSize = 2 * 1024 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showToast("File must be under 2GB.", "error");
            return;
        }
        this.selectedFile = file;
        this.fileName.textContent = file.name;
        this.fileSize.textContent = this.formatFileSize(file.size);
        this.dropZone.hidden = true;
        this.selectedFileDiv.hidden = false;
        this.startBtn.disabled = false;
    }

    removeFile() {
        this.selectedFile = null;
        this.fileInput.value = "";
        this.dropZone.hidden = false;
        this.selectedFileDiv.hidden = true;
        this.startBtn.disabled = true;
        this.resetProgress();
    }

    formatFileSize(bytes) {
        const units = ["Bytes", "KB", "MB", "GB"];
        if (bytes === 0) return "0 Bytes";
        const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
        return `${(bytes / 1024 ** i).toFixed(2)} ${units[i]}`;
    }

    buildConfigFromForm() {
        const cfg = { ...(this.defaults || {}) };
        cfg.output_name = this.outputName.value.trim() || null;
        cfg.yolo_device = this.yoloDevice.value || null;
        cfg.write_csv = true;
        cfg.segment_video = true;
        const parsedLow = parseFloat(this.low.value);
        const parsedHigh = parseFloat(this.high.value);
        const parsedMinDur = parseFloat(this.minDurSec.value);
        cfg.low = Number.isNaN(parsedLow) ? cfg.low : parsedLow;
        cfg.high = Number.isNaN(parsedHigh) ? cfg.high : parsedHigh;
        cfg.min_dur_sec = Number.isNaN(parsedMinDur) ? cfg.min_dur_sec : parsedMinDur;
        return cfg;
    }

    // ----- Processing ------------------------------------------------------ //
    async startAnalysis() {
        if (!this.selectedFile || this.isProcessing) return;
        this.markWelcomeSeen();
        this.libraryId = null;
        this.resetProgress();
        this.isProcessing = true;
        this.startBtn.disabled = true;
        this.cancelBtn.disabled = false;
        this.stageText.textContent = "Preparing";
        this.showView("processing");

        try {
            const jobId = await this.uploadFileAndStart();
            this.currentJobId = jobId;
            try { localStorage.setItem(JOB_ID_KEY, jobId); } catch (_) {}
            this.startProgressMonitoring();
        } catch (error) {
            console.error(error);
            this.showToast(this.errorText(error) || "Failed to start segmentation.", "error");
            this.resetControls();
            this.showView("upload");
        }
    }

    errorText(error) {
        const msg = (error && error.message) || "";
        try {
            return JSON.parse(msg).error || msg;
        } catch (_) {
            return msg;
        }
    }

    async uploadFileAndStart() {
        const formData = new FormData();
        formData.append("video", this.selectedFile);
        formData.append("config", JSON.stringify(this.buildConfigFromForm()));
        const response = await fetch("/api/upload-and-start", { method: "POST", body: formData });
        if (!response.ok) throw new Error(await response.text());
        const result = await response.json();
        return result.job_id;
    }

    startProgressMonitoring() {
        this.stopProgressMonitoring();
        this.pollFailures = 0;
        this.progressInterval = setInterval(() => this.updateProgress().then(() => {
            this.pollFailures = 0;
        }).catch((err) => {
            console.error(err);
            this.pollFailures += 1;
            if (this.pollFailures >= 5) {
                this.stopProgressMonitoring();
                this.onAnalysisError("Lost connection to the RallyClip backend.");
            }
        }), 1000);
    }

    async restoreJobIfAny() {
        let storedId = null;
        try { storedId = localStorage.getItem(JOB_ID_KEY); } catch (_) {}
        if (!storedId) return false;

        this.markWelcomeSeen();
        this.currentJobId = storedId;
        this.isProcessing = true;
        this.startBtn.disabled = true;
        this.cancelBtn.disabled = false;
        this.stageText.textContent = "Restoring";
        this.showView("processing");
        this.startProgressMonitoring();

        try {
            await this.updateProgress();
            return true;
        } catch (e) {
            console.warn("Could not restore job progress:", e);
            this.stopProgressMonitoring();
            this.resetControls();
            this.resetProgress();
            try { localStorage.removeItem(JOB_ID_KEY); } catch (_) {}
            return false;
        }
    }

    async updateProgress() {
        if (!this.currentJobId) return;
        const response = await fetch(`/api/progress/${this.currentJobId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const progress = await response.json();
        if (progress.weights) this.weights = progress.weights;
        this.etaSeconds = progress.eta_seconds ?? null;
        if (progress.library_id) this.libraryId = progress.library_id;
        this.displayProgress(progress);

        if (progress.status === "completed") this.onAnalysisComplete();
        else if (progress.status === "failed") this.onAnalysisError(progress.error || "Segmentation failed");
        else if (progress.status === "cancelled") this.onAnalysisCancelled();
    }

    displayProgress(progress) {
        let weightedSum = 0;
        let weightTotal = 0;
        let plainSum = 0;
        const stepState = progress.steps || {};

        this.steps.forEach((step) => {
            const stepProgress = stepState[step] || { status: "waiting", progress: 0 };
            const elements = this.progressItems[step];
            elements.status.textContent = this.formatStatus(stepProgress.status);
            elements.fill.style.width = `${stepProgress.progress}%`;
            plainSum += stepProgress.progress;
            if (this.weights && typeof this.weights[step] === "number") {
                weightedSum += stepProgress.progress * this.weights[step];
                weightTotal += this.weights[step];
            }
        });

        const overall = weightTotal > 0 ? weightedSum / weightTotal : plainSum / this.steps.length;
        this.stageText.textContent = this.currentStageLabel(stepState);
        this.overallFill.style.width = `${overall}%`;
        this.overallPercentage.textContent = `${Math.round(overall)}%`;
        this.updateEta(progress);
    }

    currentStageLabel(stepState) {
        const running = this.steps.find((step) => stepState[step]?.status === "in_progress");
        if (running) return this.stepLabels[running];
        const waiting = this.steps.find((step) => stepState[step]?.status === "waiting");
        if (waiting) return waiting === "pose" ? "Starting" : this.stepLabels[waiting];
        return "Finishing";
    }

    updateEta(progress) {
        const eta = progress.eta_seconds ?? this.etaSeconds;
        if (eta == null) {
            this.etaText.textContent = "ETA: --";
            return;
        }
        const remaining = Math.max(0, Math.round(eta));
        const minutes = Math.floor(remaining / 60);
        const seconds = remaining % 60;
        this.etaText.textContent = minutes > 0
            ? `ETA: ~${minutes}m ${String(seconds).padStart(2, "0")}s`
            : `ETA: ~${seconds}s`;
    }

    formatStatus(status) {
        return ({
            waiting: "Waiting",
            in_progress: "Running",
            completed: "Done",
            failed: "Failed",
            cancelled: "Cancelled",
        })[status] || status;
    }

    async cancelAnalysis() {
        if (!this.currentJobId || !this.isProcessing) return;
        try {
            const response = await fetch(`/api/cancel/${this.currentJobId}`, { method: "POST" });
            if (response.ok) {
                const payload = await response.json().catch(() => ({}));
                if (payload.status === "cancelled") this.onAnalysisCancelled();
            } else {
                this.showToast("Could not cancel job.", "error");
            }
        } catch (err) {
            console.error("Cancel request failed:", err);
            this.showToast("Cancel request failed.", "error");
            this.resetControls();
            this.showLibrary();
        }
    }

    onAnalysisComplete() {
        this.stopProgressMonitoring();
        this.isProcessing = false;
        this.cancelBtn.disabled = true;
        try { localStorage.removeItem(JOB_ID_KEY); } catch (_) {}
        const saved = Boolean(this.libraryId);
        this.currentJobId = null;
        this.selectedFile = null;
        this.libraryId = null;
        this.showLibrary();
        if (saved) this.showToast("Saved to your matches.", "success");
        else this.showToast("No tennis points detected in this video.", "info");
    }

    onAnalysisError(error) {
        this.stopProgressMonitoring();
        this.resetControls();
        this.resetProgress();
        this.showToast(error, "error");
        this.showView("upload");
        try { localStorage.removeItem(JOB_ID_KEY); } catch (_) {}
    }

    onAnalysisCancelled() {
        this.stopProgressMonitoring();
        this.resetControls();
        this.resetProgress();
        this.showToast("Job cancelled.", "success");
        this.showLibrary();
        try { localStorage.removeItem(JOB_ID_KEY); } catch (_) {}
    }

    stopProgressMonitoring() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    resetControls() {
        this.isProcessing = false;
        this.currentJobId = null;
        this.startBtn.disabled = !this.selectedFile;
        this.cancelBtn.disabled = true;
    }

    resetProgress() {
        Object.values(this.progressItems).forEach((item) => {
            item.status.textContent = "Waiting";
            item.fill.style.width = "0%";
        });
        this.overallFill.style.width = "0%";
        this.overallPercentage.textContent = "0%";
        this.stageText.textContent = "Ready";
        this.etaText.textContent = "ETA: --";
        if (this.progressDetails) this.progressDetails.open = false;
    }

    showToast(message, tone = "info") {
        const toast = document.createElement("div");
        toast.className = `toast ${tone}`;
        toast.textContent = message;
        this.toastStack.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.rallyClipApp = new RallyClipApp();
});
