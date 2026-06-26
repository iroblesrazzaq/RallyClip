const WELCOME_SEEN_KEY = "rallyclip_welcome_seen";
const JOB_ID_KEY = "rallyclip_job_id";
const WELCOME_TYPE_START_DELAY_MS = 504;
const WELCOME_TYPE_LINE_DELAY_MS = WELCOME_TYPE_START_DELAY_MS;
const WELCOME_TYPE_FIRST_CHAR_MS = 110;
const WELCOME_TYPE_CHAR_MS = 89;
const SYSTEM_DARK_QUERY = "(prefers-color-scheme: dark)";
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
        this.previewLookaheadChunks = 12;
        this.pointBufferSeconds = 5;
        this.currentPreviewWindowStart = 0;
        this.currentPreviewWindowDuration = 0;
        this.sourceDuration = 0;
        this.previewRequestSeq = 0;
        this.previewLoadInProgress = false;
        this.previewSpinnerTimeout = null;
        this.viewerControlsHideTimeout = null;
        this.welcomeTypeTimers = [];
        this.welcomeCursor = null;
        this.systemThemeQuery = window.matchMedia ? window.matchMedia(SYSTEM_DARK_QUERY) : null;
        this.prefetchedWindowKeys = new Set();
        this.prefetchWindowTimers = new Map();
        this.readyPreviewWindows = new Map();
        this.viewerSeekDragging = false;
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
        this.loadDefaults();
        this.restoreJobIfAny().then((restored) => {
            if (restored) return;
            this.showWelcome();
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
        this.viewerTimeline = document.getElementById("viewerTimeline");
        this.viewerSeek = document.getElementById("viewerSeek");
        this.viewerSeekWrap = document.querySelector(".viewer-seek-wrap");
        this.viewerBufferTrack = document.getElementById("viewerBufferTrack");
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
        this.viewerBackBtn.addEventListener("click", () => this.skipViewerBy(-this.viewerSkipSeconds));
        this.viewerPlayPauseBtn.addEventListener("click", (e) => this.toggleViewerPlayback(e));
        this.viewerForwardBtn.addEventListener("click", () => this.skipViewerBy(this.viewerSkipSeconds));
        this.viewerSeek.addEventListener("input", () => {
            this.viewerSeekDragging = true;
            this.updateViewerTimeline(Number(this.viewerSeek.value));
        });
        this.viewerSeek.addEventListener("change", () => {
            this.viewerSeekDragging = false;
            this.seekViewerToSourceTime(Number(this.viewerSeek.value), this.viewerHasVideo() && !this.matchVideo.paused);
        });
        this.viewerSeekWrap.addEventListener("pointerdown", (e) => this.seekViewerFromTimelinePointer(e));
        document.addEventListener("keydown", (e) => this.handleViewerKeyboardShortcuts(e));
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
        video.addEventListener("error", () => {
            if (video === this.matchVideo) this.handleViewerVideoError();
        });
        video.addEventListener("click", (e) => {
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

    hasSeenWelcome() {
        try {
            return localStorage.getItem(WELCOME_SEEN_KEY) === "1";
        } catch (_) {
            return true;
        }
    }

    markWelcomeSeen() {
        try {
            localStorage.setItem(WELCOME_SEEN_KEY, "1");
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
            this.resetViewerVideos();
            this.viewingItemId = null;
            this.pointIntervals = [];
            this.lastViewerTime = null;
            this.viewerSeekInProgress = false;
            this.previewLoadInProgress = false;
            this.currentPreviewWindowStart = 0;
            this.currentPreviewWindowDuration = 0;
            this.sourceDuration = 0;
            this.clearPrefetchTimers();
            this.prefetchedWindowKeys.clear();
            this.readyPreviewWindows.clear();
            this.renderViewerBufferedRanges();
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
        this.clearVideoElement(this.primaryMatchVideo);
        this.clearVideoElement(this.secondaryMatchVideo);
        this.matchVideo = this.primaryMatchVideo;
        this.matchVideoBuffer = this.secondaryMatchVideo;
        this.primaryMatchVideo.classList.add("is-active");
        this.secondaryMatchVideo.classList.remove("is-active");
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
                this.showViewer(item);
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
            return new Date(iso).toLocaleString();
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
        this.showViewer(item);
    }

    async showViewer(item) {
        if (!item || !item.id) return;
        this.viewingItemId = item.id;
        this.pointIntervals = [];
        this.lastViewerTime = null;
        this.viewerSeekInProgress = false;
        this.previewLoadInProgress = false;
        this.currentPreviewWindowStart = 0;
        this.currentPreviewWindowDuration = 0;
        this.sourceDuration = Number(item.duration_s) || 0;
        this.clearPrefetchTimers();
        this.prefetchedWindowKeys.clear();
        this.readyPreviewWindows.clear();
        this.renderViewerBufferedRanges();
        this.configureViewerTimeline(this.sourceDuration);
        this.viewerTitle.textContent = item.name || "Match";
        this.viewerMeta.textContent = item.metaText || this.cardMeta(item);
        this.viewerCsvBtn.hidden = !item.has_csv;
        this.resetViewerVideos();
        this.updateViewerControls();
        this.showPreviewLoading();
        this.showView("viewer");
        this.showViewerControls();
        await this.loadPointIntervals(item.id);
        const start = this.pointIntervals.length ? this.pointIntervals[0].start : 0;
        this.loadPreviewWindowAt(start, true);
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
        if (!this.previewStatus.hidden) return;
        this.clearPreviewSpinner();
        this.previewStatus.hidden = false;
        this.previewStatus.classList.remove("is-slow");
        this.previewSpinnerTimeout = setTimeout(() => {
            if (!this.previewStatus.hidden) this.previewStatus.classList.add("is-slow");
        }, 2000);
    }

    hidePreviewLoading() {
        this.clearPreviewSpinner();
        this.previewStatus.hidden = true;
        this.previewStatus.classList.remove("is-slow");
    }

    buildPreviewWindowUrl(itemId, start, duration, status = false) {
        const suffix = status ? "/status" : "";
        return `/api/library/${itemId}/preview/window${suffix}?start=${start.toFixed(3)}&duration=${duration.toFixed(3)}`;
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

    preloadNearestReadyWindow() {
        if (!this.matchVideoBuffer || !this.viewingItemId) return;
        const next = Array.from(this.readyPreviewWindows.values())
            .filter((windowInfo) => windowInfo.previewUrl && windowInfo.start > this.currentPreviewWindowStart + 0.001)
            .sort((a, b) => a.start - b.start)[0];
        if (!next) return;
        this.setPreviewVideoSource(this.matchVideoBuffer, next.previewUrl, next.start, next.duration);
    }

    activatePreviewWindow(itemId, requestId, payload, targetSourceTime, autoplay) {
        if (this.viewingItemId !== itemId || requestId !== this.previewRequestSeq) return;
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
            if (this.viewingItemId !== itemId || requestId !== this.previewRequestSeq) return;
            this.currentPreviewWindowStart = windowStart;
            this.currentPreviewWindowDuration = windowDuration;
            const offset = Math.max(0, Math.min(targetSourceTime - windowStart, targetVideo.duration || windowDuration));
            targetVideo.currentTime = offset;
            if (targetVideo !== this.matchVideo) {
                const previousVideo = this.matchVideo;
                previousVideo.pause();
                previousVideo.classList.remove("is-active");
                targetVideo.classList.add("is-active");
                targetVideo.tabIndex = 0;
                previousVideo.tabIndex = -1;
                this.matchVideo = targetVideo;
                this.matchVideoBuffer = previousVideo;
                this.clearVideoElement(previousVideo);
            }
            this.hidePreviewLoading();
            this.lastViewerTime = windowStart + offset;
            this.updateViewerTimeline(this.lastViewerTime);
            this.updateViewerControls();
            this.prefetchPointAwareWindows(this.lastViewerTime);
            this.preloadNearestReadyWindow();
            this.previewLoadInProgress = false;
            if (autoplay) this.startViewerPlayback();
        };

        if (targetVideo.readyState >= 2) finishActivation();
        else targetVideo.addEventListener("canplay", finishActivation, { once: true });
    }

    loadPreviewWindowAt(sourceTime, autoplay = true) {
        if (!this.viewingItemId) return;
        const itemId = this.viewingItemId;
        const requestId = ++this.previewRequestSeq;
        const targetSourceTime = Math.max(0, Number(sourceTime) || 0);
        this.clearPreviewPoll();
        this.previewLoadInProgress = true;
        if (this.viewerHasVideo()) this.hidePreviewLoading();
        else this.showPreviewLoading();
        const readyWindow = this.getReadyPreviewWindow(targetSourceTime);
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
                const resp = await fetch(this.buildPreviewWindowUrl(itemId, targetSourceTime, this.previewWindowDuration, true));
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const payload = await resp.json();
                if (this.viewingItemId !== itemId || requestId !== this.previewRequestSeq) return;
                if (typeof payload.source_duration === "number" && payload.source_duration > 0) {
                    this.sourceDuration = payload.source_duration;
                    this.configureViewerTimeline(this.sourceDuration);
                }
                if (payload.ready && payload.preview_url) {
                    this.activatePreviewWindow(itemId, requestId, payload, targetSourceTime, autoplay);
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

    previewChunkKeyForStart(start) {
        return `${this.viewingItemId}:${Number(start).toFixed(1)}:${this.previewWindowDuration.toFixed(1)}`;
    }

    previewChunkKey(sourceTime) {
        return this.previewChunkKeyForStart(this.canonicalPreviewChunkStart(sourceTime));
    }

    markPreviewWindowReady(start, duration = this.previewWindowDuration, previewUrl = null) {
        if (!this.viewingItemId) return;
        const canonicalStart = this.canonicalPreviewChunkStart(start);
        const safeDuration = Math.max(0.1, Number(duration) || this.previewWindowDuration);
        const key = this.previewChunkKeyForStart(canonicalStart);
        this.readyPreviewWindows.set(key, { start: canonicalStart, duration: safeDuration, previewUrl });
        this.prefetchedWindowKeys.add(key);
        if (this.prefetchWindowTimers.has(key)) {
            clearTimeout(this.prefetchWindowTimers.get(key));
            this.prefetchWindowTimers.delete(key);
        }
        this.renderViewerBufferedRanges();
        this.preloadNearestReadyWindow();
    }

    getReadyPreviewWindow(sourceTime) {
        const start = this.canonicalPreviewChunkStart(sourceTime);
        return this.readyPreviewWindows.get(this.previewChunkKeyForStart(start)) || null;
    }

    renderViewerBufferedRanges() {
        if (!this.viewerBufferTrack) return;
        this.viewerBufferTrack.innerHTML = "";
        const duration = Math.max(0, Number(this.sourceDuration) || 0);
        if (duration <= 0) return;

        Array.from(this.readyPreviewWindows.values())
            .sort((a, b) => a.start - b.start)
            .forEach((windowInfo) => {
                const start = Math.max(0, Math.min(duration, windowInfo.start));
                const end = Math.max(start, Math.min(duration, windowInfo.start + windowInfo.duration));
                if (end <= start) return;
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
        if (!this.pointIntervals.length) return -1;
        const t = Number(sourceTime) || 0;
        const active = this.pointIntervals.findIndex((seg) => t >= seg.start && t < seg.end);
        if (active >= 0) return active;
        return this.pointIntervals.findIndex((seg) => t < seg.end);
    }

    getPointAwarePrefetchStarts(sourceTime) {
        const limit = Math.max(0, Number(this.previewLookaheadChunks) || 0);
        if (!limit) return [];

        const currentStart = this.canonicalPreviewChunkStart(sourceTime);
        const chunk = Math.max(1, Number(this.previewWindowDuration) || 8);
        const selected = [];
        const seen = new Set([this.previewChunkKey(currentStart)]);

        const addChunk = (start) => {
            const canonicalStart = this.canonicalPreviewChunkStart(start);
            if (canonicalStart < currentStart - 0.001) return true;
            if (this.sourceDuration > 0 && canonicalStart >= this.sourceDuration - 0.1) return true;
            const key = this.previewChunkKey(canonicalStart);
            if (seen.has(key)) return true;
            seen.add(key);
            selected.push(canonicalStart);
            return selected.length < limit;
        };

        const addRange = (start, end) => {
            const candidates = this.previewChunkStartsForRange(start, end)
                .filter((candidate) => candidate >= currentStart + chunk - 0.001);
            for (const candidate of candidates) {
                if (!addChunk(candidate)) return false;
            }
            return true;
        };

        const addBufferedPoint = (seg) => {
            const bufferedStart = Math.max(0, seg.start - this.pointBufferSeconds);
            const bufferedEnd = seg.end + this.pointBufferSeconds;
            const candidates = this.previewChunkStartsForRange(bufferedStart, bufferedEnd)
                .filter((start) => start >= currentStart - 0.001 && !seen.has(this.previewChunkKey(start)));
            for (const start of candidates) {
                if (!addChunk(start)) return false;
            }
            return true;
        };

        const pointIdx = this.findPlaybackTargetPointIndex(sourceTime);
        if (pointIdx >= 0) {
            const target = this.pointIntervals[pointIdx];
            addRange(sourceTime, target.end + this.pointBufferSeconds);
            const next = this.pointIntervals[pointIdx + 1];
            if (next && selected.length < limit) addBufferedPoint(next);
        } else {
            for (let start = currentStart + chunk; selected.length < limit; start += chunk) {
                if (!addChunk(start)) break;
            }
        }

        return selected;
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
        if (!this.viewingItemId) return;
        this.trimPrefetchedWindowKeys(sourceTime);
        this.getPointAwarePrefetchStarts(sourceTime).forEach((start) => this.prefetchPreviewWindow(start));
    }

    prefetchPreviewWindow(sourceTime) {
        if (!this.viewingItemId || !Number.isFinite(sourceTime)) return;
        if (this.sourceDuration > 0 && sourceTime >= this.sourceDuration - 0.1) return;
        const start = this.canonicalPreviewChunkStart(sourceTime);
        const key = this.previewChunkKey(start);
        if (this.readyPreviewWindows.has(key) || this.prefetchWindowTimers.has(key)) return;
        if (this.prefetchedWindowKeys.has(key)) return;
        this.prefetchedWindowKeys.add(key);
        this.pollPrefetchPreviewWindow(start);
    }

    async pollPrefetchPreviewWindow(start, delayMs = 0) {
        if (!this.viewingItemId) return;
        const canonicalStart = this.canonicalPreviewChunkStart(start);
        const key = this.previewChunkKeyForStart(canonicalStart);
        if (this.readyPreviewWindows.has(key)) return;
        if (delayMs > 0) {
            const timer = setTimeout(() => {
                this.prefetchWindowTimers.delete(key);
                this.pollPrefetchPreviewWindow(canonicalStart);
            }, delayMs);
            this.prefetchWindowTimers.set(key, timer);
            return;
        }
        try {
            const resp = await fetch(this.buildPreviewWindowUrl(this.viewingItemId, canonicalStart, this.previewWindowDuration, true));
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const payload = await resp.json();
            if (!this.viewingItemId) return;
            if (payload.ready && payload.preview_url) {
                this.markPreviewWindowReady(Number(payload.start) || canonicalStart, Number(payload.duration) || this.previewWindowDuration, payload.preview_url);
                return;
            }
            this.pollPrefetchPreviewWindow(canonicalStart, 900);
        } catch (_) {
            this.pollPrefetchPreviewWindow(canonicalStart, 1500);
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
        [this.viewerBackBtn, this.viewerPlayPauseBtn, this.viewerForwardBtn].forEach((btn) => {
            if (btn) btn.disabled = disabled;
        });
        if (this.viewerPlayPauseBtn) {
            this.viewerPlayPauseBtn.textContent = isPaused ? "Play" : "Pause";
            this.viewerPlayPauseBtn.setAttribute("aria-label", isPaused ? "Play video" : "Pause video");
        }
    }

    toggleViewerPlayback(event) {
        if (event && typeof event.preventDefault === "function") event.preventDefault();
        if (!this.matchVideo.src) return;
        this.showViewerControls();
        if (this.matchVideo.paused) this.startViewerPlayback();
        else this.matchVideo.pause();
    }

    skipViewerBy(deltaSeconds) {
        if (!this.viewerHasVideo()) return;
        this.showViewerControls();
        const wasPlaying = !this.matchVideo.paused;
        const current = this.getViewerSourceTime();
        this.seekViewerToSourceTime(this.clampViewerSourceTime(current + deltaSeconds), wasPlaying);
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
        }
    }

    async loadPointIntervals(itemId) {
        try {
            const resp = await fetch(`/api/library/${itemId}/segments`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const payload = await resp.json();
            if (this.viewingItemId !== itemId) return;
            this.pointIntervals = (payload.segments || [])
                .map((seg) => ({ start: Number(seg.start), end: Number(seg.end) }))
                .filter((seg) => Number.isFinite(seg.start) && Number.isFinite(seg.end) && seg.end > seg.start)
                .sort((a, b) => a.start - b.start || a.end - b.end);
            this.lastViewerTime = null;
        } catch (err) {
            console.error(err);
            this.pointIntervals = [];
            this.showToast("Could not load point times.", "error");
        }
    }

    configureViewerTimeline(duration) {
        const safeDuration = Math.max(0, Number(duration) || 0);
        this.viewerTimeline.hidden = safeDuration <= 0;
        this.viewerSeek.min = "0";
        this.viewerSeek.max = safeDuration > 0 ? String(safeDuration) : "0";
        this.viewerSeek.step = "0.1";
        this.viewerDuration.textContent = this.formatClock(safeDuration);
        this.updateViewerTimeline(0);
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
        if (!this.viewerSeekDragging) {
            this.viewerSeek.value = String(Math.min(safeTime, Number(this.viewerSeek.max) || safeTime));
        }
    }

    getViewerSourceTime() {
        return this.currentPreviewWindowStart + (Number(this.matchVideo.currentTime) || 0);
    }

    seekViewerToSourceTime(sourceTime, autoplay = true) {
        const target = Math.max(0, Number(sourceTime) || 0);
        const windowEnd = this.currentPreviewWindowStart + this.currentPreviewWindowDuration;
        if (this.matchVideo.src && target >= this.currentPreviewWindowStart && target < windowEnd - 0.15) {
            this.matchVideo.currentTime = Math.max(0, target - this.currentPreviewWindowStart);
            this.lastViewerTime = target;
            this.updateViewerTimeline(target);
            if (autoplay) this.startViewerPlayback();
            return;
        }
        this.loadPreviewWindowAt(target, autoplay);
    }

    handleViewerVideoError() {
        if (!this.matchVideo.src || !this.matchVideo.error) return;
        this.showToast("Could not play this video in the viewer.", "error");
    }

    handleViewerTimeUpdate() {
        if (!this.matchVideo.src) return;
        const t = this.getViewerSourceTime();
        this.updateViewerTimeline(t);
        const remaining = this.currentPreviewWindowStart + this.currentPreviewWindowDuration - t;
        if (remaining < 8) this.prefetchPointAwareWindows(t);
        if (!this.pointIntervals.length || this.matchVideo.paused) return;
        const previous = this.lastViewerTime;
        this.lastViewerTime = t;
        if (this.viewerSeekInProgress || previous === null || t <= previous) return;

        const idx = this.findCrossedPointIndex(previous, t);
        if (idx < 0) return;

        this.advanceAfterPointIndex(idx);
    }

    handleViewerWindowEnded() {
        if (!this.matchVideo.src || !this.viewingItemId) return;
        if (this.previewLoadInProgress) return;
        const sourceTime = this.currentPreviewWindowStart + this.currentPreviewWindowDuration;
        this.updateViewerTimeline(sourceTime);
        if (this.pointIntervals.length && this.lastViewerTime !== null) {
            const idx = this.findCrossedPointIndex(this.lastViewerTime, sourceTime);
            this.lastViewerTime = sourceTime;
            if (idx >= 0) {
                this.advanceAfterPointIndex(idx);
                return;
            }
        }
        if (this.sourceDuration > 0 && sourceTime >= this.sourceDuration - 0.25) return;
        this.loadPreviewWindowAt(sourceTime, true);
    }

    findCrossedPointIndex(previous, current) {
        return this.pointIntervals.findIndex((seg) => previous < seg.end && current >= seg.end);
    }

    advanceAfterPointIndex(idx) {
        const next = this.pointIntervals[idx + 1];
        if (!next) {
            this.matchVideo.pause();
            this.lastViewerTime = this.pointIntervals[idx].end;
            this.updateViewerTimeline(this.lastViewerTime);
            return;
        }
        this.prefetchPointAwareWindows(next.start);
        this.seekViewerToSourceTime(next.start, true);
    }

    findPointIndexAt(sourceTime) {
        const t = Number(sourceTime) || 0;
        return this.pointIntervals.findIndex((seg) => t >= seg.start && t < seg.end);
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
        const selected = this.yoloDevice.value;
        if (!selected) {
            this.deviceNote.textContent = `Auto picks ${this.autoDevice.toUpperCase()} on this machine (CUDA > MPS > CPU).`;
            return;
        }
        this.deviceNote.textContent = `Using ${selected.toUpperCase()} for pose extraction.`;
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
