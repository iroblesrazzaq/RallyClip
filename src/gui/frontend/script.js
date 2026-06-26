const WELCOME_SEEN_KEY = "rallyclip_welcome_seen";
const JOB_ID_KEY = "rallyclip_job_id";

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
        this.previewWindowDuration = 45;
        this.currentPreviewWindowStart = 0;
        this.currentPreviewWindowDuration = 0;
        this.sourceDuration = 0;
        this.previewRequestSeq = 0;
        this.prefetchWindowKey = null;
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
        this.loadDefaults();
        this.restoreJobIfAny().then((restored) => {
            if (restored) return;
            if (this.hasSeenWelcome()) this.showLibrary();
            else this.showWelcome();
        });
    }

    cacheElements() {
        this.welcomeScreen = document.getElementById("welcomeScreen");
        this.welcomeStartBtn = document.getElementById("welcomeStartBtn");
        this.appShell = document.getElementById("appShell");

        this.libraryView = document.getElementById("libraryView");
        this.libraryGrid = document.getElementById("libraryGrid");
        this.libraryEmpty = document.getElementById("libraryEmpty");
        this.newMatchBtn = document.getElementById("newMatchBtn");
        this.emptyNewMatchBtn = document.getElementById("emptyNewMatchBtn");

        this.viewerView = document.getElementById("viewerView");
        this.backFromViewer = document.getElementById("backFromViewer");
        this.viewerTitle = document.getElementById("viewerTitle");
        this.viewerMeta = document.getElementById("viewerMeta");
        this.matchVideo = document.getElementById("matchVideo");
        this.previewStatus = document.getElementById("previewStatus");
        this.viewerTimeline = document.getElementById("viewerTimeline");
        this.viewerSeek = document.getElementById("viewerSeek");
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
        this.emptyNewMatchBtn.addEventListener("click", () => this.showUpload());
        this.backToLibrary.addEventListener("click", () => this.showLibrary());
        this.backFromViewer.addEventListener("click", () => this.showLibrary());
        this.viewerExportBtn.addEventListener("click", () => {
            if (this.viewingItemId) this.triggerDownload(`/api/library/${this.viewingItemId}/video`);
        });
        this.viewerCsvBtn.addEventListener("click", () => {
            if (this.viewingItemId) this.triggerDownload(`/api/library/${this.viewingItemId}/csv`);
        });
        this.matchVideo.addEventListener("timeupdate", () => this.handleViewerTimeUpdate());
        this.matchVideo.addEventListener("seeking", () => {
            this.viewerSeekInProgress = true;
        });
        this.matchVideo.addEventListener("seeked", () => {
            this.viewerSeekInProgress = false;
            this.lastViewerTime = this.getViewerSourceTime();
        });
        this.matchVideo.addEventListener("ended", () => this.handleViewerWindowEnded());
        this.matchVideo.addEventListener("error", () => this.handleViewerVideoError());
        this.viewerSeek.addEventListener("input", () => {
            this.viewerSeekDragging = true;
            this.updateViewerTimeline(Number(this.viewerSeek.value));
        });
        this.viewerSeek.addEventListener("change", () => {
            this.viewerSeekDragging = false;
            this.seekViewerToSourceTime(Number(this.viewerSeek.value), true);
        });
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
        this.showLibrary();
    }

    showWelcome() {
        this.welcomeScreen.hidden = false;
        this.appShell.hidden = true;
    }

    showView(viewName) {
        this.welcomeScreen.hidden = true;
        this.appShell.hidden = false;
        this.libraryView.hidden = viewName !== "library";
        this.viewerView.hidden = viewName !== "viewer";
        this.uploadView.hidden = viewName !== "upload";
        this.progressCard.hidden = viewName !== "processing";
        if (viewName !== "viewer" && this.matchVideo) {
            this.clearPreviewPoll();
            this.matchVideo.pause();
            this.matchVideo.removeAttribute("src");
            this.matchVideo.load();
            this.viewingItemId = null;
            this.pointIntervals = [];
            this.lastViewerTime = null;
            this.viewerSeekInProgress = false;
            this.currentPreviewWindowStart = 0;
            this.currentPreviewWindowDuration = 0;
            this.sourceDuration = 0;
            this.prefetchWindowKey = null;
            this.viewerSeekDragging = false;
            this.viewerTimeline.hidden = true;
            this.previewStatus.hidden = true;
        }
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
        this.libraryGrid.innerHTML = "";
        this.libraryEmpty.hidden = items.length > 0;
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
        this.currentPreviewWindowStart = 0;
        this.currentPreviewWindowDuration = 0;
        this.sourceDuration = Number(item.duration_s) || 0;
        this.prefetchWindowKey = null;
        this.configureViewerTimeline(this.sourceDuration);
        this.viewerTitle.textContent = item.name || "Match";
        this.viewerMeta.textContent = item.metaText || this.cardMeta(item);
        this.viewerCsvBtn.hidden = !item.has_csv;
        this.matchVideo.removeAttribute("src");
        this.matchVideo.load();
        this.previewStatus.textContent = "Preparing video preview...";
        this.previewStatus.hidden = false;
        this.showView("viewer");
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

    buildPreviewWindowUrl(itemId, start, duration, status = false) {
        const suffix = status ? "/status" : "";
        return `/api/library/${itemId}/preview/window${suffix}?start=${start.toFixed(3)}&duration=${duration.toFixed(3)}`;
    }

    loadPreviewWindowAt(sourceTime, autoplay = true) {
        if (!this.viewingItemId) return;
        const itemId = this.viewingItemId;
        const requestId = ++this.previewRequestSeq;
        const targetSourceTime = Math.max(0, Number(sourceTime) || 0);
        this.clearPreviewPoll();
        this.previewStatus.textContent = "Preparing video preview...";
        this.previewStatus.hidden = false;
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
                    this.currentPreviewWindowStart = Number(payload.start) || 0;
                    this.currentPreviewWindowDuration = Number(payload.duration) || this.previewWindowDuration;
                    this.previewStatus.hidden = true;
                    this.matchVideo.src = `${payload.preview_url}&t=${Date.now()}`;
                    this.lastViewerTime = null;
                    this.viewerSeekInProgress = false;
                    this.matchVideo.load();
                    const seekAfterLoad = () => {
                        if (this.viewingItemId !== itemId || requestId !== this.previewRequestSeq) return;
                        const offset = Math.max(0, Math.min(targetSourceTime - this.currentPreviewWindowStart, this.matchVideo.duration || this.currentPreviewWindowDuration));
                        this.matchVideo.currentTime = offset;
                        this.lastViewerTime = this.currentPreviewWindowStart + offset;
                        this.updateViewerTimeline(this.lastViewerTime);
                        this.prefetchPreviewWindow(this.currentPreviewWindowStart + this.currentPreviewWindowDuration);
                        if (autoplay) this.startViewerPlayback();
                    };
                    if (this.matchVideo.readyState >= 1) seekAfterLoad();
                    else this.matchVideo.addEventListener("loadedmetadata", seekAfterLoad, { once: true });
                    return;
                }
                this.previewStatus.textContent = "Preparing video preview...";
                this.previewStatus.hidden = false;
                this.previewPollTimeout = setTimeout(poll, 750);
            } catch (err) {
                console.error(err);
                if (this.viewingItemId !== itemId || requestId !== this.previewRequestSeq) return;
                this.previewStatus.textContent = "Still preparing preview...";
                this.previewStatus.hidden = false;
                this.previewPollTimeout = setTimeout(poll, 1500);
            }
        };
        poll();
    }

    prefetchPreviewWindow(sourceTime) {
        if (!this.viewingItemId || !Number.isFinite(sourceTime)) return;
        if (this.sourceDuration > 0 && sourceTime >= this.sourceDuration - 0.1) return;
        const start = Math.max(0, sourceTime);
        const key = `${this.viewingItemId}:${start.toFixed(1)}`;
        if (this.prefetchWindowKey === key) return;
        this.prefetchWindowKey = key;
        fetch(this.buildPreviewWindowUrl(this.viewingItemId, start, this.previewWindowDuration, true)).catch(() => {});
    }

    startViewerPlayback() {
        const playPromise = this.matchVideo.play();
        if (playPromise && typeof playPromise.catch === "function") {
            playPromise.catch(() => {
                // Browser autoplay policy can still block this if the card click
                // is not treated as the media gesture. The controls remain usable.
            });
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
        if (remaining < 8) this.prefetchPreviewWindow(this.currentPreviewWindowStart + this.currentPreviewWindowDuration);
        if (!this.pointIntervals.length || this.matchVideo.paused) return;
        const previous = this.lastViewerTime;
        this.lastViewerTime = t;
        if (this.viewerSeekInProgress || previous === null || t <= previous) return;

        const elapsed = t - previous;
        if (elapsed > 3.0) return;

        const idx = this.findCrossedPointIndex(previous, t);
        if (idx < 0) return;

        const next = this.pointIntervals[idx + 1];
        if (!next) {
            this.matchVideo.pause();
            this.lastViewerTime = this.pointIntervals[idx].end;
            this.updateViewerTimeline(this.lastViewerTime);
            return;
        }
        this.seekViewerToSourceTime(next.start, true);
    }

    handleViewerWindowEnded() {
        if (!this.matchVideo.src || !this.viewingItemId) return;
        const sourceTime = this.getViewerSourceTime();
        this.updateViewerTimeline(sourceTime);
        if (this.sourceDuration > 0 && sourceTime >= this.sourceDuration - 0.25) return;
        this.loadPreviewWindowAt(sourceTime, true);
    }

    findCrossedPointIndex(previous, current) {
        return this.pointIntervals.findIndex((seg) => previous < seg.end && current >= seg.end);
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
