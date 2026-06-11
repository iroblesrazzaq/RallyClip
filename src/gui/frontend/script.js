class RallyClipApp {
    constructor() {
        this.selectedFile = null;
        this.isProcessing = false;
        this.currentJobId = null;
        this.progressInterval = null;
        this.defaults = {};
        this.warnings = {};
        this.yoloOptions = [];
        this.availableDevices = [];
        this.autoDevice = "cpu";
        this.weights = null;
        this.etaSeconds = null;

        this.cacheElements();
        this.bindEvents();
        this.loadDefaults();
        this.restoreJobIfAny();
    }

    cacheElements() {
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
        this.outputDir = document.getElementById("outputDir");
        this.csvOutputDir = document.getElementById("csvOutputDir");
        this.yoloSize = document.getElementById("yoloSize");
        this.yoloDevice = document.getElementById("yoloDevice");
        this.deviceNote = document.getElementById("deviceNote");
        this.low = document.getElementById("low");
        this.high = document.getElementById("high");
        this.minDurSec = document.getElementById("minDurSec");
        this.lowWarning = document.getElementById("lowWarning");
        this.highWarning = document.getElementById("highWarning");
        this.minDurWarning = document.getElementById("minDurWarning");
        this.statusBadge = document.getElementById("statusBadge");
        this.statusBadgeText = document.getElementById("statusBadgeText");
        this.progressCard = document.getElementById("progress");
        this.resultsCard = document.getElementById("results");
        this.overallFill = document.getElementById("overallFill");
        this.overallPercentage = document.getElementById("overallPercentage");
        this.etaText = document.getElementById("etaText");
        this.downloadVideoBtn = document.getElementById("downloadVideoBtn");
        this.downloadCsvBtn = document.getElementById("downloadCsvBtn");
        this.newAnalysisBtn = document.getElementById("newAnalysisBtn");
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
        this.downloadVideoBtn.addEventListener("click", () => this.downloadVideo());
        this.downloadCsvBtn.addEventListener("click", () => this.downloadCsv());
        this.newAnalysisBtn.addEventListener("click", () => this.startNewAnalysis());
        this.yoloDevice.addEventListener("change", () => this.updateDeviceNote());
        document.querySelectorAll("[data-scroll]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const target = document.getElementById(btn.dataset.scroll);
                if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
            });
        });
    }

    async loadDefaults() {
        try {
            const resp = await fetch("/api/config/defaults");
            if (!resp.ok) throw new Error("Failed to load defaults");
            const payload = await resp.json();
            this.defaults = payload.defaults || {};
            this.warnings = payload.warnings || {};
            this.yoloOptions = payload.yolo_sizes || [];
            this.availableDevices = payload.available_devices || ["cpu"];
            this.autoDevice = payload.auto_device || "cpu";
            this.applyDefaults();
        } catch (err) {
            console.error(err);
            this.showToast("Could not load server defaults", "error");
        }
    }

    applyDefaults() {
        this.populateSelect(this.yoloSize, this.yoloOptions, this.defaults.yolo_size || "small");
        this.populateDeviceSelect();
        this.outputName.value = "";
        this.outputDir.value = this.defaults.output_dir || "";
        this.csvOutputDir.value = this.defaults.csv_output_dir || "";
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
        if (file.type !== "video/mp4" && !file.name.toLowerCase().endsWith(".mp4")) {
            this.showToast("Only MP4 videos are supported.", "error");
            return;
        }
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
        this.setStatus("Ready", "ready");
    }

    removeFile() {
        this.selectedFile = null;
        this.fileInput.value = "";
        this.dropZone.hidden = false;
        this.selectedFileDiv.hidden = true;
        this.startBtn.disabled = true;
        this.resetProgress();
        this.setStatus("Ready", "ready");
    }

    formatFileSize(bytes) {
        const units = ["Bytes", "KB", "MB", "GB"];
        if (bytes === 0) return "0 Bytes";
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return `${(bytes / 1024 ** i).toFixed(2)} ${units[i]}`;
    }

    buildConfigFromForm() {
        const cfg = { ...(this.defaults || {}) };
        cfg.output_name = this.outputName.value.trim() || null;
        cfg.output_dir = this.outputDir.value.trim() || cfg.output_dir;
        cfg.csv_output_dir = this.csvOutputDir.value.trim() || cfg.csv_output_dir;
        cfg.yolo_size = this.yoloSize.value || cfg.yolo_size;
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

    async startAnalysis() {
        if (!this.selectedFile || this.isProcessing) return;
        this.isProcessing = true;
        this.startBtn.disabled = true;
        this.cancelBtn.disabled = false;
        this.progressCard.hidden = false;
        this.resultsCard.hidden = true;
        this.setStatus("Processing", "processing");

        try {
            const jobId = await this.uploadFileAndStart();
            this.currentJobId = jobId;
            try { localStorage.setItem("rallyclip_job_id", jobId); } catch (_) {}
            this.startProgressMonitoring();
        } catch (error) {
            console.error(error);
            this.showToast("Failed to start segmentation.", "error");
            this.resetControls();
            this.setStatus("Error", "error");
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
        try { storedId = localStorage.getItem("rallyclip_job_id"); } catch (_) {}
        if (!storedId) return;
        this.currentJobId = storedId;
        this.isProcessing = true;
        this.progressCard.hidden = false;
        this.startBtn.disabled = true;
        this.cancelBtn.disabled = false;
        this.setStatus("Processing", "processing");
        this.startProgressMonitoring();
        try {
            await this.updateProgress();
        } catch (e) {
            console.warn("Could not restore job progress:", e);
            this.stopProgressMonitoring();
            this.resetControls();
            this.resetProgress();
            this.setStatus("Ready", "ready");
            try { localStorage.removeItem("rallyclip_job_id"); } catch (_) {}
        }
    }

    async updateProgress() {
        if (!this.currentJobId) return;
        const response = await fetch(`/api/progress/${this.currentJobId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const progress = await response.json();
        if (progress.weights) this.weights = progress.weights;
        this.etaSeconds = progress.eta_seconds ?? null;
        this.displayProgress(progress);

        if (progress.status === "completed") this.onAnalysisComplete();
        else if (progress.status === "failed") this.onAnalysisError(progress.error || "Segmentation failed");
        else if (progress.status === "cancelled") this.onAnalysisCancelled();
    }

    displayProgress(progress) {
        const steps = ["pose", "preprocess", "feature", "inference", "output"];
        let weightedSum = 0;
        let weightTotal = 0;
        let plainSum = 0;

        steps.forEach((step) => {
            const stepProgress = progress.steps[step] || { status: "waiting", progress: 0 };
            const elements = this.progressItems[step];
            elements.status.textContent = this.formatStatus(stepProgress.status);
            elements.fill.style.width = `${stepProgress.progress}%`;
            plainSum += stepProgress.progress;
            if (this.weights && typeof this.weights[step] === "number") {
                weightedSum += stepProgress.progress * this.weights[step];
                weightTotal += this.weights[step];
            }
        });

        const overall = weightTotal > 0 ? weightedSum / weightTotal : plainSum / steps.length;
        this.overallFill.style.width = `${overall}%`;
        this.overallPercentage.textContent = `${Math.round(overall)}%`;
        this.updateEta(progress);
    }

    updateEta(progress) {
        const eta = progress.eta_seconds ?? this.etaSeconds;
        if (eta == null) {
            this.etaText.textContent = "Est. remaining: --";
            return;
        }
        const remaining = Math.max(0, Math.round(eta));
        const minutes = Math.floor(remaining / 60);
        const seconds = remaining % 60;
        this.etaText.textContent = minutes > 0
            ? `Est. remaining: ~${minutes}m ${String(seconds).padStart(2, "0")}s`
            : `Est. remaining: ~${seconds}s`;
    }

    formatStatus(status) {
        return ({
            waiting: "Waiting",
            in_progress: "Running",
            completed: "Done",
            failed: "Failed",
        })[status] || status;
    }

    async cancelAnalysis() {
        if (!this.currentJobId || !this.isProcessing) return;
        try {
            const response = await fetch(`/api/cancel/${this.currentJobId}`, { method: "POST" });
            if (response.ok) {
                const payload = await response.json().catch(() => ({}));
                // The job may have finished before the cancel landed; let the
                // next progress poll surface the real terminal state.
                if (payload.status === "cancelled") this.onAnalysisCancelled();
            } else {
                this.showToast("Could not cancel job.", "error");
            }
        } catch (err) {
            console.error("Cancel request failed:", err);
            this.showToast("Cancel request failed.", "error");
            this.resetControls();
            this.setStatus("Ready", "ready");
        }
    }

    onAnalysisComplete() {
        this.stopProgressMonitoring();
        this.isProcessing = false;
        this.cancelBtn.disabled = true;
        this.resultsCard.hidden = false;
        this.setStatus("Done", "done");
        this.showToast("Segmentation complete.", "success");
        this.etaText.textContent = "Est. remaining: 0s";
        try { localStorage.removeItem("rallyclip_job_id"); } catch (_) {}
    }

    onAnalysisError(error) {
        this.stopProgressMonitoring();
        this.resetControls();
        this.showToast(error, "error");
        this.setStatus("Error", "error");
        try { localStorage.removeItem("rallyclip_job_id"); } catch (_) {}
    }

    onAnalysisCancelled() {
        this.stopProgressMonitoring();
        this.resetControls();
        this.resetProgress();
        this.showToast("Job cancelled.", "success");
        this.setStatus("Ready", "ready");
        try { localStorage.removeItem("rallyclip_job_id"); } catch (_) {}
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
        this.progressCard.hidden = true;
        this.resultsCard.hidden = true;
        Object.values(this.progressItems).forEach((item) => {
            item.status.textContent = "Waiting";
            item.fill.style.width = "0%";
        });
        this.overallFill.style.width = "0%";
        this.overallPercentage.textContent = "0%";
        this.etaText.textContent = "Est. remaining: --";
    }

    async downloadVideo() {
        if (!this.currentJobId) return;
        try {
            const response = await fetch(`/api/download/video/${this.currentJobId}`);
            if (!response.ok) return this.showToast("Video not ready.", "error");
            const blob = await response.blob();
            this.downloadBlob(blob, "rallyclip_segmented.mp4");
        } catch (err) {
            console.error("Video download failed:", err);
            this.showToast("Video download failed.", "error");
        }
    }

    async downloadCsv() {
        if (!this.currentJobId) return;
        try {
            const response = await fetch(`/api/download/csv/${this.currentJobId}`);
            if (!response.ok) return this.showToast("CSV not ready.", "error");
            const blob = await response.blob();
            this.downloadBlob(blob, "rallyclip_segments.csv");
        } catch (err) {
            console.error("CSV download failed:", err);
            this.showToast("CSV download failed.", "error");
        }
    }

    downloadBlob(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        window.URL.revokeObjectURL(url);
    }

    startNewAnalysis() {
        this.stopProgressMonitoring();
        this.removeFile();
        this.resetProgress();
        this.resetControls();
    }

    setStatus(text, tone) {
        this.statusBadgeText.textContent = text;
        this.statusBadge.className = `status-badge ${tone}`;
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
    new RallyClipApp();
});
