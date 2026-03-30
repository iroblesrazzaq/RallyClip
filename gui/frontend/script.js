class RallyClip {
    constructor() {
        this.selectedFile = null;
        this.isProcessing = false;
        this.currentJobId = null;
        this.progressInterval = null;
        this.defaults = {};
        this.warnings = {};
        this.yoloOptions = [];
        this.weights = null;
        this.etaSeconds = null;
        this.poseEtaSeconds = null;
        this.poseFps = null;

        this.initializeElements();
        this.bindEvents();
        this.loadDefaults();
        this.restoreJobIfAny();
    }

    initializeElements() {
        // File upload elements
        this.dropZone = document.getElementById('dropZone');
        this.fileInput = document.getElementById('fileInput');
        this.browseBtn = document.getElementById('browseBtn');
        this.selectedFileDiv = document.getElementById('selectedFile');
        this.fileName = document.getElementById('fileName');
        this.fileSize = document.getElementById('fileSize');
        this.removeFileBtn = document.getElementById('removeFileBtn');

        // Control elements
        this.startBtn = document.getElementById('startBtn');
        this.cancelBtn = document.getElementById('cancelBtn');
        this.advancedToggle = document.getElementById('advancedToggle');
        this.advancedPanel = document.getElementById('advancedPanel');
        this.resetAdvanced = document.getElementById('resetAdvanced');

        // Advanced inputs
        this.outputName = document.getElementById('outputName');
        this.yoloSize = document.getElementById('yoloSize');
        this.yoloDevice = document.getElementById('yoloDevice');
        this.low = document.getElementById('low');
        this.high = document.getElementById('high');
        this.minDurSec = document.getElementById('minDurSec');
        this.lowWarning = document.getElementById('lowWarning');
        this.highWarning = document.getElementById('highWarning');
        this.minDurWarning = document.getElementById('minDurWarning');

        // Progress elements
        this.progressSection = document.getElementById('progressSection');
        this.progressItems = {
            pose: {
                status: document.getElementById('poseStatus'),
                fill: document.getElementById('poseFill'),
                percentage: document.getElementById('posePercentage'),
                item: null
            },
            preprocess: {
                status: document.getElementById('preprocessStatus'),
                fill: document.getElementById('preprocessFill'),
                percentage: document.getElementById('preprocessPercentage'),
                item: null
            },
            feature: {
                status: document.getElementById('featureStatus'),
                fill: document.getElementById('featureFill'),
                percentage: document.getElementById('featurePercentage'),
                item: null
            },
            inference: {
                status: document.getElementById('inferenceStatus'),
                fill: document.getElementById('inferenceFill'),
                percentage: document.getElementById('inferencePercentage'),
                item: null
            },
            output: {
                status: document.getElementById('outputStatus'),
                fill: document.getElementById('outputFill'),
                percentage: document.getElementById('outputPercentage'),
                item: null
            }
        };

        // Set progress item references
        const progressItemsElements = document.querySelectorAll('.progress-item');
        const steps = ['pose', 'preprocess', 'feature', 'inference', 'output'];
        steps.forEach((step, index) => {
            this.progressItems[step].item = progressItemsElements[index];
        });

        // Overall progress
        this.overallFill = document.getElementById('overallFill');
        this.overallPercentage = document.getElementById('overallPercentage');
        this.etaText = document.getElementById('etaText');
        this.etaPill = document.getElementById('etaPill');

        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.downloadVideoBtn = document.getElementById('downloadVideoBtn');
        this.downloadCsvBtn = document.getElementById('downloadCsvBtn');
        this.newAnalysisBtn = document.getElementById('newAnalysisBtn');
    }

    bindEvents() {
        // File upload events
        this.dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        this.dropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.dropZone.addEventListener('drop', this.handleDrop.bind(this));
        this.dropZone.addEventListener('click', () => this.fileInput.click());
        
        this.browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.fileInput.click();
        });
        
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        this.removeFileBtn.addEventListener('click', this.removeFile.bind(this));

        // Control events
        this.startBtn.addEventListener('click', this.startAnalysis.bind(this));
        this.cancelBtn.addEventListener('click', this.cancelAnalysis.bind(this));
        this.advancedToggle.addEventListener('click', this.toggleAdvanced.bind(this));
        this.resetAdvanced.addEventListener('click', this.resetAdvancedSettings.bind(this));

        // Download events
        this.downloadVideoBtn.addEventListener('click', this.downloadVideo.bind(this));
        this.downloadCsvBtn.addEventListener('click', this.downloadCsv.bind(this));
        this.newAnalysisBtn.addEventListener('click', this.startNewAnalysis.bind(this));
    }

    async loadDefaults() {
        try {
            const resp = await fetch('/api/config/defaults');
            if (!resp.ok) throw new Error('Failed to load defaults');
            const payload = await resp.json();
            this.defaults = payload.defaults || {};
            this.warnings = payload.warnings || {};
            this.yoloOptions = payload.yolo_sizes || [];
            this.applyDefaults();
        } catch (err) {
            console.error('Failed to fetch defaults', err);
            this.showError('Could not load defaults; using built-in values.');
            this.defaults = this.defaults || {};
        }
    }

    applyDefaults() {
        if (!this.defaults) return;
        this.populateSelect(this.yoloSize, this.yoloOptions, this.defaults.yolo_size || 'small');
        this.yoloDevice.value = this.defaults.yolo_device || '';
        this.low.value = this.defaults.low ?? 0.45;
        this.high.value = this.defaults.high ?? 0.8;
        this.minDurSec.value = this.defaults.min_dur_sec ?? 0.5;
        this.outputName.value = '';

        // Warnings
        this.lowWarning.textContent = this.warnings.low || '';
        this.highWarning.textContent = this.warnings.high || '';
        this.minDurWarning.textContent = this.warnings.min_dur_sec || '';
    }

    populateSelect(selectEl, options, selected) {
        if (!selectEl) return;
        selectEl.innerHTML = '';
        options.forEach(opt => {
            const o = document.createElement('option');
            o.value = opt;
            o.textContent = opt;
            if (opt === selected) o.selected = true;
            selectEl.appendChild(o);
        });
        if (selected && !options.includes(selected)) {
            const custom = document.createElement('option');
            custom.value = selected;
            custom.textContent = selected;
            custom.selected = true;
            selectEl.appendChild(custom);
        }
    }

    toggleAdvanced() {
        const isHidden = this.advancedPanel.style.display === 'none';
        this.advancedPanel.style.display = isHidden ? 'block' : 'none';
    }

    resetAdvancedSettings() {
        this.applyDefaults();
        this.showInfo('Advanced settings reset to defaults');
    }

    // File handling methods
    handleDragOver(e) {
        e.preventDefault();
        this.dropZone.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        if (!this.dropZone.contains(e.relatedTarget)) {
            this.dropZone.classList.remove('dragover');
        }
    }

    handleDrop(e) {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    processFile(file) {
        // Validate file type - only MP4 is tested and supported
        if (file.type !== 'video/mp4' && !file.name.toLowerCase().endsWith('.mp4')) {
            this.showError('Please select an MP4 video file. Other formats are not currently supported.');
            return;
        }

        // Validate file size (2GB limit)
        const maxSize = 2 * 1024 * 1024 * 1024; // 2GB
        if (file.size > maxSize) {
            this.showError('File size must be less than 2GB.');
            return;
        }

        this.selectedFile = file;
        this.displaySelectedFile();
    }

    displaySelectedFile() {
        this.fileName.textContent = this.selectedFile.name;
        this.fileSize.textContent = this.formatFileSize(this.selectedFile.size);
        
        this.dropZone.style.display = 'none';
        this.selectedFileDiv.style.display = 'block';
        this.startBtn.disabled = false;
    }

    removeFile() {
        this.selectedFile = null;
        this.fileInput.value = '';
        
        this.dropZone.style.display = 'block';
        this.selectedFileDiv.style.display = 'none';
        this.startBtn.disabled = true;
        
        this.resetProgress();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Analysis control methods
    async startAnalysis() {
        if (!this.selectedFile || this.isProcessing) return;

        this.isProcessing = true;
        this.startBtn.disabled = true;
        this.cancelBtn.disabled = false;
        this.progressSection.style.display = 'block';
        this.resultsSection.style.display = 'none';

        try {
            const jobId = await this.uploadFileAndStart();
            this.currentJobId = jobId;
            try { localStorage.setItem('rallyclip_job_id', jobId); } catch (_) {}
            this.startProgressMonitoring();
        } catch (error) {
            console.error('Error starting analysis:', error);
            this.showError('Failed to start analysis. Please try again.');
            this.resetControls();
        }
    }

    buildConfigFromForm() {
        const cfg = { ...(this.defaults || {}) };
        cfg.output_name = this.outputName.value.trim() || null;
        cfg.yolo_size = this.yoloSize.value || cfg.yolo_size;
        cfg.yolo_device = this.yoloDevice.value || null;
        cfg.write_csv = true; // always write CSV for GUI
        cfg.segment_video = true; // always write segmented MP4 in GUI
        cfg.low = parseFloat(this.low.value) || cfg.low;
        cfg.high = parseFloat(this.high.value) || cfg.high;
        cfg.min_dur_sec = parseFloat(this.minDurSec.value) || cfg.min_dur_sec;
        return cfg;
    }

    async uploadFileAndStart() {
        const formData = new FormData();
        formData.append('video', this.selectedFile);
        formData.append('config', JSON.stringify(this.buildConfigFromForm()));

        const response = await fetch('/api/upload-and-start', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const msg = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, ${msg}`);
        }

        const result = await response.json();
        return result.job_id;
    }

    startProgressMonitoring() {
        this.progressInterval = setInterval(async () => {
            try {
                await this.updateProgress();
            } catch (error) {
                console.error('Error updating progress:', error);
                this.stopProgressMonitoring();
            }
        }, 1000); // Update every second
    }

    async restoreJobIfAny() {
        let storedId = null;
        try { storedId = localStorage.getItem('rallyclip_job_id'); } catch (_) {}
        if (!storedId) return;

        this.currentJobId = storedId;
        this.progressSection.style.display = 'block';
        this.startBtn.disabled = true;
        this.cancelBtn.disabled = false;
        this.startProgressMonitoring();
        try {
            await this.updateProgress();
        } catch (e) {
            console.warn('Failed to restore job progress:', e);
        }
    }

    async updateProgress() {
        if (!this.currentJobId) return;

        const response = await fetch(`/api/progress/${this.currentJobId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const progress = await response.json();
        if (progress.weights) {
            this.weights = progress.weights;
        }
        this.etaSeconds = progress.eta_seconds ?? null;
        this.poseEtaSeconds = progress.pose_eta_seconds ?? null;
        this.poseFps = progress.pose_throughput_fps ?? null;
        this.displayProgress(progress);

        if (progress.status === 'completed') {
            this.onAnalysisComplete();
        } else if (progress.status === 'failed') {
            this.onAnalysisError(progress.error || 'Analysis failed');
        } else if (progress.status === 'cancelled') {
            this.onAnalysisCancelled();
        }
    }

    displayProgress(progress) {
        const steps = ['pose', 'preprocess', 'feature', 'inference', 'output'];
        let totalProgress = 0;
        const weights = this.weights || progress.weights || null;
        let weightedSum = 0;
        let weightTotal = 0;

        steps.forEach((step) => {
            const stepProgress = progress.steps[step] || { status: 'waiting', progress: 0 };
            const elements = this.progressItems[step];

            elements.status.textContent = this.formatStatus(stepProgress.status);
            elements.status.className = `progress-status ${stepProgress.status}`;

            elements.fill.style.width = `${stepProgress.progress}%`;
            elements.percentage.textContent = `${Math.round(stepProgress.progress)}%`;

            elements.item.classList.remove('active', 'completed');
            if (stepProgress.status === 'in_progress') {
                elements.item.classList.add('active');
            } else if (stepProgress.status === 'completed') {
                elements.item.classList.add('completed');
            }

            totalProgress += stepProgress.progress;
            if (weights && typeof weights[step] === 'number') {
                weightedSum += stepProgress.progress * weights[step];
                weightTotal += weights[step];
            }
        });

        const overallProgress = (weightTotal > 0)
            ? (weightedSum / weightTotal)
            : (totalProgress / steps.length);
        this.overallFill.style.width = `${overallProgress}%`;
        this.overallPercentage.textContent = `${Math.round(overallProgress)}%`;
        this.updateEta(progress);
    }

    updateEta(progress) {
        if (!this.etaText) return;
        const eta = (progress.eta_seconds !== null && progress.eta_seconds !== undefined)
            ? progress.eta_seconds
            : (this.weights && this.weights.eta_seconds ? this.weights.eta_seconds : null);
        if (eta === null || eta === undefined) {
            this.etaText.textContent = 'Est. remaining: --';
            if (this.etaPill) this.etaPill.style.display = 'none';
            return;
        }
        const remaining = Math.max(0, Math.round(eta));
        const minutes = Math.floor(remaining / 60);
        const seconds = remaining % 60;
        if (minutes > 0) {
            this.etaText.textContent = `Est. remaining: ~${minutes}m ${seconds.toString().padStart(2, '0')}s`;
        } else {
            this.etaText.textContent = `Est. remaining: ~${seconds}s`;
        }
        if (this.etaPill) {
            const pretty = minutes > 0 ? `${minutes}m ${seconds.toString().padStart(2, '0')}s` : `${seconds}s`;
            this.etaPill.textContent = `ETA ~${pretty}`;
            this.etaPill.style.display = 'inline-flex';
        }
    }

    formatStatus(status) {
        const statusMap = {
            'waiting': 'Waiting...',
            'in_progress': 'In Progress...',
            'completed': 'Completed',
            'failed': 'Failed'
        };
        return statusMap[status] || status;
    }

    async cancelAnalysis() {
        if (!this.currentJobId || !this.isProcessing) return;

        try {
            const response = await fetch(`/api/cancel/${this.currentJobId}`, {
                method: 'POST'
            });

            if (response.ok) {
                this.onAnalysisCancelled();
            } else {
                throw new Error('Failed to cancel analysis');
            }
        } catch (error) {
            console.error('Error cancelling analysis:', error);
            this.showError('Failed to cancel analysis');
        }
    }

    onAnalysisComplete() {
        this.stopProgressMonitoring();
        this.isProcessing = false;
        this.cancelBtn.disabled = true;
        this.resultsSection.style.display = 'block';
        
        this.showSuccess('Analysis completed successfully!');
        if (this.etaText) {
            this.etaText.textContent = 'Est. remaining: 0s';
        }
        try { localStorage.removeItem('rallyclip_job_id'); } catch (_) {}
    }

    onAnalysisError(error) {
        this.stopProgressMonitoring();
        this.resetControls();
        this.showError(`Analysis failed: ${error}`);
        try { localStorage.removeItem('rallyclip_job_id'); } catch (_) {}
    }

    onAnalysisCancelled() {
        this.stopProgressMonitoring();
        this.resetControls();
        this.resetProgress();
        this.showInfo('Analysis cancelled');
        try { localStorage.removeItem('rallyclip_job_id'); } catch (_) {}
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
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'none';

        Object.values(this.progressItems).forEach(item => {
            item.status.textContent = 'Waiting...';
            item.status.className = 'progress-status';
            item.fill.style.width = '0%';
            item.percentage.textContent = '0%';
            item.item.classList.remove('active', 'completed');
        });

        this.overallFill.style.width = '0%';
        this.overallPercentage.textContent = '0%';
        if (this.etaText) {
            this.etaText.textContent = 'Est. remaining: --';
        }
    }

    // Download methods
    async downloadVideo() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/api/download/video/${this.currentJobId}`);
            if (response.ok) {
                const blob = await response.blob();
                this.downloadBlob(blob, 'rallyclip_analysis_video.mp4');
            } else {
                throw new Error('Failed to download video');
            }
        } catch (error) {
            console.error('Error downloading video:', error);
            this.showError('Failed to download video');
        }
    }

    async downloadCsv() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/api/download/csv/${this.currentJobId}`);
            if (response.ok) {
                const blob = await response.blob();
                this.downloadBlob(blob, 'rallyclip_analysis_data.csv');
            } else {
                throw new Error('Failed to download CSV');
            }
        } catch (error) {
            console.error('Error downloading CSV:', error);
            this.showError('Failed to download CSV');
        }
    }

    downloadBlob(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    startNewAnalysis() {
        this.removeFile();
        this.resetProgress();
        this.resetControls();
    }

    // Utility methods
    showError(message) {
        this.showNotification(message, 'error');
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showInfo(message) {
        this.showNotification(message, 'info');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;

        document.body.appendChild(notification);
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 50);

        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new RallyClip();
});
