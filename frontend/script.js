class DeepMatch {
    constructor() {
        this.selectedFile = null;
        this.isProcessing = false;
        this.currentJobId = null;
        this.progressInterval = null;
        
        this.initializeElements();
        this.bindEvents();
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

        // Download events
        this.downloadVideoBtn.addEventListener('click', this.downloadVideo.bind(this));
        this.downloadCsvBtn.addEventListener('click', this.downloadCsv.bind(this));
        this.newAnalysisBtn.addEventListener('click', this.startNewAnalysis.bind(this));
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
            // Upload file and start processing
            const jobId = await this.uploadFileAndStart();
            this.currentJobId = jobId;
            try { localStorage.setItem('deepmatch_job_id', jobId); } catch (_) {}
            
            // Start progress monitoring
            this.startProgressMonitoring();
            
        } catch (error) {
            console.error('Error starting analysis:', error);
            this.showError('Failed to start analysis. Please try again.');
            this.resetControls();
        }
    }

    async uploadFileAndStart() {
        const formData = new FormData();
        formData.append('video', this.selectedFile);

        const response = await fetch('/api/upload-and-start', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
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
        // On reload, try to restore an existing job
        let storedId = null;
        try { storedId = localStorage.getItem('deepmatch_job_id'); } catch (_) {}
        if (!storedId) return;

        this.currentJobId = storedId;
        this.progressSection.style.display = 'block';
        this.startBtn.disabled = true;
        this.cancelBtn.disabled = false;
        // We don't know selected file anymore; keep UI minimal
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
        this.displayProgress(progress);

        // Check if completed
        if (progress.status === 'completed') {
            this.onAnalysisComplete();
        } else if (progress.status === 'failed') {
            this.onAnalysisError(progress.error || 'Analysis failed');
        }
    }

    displayProgress(progress) {
        const steps = ['pose', 'preprocess', 'feature', 'inference', 'output'];
        let totalProgress = 0;
        let completedSteps = 0;

        steps.forEach((step, index) => {
            const stepProgress = progress.steps[step] || { status: 'waiting', progress: 0 };
            const elements = this.progressItems[step];

            // Update status
            elements.status.textContent = this.formatStatus(stepProgress.status);
            elements.status.className = `progress-status ${stepProgress.status}`;

            // Update progress bar
            elements.fill.style.width = `${stepProgress.progress}%`;
            elements.percentage.textContent = `${Math.round(stepProgress.progress)}%`;

            // Update item styling
            elements.item.classList.remove('active', 'completed');
            if (stepProgress.status === 'in_progress') {
                elements.item.classList.add('active');
            } else if (stepProgress.status === 'completed') {
                elements.item.classList.add('completed');
                completedSteps++;
            }

            totalProgress += stepProgress.progress;
        });

        // Update overall progress
        const overallProgress = totalProgress / steps.length;
        this.overallFill.style.width = `${overallProgress}%`;
        this.overallPercentage.textContent = `${Math.round(overallProgress)}%`;
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
        try { localStorage.removeItem('deepmatch_job_id'); } catch (_) {}
    }

    onAnalysisError(error) {
        this.stopProgressMonitoring();
        this.resetControls();
        this.showError(`Analysis failed: ${error}`);
        try { localStorage.removeItem('deepmatch_job_id'); } catch (_) {}
    }

    onAnalysisCancelled() {
        this.stopProgressMonitoring();
        this.resetControls();
        this.resetProgress();
        this.showInfo('Analysis cancelled');
        try { localStorage.removeItem('deepmatch_job_id'); } catch (_) {}
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

        // Reset all progress items
        Object.values(this.progressItems).forEach(item => {
            item.status.textContent = 'Waiting...';
            item.status.className = 'progress-status';
            item.fill.style.width = '0%';
            item.percentage.textContent = '0%';
            item.item.classList.remove('active', 'completed');
        });

        // Reset overall progress
        this.overallFill.style.width = '0%';
        this.overallPercentage.textContent = '0%';
    }

    // Download methods
    async downloadVideo() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/api/download/video/${this.currentJobId}`);
            if (response.ok) {
                const blob = await response.blob();
                this.downloadBlob(blob, 'deepmatch_analysis_video.mp4');
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
                this.downloadBlob(blob, 'deepmatch_analysis_data.csv');
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
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;

        // Add notification styles
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '15px 20px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: '600',
            zIndex: '10000',
            maxWidth: '400px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            transform: 'translateX(100%)',
            transition: 'transform 0.3s ease'
        });

        // Set background color based on type
        const colors = {
            error: '#e53e3e',
            success: '#48bb78',
            info: '#667eea'
        };
        notification.style.backgroundColor = colors[type] || colors.info;

        // Add to DOM and animate in
        document.body.appendChild(notification);
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Remove after 5 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DeepMatch();
});
