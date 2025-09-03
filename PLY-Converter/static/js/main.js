class PLYConverter {
    constructor() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.browseBtn = document.getElementById('browseBtn');
        this.convertBtn = document.getElementById('convertBtn');
        this.progressCard = document.getElementById('progressCard');
        this.resultsCard = document.getElementById('resultsCard');
        this.errorAlert = document.getElementById('errorAlert');
        this.newConversionBtn = document.getElementById('newConversionBtn');
        
        this.currentFile = null;
        this.conversionId = null;
        this.progressInterval = null;
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // File input events
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.browseBtn.addEventListener('click', () => this.fileInput.click());
        this.convertBtn.addEventListener('click', () => this.startConversion());
        this.newConversionBtn.addEventListener('click', () => this.resetForm());
        
        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        
        // Format checkbox events
        const formatCheckboxes = document.querySelectorAll('input[type="checkbox"][value]');
        formatCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => this.updateConvertButton());
        });
    }
    
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.setSelectedFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        if (e.target.files.length > 0) {
            this.setSelectedFile(e.target.files[0]);
        }
    }
    
    setSelectedFile(file) {
        // Validate file type
        if (!file.name.toLowerCase().endsWith('.ply')) {
            this.showError('Please select a PLY file (.ply extension)');
            return;
        }
        
        // Check file size (500MB limit)
        const maxSize = 500 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File size too large. Maximum size is 500MB.');
            return;
        }
        
        this.currentFile = file;
        this.updateUploadArea();
        this.updateConvertButton();
        this.hideError();
    }
    
    updateUploadArea() {
        if (this.currentFile) {
            this.uploadArea.classList.add('file-selected');
            this.uploadArea.innerHTML = `
                <div class="upload-content">
                    <i class="fas fa-file-alt upload-icon mb-3"></i>
                    <h5>File Selected</h5>
                    <p class="text-muted mb-2">${this.currentFile.name}</p>
                    <p class="small text-muted">${this.formatFileSize(this.currentFile.size)}</p>
                    <button type="button" class="btn btn-outline-secondary btn-sm mt-2">
                        <i class="fas fa-times me-1"></i>Change File
                    </button>
                </div>
            `;
        }
    }
    
    updateConvertButton() {
        const hasFile = this.currentFile !== null;
        const hasFormats = this.getSelectedFormats().length > 0;
        
        this.convertBtn.disabled = !hasFile || !hasFormats;
        
        if (!hasFormats && hasFile) {
            this.convertBtn.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Select Output Format(s)';
        } else if (hasFile && hasFormats) {
            this.convertBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Convert PLY File';
        } else {
            this.convertBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Convert PLY File';
        }
    }
    
    getSelectedFormats() {
        const checkboxes = document.querySelectorAll('input[type="checkbox"][value]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }
    
    async startConversion() {
        if (!this.currentFile) {
            this.showError('Please select a PLY file first');
            return;
        }
        
        const formats = this.getSelectedFormats();
        if (formats.length === 0) {
            this.showError('Please select at least one output format');
            return;
        }
        
        this.hideError();
        this.showProgress();
        
        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('file', this.currentFile);
            formats.forEach(format => formData.append('formats', format));
            
            // Upload and start conversion
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Upload failed');
            }
            
            const result = await response.json();
            this.conversionId = result.conversion_id;
            
            // Update file info
            document.getElementById('fileInfo').innerHTML = `
                <strong>File:</strong> ${result.input_file}<br>
                <strong>Output Formats:</strong> ${result.output_formats.join(', ').toUpperCase()}<br>
                <strong>Conversion ID:</strong> ${result.conversion_id}
            `;
            
            // Start progress polling
            this.startProgressPolling();
            
        } catch (error) {
            this.hideProgress();
            this.showError(`Conversion failed: ${error.message}`);
        }
    }
    
    startProgressPolling() {
        this.progressInterval = setInterval(() => {
            this.updateProgress();
        }, 1000);
    }
    
    async updateProgress() {
        if (!this.conversionId) return;
        
        try {
            const response = await fetch(`/progress/${this.conversionId}`);
            if (!response.ok) {
                throw new Error('Failed to get progress');
            }
            
            const progress = await response.json();
            
            // Update progress bar
            const progressBar = document.getElementById('progressBar');
            const progressPercent = document.getElementById('progressPercent');
            const progressMessage = document.getElementById('progressMessage');
            
            progressBar.style.width = `${progress.progress || 0}%`;
            progressBar.setAttribute('aria-valuenow', progress.progress || 0);
            progressPercent.textContent = `${progress.progress || 0}%`;
            progressMessage.textContent = progress.message || 'Processing...';
            
            // Check if completed
            if (progress.status === 'completed') {
                clearInterval(this.progressInterval);
                this.showResults(progress);
            } else if (progress.status === 'error') {
                clearInterval(this.progressInterval);
                this.hideProgress();
                this.showError(progress.message || 'Conversion failed');
            }
            
        } catch (error) {
            clearInterval(this.progressInterval);
            this.hideProgress();
            this.showError(`Progress update failed: ${error.message}`);
        }
    }
    
    showResults(progress) {
        this.hideProgress();
        
        const downloadLinks = document.getElementById('downloadLinks');
        downloadLinks.innerHTML = '';
        
        if (progress.download_links) {
            Object.entries(progress.download_links).forEach(([format, url]) => {
                const link = document.createElement('a');
                link.href = url;
                link.className = 'download-link';
                link.innerHTML = `
                    <i class="fas fa-download"></i>
                    Download ${format.toUpperCase()} File
                `;
                downloadLinks.appendChild(link);
            });
        }
        
        this.resultsCard.style.display = 'block';
        this.resultsCard.scrollIntoView({ behavior: 'smooth' });
    }
    
    showProgress() {
        this.progressCard.style.display = 'block';
        this.resultsCard.style.display = 'none';
        
        // Reset progress
        const progressBar = document.getElementById('progressBar');
        const progressPercent = document.getElementById('progressPercent');
        const progressMessage = document.getElementById('progressMessage');
        
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        progressPercent.textContent = '0%';
        progressMessage.textContent = 'Initializing...';
        
        this.progressCard.scrollIntoView({ behavior: 'smooth' });
    }
    
    hideProgress() {
        this.progressCard.style.display = 'none';
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }
    
    showError(message) {
        this.errorAlert.style.display = 'block';
        document.getElementById('errorMessage').textContent = message;
        this.errorAlert.scrollIntoView({ behavior: 'smooth' });
    }
    
    hideError() {
        this.errorAlert.style.display = 'none';
    }
    
    async resetForm() {
        // Cleanup previous conversion
        if (this.conversionId) {
            try {
                await fetch(`/cleanup/${this.conversionId}`, { method: 'POST' });
            } catch (error) {
                console.warn('Cleanup failed:', error);
            }
        }
        
        // Reset form state
        this.currentFile = null;
        this.conversionId = null;
        
        // Reset UI
        this.uploadArea.classList.remove('file-selected');
        this.uploadArea.innerHTML = `
            <div class="upload-content">
                <i class="fas fa-cloud-upload-alt upload-icon mb-3"></i>
                <h5>Drag & Drop PLY File Here</h5>
                <p class="text-muted mb-3">or click to browse files</p>
                <button type="button" class="btn btn-outline-primary" id="browseBtn">
                    <i class="fas fa-folder-open me-2"></i>Browse Files
                </button>
            </div>
        `;
        
        // Re-attach event listener for new browse button
        document.getElementById('browseBtn').addEventListener('click', () => this.fileInput.click());
        
        this.fileInput.value = '';
        this.hideProgress();
        this.hideError();
        this.resultsCard.style.display = 'none';
        this.updateConvertButton();
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the converter when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new PLYConverter();
});
