// API base URL
const API_BASE_URL = 'http://localhost:8000';

// DOM elements
const frameUploadArea = document.getElementById('frameUploadArea');
const frame1Input = document.getElementById('frame1Input');
const frame2Input = document.getElementById('frame2Input');
const framePreview = document.getElementById('framePreview');
const frame1Preview = document.getElementById('frame1Preview');
const frame2Preview = document.getElementById('frame2Preview');
const interpolateBtn = document.getElementById('interpolateBtn');

const videoUploadArea = document.getElementById('videoUploadArea');
const videoInput = document.getElementById('videoInput');
const videoPreview = document.getElementById('videoPreview');
const videoPreviewElement = document.getElementById('videoPreviewElement');
const interpolationFactor = document.getElementById('interpolationFactor');
const interpolateVideoBtn = document.getElementById('interpolateVideoBtn');

const loading = document.getElementById('loading');
const resultContainer = document.getElementById('resultContainer');
const resultImage = document.getElementById('resultImage');
const metrics = document.getElementById('metrics');
const ssimValue = document.getElementById('ssimValue');
const psnrValue = document.getElementById('psnrValue');
const downloadBtn = document.getElementById('downloadBtn');

// State variables
let frame1File = null;
let frame2File = null;
let videoFile = null;
let currentResult = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    checkAPIHealth();
});

function setupEventListeners() {
    // Frame interpolation events
    frameUploadArea.addEventListener('click', () => {
        frame1Input.click();
    });
    
    frame1Input.addEventListener('change', (e) => handleFrame1Selection(e));
    frame2Input.addEventListener('change', (e) => handleFrame2Selection(e));
    
    interpolateBtn.addEventListener('click', handleFrameInterpolation);
    
    // Video interpolation events
    videoUploadArea.addEventListener('click', () => {
        videoInput.click();
    });
    
    videoInput.addEventListener('change', (e) => handleVideoSelection(e));
    interpolateVideoBtn.addEventListener('click', handleVideoInterpolation);
    
    // Download button
    downloadBtn.addEventListener('click', handleDownload);
    
    // Drag and drop events
    setupDragAndDrop();
}

function setupDragAndDrop() {
    // Frame upload area drag and drop
    frameUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        frameUploadArea.classList.add('dragover');
    });
    
    frameUploadArea.addEventListener('dragleave', () => {
        frameUploadArea.classList.remove('dragover');
    });
    
    frameUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        frameUploadArea.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        if (files.length >= 2) {
            handleDroppedFrames(files[0], files[1]);
        }
    });
    
    // Video upload area drag and drop
    videoUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        videoUploadArea.classList.add('dragover');
    });
    
    videoUploadArea.addEventListener('dragleave', () => {
        videoUploadArea.classList.remove('dragover');
    });
    
    videoUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        videoUploadArea.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0 && files[0].type.startsWith('video/')) {
            handleDroppedVideo(files[0]);
        }
    });
}

function handleDroppedFrames(file1, file2) {
    if (file1.type.startsWith('image/') && file2.type.startsWith('image/')) {
        frame1File = file1;
        frame2File = file2;
        updateFramePreviews();
        updateInterpolateButton();
    }
}

function handleDroppedVideo(file) {
    if (file.type.startsWith('video/')) {
        videoFile = file;
        updateVideoPreview();
        updateVideoInterpolateButton();
    }
}

function handleFrame1Selection(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        frame1File = file;
        updateFramePreviews();
        updateInterpolateButton();
    }
}

function handleFrame2Selection(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        frame2File = file;
        updateFramePreviews();
        updateInterpolateButton();
    }
}

function handleVideoSelection(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('video/')) {
        videoFile = file;
        updateVideoPreview();
        updateVideoInterpolateButton();
    }
}

function updateFramePreviews() {
    if (frame1File && frame2File) {
        const reader1 = new FileReader();
        const reader2 = new FileReader();
        
        reader1.onload = (e) => {
            frame1Preview.src = e.target.result;
        };
        
        reader2.onload = (e) => {
            frame2Preview.src = e.target.result;
        };
        
        reader1.readAsDataURL(frame1File);
        reader2.readAsDataURL(frame2File);
        
        framePreview.style.display = 'grid';
    }
}

function updateVideoPreview() {
    if (videoFile) {
        const url = URL.createObjectURL(videoFile);
        videoPreviewElement.src = url;
        videoPreview.style.display = 'grid';
    }
}

function updateInterpolateButton() {
    interpolateBtn.disabled = !(frame1File && frame2File);
}

function updateVideoInterpolateButton() {
    interpolateVideoBtn.disabled = !videoFile;
}

async function handleFrameInterpolation() {
    if (!frame1File || !frame2File) return;
    
    showLoading(true);
    hideResult();
    
    try {
        const formData = new FormData();
        formData.append('frame1', frame1File);
        formData.append('frame2', frame2File);
        
        const response = await fetch(`${API_BASE_URL}/interpolate-frames`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        currentResult = {
            type: 'frame',
            url: url,
            filename: 'interpolated_frame.jpg'
        };
        
        showResult(url);
        showSuccess('Frame interpolation completed successfully!');
        
    } catch (error) {
        console.error('Error during frame interpolation:', error);
        showError('Error during frame interpolation: ' + error.message);
    } finally {
        showLoading(false);
    }
}

async function handleVideoInterpolation() {
    if (!videoFile) return;
    
    showLoading(true);
    hideResult();
    
    try {
        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('interpolation_factor', interpolationFactor.value);
        
        const response = await fetch(`${API_BASE_URL}/interpolate-video`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        currentResult = {
            type: 'video',
            url: url,
            filename: 'interpolated_video.mp4'
        };
        
        showResult(url);
        showSuccess('Video interpolation completed successfully!');
        
    } catch (error) {
        console.error('Error during video interpolation:', error);
        showError('Error during video interpolation: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function showResult(url) {
    if (currentResult.type === 'frame') {
        resultImage.src = url;
        resultImage.style.display = 'block';
        metrics.style.display = 'none';
    } else {
        // For video results, show a download link instead of image
        resultImage.style.display = 'none';
        resultImage.innerHTML = `
            <div style="padding: 20px; background: #f8f9ff; border-radius: 15px;">
                <i class="fas fa-video" style="font-size: 3rem; color: #667eea; margin-bottom: 15px;"></i>
                <p style="color: #666; margin-bottom: 15px;">Video interpolation completed!</p>
                <p style="color: #888; font-size: 0.9rem;">Click download to get your interpolated video</p>
            </div>
        `;
    }
    
    resultContainer.style.display = 'block';
    resultContainer.scrollIntoView({ behavior: 'smooth' });
}

function hideResult() {
    resultContainer.style.display = 'none';
    currentResult = null;
}

function showLoading(show) {
    loading.style.display = show ? 'block' : 'none';
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
    
    // Remove any existing error messages
    const existingErrors = document.querySelectorAll('.error');
    existingErrors.forEach(err => err.remove());
    
    // Add new error message
    document.querySelector('.container').appendChild(errorDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success';
    successDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
    
    // Remove any existing success messages
    const existingSuccess = document.querySelectorAll('.success');
    existingSuccess.forEach(succ => succ.remove());
    
    // Add new success message
    document.querySelector('.container').appendChild(successDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        successDiv.remove();
    }, 5000);
}

function handleDownload() {
    if (!currentResult) return;
    
    const link = document.createElement('a');
    link.href = currentResult.url;
    link.download = currentResult.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            if (data.interpolator_ready) {
                console.log('API is healthy and ready');
            } else {
                console.warn('API is running but interpolator is not ready');
                showError('Warning: The AI model is not loaded. Some features may not work.');
            }
        } else {
            console.error('API health check failed');
            showError('Warning: Cannot connect to the API server. Please ensure the backend is running.');
        }
    } catch (error) {
        console.error('API health check error:', error);
        showError('Warning: Cannot connect to the API server. Please ensure the backend is running.');
    }
}

// Utility function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
