// Global variables
let frame1File = null;
let frame2File = null;
let currentVideoBlob = null;

// API configuration
const API_BASE_URL = 'http://localhost:8000'; // Change this if your API runs on a different port

// DOM elements
const uploadBox1 = document.getElementById('upload-box-1');
const uploadBox2 = document.getElementById('upload-box-2');
const fileInput1 = document.getElementById('file-input-1');
const fileInput2 = document.getElementById('file-input-2');
const generateBtn = document.getElementById('generate-btn');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('result-section');
const resultVideo = document.getElementById('result-video');
const downloadBtn = document.getElementById('download-btn');
const errorMessage = document.getElementById('error-message');
const successMessage = document.getElementById('success-message');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupFileInputs();
    setupDragAndDrop();
    updateGenerateButton();
});

// Setup file input event listeners
function setupFileInputs() {
    fileInput1.addEventListener('change', (e) => handleFileSelect(e, 1));
    fileInput2.addEventListener('change', (e) => handleFileSelect(e, 2));
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    [uploadBox1, uploadBox2].forEach((box, index) => {
        box.addEventListener('dragover', (e) => {
            e.preventDefault();
            box.classList.add('dragover');
        });

        box.addEventListener('dragleave', (e) => {
            e.preventDefault();
            box.classList.remove('dragover');
        });

        box.addEventListener('drop', (e) => {
            e.preventDefault();
            box.classList.remove('dragover');
            handleFileDrop(e, index + 1);
        });
    });
}

// Handle file selection from file input
function handleFileSelect(event, frameNumber) {
    const file = event.target.files[0];
    if (file && validateImageFile(file)) {
        setFrameFile(file, frameNumber);
    } else {
        showError(`Invalid file selected for Frame ${frameNumber}. Please select a valid image file.`);
    }
}

// Handle file drop from drag and drop
function handleFileDrop(event, frameNumber) {
    const file = event.dataTransfer.files[0];
    if (file && validateImageFile(file)) {
        setFrameFile(file, frameNumber);
        // Update the file input to reflect the dropped file
        const fileInput = frameNumber === 1 ? fileInput1 : fileInput2;
        fileInput.files = event.dataTransfer.files;
    } else {
        showError(`Invalid file dropped for Frame ${frameNumber}. Please drop a valid image file.`);
    }
}

// Validate image file
function validateImageFile(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'image/tif'];
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (!allowedTypes.includes(file.type)) {
        return false;
    }

    if (file.size > maxSize) {
        return false;
    }

    return true;
}

// Set frame file and update UI
function setFrameFile(file, frameNumber) {
    if (frameNumber === 1) {
        frame1File = file;
        updateUploadBox(uploadBox1, file);
    } else {
        frame2File = file;
        updateUploadBox(uploadBox2, file);
    }
    
    updateGenerateButton();
    hideMessages();
}

// Update upload box appearance
function updateUploadBox(box, file) {
    box.classList.add('has-file');
    const uploadText = box.querySelector('.upload-text');
    const fileInfo = box.querySelector('.file-info');
    
    uploadText.textContent = file.name;
    fileInfo.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
}

// Update generate button state
function updateGenerateButton() {
    generateBtn.disabled = !(frame1File && frame2File);
}

// Generate video function
async function generateVideo() {
    if (!frame1File || !frame2File) {
        showError('Please select both frame files before generating video.');
        return;
    }

    // Get parameters
    const numIntermediate = parseInt(document.getElementById('num-intermediate').value);
    const fps = parseInt(document.getElementById('fps').value);

    // Show loading state
    setLoading(true);
    hideMessages();
    hideResult();

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('frame1', frame1File);
        formData.append('frame2', frame2File);
        formData.append('num_intermediate', numIntermediate);
        formData.append('fps', fps);

        // Make API call
        const response = await fetch(`${API_BASE_URL}/interpolate`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        // Get video blob
        const videoBlob = await response.blob();
        currentVideoBlob = videoBlob;

        // Display video
        displayVideo(videoBlob);
        showSuccess(`Successfully generated video with ${numIntermediate} intermediate frames at ${fps} FPS!`);

    } catch (error) {
        console.error('Error generating video:', error);
        showError(`Failed to generate video: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

// Display video result
function displayVideo(videoBlob) {
    const videoUrl = URL.createObjectURL(videoBlob);
    
    resultVideo.src = videoUrl;
    resultVideo.load();
    
    // Update download button
    downloadBtn.href = videoUrl;
    downloadBtn.download = `interpolated_video_${Date.now()}.mp4`;
    
    // Show result section
    resultSection.style.display = 'block';
    
    // Scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// Set loading state
function setLoading(isLoading) {
    loading.style.display = isLoading ? 'block' : 'none';
    generateBtn.disabled = isLoading;
    
    if (isLoading) {
        generateBtn.textContent = '⏳ Processing...';
    } else {
        generateBtn.textContent = '🎬 Generate Video';
    }
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    successMessage.style.display = 'none';
}

// Show success message
function showSuccess(message) {
    successMessage.textContent = message;
    successMessage.style.display = 'block';
    errorMessage.style.display = 'none';
}

// Hide all messages
function hideMessages() {
    errorMessage.style.display = 'none';
    successMessage.style.display = 'none';
}

// Hide result section
function hideResult() {
    resultSection.style.display = 'none';
}

// Reset application state
function resetApp() {
    frame1File = null;
    frame2File = null;
    currentVideoBlob = null;
    
    // Reset file inputs
    fileInput1.value = '';
    fileInput2.value = '';
    
    // Reset upload boxes
    uploadBox1.classList.remove('has-file');
    uploadBox2.classList.remove('has-file');
    uploadBox1.querySelector('.upload-text').textContent = 'Click to upload Frame 1';
    uploadBox1.querySelector('.file-info').textContent = 'or drag & drop';
    uploadBox2.querySelector('.upload-text').textContent = 'Click to upload Frame 2';
    uploadBox2.querySelector('.file-info').textContent = 'or drag & drop';
    
    // Reset controls
    document.getElementById('num-intermediate').value = '3';
    document.getElementById('fps').value = '30';
    
    // Hide results and messages
    hideResult();
    hideMessages();
    
    // Update button state
    updateGenerateButton();
}

// Add reset button functionality (optional)
function addResetButton() {
    const resetBtn = document.createElement('button');
    resetBtn.textContent = '🔄 Reset';
    resetBtn.className = 'download-btn';
    resetBtn.style.background = '#ff9800';
    resetBtn.onclick = resetApp;
    
    const downloadSection = document.querySelector('.download-section');
    downloadSection.appendChild(resetBtn);
}

// Initialize reset button
document.addEventListener('DOMContentLoaded', function() {
    addResetButton();
});

// Handle video errors
resultVideo.addEventListener('error', function() {
    showError('Error loading video. Please try again.');
});

// Handle video load success
resultVideo.addEventListener('loadeddata', function() {
    console.log('Video loaded successfully');
});

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to generate video
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (!generateBtn.disabled) {
            generateVideo();
        }
    }
    
    // Escape to hide messages
    if (e.key === 'Escape') {
        hideMessages();
    }
});

// Add file size validation feedback
function validateFileSize(file) {
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        showError(`File ${file.name} is too large. Maximum size is 50MB.`);
        return false;
    }
    return true;
}

// Add file type validation feedback
function validateFileType(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'image/tif'];
    if (!allowedTypes.includes(file.type)) {
        showError(`File ${file.name} is not a supported image type. Please use JPG, PNG, BMP, or TIFF.`);
        return false;
    }
    return true;
}

// Enhanced file validation
function validateImageFile(file) {
    if (!validateFileType(file)) {
        return false;
    }
    
    if (!validateFileSize(file)) {
        return false;
    }
    
    return true;
}
