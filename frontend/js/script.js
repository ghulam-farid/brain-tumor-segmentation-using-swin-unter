const API_BASE = 'http://localhost:5000/api';
let currentSessionId = null;

// DOM Elements
const fileInputs = {
    t1: document.getElementById('t1'),
    t1c: document.getElementById('t1c'),
    t2: document.getElementById('t2'),
    flair: document.getElementById('flair')
};

const statusIndicators = {
    t1: document.getElementById('t1-status'),
    t1c: document.getElementById('t1c-status'),
    t2: document.getElementById('t2-status'),
    flair: document.getElementById('flair-status')
};

const submitBtn = document.getElementById('submit-btn');
const resetBtn = document.getElementById('reset-btn');
const errorResetBtn = document.getElementById('error-reset-btn');
const downloadBtn = document.getElementById('download-btn');

// Event Listeners
Object.entries(fileInputs).forEach(([modality, input]) => {
    input.addEventListener('change', (e) => handleFileSelect(modality, e));

    // Make entire box clickable
    document.querySelector(`label[for="${modality}"]`).parentElement.addEventListener('click', () => {
        input.click();
    });
});

submitBtn.addEventListener('click', submitFiles);
resetBtn.addEventListener('click', resetUI);
errorResetBtn.addEventListener('click', resetUI);
downloadBtn.addEventListener('click', downloadResults);

// File Selection Handler
function handleFileSelect(modality, event) {
    const file = event.target.files[0];
    const indicator = statusIndicators[modality];

    if (!file) return;

    if (file.name.endsWith('.nii.gz') || file.name.endsWith('.nii')) {
        indicator.textContent = `✓ ${file.name}`;
        indicator.style.color = '#27ae60';
        checkAllFilesSelected();
    } else {
        indicator.textContent = '✗ Invalid format (use .nii or .nii.gz)';
        indicator.classList.add('error');
        checkAllFilesSelected();
    }
}

// Check if all files are selected
function checkAllFilesSelected() {
    const allSelected = Object.values(fileInputs).every(input => input.files.length > 0);
    submitBtn.disabled = !allSelected;
}

// Submit Files
async function submitFiles() {
    try {
        showSection('processing-section');
        updateStatus('Uploading files...');
        updateProgress(10);

        // Create FormData
        const formData = new FormData();

        Object.entries(fileInputs).forEach(([modality, input]) => {
            formData.append(modality, input.files[0]);
        });

        // Upload
        updateStatus('Preprocessing...');
        updateProgress(30);

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Upload failed');
        }

        const data = await response.json();
        currentSessionId = data.session_id;

        updateProgress(70);
        updateStatus('Running segmentation...');

        // Wait a moment then fetch results
        await new Promise(resolve => setTimeout(resolve, 2000));

        updateProgress(90);
        updateStatus('Retrieving results...');

        await displayResults(data);

        updateProgress(100);
        showSection('results-section');

    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    }
}

// Display Results
async function displayResults(data) {
    // Display statistics
    const statsHtml = Object.entries(data.tumor_statistics).map(([label, stats]) => `
        <div class="stat-category">
            <h4>${label}</h4>
            <div class="stat-row">
                <span class="stat-label">Voxel Count:</span>
                <span class="stat-value">${stats.voxel_count.toLocaleString()}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Volume (mm³):</span>
                <span class="stat-value">${stats.volume_mm3.toFixed(2)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Volume (cm³):</span>
                <span class="stat-value">${stats.volume_cm3.toFixed(3)}</span>
            </div>
        </div>
    `).join('');

    document.getElementById('statistics').innerHTML = statsHtml;

    // Display preview image
    const previewUrl = `${API_BASE.replace('/api', '')}/predictions/${currentSessionId}/preview.png`;
    document.getElementById('preview-image').src = previewUrl;
}

// Download Results
function downloadResults() {
    const downloadUrl = `${API_BASE.replace('/api', '')}/predictions/${currentSessionId}/segmentation_mask.nii.gz`;
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `segmentation_${currentSessionId}.nii.gz`;
    link.click();
}

// UI Helpers
function showSection(sectionId) {
    document.querySelectorAll('section').forEach(s => s.style.display = 'none');
    document.querySelector(`.${sectionId}`).style.display = 'block';
}

function updateStatus(message) {
    document.getElementById('processing-status').textContent = message;
}

function updateProgress(percent) {
    document.getElementById('progress-fill').style.width = percent + '%';
}

function showError(message) {
    document.getElementById('error-message').textContent = message;
    showSection('error-section');
}

function resetUI() {
    Object.values(fileInputs).forEach(input => input.value = '');
    Object.values(statusIndicators).forEach(indicator => {
        indicator.textContent = '';
        indicator.classList.remove('error');
    });
    updateProgress(0);
    showSection('upload-section');
    submitBtn.disabled = true;
    currentSessionId = null;
}

// Health check on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const health = await response.json();
        console.log('✓ Server healthy:', health);
    } catch (error) {
        console.error('✗ Server unreachable:', error);
    }
});