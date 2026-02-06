const fileInput = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');
const detailSlider = document.getElementById('detail');
const detailReadout = document.getElementById('detail-readout');
const maxDimensionInput = document.getElementById('max-dimension');
const formatSelect = document.getElementById('format');
const convertForm = document.getElementById('convert-form');
const statusEl = document.getElementById('status');
const originalPreview = document.getElementById('original-preview');
const meshPreview = document.getElementById('mesh-preview');
const downloadLink = document.getElementById('download-link');
const resetBtn = document.getElementById('reset-btn');

const statMap = {
    vertices: document.querySelector('[data-stat="vertices"]'),
    triangles: document.querySelector('[data-stat="triangles"]'),
    canvas: document.querySelector('[data-stat="canvas"]'),
    time: document.querySelector('[data-stat="time"]'),
    scale: document.querySelector('[data-stat="scale"]'),
};

let selectedFile = null;
let previewUrl = null;
let resultUrl = null;

const setStatus = (message, mode = 'idle') => {
    statusEl.textContent = message;
    statusEl.classList.remove('running', 'error');
    if (mode === 'running') statusEl.classList.add('running');
    if (mode === 'error') statusEl.classList.add('error');
};

const resetStats = () => {
    Object.values(statMap).forEach((node) => {
        node.textContent = '—';
    });
};

const revokeUrl = (url) => {
    if (url) URL.revokeObjectURL(url);
};

const updatePreview = (imgEl, file, assignUrlCb) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    imgEl.src = url;
    if (assignUrlCb) assignUrlCb(url);
};

const handleFiles = (files) => {
    if (!files || !files.length) return;
    selectedFile = files[0];
    revokeUrl(previewUrl);
    updatePreview(originalPreview, selectedFile, (url) => {
        previewUrl = url;
    });
    setStatus('Image loaded. Press generate when ready.');
};

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        fileInput.click();
    }
});

uploadZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadZone.classList.add('dragging');
});

uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragging'));

uploadZone.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadZone.classList.remove('dragging');
    handleFiles(event.dataTransfer.files);
});

fileInput.addEventListener('change', (event) => handleFiles(event.target.files));

detailSlider.addEventListener('input', () => {
    detailReadout.textContent = `${detailSlider.value} vertices`;
});

resetBtn.addEventListener('click', () => {
    convertForm.reset();
    selectedFile = null;
    detailReadout.textContent = `${detailSlider.value} vertices`;
    originalPreview.removeAttribute('src');
    meshPreview.removeAttribute('src');
    downloadLink.classList.add('hidden');
    resetStats();
    setStatus('Waiting for an image…');
    revokeUrl(previewUrl);
    revokeUrl(resultUrl);
    previewUrl = null;
    resultUrl = null;
});

const updateStats = (headers) => {
    const vertices = headers.get('X-Mesh-Vertices');
    const triangles = headers.get('X-Mesh-Triangles');
    const width = headers.get('X-Mesh-Width');
    const height = headers.get('X-Mesh-Height');
    const seconds = headers.get('X-Mesh-Seconds');
    const scale = headers.get('X-Mesh-Scale');

    if (vertices) statMap.vertices.textContent = vertices;
    if (triangles) statMap.triangles.textContent = triangles;
    if (width && height) statMap.canvas.textContent = `${width} × ${height}`;
    if (seconds) statMap.time.textContent = `${seconds}s`;
    if (scale) statMap.scale.textContent = scale;
};

convertForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!selectedFile) {
        setStatus('Please load an image first.', 'error');
        return;
    }

    setStatus('Triangulating in Python…', 'running');
    downloadLink.classList.add('hidden');
    revokeUrl(resultUrl);
    resultUrl = null;
    resetStats();

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('detail', detailSlider.value);
    formData.append('maxDimension', maxDimensionInput.value);
    formData.append('format', formatSelect.value);

    try {
        const response = await fetch('/api/filter', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            let errorMessage = 'Processing failed.';
            try {
                const payload = await response.json();
                if (payload?.error) errorMessage = payload.error;
            } catch (_) {
                // Ignore JSON parse problems and use the fallback
            }
            throw new Error(errorMessage);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        meshPreview.src = url;
        downloadLink.href = url;
        downloadLink.download = `math-mesh.${formatSelect.value === 'jpg' ? 'jpeg' : formatSelect.value}`;
        downloadLink.classList.remove('hidden');
        resultUrl = url;

        updateStats(response.headers);
        setStatus('Mesh generated. Download or tweak settings.');
    } catch (error) {
        setStatus(error.message, 'error');
    }
});
