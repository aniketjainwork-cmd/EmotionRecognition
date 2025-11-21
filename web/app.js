// ========================================
// Global Variables
// ========================================
let webcam = null;
let canvas = null;
let ctx = null;
let socket = null;
let isDetecting = false;
let frameCount = 0;
let detectionInterval = null;

// DOM Elements
const webcamElement = document.getElementById('webcam');
const overlayCanvas = document.getElementById('overlay');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusMessage = document.getElementById('status-message');
const emotionDisplay = document.getElementById('emotion-display');
const emotionEmoji = document.getElementById('emotion-emoji');
const emotionLabel = document.getElementById('emotion-label');
const confidenceScore = document.getElementById('confidence-score');
const connectionStatus = document.getElementById('connection-status');
const framesCountEl = document.getElementById('frames-count');
const lastUpdateEl = document.getElementById('last-update');

// Emotion to Emoji mapping
const emotionEmojis = {
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😠',
    'Surprise': '😲',
    'Fear': '😨',
    'Disgust': '🤢',
    'Neutral': '😐'
};

// ========================================
// Initialization
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎭 Emotion Detection App Initialized');
    setupEventListeners();
    setupCanvas();
});

// ========================================
// Event Listeners
// ========================================
function setupEventListeners() {
    startBtn.addEventListener('click', startDetection);
    stopBtn.addEventListener('click', stopDetection);
}

// ========================================
// Canvas Setup
// ========================================
function setupCanvas() {
    canvas = overlayCanvas;
    ctx = canvas.getContext('2d');
}

// ========================================
// WebSocket Connection (Flask-SocketIO)
// ========================================
function connectWebSocket() {
    console.log('📡 Connecting to Flask server...');
    
    // Connect to Flask-SocketIO server
    socket = io('http://localhost:5000');
    
    socket.on('connect', () => {
        console.log('✅ Connected to server');
        updateConnectionStatus(true);
    });
    
    socket.on('disconnect', () => {
        console.log('❌ Disconnected from server');
        updateConnectionStatus(false);
    });
    
    socket.on('emotion_result', (data) => {
        handleEmotionResult(data);
    });
    
    socket.on('error', (data) => {
        console.error('❌ Server error:', data.message);
    });
    
    socket.on('connection_response', (data) => {
        console.log('🎉 Server responded:', data.status);
    });
}

function disconnectWebSocket() {
    if (socket) {
        socket.disconnect();
    }
    updateConnectionStatus(false);
}

// ========================================
// Camera Functions
// ========================================
async function startCamera() {
    try {
        showStatus('Requesting camera access...');
        
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        webcamElement.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            webcamElement.onloadedmetadata = () => {
                resolve();
            };
        });
        
        // Set canvas size to match video
        canvas.width = webcamElement.videoWidth;
        canvas.height = webcamElement.videoHeight;
        
        console.log('📷 Camera started successfully');
        hideStatus();
        return true;
        
    } catch (error) {
        console.error('❌ Camera error:', error);
        showStatus('Camera access denied. Please allow camera access.');
        return false;
    }
}

function stopCamera() {
    const stream = webcamElement.srcObject;
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        webcamElement.srcObject = null;
    }
    console.log('📷 Camera stopped');
}

// ========================================
// Detection Control
// ========================================
async function startDetection() {
    if (isDetecting) return;
    
    console.log('▶️ Starting detection...');
    
    // Start camera
    const cameraStarted = await startCamera();
    if (!cameraStarted) return;
    
    // Connect to server (when ready)
    connectWebSocket();
    
    // Update UI
    isDetecting = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    emotionDisplay.classList.remove('hidden');
    
    // Start sending frames for detection
    startFrameCapture();
    
    console.log('✅ Detection started');
}

function stopDetection() {
    if (!isDetecting) return;
    
    console.log('⏸️ Stopping detection...');
    
    // Stop frame capture
    stopFrameCapture();
    
    // Disconnect from server
    disconnectWebSocket();
    
    // Stop camera
    stopCamera();
    
    // Update UI
    isDetecting = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    emotionDisplay.classList.add('hidden');
    showStatus('Click "Start Detection" to begin');
    
    // Reset stats
    frameCount = 0;
    updateFrameCount();
    
    console.log('✅ Detection stopped');
}

// ========================================
// Frame Capture & Processing
// ========================================
function startFrameCapture() {
    // Capture and send frames every 1.5 seconds for better performance
    detectionInterval = setInterval(() => {
        captureAndSendFrame();
    }, 1500);
}

function stopFrameCapture() {
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
}

function captureAndSendFrame() {
    if (!isDetecting || !webcamElement.videoWidth) return;
    
    // Use smaller canvas for faster processing
    const targetWidth = 480;
    const targetHeight = Math.floor(webcamElement.videoHeight * (targetWidth / webcamElement.videoWidth));
    
    // Temporarily set canvas to smaller size for capture
    const originalWidth = canvas.width;
    const originalHeight = canvas.height;
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    
    // Draw current video frame to smaller canvas
    ctx.drawImage(webcamElement, 0, 0, targetWidth, targetHeight);
    
    // Convert canvas to base64 image with lower quality for speed
    const imageData = canvas.toDataURL('image/jpeg', 0.6);
    
    // Restore canvas size
    canvas.width = originalWidth;
    canvas.height = originalHeight;
    
    // Send to server
    sendFrameToServer(imageData);
}

function sendFrameToServer(imageData) {
    // Send frame to Flask server via WebSocket
    if (socket && socket.connected) {
        socket.emit('process_frame', { image: imageData });
        console.log('📤 Frame sent to server');
    } else {
        console.warn('⚠️ Socket not connected');
    }
}

// ========================================
// Emotion Result Handling
// ========================================
function handleEmotionResult(data) {
    // data format: { emotion: "Happy", confidence: 0.87 }
    const { emotion, confidence } = data;
    
    // Update UI
    updateEmotionDisplay(emotion, confidence);
    
    // Update stats
    frameCount++;
    updateFrameCount();
    updateLastUpdate();
    
    console.log(`😊 Detected: ${emotion} (${(confidence * 100).toFixed(1)}%)`);
}

function updateEmotionDisplay(emotion, confidence) {
    // Update emoji
    emotionEmoji.textContent = emotionEmojis[emotion] || '😐';
    
    // Update label
    emotionLabel.textContent = emotion;
    emotionLabel.className = `emotion-${emotion.toLowerCase()}`;
    
    // Update confidence
    confidenceScore.textContent = `${(confidence * 100).toFixed(1)}%`;
    
    // Show emotion display if hidden
    emotionDisplay.classList.remove('hidden');
}


// ========================================
// UI Update Functions
// ========================================
function showStatus(message) {
    statusMessage.textContent = message;
    statusMessage.classList.remove('hidden');
}

function hideStatus() {
    statusMessage.classList.add('hidden');
}

function updateConnectionStatus(connected) {
    const statusEl = connectionStatus;
    if (connected) {
        statusEl.textContent = 'Connected';
        statusEl.classList.remove('disconnected');
        statusEl.classList.add('connected');
    } else {
        statusEl.textContent = 'Disconnected';
        statusEl.classList.remove('connected');
        statusEl.classList.add('disconnected');
    }
}

function updateFrameCount() {
    framesCountEl.textContent = frameCount;
}

function updateLastUpdate() {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    lastUpdateEl.textContent = timeString;
}

// ========================================
// Cleanup on page unload
// ========================================
window.addEventListener('beforeunload', () => {
    if (isDetecting) {
        stopDetection();
    }
});

// ========================================
// Console Art
// ========================================
console.log(`
%c🎭 Emotion Detection App
%cFrontend Ready! 
%cReady to connect to Flask server at http://localhost:5000
`, 
'font-size: 20px; font-weight: bold; color: #667eea;',
'font-size: 14px; color: #28a745;',
'font-size: 12px; color: #6c757d;'
);
