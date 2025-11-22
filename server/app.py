"""
Flask Server for Emotion Detection Web Application
Handles WebSocket connections and processes frames for emotion detection
"""

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.emotion_processor import EmotionProcessor

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../web',
            template_folder='../web')
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Enable CORS for all routes
CORS(app)

# Initialize SocketIO with CORS support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize emotion processor (shared module)
print("=" * 60)
print("Initializing Emotion Detection Server")
print("=" * 60)

emotion_processor = EmotionProcessor()

print("🚀 Server ready to process frames!")
print("=" * 60)

# ========================================
# Routes
# ========================================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('../web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS)"""
    return send_from_directory('../web', path)

# ========================================
# WebSocket Event Handlers
# ========================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('🔌 Client connected')
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('🔌 Client disconnected')

@socketio.on('process_frame')
def handle_frame(data):
    """
    Process frame from client and return emotion detection result
    Uses shared EmotionProcessor module
    
    Args:
        data (dict): Contains 'image' as base64 encoded string
    """
    try:
        # Extract base64 image data
        image_data = data.get('image', '')
        
        # Remove base64 header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format (BGR)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process frame using shared module (with optimization)
        result = emotion_processor.process_frame(frame, resize_for_speed=True)
        
        if result['face_detected'] and len(result['emotions']) > 0:
            # Get first detected emotion
            emotion_data = result['emotions'][0]
            
            # Send result back to client
            response = {
                'emotion': emotion_data['emotion'],
                'confidence': emotion_data['confidence'],
                'face_detected': True
            }
            
            print(f"😊 Detected: {emotion_data['emotion']} ({emotion_data['confidence']*100:.1f}%)")
            emit('emotion_result', response)
        else:
            # No face detected
            emit('emotion_result', {
                'emotion': 'No face detected',
                'confidence': 0,
                'face_detected': False
            })
        
    except Exception as e:
        print(f"❌ Error processing frame: {str(e)}")
        emit('error', {'message': str(e)})

# ========================================
# Main
# ========================================

if __name__ == '__main__':
    import os
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "=" * 60)
    print("🎭 Emotion Detection Web Server")
    print("=" * 60)
    print(f"Server running on port: {port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Run server (debug=False for production)
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
