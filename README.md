# 🎭 Emotion Recognition Application

A real-time emotion detection system with **Desktop** and **Web** versions that uses your webcam to detect and classify emotions from facial expressions.

## 🌟 Two Ways to Use

### 🖥️ Desktop Application
- Standalone OpenCV-based application
- Direct camera access
- Fast and lightweight
- No server setup required

### 🌐 Web Application  
- Browser-based interface
- Beautiful modern UI
- Accessible from any device
- Real-time WebSocket communication

---

## Features

✅ **Real-Time Emotion Detection**
- Detects 7 emotions: Happy, Sad, Angry, Surprised, Neutral, Fear, Disgust
- Color-coded labels for easy identification
- Confidence scores for each prediction
- Smooth, stable emotion updates (~1 per second)

✅ **Powered by DeepFace**
- Uses state-of-the-art deep learning models
- Automatic model downloading and management
- High accuracy emotion classification
- Works offline after initial setup

✅ **Simple & Clean Interface**
- Just press 'q' to quit
- No complex controls
- Instant feedback
- Professional visualization

## Tech Stack

- **OpenCV** - Camera handling and image processing
- **DeepFace** - Emotion recognition AI model
- **Python** - Core programming language
- **NumPy** - Numerical computations

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam/camera
- Internet connection (for initial model download)

### Setup Instructions

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd EmotionRecognition
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Desktop Application

**Run the desktop app:**
```bash
cd src
python main.py
```

**Controls:**
- Press **'q'** to quit

**What it does:**
- Opens OpenCV window
- Shows live camera feed
- Displays emotions directly on video
- Updates every ~1 second

---

### Option 2: Web Application

**Step 1: Start the server**
```bash
python server/app.py
```

You should see:
```
🎭 Emotion Detection Web Server
Server running at: http://localhost:5000
```

**Step 2: Open in browser**
```
http://localhost:5000
```

**Step 3: Use the app**
1. Click **"Start Detection"** button
2. Allow camera access when prompted
3. Watch your emotions detected in real-time! 😊
4. Click **"Stop Detection"** when done

---

## How It Works

### 1. Face Detection
- Uses OpenCV's Haar Cascade classifier
- Detects faces in real-time from camera feed
- Fast and efficient (~30 FPS)

### 2. Emotion Recognition
- Extracts detected face region
- Feeds to DeepFace emotion model
- Predicts emotion using deep learning
- Returns emotion label and confidence score

### 3. Visualization
- Draws color-coded box around face
- Displays emotion label with confidence
- Updates smoothly every ~1 second
- Professional, clean interface

## Emotion Colors

Each emotion is displayed with a unique color:

- 🟢 **Happy** - Green
- 🔵 **Sad** - Blue
- 🔴 **Angry** - Red
- 🟡 **Surprise** - Yellow
- 🟣 **Fear** - Magenta
- 🟣 **Disgust** - Purple
- ⚪ **Neutral** - Gray

## 📁 Project Structure

```
EmotionRecognition/
├── src/                         # Core emotion detection modules
│   ├── main.py                 # Desktop app entry point
│   ├── emotion_processor.py    # Shared processing logic
│   ├── face_detector.py        # Face detection
│   ├── emotion_classifier.py   # Emotion classification
│   ├── camera_handler.py       # Camera operations
│   └── __init__.py
│
├── server/                      # Web application backend
│   └── app.py                  # Flask + Socket.IO server
│
├── web/                         # Web application frontend
│   ├── index.html              # Main HTML page
│   ├── style.css               # Styling
│   └── app.js                  # JavaScript logic
│
├── models/                      # Model cache (auto-created)
│   └── emotion_model.h5
│
├── requirements.txt             # Python dependencies (all apps)
├── ARCHITECTURE.md             # Technical documentation
├── REFACTORING_SUMMARY.md     # Code organization details
└── README.md                   # This file
```

## Dependencies

### Core Dependencies (Desktop + Web)
```
opencv-python    # Computer vision
numpy            # Numerical operations
tensorflow       # Deep learning framework
tf-keras         # Keras for TensorFlow 2.x
pillow           # Image processing
deepface         # Emotion recognition
```

### Web Application Additional Dependencies
```
flask            # Web framework
flask-socketio   # WebSocket support
flask-cors       # CORS support
python-socketio  # Socket.IO client
python-engineio  # Engine.IO client
```

All dependencies are in `requirements.txt` for easy installation.

## Model Information

**DeepFace Emotion Model:**
- Based on FER-2013 dataset
- ~67% accuracy on standard benchmarks
- Trained on 35,000+ facial expressions
- Recognizes 7 emotion categories
- Model size: ~5MB
- Inference speed: Real-time capable

## Troubleshooting

### Camera not opening?
- Ensure no other application is using the camera
- Check camera permissions in system settings
- Try different camera index (change `camera_source=0` in main.py)

### Module import errors?
```bash
pip install -r requirements.txt
```

### Model not loading?
```bash
python setup_model.py
```

### TensorFlow warnings?
These are normal CPU optimization warnings. To suppress:
```bash
export TF_CPP_MIN_LOG_LEVEL=2  # Linux/Mac
set TF_CPP_MIN_LOG_LEVEL=2     # Windows
```

### Slow performance?
- Close other applications using the camera
- Reduce camera resolution in `camera_handler.py`
- Ensure you have adequate RAM (~4GB minimum)

## Performance

- **Face Detection**: ~30 FPS on CPU
- **Emotion Prediction**: Updates every ~1 second
- **Overall**: Smooth, real-time experience
- **Memory**: ~500MB RAM usage
- **Model Size**: ~5MB disk space

## 🏗️ Architecture

### Desktop App Flow
```
Camera → Face Detection → Emotion Classification → OpenCV Display
```

### Web App Flow
```
Browser (Webcam) 
    ↓ WebSocket
Flask Server 
    ↓
EmotionProcessor (Shared)
    ↓ Face Detection
    ↓ Emotion Classification
Flask Server
    ↓ WebSocket
Browser (Display)
```

Both apps share the same **EmotionProcessor** module for consistent results!

---

## 🔧 Configuration

### Desktop App
- Adjust frame rate in `src/main.py` (line ~34): `prediction_interval = 30`
- Change camera source: `camera_source=0` (try 1, 2 for other cameras)

### Web App
- Change server port in `server/app.py` (line ~140): `port=5000`
- Adjust frame rate in `web/app.js` (line ~235): `1500` (milliseconds)
- Update server URL in `web/app.js` (line ~68): `'http://localhost:5000'`

---

## 🐛 Troubleshooting

### Desktop App Issues

**Camera not opening?**
- Ensure no other app is using camera
- Try different camera index (0, 1, 2)
- Check camera permissions

**Performance issues?**
- Close other apps
- Reduce `prediction_interval` in main.py

### Web App Issues

**Server won't start:**
```bash
# Check dependencies
pip install -r requirements.txt

# Check if port 5000 is in use
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows
```

**Camera not working in browser:**
- Allow camera permissions
- Use Chrome or Firefox (recommended)
- Check if HTTPS required (some browsers)

**Socket connection fails:**
- Verify server is running at http://localhost:5000
- Check browser console (F12) for errors
- Check firewall settings

**Lag/slow performance:**
- Increase interval in `web/app.js` to 2000ms
- Close other browser tabs
- Check terminal for processing time

---

## 🚀 Deployment (Web App)

### Local Network Access
```bash
# Server will be accessible at http://YOUR_IP:5000
python server/app.py
```

### Production Deployment

**Option 1: Using Gunicorn**
```bash
pip install gunicorn
gunicorn --worker-class eventlet -w 1 server.app:app --bind 0.0.0.0:5000
```

**Option 2: Docker**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "server/app.py"]
```

**Option 3: Cloud Platforms**
- Heroku
- AWS Elastic Beanstalk
- Google Cloud Run
- Azure App Service

---

## 🎯 Future Enhancements

Potential features to add:
- 📊 Emotion statistics tracking over time
- 📹 Video file processing capability
- 📸 Screenshot capture in web app
- 🎯 Multiple face tracking
- 📈 Real-time emotion graphs
- 💾 Session recording and playback
- 🌙 Dark mode for web interface
- 📱 Mobile app version

## Technical Details

For detailed technical documentation, see [ARCHITECTURE.md](ARCHITECTURE.md)

Topics covered:
- System architecture
- Component design
- Data flow
- Model specifications
- API documentation

## License

This project is for educational purposes.

## Credits

- **Face Detection**: OpenCV Haar Cascades
- **Emotion Recognition**: DeepFace library
- **Dataset**: FER-2013 (Facial Expression Recognition)

## Contributing

Feel free to fork, improve, and submit pull requests!

## 📚 Additional Documentation

- **ARCHITECTURE.md** - Detailed technical architecture
- **REFACTORING_SUMMARY.md** - Code organization and shared modules

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Fork the repository
- Create a feature branch
- Submit a pull request

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review ARCHITECTURE.md for technical details
3. Open an issue on GitHub

---

**Built with ❤️ for learning and fun!**

Enjoy detecting emotions in real-time! 😊😢😠😲
