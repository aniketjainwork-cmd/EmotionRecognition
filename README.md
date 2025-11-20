# Emotion Recognition Application

A real-time emotion detection system that uses your webcam to detect and classify emotions from facial expressions.

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

## Usage

### Run the Application

```bash
cd src
python main.py
```

### Controls

- **q** - Quit the application

That's it! The app automatically:
- Opens your camera
- Detects your face
- Shows your current emotion with confidence score
- Updates every second for stable readings

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

## Project Structure

```
EmotionRecognition/
├── src/
│   ├── main.py                 # Application entry point
│   ├── camera_handler.py       # Camera operations
│   ├── face_detector.py        # Face detection logic
│   ├── emotion_classifier.py   # Emotion recognition
│   └── __init__.py
│
├── models/                      # Model cache (auto-created)
│   └── emotion_model.h5
│
├── requirements.txt             # Python dependencies
├── setup_model.py              # Model setup script
├── README.md                   # This file
├── ARCHITECTURE.md             # Technical documentation
└── .gitignore                  # Git ignore rules
```

## Dependencies

```
opencv-python    # Computer vision
numpy            # Numerical operations
tensorflow       # Deep learning framework
tf-keras         # Keras for TensorFlow 2.x
pillow           # Image processing
deepface         # Emotion recognition
```

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

## Future Enhancements

Potential features to add:
- 📊 Emotion statistics tracking over time
- 📹 Video file processing capability
- 📸 Photo mode for still images
- 🎯 Multiple face tracking
- 📈 Real-time emotion graphs
- 💾 Session recording and playback
- 🌐 Web interface version

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

## Support

If you encounter any issues:
1. Check the troubleshooting section
2. Review the architecture documentation
3. Open an issue on GitHub

---

**Built with ❤️ for learning and fun!**

Enjoy detecting emotions in real-time! 😊😢😠😲
