# Face Recognition Application

A Python-based face emotion recognition system that detects faces and classifies emotions in real-time.

## Current Features

✅ **Module 1: Camera Module**
- Opens and initializes webcam
- Displays live camera feed
- Captures and saves images with timestamp
- Clean resource management

✅ **Module 2: Face Detection**
- Real-time face detection using Haar Cascades
- Draws bounding boxes around detected faces
- Shows face count on screen
- Toggle detection ON/OFF
- Detects multiple faces simultaneously

✅ **Module 3: Emotion Recognition**
- Classifies 7 emotions: Happy, Sad, Angry, Surprised, Neutral, Fear, Disgust
- Color-coded emotion labels for each face
- Real-time emotion prediction
- Toggle emotion recognition ON/OFF
- Works with or without pre-trained model (demo mode)
- Confidence scores displayed (when model is loaded)

## Setup Instructions

### 1. Install Dependencies

First, make sure you have Python 3.7+ installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

**Note:** TensorFlow installation may take a few minutes.

### 2. Run the Application

Navigate to the `src` directory and run:

```bash
cd src
python main.py
```

## How to Use

1. **Launch the application**: Run `python main.py` from the `src` directory
2. **View live feed**: Your camera will open and display real-time emotion detection
3. **See emotions**: Detected emotions appear above each face with color-coded boxes
4. **Capture image**: Press `c` to save the current frame with emotions
5. **Toggle features**: Use `d` and `e` to turn detection/emotions on/off
6. **Quit**: Press `q` to exit

### Keyboard Controls

- `c` - Capture and save current frame as image
- `d` - Toggle face detection ON/OFF
- `e` - Toggle emotion recognition ON/OFF
- `q` - Quit the application

## Emotion Colors

Each emotion is displayed with a unique color:

- 🟢 **Happy** - Green
- 🔵 **Sad** - Blue
- 🔴 **Angry** - Red
- 🟡 **Surprise** - Yellow
- 🟣 **Fear** - Magenta
- 🟣 **Disgust** - Purple
- ⚪ **Neutral** - Gray

## Demo Mode vs Model Mode

### Demo Mode (Default)
If no emotion model is found, the app runs in **DEMO MODE**:
- Shows random emotions for testing purposes
- Labels show "(DEMO)" indicator
- All features work normally for testing the UI

### Model Mode (Optional)
To use real emotion predictions, add a pre-trained model:

1. Download a pre-trained FER-2013 emotion model (.h5 file)
2. Create a `models/` directory in the project root
3. Place the model as `models/emotion_model.h5`
4. Restart the application

**Model sources:**
- GitHub: Search for "FER-2013 emotion model"
- Kaggle: FER-2013 dataset with trained models
- DeepFace library releases

## Output

Captured images are automatically saved in the `output/` directory with timestamps:
- Format: `capture_YYYYMMDD_HHMMSS.jpg`
- Example: `capture_20231120_141530.jpg`
- Images include face boxes and emotion labels

## Project Structure

```
Face_Recognition/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Main application entry point
│   ├── camera_handler.py       # Camera handling logic
│   ├── face_detector.py        # Face detection module
│   └── emotion_classifier.py   # Emotion classification module
├── models/                      # Pre-trained models (optional)
│   └── emotion_model.h5        # Emotion classification model
├── output/                      # Saved images (auto-created)
├── requirements.txt             # Python dependencies
├── ARCHITECTURE.md              # Detailed architecture documentation
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## How It Works

### 1. Camera Capture
- Captures video frames at 30 FPS
- Each frame is processed independently

### 2. Face Detection
- Uses OpenCV's Haar Cascade classifier
- Converts frame to grayscale
- Detects face regions (bounding boxes)
- Fast and efficient (~30 FPS on CPU)

### 3. Emotion Classification
- Extracts each detected face region
- Resizes to 48x48 pixels and normalizes
- Feeds to CNN model (or demo mode)
- Predicts emotion with confidence score
- Displays result with color-coded label

## Troubleshooting

**Camera not opening?**
- Make sure no other application is using the camera
- Check camera permissions on your system
- Try changing camera source in `main.py` (default is 0)

**Import errors?**
- Ensure you're running from the `src` directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

**TensorFlow warnings?**
- TensorFlow may show warnings about CPU optimization
- These are normal and don't affect functionality
- To suppress: `export TF_CPP_MIN_LOG_LEVEL=2` (Linux/Mac)

**Slow performance?**
- Demo mode runs faster than model mode
- Close other applications using the camera
- Lower the camera resolution in `camera_handler.py`

## Requirements

- Python 3.7+
- OpenCV (opencv-python)
- NumPy
- TensorFlow (for emotion model)
- Pillow (image processing)
- Working webcam/camera

## Future Enhancements

- 📊 Emotion statistics and tracking
- 📹 Video file processing
- 🎯 Custom model training interface
- 📱 Mobile app version
- ☁️ Cloud deployment
- 🔊 Audio emotion analysis

## License

This project is for educational purposes.

## Credits

- Face Detection: OpenCV Haar Cascades
- Emotion Classification: CNN architecture based on FER-2013 dataset
