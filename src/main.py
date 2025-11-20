"""
Main Application Entry Point
Real-time emotion detection from camera
"""

import sys
import cv2
from camera_handler import CameraHandler
from face_detector import FaceDetector
from emotion_classifier import EmotionClassifier


def main():
    """
    Main function to run emotion detection
    """
    print("=" * 50)
    print("Real-Time Emotion Detection")
    print("=" * 50)
    print("\nPress 'q' to quit")
    print("=" * 50)
    
    # Initialize components
    camera = CameraHandler(camera_source=0)
    face_detector = FaceDetector()
    emotion_classifier = EmotionClassifier()
    
    # Start camera
    if not camera.start_camera():
        print("Failed to start camera. Exiting...")
        return
    
    print("\n✅ Ready! Detecting emotions...\n")
    
    # Variables for throttling emotion predictions
    frame_count = 0
    prediction_interval = 30  # Predict emotion every 30 frames (~1 second at 30 FPS)
    last_emotions = {}  # Store last emotion for each face
    
    try:
        while True:
            # Capture frame
            success, frame = camera.get_frame()
            
            if not success:
                print("Failed to get frame")
                break
            
            frame_count += 1
            
            # Detect faces
            faces = face_detector.detect_faces(frame)
            
            # Process each detected face
            for idx, (x, y, w, h) in enumerate(faces):
                # Only predict emotion every N frames to slow down refresh rate
                if frame_count % prediction_interval == 0:
                    # Extract face region
                    face_roi = face_detector.extract_face_roi(frame, (x, y, w, h))
                    
                    if face_roi is not None:
                        # Predict emotion
                        emotion, confidence = emotion_classifier.predict_emotion(face_roi)
                        
                        if emotion:
                            # Store this emotion for this face
                            last_emotions[idx] = (emotion, confidence)
                
                # Use the last predicted emotion for display
                if idx in last_emotions:
                    emotion, confidence = last_emotions[idx]
                    
                    # Get color for emotion
                    color = emotion_classifier.get_emotion_color(emotion)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Display emotion label
                    if confidence > 0:
                        label = f"{emotion} {confidence*100:.0f}%"
                    else:
                        label = f"{emotion}"
                    
                    # Add background for text
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                    )
                    cv2.rectangle(frame, (x, y-35), (x+label_width+10, y), color, -1)
                    
                    cv2.putText(frame, label, (x+5, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Display instruction
            cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            camera.display_frame(frame, "Emotion Detection")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    
    finally:
        # Clean up
        print("\nCleaning up...")
        camera.release()
        print("Application closed successfully!")


if __name__ == "__main__":
    main()
