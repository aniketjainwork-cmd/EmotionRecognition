"""
Emotion Processor Module
Shared processing logic for both desktop and web applications
"""

import cv2
import numpy as np
from .face_detector import FaceDetector
from .emotion_classifier import EmotionClassifier


class EmotionProcessor:
    """
    Unified emotion processing for desktop and web applications
    Handles face detection and emotion classification
    """
    
    def __init__(self):
        """Initialize face detector and emotion classifier"""
        self.face_detector = FaceDetector()
        self.emotion_classifier = EmotionClassifier()
        print("✅ Emotion processor initialized")
    
    def process_frame(self, frame, resize_for_speed=False):
        """
        Process a single frame and return emotion results
        
        Args:
            frame: OpenCV image (BGR format)
            resize_for_speed (bool): Resize for faster processing (web app)
        
        Returns:
            dict: {
                'faces': [(x, y, w, h), ...],
                'emotions': [
                    {
                        'emotion': 'Happy',
                        'confidence': 0.87,
                        'bbox': (x, y, w, h)
                    },
                    ...
                ],
                'face_detected': bool
            }
        """
        result = {
            'faces': [],
            'emotions': [],
            'face_detected': False
        }
        
        try:
            # Optionally resize for web app performance
            if resize_for_speed:
                max_dimension = 480
                height, width = frame.shape[:2]
                if width > max_dimension or height > max_dimension:
                    ratio = min(max_dimension / width, max_dimension / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    frame = cv2.resize(frame, new_size)
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            result['faces'] = faces
            
            if len(faces) > 0:
                result['face_detected'] = True
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = self.face_detector.extract_face_roi(frame, (x, y, w, h))
                    
                    if face_roi is not None:
                        # Optionally resize face for faster processing
                        if resize_for_speed:
                            face_roi = cv2.resize(face_roi, (224, 224))
                        
                        # Predict emotion
                        emotion, confidence = self.emotion_classifier.predict_emotion(face_roi)
                        
                        if emotion:
                            result['emotions'].append({
                                'emotion': emotion,
                                'confidence': float(confidence),
                                'bbox': (x, y, w, h)
                            })
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing frame: {str(e)}")
            return result
    
    def get_emotion_color(self, emotion):
        """
        Get BGR color for emotion visualization
        
        Args:
            emotion (str): Emotion label
        
        Returns:
            tuple: BGR color
        """
        return self.emotion_classifier.get_emotion_color(emotion)
    
    def draw_emotion_on_frame(self, frame, emotion_data):
        """
        Draw emotion label and bounding box on frame
        
        Args:
            frame: OpenCV image
            emotion_data (dict): Single emotion result from process_frame
        
        Returns:
            frame: Modified frame with drawings
        """
        emotion = emotion_data['emotion']
        confidence = emotion_data['confidence']
        x, y, w, h = emotion_data['bbox']
        
        # Get color for this emotion
        color = self.get_emotion_color(emotion)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Create label
        if confidence > 0:
            label = f"{emotion} {confidence*100:.0f}%"
        else:
            label = emotion
        
        # Draw label background
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )
        cv2.rectangle(frame, (x, y-35), (x+label_width+10, y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        return frame
