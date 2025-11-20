"""
Emotion Classifier Module
Classifies emotions from face images using DeepFace
"""

import cv2
import numpy as np
import os


class EmotionClassifier:
    """
    Classifies emotions from face images using DeepFace emotion model
    
    DeepFace automatically manages model files in ~/.deepface/weights/
    No manual model path needed!
    """
    
    # Standard emotion labels (FER-2013 dataset format)
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self):
        """
        Initialize emotion classifier using DeepFace
        
        DeepFace handles model downloading and caching automatically
        """
        self.deepface = None
        self.is_loaded = False
        
        # Load DeepFace
        self.load_model()
    
    def load_model(self):
        """
        Load DeepFace for emotion analysis
        
        DeepFace automatically:
        - Downloads emotion model if not present
        - Stores it in ~/.deepface/weights/
        - Loads it when needed
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.is_loaded = True
            print(f"✅ DeepFace emotion model ready")
            print(f"   (Models stored in ~/.deepface/weights/)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading DeepFace: {str(e)}")
            print("   Running in DEMO mode")
            self.is_loaded = False
            return False
    
    def predict_emotion(self, face_image):
        """
        Predict emotion from a face image using DeepFace
        
        Args:
            face_image: Face region extracted from frame (BGR format)
        
        Returns:
            tuple: (emotion_label, confidence) or (None, 0) if prediction fails
        """
        if face_image is None or face_image.size == 0:
            return None, 0
        
        try:
            if self.is_loaded and self.deepface is not None:
                # Use DeepFace to analyze emotion
                result = self.deepface.analyze(
                    face_image,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                # Extract emotion and confidence
                if isinstance(result, list):
                    result = result[0]
                
                emotion_scores = result['emotion']
                dominant_emotion = result['dominant_emotion']
                confidence = emotion_scores[dominant_emotion] / 100.0  # Convert to 0-1
                
                # Capitalize first letter to match our format
                emotion = dominant_emotion.capitalize()
                
                return emotion, confidence
                
            else:
                # Demo mode: return random emotion (for testing without model)
                emotion_idx = np.random.randint(0, len(self.EMOTIONS))
                emotion = self.EMOTIONS[emotion_idx].capitalize()
                confidence = 0.0  # 0 confidence indicates demo mode
                
                return emotion, confidence
            
        except Exception as e:
            # Silently fail and return None for individual frames
            return None, 0
    
    def get_emotion_color(self, emotion):
        """
        Get color for emotion label (for visualization)
        
        Args:
            emotion (str): Emotion label
        
        Returns:
            tuple: BGR color tuple
        """
        color_map = {
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Angry': (0, 0, 255),      # Red
            'Surprise': (0, 255, 255), # Yellow
            'Fear': (255, 0, 255),     # Magenta
            'Disgust': (128, 0, 128),  # Purple
            'Neutral': (128, 128, 128) # Gray
        }
        
        return color_map.get(emotion, (255, 255, 255))  # White as default
