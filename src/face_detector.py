"""
Face Detector Module
Detects faces in images/video frames using Haar Cascades
"""

import cv2
import os


class FaceDetector:
    """
    Detects faces in images using OpenCV's Haar Cascade classifier
    """
    
    def __init__(self, cascade_path=None):
        """
        Initialize face detector
        
        Args:
            cascade_path (str): Path to Haar Cascade XML file
                              If None, uses OpenCV's built-in cascade
        """
        if cascade_path is None:
            # Use OpenCV's built-in Haar Cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.cascade_path = cascade_path
        self.face_cascade = None
        self.load_cascade()
    
    def load_cascade(self):
        """
        Load the Haar Cascade classifier
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.cascade_path):
                print(f"Error: Cascade file not found at {self.cascade_path}")
                return False
            
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            
            if self.face_cascade.empty():
                print("Error: Failed to load cascade classifier")
                return False
            
            print("Face detector loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading cascade: {str(e)}")
            return False
    
    def detect_faces(self, frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in a frame
        
        Args:
            frame: Input image frame
            scale_factor (float): How much the image size is reduced at each scale
                                 (1.1 = 10% reduction, smaller = more accurate but slower)
            min_neighbors (int): How many neighbors each candidate rectangle should have
                               (higher = fewer false positives but may miss faces)
            min_size (tuple): Minimum face size to detect (width, height)
        
        Returns:
            list: List of face coordinates as (x, y, w, h) tuples
        """
        if self.face_cascade is None:
            print("Error: Face cascade not loaded")
            return []
        
        if frame is None:
            print("Error: No frame provided")
            return []
        
        try:
            # Convert to grayscale (Haar Cascades work on grayscale images)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
            
        except Exception as e:
            print(f"Error detecting faces: {str(e)}")
            return []
    
    def draw_faces(self, frame, faces, color=(0, 255, 0), thickness=2, show_count=True):
        """
        Draw bounding boxes around detected faces
        
        Args:
            frame: Input image frame
            faces: List of face coordinates from detect_faces()
            color (tuple): BGR color for rectangle (default: green)
            thickness (int): Rectangle line thickness
            show_count (bool): Whether to show face count on frame
        
        Returns:
            frame: Frame with drawn rectangles
        """
        if frame is None:
            return None
        
        # Draw rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Optionally add "Face" label above the box
            cv2.putText(frame, "Face", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show face count
        if show_count:
            count_text = f"Faces detected: {len(faces)}"
            cv2.putText(frame, count_text, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def extract_face_roi(self, frame, face_coords):
        """
        Extract face region of interest (ROI) from frame
        
        Args:
            frame: Input image frame
            face_coords: Face coordinates (x, y, w, h)
        
        Returns:
            numpy array: Cropped face image, or None if invalid
        """
        if frame is None or face_coords is None:
            return None
        
        try:
            x, y, w, h = face_coords
            face_roi = frame[y:y+h, x:x+w]
            return face_roi
        except Exception as e:
            print(f"Error extracting face ROI: {str(e)}")
            return None
