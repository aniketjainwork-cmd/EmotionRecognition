"""
Camera Handler Module
Handles camera initialization, frame capture, and image saving
"""

import cv2
import numpy as np
from datetime import datetime
import os


class CameraHandler:
    """
    Handles camera operations including initialization, 
    frame capture, and image saving
    """
    
    def __init__(self, camera_source=0):
        """
        Initialize camera handler
        
        Args:
            camera_source (int): Camera index (0 for default webcam)
        """
        self.camera_source = camera_source
        self.capture = None
        self.is_camera_opened = False
        
    def start_camera(self):
        """
        Initialize and start the camera
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            self.capture = cv2.VideoCapture(self.camera_source)
            
            if not self.capture.isOpened():
                print(f"Error: Could not open camera {self.camera_source}")
                return False
            
            # Set camera properties for better quality
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_camera_opened = True
            print(f"Camera {self.camera_source} opened successfully!")
            return True
            
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            return False
    
    def get_frame(self):
        """
        Capture a single frame from the camera
        
        Returns:
            tuple: (success, frame) where success is bool and frame is numpy array
        """
        if not self.is_camera_opened or self.capture is None:
            print("Error: Camera is not opened")
            return False, None
        
        success, frame = self.capture.read()
        
        if not success:
            print("Error: Failed to capture frame")
            return False, None
            
        return True, frame
    
    def save_image(self, frame, output_dir="output", filename=None):
        """
        Save a captured frame as an image file
        
        Args:
            frame: The image frame to save
            output_dir (str): Directory to save the image
            filename (str): Optional filename, auto-generated if None
            
        Returns:
            str: Path to saved image, or None if failed
        """
        if frame is None:
            print("Error: No frame to save")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            print(f"Image saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return None
    
    def display_frame(self, frame, window_name="Camera Feed"):
        """
        Display a frame in a window
        
        Args:
            frame: The image frame to display
            window_name (str): Name of the display window
        """
        if frame is not None:
            cv2.imshow(window_name, frame)
    
    def release(self):
        """
        Release the camera and close all windows
        """
        if self.capture is not None:
            self.capture.release()
            self.is_camera_opened = False
            print("Camera released")
        
        cv2.destroyAllWindows()
        print("All windows closed")
    
    def __del__(self):
        """
        Destructor to ensure camera is released
        """
        self.release()
