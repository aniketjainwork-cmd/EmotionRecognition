"""
Main Application Entry Point
Real-time emotion detection from camera using shared EmotionProcessor
"""

import sys
import cv2
from camera_handler import CameraHandler
from emotion_processor import EmotionProcessor


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
    emotion_processor = EmotionProcessor()
    
    # Start camera
    if not camera.start_camera():
        print("Failed to start camera. Exiting...")
        return
    
    print("\n✅ Ready! Detecting emotions...\n")
    
    # Variables for throttling emotion predictions
    frame_count = 0
    prediction_interval = 30  # Predict emotion every 30 frames (~1 second at 30 FPS)
    last_result = None  # Store last processing result
    
    try:
        while True:
            # Capture frame
            success, frame = camera.get_frame()
            
            if not success:
                print("Failed to get frame")
                break
            
            frame_count += 1
            
            # Process frame using shared module every N frames
            if frame_count % prediction_interval == 0:
                last_result = emotion_processor.process_frame(frame, resize_for_speed=False)
            
            # Draw emotions if we have results
            if last_result and last_result['face_detected']:
                for emotion_data in last_result['emotions']:
                    frame = emotion_processor.draw_emotion_on_frame(frame, emotion_data)
            
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
