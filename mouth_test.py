"""
Mouth Detection Test Script - Updated for MediaPipe 0.10.30+
B.Tech Mini-Project: Visual Fatigue Detection System
"""

import cv2
import mediapipe as mp
import numpy as np

print("=" * 60)
print("MediaPipe Face Landmarker - Mouth Detection Test")
print("=" * 60)
print(f"MediaPipe Version: {mp.__version__}")
print("=" * 60)

# Initialize Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download model if not exists
import urllib.request
import os

model_path = 'models/face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face landmarker model (this may take a minute)...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    try:
        urllib.request.urlretrieve(url, model_path)
        print("✓ Model downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("Please check your internet connection")
        exit()

# Create FaceLandmarker options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# Create the landmarker
try:
    landmarker = FaceLandmarker.create_from_options(options)
    print("✓ Face Landmarker initialized")
except Exception as e:
    print(f"✗ Error initializing landmarker: {e}")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ ERROR: Cannot access webcam")
    exit()

print("✓ Webcam opened successfully")
print("\nInstructions:")
print("  - Align your face with the camera")
print("  - Green dots will mark key mouth landmarks")
print("  - Press 'q' to quit")
print("=" * 60)
print()

# Mouth landmark indices
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_CORNER = 78
RIGHT_CORNER = 308

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Flip for selfie view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect landmarks
    detection_result = landmarker.detect_for_video(mp_image, frame_count)
    
    # Process results
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            h, w, _ = frame.shape
            
            # Get mouth landmarks
            mouth_indices = [UPPER_LIP, LOWER_LIP, LEFT_CORNER, RIGHT_CORNER]
            
            for idx in mouth_indices:
                landmark = face_landmarks[idx]
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # Draw green circles
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # Add labels
                cv2.putText(frame, str(idx), (cx + 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Calculate Mouth Aspect Ratio (MAR)
            upper = face_landmarks[UPPER_LIP]
            lower = face_landmarks[LOWER_LIP]
            left = face_landmarks[LEFT_CORNER]
            right = face_landmarks[RIGHT_CORNER]
            
            # Convert to pixel coordinates
            left_px = (int(left.x * w), int(left.y * h))
            right_px = (int(right.x * w), int(right.y * h))
            upper_px = (int(upper.x * w), int(upper.y * h))
            lower_px = (int(lower.x * w), int(lower.y * h))
            
            # Draw lines
            cv2.line(frame, left_px, right_px, (255, 0, 0), 2)  # Blue horizontal
            cv2.line(frame, upper_px, lower_px, (0, 0, 255), 2)  # Red vertical
            
            # Calculate distances
            mouth_width = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
            mouth_height = np.sqrt((upper.x - lower.x)**2 + (upper.y - lower.y)**2)
            
            if mouth_width > 0:
                mar = mouth_height / mouth_width
                
                # Display MAR
                cv2.putText(frame, f"MAR: {mar:.3f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Yawn detection
                if mar > 0.6:
                    cv2.putText(frame, "POTENTIAL YAWN DETECTED", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show frame
    cv2.imshow('Mouth Landmark Test - B.Tech Mini-Project', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("\n✓ Test completed successfully!")