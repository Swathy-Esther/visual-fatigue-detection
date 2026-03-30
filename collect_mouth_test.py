import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. CONFIGURATION ---
LABEL = "not_yawn"  # Switch to "not_yawn" for the second collection run
SAVE_PATH = f"Dataset_test/test_crops/mouth/{LABEL}"
os.makedirs(SAVE_PATH, exist_ok=True)

# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
count = 0
recording = False

print(f"Target: {LABEL.upper()} | Press 'S' to start/pause capturing, 'ESC' to quit.")

while count < 50:  # Stops once you have 50 clean images
    success, frame = cap.read()
    if not success: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.face_landmarks and recording:
        lms = result.face_landmarks[0]
        h, w, _ = frame.shape
        
        # MOUTH INDICES AND PADDING
        indices = [61, 291, 13, 14] 
        pad = 25 
        
        coords = [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices]
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        
        # Extract and Resize Mouth
        mouth_crop = frame[max(0, y_min-pad):min(h, y_max+pad), max(0, x_min-pad):min(w, x_max+pad)]
        
        if mouth_crop.size > 0:
            mouth_resized = cv2.resize(mouth_crop, (100, 100)) # Standard for Yawn CNN
            cv2.imwrite(f"{SAVE_PATH}/mouth_{count}.jpg", mouth_resized)
            count += 1
            cv2.putText(frame, f"Saved: {count}/50", (50, 50), 1, 1, (0, 255, 0), 2)

    status = "RECORDING" if recording else "PAUSED (Press S)"
    cv2.putText(frame, status, (50, 80), 1, 1, (255, 255, 0), 2)
    cv2.imshow("Mouth Collector", frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'): recording = not recording
    if key == 27: break

cap.release()
cv2.destroyAllWindows()
print(f"Successfully saved 50 images to {SAVE_PATH}")