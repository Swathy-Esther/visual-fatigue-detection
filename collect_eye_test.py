import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# --- CONFIG ---
LABEL = "closed" # Change to "closed" for the second run
SAVE_PATH = f"Dataset_test/test_crops/eyes/{LABEL}"
os.makedirs(SAVE_PATH, exist_ok=True)

# MediaPipe Setup (Same as master_detector)
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
count = 0

print(f"Collecting 100 images for: {LABEL.upper()}")
print("Press 'S' to start/pause, 'ESC' to quit.")

recording = False
while count < 100:
    success, frame = cap.read()
    if not success: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.face_landmarks and recording:
        lms = result.face_landmarks[0]
        # Same crop logic as master_detector
        h, w, _ = frame.shape
        indices = [33, 133, 145, 159]; pad = 5
        coords = [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices]
        x_min, y_min = np.min(coords, axis=0); x_max, y_max = np.max(coords, axis=0)
        eye_crop = frame[max(0, y_min-pad):min(h, y_max+pad), max(0, x_min-pad):min(w, x_max+pad)]
        
        if eye_crop.size > 0:
            eye_resized = cv2.resize(eye_crop, (64, 64))
            cv2.imwrite(f"{SAVE_PATH}/eye_{count}.jpg", eye_resized)
            count += 1
            cv2.putText(frame, f"Saved: {count}/100", (50, 50), 1, 1, (0, 255, 0), 2)

    cv2.imshow("Collector", frame)
    key = cv2.waitKey(1)
    if key == ord('s'): recording = not recording
    if key == 27: break

cap.release(); cv2.destroyAllWindows()