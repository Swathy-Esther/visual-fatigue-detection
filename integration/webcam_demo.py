import cv2
import torch
import numpy as np
import os
import sys

# Ensure the script can find your 'models' and 'temporal' folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.train_eye_cnn import EyeCNN
from temporal.eye_temporal_logic import EyeTemporalTracker

# 1. Load Detectors and Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

model = EyeCNN()
model.load_state_dict(torch.load("models/eye_cnn.pth", map_location=torch.device('cpu')))
model.eval()

tracker_obj = EyeTemporalTracker(window_size=20)

# 2. Initialize Tracker Variables (NEW)
eye_tracker = cv2.TrackerCSRT_create()
tracking_active = False
frame_count = 0
RESET_INTERVAL = 150  # Re-detect eyes every ~5-6 seconds to prevent drift

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]

        # --- TRACKING LOGIC START ---
        
        # Auto-reset: Every 150 frames, force a re-detection to stay accurate
        if frame_count % RESET_INTERVAL == 0:
            tracking_active = False

        if not tracking_active:
            # SEARCH MODE: Use Haar Cascade
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            if len(eyes) > 0:
                (ex, ey, ew, eh) = eyes[0]
                # Reset and initialize the tracker on the new eye coordinates
                eye_tracker = cv2.TrackerCSRT_create()
                eye_tracker.init(frame, (x + ex, y + ey, ew, eh))
                tracking_active = True
                current_frame_state = 0
        else:
            # TRACKING MODE: Follow the eye found previously
            success, bbox = eye_tracker.update(frame)
            if success:
                tx, ty, tw, th = [int(v) for v in bbox]
                
                # Crop and Predict
                eye_img = gray[max(0, ty):ty+th, max(0, tx):tx+tw]
                if eye_img.size > 0:
                    eye_img = cv2.resize(eye_img, (64, 64)).astype('float32') / 255.0
                    tensor = torch.tensor(eye_img).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        output = model(tensor)
                        prediction = torch.argmax(output, dim=1).item()
                    
                    current_frame_state = prediction
                    
                    # Cyan box indicates active tracking
                    color = (0, 0, 255) if prediction == 1 else (255, 255, 0)
                    cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), color, 2)
            else:
                # If tracker fails (success=False), switch back to Search Mode
                tracking_active = False
                current_frame_state = 1 # Assume eyes closed if lost

        # --- TRACKING LOGIC END ---

        tracker_obj.update(current_frame_state)
        fatigue_state, perclos = tracker_obj.fatigue_level()

        # Display UI
        display_color = (0, 0, 255) if fatigue_state == "HIGH FATIGUE" else (0, 255, 0)
        cv2.putText(frame, f"State: {fatigue_state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)

    cv2.imshow('B.Tech Mini-Project: Fatigue Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()