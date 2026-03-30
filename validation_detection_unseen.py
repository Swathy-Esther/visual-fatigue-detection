import cv2
import torch
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import deque
from temporal.eye_temporal_logic import EyeTemporalTracker
from temporal.yawn_temporal_logic import YawnTemporalTracker
from temporal.pose_temporal_logic import PoseTemporalTracker
import time

# --- 1. MODEL DEFINITIONS (Eye, Yawn, Pose only) ---
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2)
        )
    def forward(self, x): return self.net(x)

class HybridYawnCNN(nn.Module):
    def __init__(self):
        super(HybridYawnCNN, self).__init__()
        self.base = models.mobilenet_v2(weights=None)
        self.base.classifier = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2)
        )
    def forward(self, x): return self.base(x)

class PoseRegressor(nn.Module):
    def __init__(self):
        super(PoseRegressor, self).__init__()
        self.base = models.mobilenet_v2(weights=None)
        self.base.classifier = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 3)
        )
    def forward(self, x): return self.base(x)

# --- 2. INITIALIZATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eye_model = EyeCNN().to(device)
eye_model.load_state_dict(torch.load("models/eye_cnn.pth", map_location=device))
eye_model.eval()

yawn_model = HybridYawnCNN().to(device)
yawn_model.load_state_dict(torch.load("models/yawn_hybrid_cnn.pth", map_location=device))
yawn_model.eval()

pose_model = PoseRegressor().to(device)
pose_model.load_state_dict(torch.load("models/pose_regressor.pth", map_location=device))
pose_model.eval()

# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Trackers
eye_tracker = EyeTemporalTracker(window_size=60)
yawn_tracker = YawnTemporalTracker(window_size=150)
pose_tracker = PoseTemporalTracker(window_size=150)
pitch_buffer = deque(maxlen=10)
pitch_offset = 0.0
calibrated = False

# Transforms
imagenet_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
yawn_tf = transforms.Compose([
    transforms.Resize((100, 100)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
eye_tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

def get_crop(img, landmarks, indices, pad=15):
    h, w, _ = img.shape
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_min, y_min = np.min(coords, axis=0); x_max, y_max = np.max(coords, axis=0)
    return img[max(0, y_min-pad):min(h, y_max+pad), max(0, x_min-pad):min(w, x_max+pad)]

# --- RECORDING CONFIG ---
PHASE_DURATION = 60 # 1 minute per state
SAVE_FILE = "validation_data_unseen1.csv"
current_phase = 0 # 0=Wait, 1=Alert, 2=Fatigued, 3=Save/Done
start_time = 0
data_buffer = []

cap = cv2.VideoCapture(0)

print("Position yourself. Press 'S' to start the 2-minute validation recording.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Pre-calculate tracker scores for UI
    perclos = eye_tracker.compute_perclos()
    t_yawn = yawn_tracker.compute_score()
    t_pose = pose_tracker.compute_score()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    face_detected = len(result.face_landmarks) > 0

    if face_detected:
        lms = result.face_landmarks[0]
        
        # A. POSE
        face_img = get_crop(rgb_frame, lms, [10, 152, 234, 454], pad=40)
        if face_img.size > 0:
            pose_in = imagenet_tf(Image.fromarray(face_img)).unsqueeze(0).to(device)
            with torch.no_grad():
                raw_p = pose_model(pose_in).cpu().numpy()[0][0] * 90.0
            pitch_buffer.append(raw_p)
            smooth_p = sum(pitch_buffer) / len(pitch_buffer)
            pitch = smooth_p - pitch_offset if calibrated else 0.0
            pose_tracker.update(pitch)

        # B. EYE
        eye_img = get_crop(rgb_frame, lms, [33, 133, 145, 159], pad=5)
        if eye_img.size > 0:
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            eye_pre = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
            eye_in = eye_tf(Image.fromarray(eye_pre)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(eye_model(eye_in), dim=1).cpu().numpy()[0]
                eye_tracker.update(1 if probs[1] > 0.5 else 0)

        # C. YAWN
        mouth_img = get_crop(rgb_frame, lms, [61, 291, 13, 14], pad=25)
        if mouth_img.size > 0:
            m_in = yawn_tf(Image.fromarray(mouth_img)).unsqueeze(0).to(device)
            with torch.no_grad():
                y_prob = torch.softmax(yawn_model(m_in), dim=1).cpu().numpy()[0][1]
                yawn_tracker.update(y_prob)

    # --- PHASE LOGIC & RECORDING ---
    current_time = time.time()
    elapsed = current_time - start_time
    time_left = max(0, int(PHASE_DURATION - elapsed))

    if current_phase == 1: # ALERT
        msg, color = f"PHASE: ALERT | TIME: {time_left}s", (0, 255, 0)
        if face_detected: data_buffer.append([perclos, t_yawn, t_pose, 0])
        if elapsed >= PHASE_DURATION:
            current_phase = 2
            start_time = time.time()
    
    elif current_phase == 2: # FATIGUED
        msg, color = f"PHASE: FATIGUED | TIME: {time_left}s", (0, 0, 255)
        if face_detected: data_buffer.append([perclos, t_yawn, t_pose, 1])
        if elapsed >= PHASE_DURATION:
            current_phase = 3
    
    elif current_phase == 3: # SAVE
        with open(SAVE_FILE, mode='w', newline='') as f:
            csv.writer(f).writerows(data_buffer)
        print(f"Saved {len(data_buffer)} rows. RECORDING COMPLETE.")
        break
    
    else: # WAITING
        msg, color = "READY: PRESS 'S' TO START | 'C' TO CALIBRATE", (255, 255, 255)

    # --- UI & KEYS ---
    indicator_color = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.circle(frame, (30, 25), 8, indicator_color, -1)
    cv2.putText(frame, msg, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Data: {len(data_buffer)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.imshow("Validation Collector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and current_phase == 0:
        current_phase = 1
        start_time = time.time()
    elif key == ord('c'):
        pitch_offset = smooth_p if face_detected else 0.0
        calibrated = True
        print("Calibrated!")
    elif key == 27: break

cap.release()
cv2.destroyAllWindows()