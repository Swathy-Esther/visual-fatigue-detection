import cv2
import torch
import torch.nn as nn
import csv
from torchvision import transforms, models
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
import time
from fatigue_logic import FatigueDetector
from temporal.eye_temporal_logic import EyeTemporalTracker # Import the tracker

# --- 1. MODEL DEFINITIONS (Exact Matches) ---
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
logic = FatigueDetector()
# Instantiate the temporal tracker (Window size = 60 frames, approx 2-3 seconds)
tracker = EyeTemporalTracker(window_size=60)


# Initialize Calibration
pitch_offset = 0.0
calibrated = False

# Load Models
eye_model = EyeCNN().to(device)
eye_model.load_state_dict(torch.load("models/eye_cnn.pth", map_location=device))
eye_model.eval()

yawn_model = HybridYawnCNN().to(device)
yawn_model.load_state_dict(torch.load("models/yawn_hybrid_cnn.pth", map_location=device))
yawn_model.eval()

pose_model = PoseRegressor().to(device)
pose_model.load_state_dict(torch.load("models/pose_regressor.pth", map_location=device))
pose_model.eval()

# MediaPipe Task API Setup
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# UNIFIED PREPROCESSING PIPELINE
# MobileNetV2 strictly requires ImageNet Normalization
imagenet_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Eye model matches your train_eye_cnn.py (just scaling to 0-1)
eye_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

'''# Transforms
eye_tf = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.Grayscale(), 
    transforms.ToTensor() # Scaled to [0, 1] automatically. NO NORMALIZE!
])
yawn_pose_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# --- 3. BLINK RATE LOGIC SETUP ---
start_time = time.time()
blink_timestamps = []'''

def get_crop(img, landmarks, indices, pad=15):
    h, w, _ = img.shape
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_min, y_min = np.min(coords, axis=0); x_max, y_max = np.max(coords, axis=0)
    return img[max(0, y_min-pad):min(h, y_max+pad), max(0, x_min-pad):min(w, x_max+pad)]

frame_count = 0
LOG_INTERVAL = 5

# --- BEFORE THE WHILE LOOP ---
from collections import deque
pitch_buffer = deque(maxlen=10) # Smooths the last 10 frames of pitch


# --- 4. MAIN LOOP ---
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # INITIALIZE DEFAULTS FOR THIS FRAME (Safety first!)
    pitch, yawn_prob, perclos = 0.0, 0.0, 0.0
    status_msg, color = "SEARCHING...", (255, 255, 255)
    
    key = cv2.waitKey(1) & 0xFF
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    
    if result.face_landmarks:
        lms = result.face_landmarks[0]
        
        # A. Pose (Hybrid Model)
        pose_in = imagenet_tf(Image.fromarray(rgb_frame)).unsqueeze(0).to(device)
        with torch.no_grad():
            raw_pitch, _, _ = pose_model(pose_in).cpu().numpy()[0] * 90.0

        pitch_buffer.append(raw_pitch)
        smooth_pitch = sum(pitch_buffer) / len(pitch_buffer)
        pitch = smooth_pitch - pitch_offset if calibrated else smooth_pitch
        
        if key == ord('c'):
            pitch_offset = raw_pitch
            calibrated = True
        pitch = raw_pitch - pitch_offset if calibrated else raw_pitch

        # B. Eye (Custom CNN + CLAHE Preprocessing)
        eye_img = get_crop(rgb_frame, lms, [33, 133, 145, 159], pad=5)
        if eye_img.size > 0:
            # CLAHE: Handles lighting inconsistency in your room
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            eye_pre = clahe.apply(gray)
            
            eye_in = eye_tf(Image.fromarray(eye_pre)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(eye_model(eye_in), dim=1).cpu().numpy()[0]
                # Map to tracker: 1 = Closed, 0 = Open
                eye_state = 1 if probs[1] > 0.5 else 0
                tracker.update(eye_state) # TEMPORAL TRACKING
        # C. Yawn (Hybrid Model)
        yawn_prob = 0.0
        mouth_img = get_crop(rgb_frame, lms, [61, 291, 13, 14], pad=10)
        if mouth_img.size > 0:
            m_in = imagenet_tf(Image.fromarray(mouth_img)).unsqueeze(0).to(device)
            with torch.no_grad():
                yawn_prob = torch.softmax(yawn_model(m_in), dim=1).cpu().numpy()[0][1]

        # D. FEATURE FUSION & DECISION
        perclos = tracker.compute_perclos()
        total_score, alarm, warning, msg, color = logic.get_fusion_status(perclos, yawn_prob, pitch)
        

        if frame_count % LOG_INTERVAL == 0:
            with open('fusion_training_data.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                # Features: [PERCLOS, Yawn_Prob, Pitch] | Target: [total_score]
                writer.writerow([perclos, yawn_prob, pitch, total_score])

        # UI
        # --- SECTION E: UI DISPLAY (Corrected Spacing) ---
        # Status Message (Top)
        cv2.putText(frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Feature Stats (Red for visibility - increased Y-spacing)
        cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"YAWN PROB: {yawn_prob:.2f}", (30, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"PITCH: {pitch:.1f}", (30, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # DEBUG: Show the total score so you know how close you are to an alarm
        cv2.putText(frame, f"TOTAL SCORE: {total_score:.2f}", (30, 220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
    cv2.imshow('Visual Fatigue Detection v2.0', frame)
    if key == 27: break
    frame_count += 1

detector.close()
cap.release()
cv2.destroyAllWindows()