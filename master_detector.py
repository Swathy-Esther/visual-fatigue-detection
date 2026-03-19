import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
import time
from fatigue_logic import FatigueDetector

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

# Transforms
eye_tf = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.Grayscale(), 
    transforms.ToTensor() # Scaled to [0, 1] automatically. NO NORMALIZE!
])
yawn_pose_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# --- 3. BLINK RATE LOGIC SETUP ---
start_time = time.time()
blink_timestamps = []

def get_crop(img, landmarks, indices, pad=15):
    h, w, _ = img.shape
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_min, y_min = np.min(coords, axis=0); x_max, y_max = np.max(coords, axis=0)
    return img[max(0, y_min-pad):min(h, y_max+pad), max(0, x_min-pad):min(w, x_max+pad)]

# --- 4. MAIN LOOP ---
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    key = cv2.waitKey(1) & 0xFF

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    
    if result.face_landmarks:
        lms = result.face_landmarks[0]

        pitch, yaw, roll = 0.0, 0.0, 0.0
        eye_state = 1  # Assume Open
        yawn_prob = 0.0 # Assume No Yawn
        
        # A. Pose Regression
        pose_in = yawn_pose_tf(Image.fromarray(rgb_frame)).unsqueeze(0).to(device)
        with torch.no_grad():
            raw_pitch, yaw, roll = pose_model(pose_in).cpu().numpy()[0] * 90.0

        # --- CALIBRATION LOGIC (Must be after raw_pitch is calculated) ---
        if key == ord('c'):
            pitch_offset = raw_pitch
            calibrated = True
            print(f"Calibrated! Neutral Pitch set to: {pitch_offset:.1f}")

        # If calibrated, subtract the offset. If not, use raw_pitch.
        pitch = raw_pitch - pitch_offset if calibrated else raw_pitch

        # B. Eye Inference & Blink Counting
        eye_img = get_crop(rgb_frame, lms, [33, 133, 159, 145], pad=5)
        if eye_img.size > 0:
            # Resize just for display so you can actually see it
            debug_eye = cv2.resize(eye_img, (200, 200))
            cv2.imshow('Eye_Crop_Check', debug_eye)
        if eye_img.size > 0:
            eye_in = eye_tf(Image.fromarray(eye_img)).unsqueeze(0).to(device)
            with torch.no_grad():
                # 1. Get raw scores (logits)
                outputs = eye_model(eye_in)
                
                # 2. Convert to probabilities (Softmax)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

                if probs[0] > 0.50: 
                    eye_state = 1 # Open
                else:
                    eye_state = 0 # Closed

                print(f"RAW -> OpenScore: {probs[0]:.2f} | ClosedScore: {probs[1]:.2f} | State: {eye_state}")
                
                # 3. Get the prediction
                #eye_state = np.argmax(probs)
                
                # --- DEBUG: Print the actual probabilities ---
                # Class 0: Closed, Class 1: Open
                #print(f"RAW PROBS -> Closed: {probs[0]:.2f} | Open: {probs[1]:.2f}")
            
            # Blink logic: transition from Closed (0) to Open (1)
            if logic.last_eye_state == 0 and eye_state == 1:
                blink_timestamps.append(time.time())
            logic.last_eye_state = eye_state

        # C. Yawn Inference
        mouth_img = get_crop(rgb_frame, lms, [61, 291, 13, 14])
        if mouth_img.size > 0:
            m_in = yawn_pose_tf(Image.fromarray(mouth_img)).unsqueeze(0).to(device)
            with torch.no_grad():
                yawn_prob = torch.softmax(yawn_model(m_in), dim=1)[0][1].item()

        # D. Calculate BPM (Last 60 seconds)
        current_time = time.time()
        blink_timestamps = [t for t in blink_timestamps if current_time - t < 60]
        bpm = len(blink_timestamps)


        # Add this near the bottom of your loop
        print(f"DEBUG -> Eye: {eye_state} | Yawn Prob: {yawn_prob:.2f} | Pitch: {pitch:.1f}")

        # E. Decision & UI
        alarm, warning, msg = logic.check_fatigue(eye_state, yawn_prob, pitch, yaw)
        if alarm:
            color = (0, 0, 255)      # RED for Danger
        elif warning:
            color = (0, 165, 255)    # ORANGE for Yawn/Posture
        else:
            color = (0, 255, 0)
        
        cv2.putText(frame, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"BPM: {bpm}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow('Visual Fatigue Monitor', frame)
    if key == 27: break
detector.close() # Properly shut down the MediaPipe Task
cap.release()
cv2.destroyAllWindows()