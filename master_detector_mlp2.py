import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
from collections import deque
from temporal.eye_temporal_logic import EyeTemporalTracker
from temporal.yawn_temporal_logic import YawnTemporalTracker
from temporal.pose_temporal_logic import PoseTemporalTracker

# --- 1. UPDATED MODEL DEFINITION ---
class FatigueFusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(8, 1) # REMOVED: Sigmoid is now applied manually 
        )
    def forward(self, x): return self.net(x)

# (EyeCNN, HybridYawnCNN, and PoseRegressor remain unchanged )
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

fusion_model = FatigueFusionMLP().to(device)
fusion_model.load_state_dict(torch.load("models/fusion_mlp.pth", map_location=device))
fusion_model.eval()

# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Trackers & Buffers 
eye_tracker = EyeTemporalTracker(window_size=60)
yawn_tracker = YawnTemporalTracker(window_size=150)
pose_tracker = PoseTemporalTracker(window_size=150)

# NEW: Temporal Buffers for Fusion 
mlp_score_buffer = deque(maxlen=180) # 6 seconds of memory for smoothing
sustain_counter = 0 # Frames to keep alarm "latched"
THRESHOLD = 0.25 # Lower threshold as used in 93% accuracy test 

pitch_buffer = deque(maxlen=10)
pitch_offset = 0.0
calibrated = False

# Transforms (Remain unchanged )
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

# --- 3. MAIN LOOP ---
cap = cv2.VideoCapture(0)
print("System Starting... Press 'C' to calibrate. Press ESC to quit.")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Display placeholders
        msg, color = "SEARCHING FOR FACE...", (255, 255, 255)
        perclos, t_yawn, t_pose = 0.0, 0.0, 0.0
        
        key = cv2.waitKey(1) & 0xFF
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)
        
        if result.face_landmarks:
            lms = result.face_landmarks[0]
            
            # A. POSE
            face_img = get_crop(rgb_frame, lms, [10, 152, 234, 454], pad=40)
            if face_img.size > 0:
                pose_in = imagenet_tf(Image.fromarray(face_img)).unsqueeze(0).to(device)
                with torch.no_grad():
                    raw_p = pose_model(pose_in).cpu().numpy()[0][0] * 90.0
                pitch_buffer.append(raw_p)
                smooth_p = sum(pitch_buffer) / len(pitch_buffer)
                if key == ord('c'):
                    pitch_offset, calibrated = smooth_p, True
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

            # D. NEURAL FUSION WITH SMOOTHING 
            perclos = eye_tracker.compute_perclos()
            t_yawn = yawn_tracker.compute_score()
            t_pose = pose_tracker.compute_score()
            
            fusion_in = torch.tensor([[perclos, t_yawn, t_pose]], dtype=torch.float32).to(device)
            with torch.no_grad():
                # Manually apply Sigmoid since it was removed from the model class 
                raw_logits = fusion_model(fusion_in)
                current_mlp_score = torch.sigmoid(raw_logits).item()
            
            # Apply Moving Average 
            mlp_score_buffer.append(current_mlp_score)
            smoothed_score = sum(mlp_score_buffer) / len(mlp_score_buffer)

            # Apply Sustain Logic 
            if smoothed_score > THRESHOLD:
                sustain_counter = 90 # Lock "Fatigue" state for ~3 seconds
            
            if sustain_counter > 0:
                msg, color = "ALARM: FATIGUE DETECTED", (0, 0, 255)
                sustain_counter -= 1
            else:
                msg, color = "STATUS: ALERT", (0, 255, 0)

        # E. UI DISPLAY 
        cv2.putText(frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"T_YAWN: {t_yawn:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"T_POSE: {t_pose:.2f}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"SMOOTHED SCORE: {smoothed_score:.2f}", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Final Fatigue System Demo', frame)
        if key == 27: break

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close() # Prevents NoneType error