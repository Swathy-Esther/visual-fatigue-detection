import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import cv2
import os
from temporal.eye_temporal_logic import EyeTemporalTracker

# ----------------------------
# Eye CNN definition
# ----------------------------
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Captures basic edges
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 2: Captures eye shapes (circle vs flat line)
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 3: Deeper features for fatigue detection
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Prevents the model from just "memorizing"
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)
# ----------------------------
# Load model
# ----------------------------
model = EyeCNN()
model.load_state_dict(torch.load("models/eye_cnn.pth", map_location="cpu"))
model.eval()

# ----------------------------
# Initialize temporal tracker
# ----------------------------
tracker = EyeTemporalTracker(window_size=20)

# ----------------------------
# Simulated frame sequence
# ----------------------------
# Use sleepy folder to simulate fatigue
#folder = "Data/Eyes/processed/awake"
#files = sorted(os.listdir(folder))[:30]  # simulate 30 frames
#awake frames
awake_folder = "Data/Eyes/processed/awake"
awake_files = sorted(os.listdir(awake_folder))[:20]

#sleepy frames
sleepy_folder = "Data/Eyes/processed/sleepy"
sleepy_files = sorted(os.listdir(sleepy_folder))[:20]

print("Starting eye fatigue simulation...\n")
test_files = [(awake_folder, f) for f in awake_files] + [(sleepy_folder, f) for f in sleepy_files]

for i, (folder_path, file) in enumerate(test_files):
    img_path = os.path.join(folder_path, file)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0

    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

        
    # 0 = OPEN, 1 = CLOSED
    print(f"Raw eye prediction: {pred}")
    tracker.update(pred)

    fatigue_state, perclos = tracker.fatigue_level()

    print(
        f"Frame {i+1:02d} | "
        f"Eye={'CLOSED' if pred else 'OPEN'} | "
        f"PERCLOS={perclos:.2f} | "
        f"Fatigue={fatigue_state}"
    )
