import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# CNN definition (must match training)
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = EyeCNN()
model.load_state_dict(torch.load("models/eye_cnn.pth", map_location="cpu"))
model.eval()

# Load a test image (change path if needed)
img_path = "Data/Eyes/processed/awake/" + os.listdir("Data/Eyes/processed/awake")[0]

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64))
img = img / 255.0

# Convert to tensor
tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(tensor)
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(probs).item()

label = "OPEN" if pred == 0 else "CLOSED"
confidence = probs[0][pred].item()

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.4f}")
