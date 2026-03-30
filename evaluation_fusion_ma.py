import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. SETUP
# Ensure this is your UNSEEN validation file, not the training file!
VAL_FILE = 'validation_data_unseen1.csv' 
MODEL_PATH = "models/fusion_mlp.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. LOAD DATA
df = pd.read_csv(VAL_FILE, header=None)
X = torch.tensor(df.iloc[:, :3].values.astype('float32')).to(device)
y_true = df.iloc[:, 3].values.astype('int')

# 3. LOAD MODEL (Ensure architecture matches train_fusion.py)
from train_fusion import FatigueFusionMLP 
model = FatigueFusionMLP().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 4. INFERENCE (The Critical Change)
with torch.no_grad():
    logits = model(X)
    # Since model has no Sigmoid, we apply it here manually:
    y_probs = torch.sigmoid(logits).cpu().numpy().flatten()

# 5. MOVING AVERAGE (Temporal Smoothing)
window_size = 180 # ~1.5 seconds of "memory"
y_probs_smoothed = pd.Series(y_probs).rolling(window=window_size, min_periods=1).mean()

# 6. EVALUATE
threshold = 0.25 # Start with 0.5 now that the model is weighted
y_pred = (y_probs_smoothed > threshold).astype(int)

# ... after calculating y_pred ...

# NEW: Sustain Logic (The "Safety Latch")
sustain_frames = 90 # Stay 'Fatigued' for at least 3 seconds after a detection
y_pred_sustained = y_pred.copy()

for i in range(len(y_pred)):
    if y_pred[i] == 1:
        # If we detect fatigue, force the next 90 frames to be fatigue too
        y_pred_sustained[i : i + sustain_frames] = 1

# Now print the metrics for the Sustained version

print("\n--- FINAL EVALUATION (Weighted Model + Moving Average +Sustained alarm) ---")
print(classification_report(y_true, y_pred_sustained, target_names=['Alert', 'Fatigued']))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_sustained))

# 7. VISUALIZE
plt.figure(figsize=(12, 5))
plt.plot(y_probs_smoothed, label='Smoothed Fatigue Probability', color='blue')
plt.axhline(y=threshold, color='red', linestyle='--', label='Alarm Threshold')
plt.fill_between(range(len(y_true)), 0, 1, where=y_true==1, color='red', alpha=0.1, label='Actual Fatigue Area')
plt.title("Performance on Unseen Data")
plt.legend()
plt.show()