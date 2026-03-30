import pandas as pd
import numpy as np

# 1. LOAD DATA
INPUT_FILE = 'fusion_training_data.csv'
OUTPUT_FILE = 'fusion_training_data_refined.csv'

try:
    df = pd.read_csv(INPUT_FILE, header=None)
    df.columns = ['PERCLOS', 'T_YAWN', 'T_POSE', 'Label']
    print(f"Initial Dataset: {len(df)} rows")
except Exception as e:
    print(f"Error: {e}")
    exit()

# 2. DATA CLEANING PIPELINE

# --- A. Outlier Removal (The T_POSE Fix) ---
# If Label is Alert (0) but T_POSE is > 0.4 (looking at keyboard/phone), 
# it's confusing the model. We remove these noisy rows.
before_count = len(df)
df = df.drop(df[(df['Label'] < 0.5) & (df['T_POSE'] > 0.4)].index)
print(f"Removed {before_count - len(df)} T_POSE outliers (Alert but head tilted).")

# --- B. Quality Yawn Filter (The Label Correction) ---
# Logic: If mouth is wide (>0.5) AND eyes are squinting (>0.2), 
# it's a Fatigue Yawn. Force Label to 1.
yawn_mask = (df['T_YAWN'] > 0.5) & (df['PERCLOS'] > 0.2)
df.loc[yawn_mask, 'Label'] = 1
print(f"Refined {yawn_mask.sum()} yawn labels to 'Fatigued'.")

# --- C. Dead-Zone Removal ---
# If all features are exactly 0, MediaPipe likely failed. 
# These rows add no value.
before_count = len(df)
df = df.drop(df[(df['PERCLOS'] == 0) & (df['T_YAWN'] == 0) & (df['T_POSE'] == 0)].index)
print(f"Removed {before_count - len(df)} dead-zone rows (All zeros).")

# --- D. Final Binarization ---
# Ensure Label is strictly 0 or 1 integers
df['Label'] = (df['Label'] > 0.5).astype(int)

# 3. SAVE AND SUMMARY
df.to_csv(OUTPUT_FILE, index=False, header=None)
print(f"\n✓ SUCCESS: Refined dataset saved as '{OUTPUT_FILE}'")
print(f"Final Count: {len(df)} rows")
print("New Class Balance:\n", df['Label'].value_counts())