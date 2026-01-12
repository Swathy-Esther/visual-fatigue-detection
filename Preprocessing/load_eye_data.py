import os
import cv2
import numpy as np

DATA_DIR = "Data/Eyes/processed"
IMG_SIZE = 64

X = []
y = []

label_map = {
    "awake": 0,   # eyes open
    "sleepy": 1   # eyes closed
}

for label_name, label in label_map.items():
    folder = os.path.join(DATA_DIR, label_name)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Add channel dimension for CNN: (N, 1, 64, 64)
X = X.reshape(-1, 1, IMG_SIZE, IMG_SIZE)

print("Data loaded successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)
np.save("Data/Eyes/X.npy", X)
np.save("Data/Eyes/y.npy", y)

