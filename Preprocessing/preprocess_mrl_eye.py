import os
import cv2

RAW_DIR = "Data/Eyes/raw/mrl/data/train"
OUT_DIR = "Data/Eyes/processed"
IMG_SIZE = 64

os.makedirs(OUT_DIR, exist_ok=True)

for label_folder in ["awake", "sleepy"]:
    input_dir = os.path.join(RAW_DIR, label_folder)
    output_dir = os.path.join(OUT_DIR, label_folder)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        save_path = os.path.join(output_dir, file)
        cv2.imwrite(save_path, (img * 255).astype("uint8"))
        count += 1

    print(f"{label_folder}: {count} images processed")

print("MRL preprocessing complete")
