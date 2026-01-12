import os
import cv2

# Path to MRL awake images
awake_dir = "Data/Eyes/raw/mrl/data/train/awake"

# List image files
files = os.listdir(awake_dir)
print("Number of awake images:", len(files))

# Read first image
img_path = os.path.join(awake_dir, files[0])
img = cv2.imread(img_path)

# Check if image loaded
if img is None:
    print("Failed to load image")
else:
    print("Image shape:", img.shape)
    cv2.imshow("MRL Awake Eye", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

