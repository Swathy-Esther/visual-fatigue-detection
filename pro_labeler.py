import cv2
import os
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. SETTINGS ---
VIDEO_DIR = "Data/Yawn/YawDD/Dash"
SAVE_DIR = "Data/Yawn/processed_yawn"
os.makedirs(f"{SAVE_DIR}/yawn", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/no_yawn", exist_ok=True)

# --- 2. MEDIAPIPE SETUP ---
model_path = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.IMAGE)
detector = vision.FaceLandmarker.create_from_options(options)

def get_mouth_crop(image, landmarks):
    h, w, _ = image.shape
    mouth_indices = [61, 291, 13, 14, 17, 0] 
    pts = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in mouth_indices])
    x, y, mw, mh = cv2.boundingRect(pts)
    pad = 25
    crop = image[max(0, y-pad):min(h, y+mh+pad), max(0, x-pad):min(w, x+mw+pad)]
    return cv2.resize(crop, (100, 100))

# Get initial counts
count_yawn = len(os.listdir(f"{SAVE_DIR}/yawn"))
count_normal = len(os.listdir(f"{SAVE_DIR}/no_yawn"))

print("-" * 50)
print(f"DATABASE READY")
print(f"Current Yawn Images: {count_yawn}")
print(f"Current Normal Images: {count_normal}")
print("Controls: [y] Save Yawn | [n] Save Normal | [s] Skip | [q] Quit")
print("-" * 50)

# --- 3. RECURSIVE LOOP ---
for root, dirs, files in os.walk(VIDEO_DIR):
    for video_file in files:
        if not video_file.lower().endswith(('.avi', '.mp4')):
            continue
        
        video_path = os.path.join(root, video_file)
        print(f"\n>>> NOW OPENING: {video_file}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % 5 != 0: continue 

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks:
                mouth_img = get_mouth_crop(frame, detection_result.face_landmarks[0])
                
                # Show the images
                cv2.imshow("Original Video", cv2.resize(frame, (640, 480)))
                cv2.imshow("Mouth Patch", mouth_img)
                
                key = cv2.waitKey(0) & 0xFF
                img_name = f"{video_file}_f{frame_count}.jpg"
                
                if key == ord('y'):
                    cv2.imwrite(f"{SAVE_DIR}/yawn/{img_name}", mouth_img)
                    count_yawn += 1
                    print(f"Saved YAWN! Total: {count_yawn}", end='\r')
                elif key == ord('n'):
                    cv2.imwrite(f"{SAVE_DIR}/no_yawn/{img_name}", mouth_img)
                    count_normal += 1
                    print(f"Saved NORMAL! Total: {count_normal}", end='\r')
                elif key == ord('s'):
                    # No print needed for skip to keep terminal clean
                    continue
                elif key == ord('q'):
                    print("\nQuitting script...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
        
        cap.release()

cv2.destroyAllWindows()