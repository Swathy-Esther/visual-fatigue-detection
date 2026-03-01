import cv2
import os
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- SETTINGS ---
VIDEO_DIR = "Data/Yawn/YawDD/Mirror"
SAVE_DIR = "Data/Yawn/processed_yawn"
os.makedirs(f"{SAVE_DIR}/yawn", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/no_yawn", exist_ok=True)

# --- MEDIAPIPE SETUP ---
model_path = 'models/face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
detector = vision.FaceLandmarker.create_from_options(options)

def get_mouth_crop(image, landmarks):
    h, w, _ = image.shape
    mouth_indices = [61, 291, 13, 14, 17, 0] 
    pts = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in mouth_indices])
    x, y, mw, mh = cv2.boundingRect(pts)
    pad = 25
    crop = image[max(0, y-pad):min(h, y+mh+pad), max(0, x-pad):min(w, x+mw+pad)]
    return cv2.resize(crop, (100, 100))

count_yawn = len(os.listdir(f"{SAVE_DIR}/yawn"))
count_normal = len(os.listdir(f"{SAVE_DIR}/no_yawn"))

print(f"DATABASE READY | Y: {count_yawn} | N: {count_normal}")
print("Controls: [y] Save Yawn | [n] Save Normal | [a] AUTO-YAWN | [p] AUTO-NORMAL | [s] Skip | [q] Quit")

for root, dirs, files in os.walk(VIDEO_DIR):
    for video_file in files:
        if not video_file.lower().endswith(('.avi', '.mp4')): continue
        
        cap = cv2.VideoCapture(os.path.join(root, video_file))
        auto_mode = None # Can be 'yawn', 'normal', or None
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
                img_name = f"{video_file}_f{frame_count}.jpg"
                
                # --- AUTO-LABELING LOGIC ---
                if auto_mode == 'yawn':
                    cv2.imwrite(f"{SAVE_DIR}/yawn/{img_name}", mouth_img)
                    count_yawn += 1
                elif auto_mode == 'normal':
                    cv2.imwrite(f"{SAVE_DIR}/no_yawn/{img_name}", mouth_img)
                    count_normal += 1

                # Display with Mode Indicator
                disp = frame.copy()
                mode_color = (0, 0, 255) if auto_mode else (0, 255, 0)
                cv2.putText(disp, f"MODE: {auto_mode or 'MANUAL'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
                cv2.imshow("Video", cv2.resize(disp, (640, 480)))
                cv2.imshow("Mouth", mouth_img)

                # Control Logic
                wait_time = 30 if auto_mode else 0 # Run automatically if in auto_mode
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('y'): 
                    cv2.imwrite(f"{SAVE_DIR}/yawn/{img_name}", mouth_img)
                    count_yawn += 1
                elif key == ord('n'): 
                    cv2.imwrite(f"{SAVE_DIR}/no_yawn/{img_name}", mouth_img)
                    count_normal += 1
                elif key == ord('a'): auto_mode = 'yawn' if auto_mode != 'yawn' else None
                elif key == ord('p'): auto_mode = 'normal' if auto_mode != 'normal' else None
                elif key == ord('s'): auto_mode = None # Reset and skip
                elif key == ord('q'): exit()
                
                print(f"Yawn: {count_yawn} | Normal: {count_normal} | Mode: {auto_mode}", end='\r')
        cap.release()
cv2.destroyAllWindows()