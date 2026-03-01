import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms
from PIL import Image
from train_yawn_hybrid import HybridYawnCNN 

# 1. Setup Paths and Model
MODEL_PATH = "models/yawn_hybrid_cnn.pth"
LANDMARKER_PATH = "models/face_landmarker.task"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the "Brain"
model = HybridYawnCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 2. Image Pre-processing
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. MediaPipe Tasks API Setup
base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
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
    return crop

# 4. Live Webcam Loop
cap = cv2.VideoCapture(0)
print(f"Starting Live Yawn Detection on {device}...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # MediaPipe Tasks requires mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    
    label = "Searching for Face..."
    color = (255, 255, 255)

    if detection_result.face_landmarks:
        # Get the first face found
        face_lms = detection_result.face_landmarks[0]
        mouth_crop = get_mouth_crop(frame, face_lms)
        
        if mouth_crop.size > 0:
            # Prepare for CNN
            mouth_img = Image.fromarray(cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(mouth_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                
            # Folder order: no_yawn=0, yawn=1
            if predicted.item() == 1:
                label = "YAWN DETECTED"
                color = (0, 0, 255) # Red
            else:
                label = "Normal"
                color = (0, 255, 0) # Green
            
            cv2.imshow("Mouth Crop (CNN Input)", cv2.resize(mouth_crop, (150, 150)))

    cv2.putText(frame, f"Status: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Driver Monitoring - Yawn Module", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()