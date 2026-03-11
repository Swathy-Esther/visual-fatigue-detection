import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms
from PIL import Image
from train_yawn_hybrid import HybridYawnCNN 

# 1. Setup
MODEL_PATH = "models/yawn_hybrid_cnn.pth"
LANDMARKER_PATH = "models/face_landmarker.task"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridYawnCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
detector = vision.FaceLandmarker.create_from_options(options)

def get_mouth_crop(image, landmarks):
    h, w, _ = image.shape
    mouth_indices = [61, 291, 13, 14, 17, 0] 
    pts = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in mouth_indices])
    x, y, mw, mh = cv2.boundingRect(pts)
    pad = 25
    return image[max(0, y-pad):min(h, y+mh+pad), max(0, x-pad):min(w, x+mw+pad)]

# 2. Variables for Temporal Smoothing
yawn_history = []
BUFFER_SIZE = 20 # Increased slightly for better stability

cap = cv2.VideoCapture(0)
print(f"Starting Live Yawn Detection on {device}...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)
        
        label = "Normal"
        color = (0, 255, 0)

        if detection_result.face_landmarks:
            mouth_crop = get_mouth_crop(frame, detection_result.face_landmarks[0])
            
            if mouth_crop.size > 0:
                mouth_img = Image.fromarray(cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(mouth_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    # Get probabilities to see how "sure" the model is
                    probs = torch.nn.functional.softmax(output, dim=1)
                    conf, predicted = torch.max(probs, 1)
                    
                prediction_idx = predicted.item()
                yawn_history.append(prediction_idx)
                if len(yawn_history) > 15:
                    yawn_history.pop(0)

                # DEBUG: Print to terminal to see what's happening
                # If these numbers are wrong (e.g. 1 when you are normal), we swap them.
                print(f"Pred: {prediction_idx} | Conf: {conf.item():.2f} | Buffer Avg: {sum(yawn_history)/len(yawn_history):.2f}", end='\r')

                # Only evaluate if buffer is full and 80% of frames are yawns
                if len(yawn_history) > 5 and (sum(yawn_history) / len(yawn_history)) > 0.5:
                    label = "CONFIRMED YAWN"
                    color = (0, 0, 255)
                
                cv2.imshow("Mouth Crop", cv2.resize(mouth_crop, (150, 150)))

        cv2.putText(frame, f"Status: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Driver Monitoring", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    print("\nClosing safely...")
    detector.close() # Fixes the TypeError
    cap.release()
    cv2.destroyAllWindows()