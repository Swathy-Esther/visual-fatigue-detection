import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# --- 1. MODEL DEFINITIONS ---
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2)
        )
    def forward(self, x): return self.net(x)

class HybridYawnCNN(nn.Module):
    def __init__(self):
        super(HybridYawnCNN, self).__init__()
        self.base = models.mobilenet_v2(weights=None)
        self.base.classifier = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2)
        )
    def forward(self, x): return self.base(x)

# --- 2. EVALUATION LOGIC ---
def run_evaluation(model, data_path, transform, model_name, is_gray=False, flip_labels=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []

    print(f"\n--- Evaluating {model_name} ---")
    with torch.no_grad():
        for inputs, labels in loader:
            if is_gray:
                # Convert 3-channel RGB to 1-channel Gray for EyeCNN
                inputs = inputs.mean(dim=1, keepdim=True)
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # THE FIX: Only flip labels for EyeCNN if the logic is inverted
            if flip_labels:
                preds = 1 - preds
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Report & Matrix
    target_names = dataset.classes
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(f"cm_{model_name.lower()}.png")
    plt.show()

# --- 3. EXECUTION ---
if __name__ == "__main__":
    # A. EVALUATE EYES (Requires label flip: 0=Open, 1=Closed vs Folder C, O)
    eye_model = EyeCNN()
    eye_model.load_state_dict(torch.load("models/eye_cnn.pth", map_location='cpu'))
    eye_tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    
    if os.path.exists("Dataset_test/test_crops/eyes"):
        run_evaluation(eye_model, "Dataset_test/test_crops/eyes", eye_tf, "Eye_CNN", is_gray=True, flip_labels=True)
    else:
        print("Eye test directory not found.")

    # B. EVALUATE YAWN (No flip needed: Folder N, Y matches 0, 1)
    yawn_model = HybridYawnCNN()
    yawn_model.load_state_dict(torch.load("models/yawn_hybrid_cnn.pth", map_location='cpu'))
    yawn_tf = transforms.Compose([
        transforms.Resize((100, 100)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if os.path.exists("Dataset_test/test_crops/mouth"):
        run_evaluation(yawn_model, "Dataset_test/test_crops/mouth", yawn_tf, "Yawn_CNN", flip_labels=False)
    else:
        print("Mouth test directory not found.")