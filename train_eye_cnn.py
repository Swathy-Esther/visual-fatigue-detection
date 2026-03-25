import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

# 1. ENHANCED EYE DATASET (With CLAHE and Augmentation)
class EyeDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Image is (1, 64, 64) in your .npy, we need (64, 64) for CV2
        img = self.X[idx].squeeze().astype(np.uint8)
        
        # A. APPLY CLAHE (To match live pipeline)
        img = self.clahe.apply(img)
        
        # Convert to PIL for Torchvision transforms
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return img, label

# 2. CNN DEFINITION (Keep exact match for loading)
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x): return self.net(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 3. LOAD RAW DATA
    X_raw = np.load("Data/Eyes/X.npy") # Shape (N, 1, 64, 64)
    y_raw = np.load("Data/Eyes/y.npy")

    # 4. DATA AUGMENTATION (Fixes "Brittle" model issue)
    train_tf = transforms.Compose([
        transforms.RandomRotation(15),      # Handles head tilts
        transforms.RandomHorizontalFlip(), # Left/Right eye symmetry
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handles lighting
        transforms.ToTensor()              # Automatically scales 0-1
    ])

    dataset = EyeDataset(X_raw, y_raw, transform=train_tf)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 5. INITIALIZE
    model = EyeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. TRAINING LOOP
    print("Starting Eye CNN Retraining...")
    for epoch in range(15): # 15 epochs is enough for 50k images
        model.train()
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # 7. SAVE
    torch.save(model.state_dict(), "models/eye_cnn.pth")
    print("Successfully retrained and saved eye_cnn.pth")