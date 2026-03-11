import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


# 4. Hybrid Architecture (MobileNetV2 + Custom Head)
class HybridYawnCNN(nn.Module):
    def __init__(self):
        super(HybridYawnCNN, self).__init__()
        # Use Weights instead of pretrained=True (Modern PyTorch)
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze the base layers (Keep the pre-trained knowledge)
        for param in self.base.parameters():
            param.requires_grad = False
            
        # Replace the head with a Custom 3-Layer Classifier
        self.base.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Output: [No_Yawn, Yawn]
        )

    def forward(self, x):
        return self.base(x)

if __name__ == "__main__":


    # 1. Configuration
    DATA_DIR = "Data/Yawn/processed_yawn"
    MODEL_SAVE_PATH = "models/yawn_hybrid_cnn.pth"
    BATCH_SIZE = 32
    EPOCHS = 20 
    LEARNING_RATE = 0.001

    # 5. Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridYawnCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.base.classifier.parameters(), lr=LEARNING_RATE)


    # 2. Data Augmentation (This makes your 2200 images act like 5000)
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(), # Flips mouth left-right
        transforms.RandomRotation(15),      # Handles head tilts
        transforms.ColorJitter(brightness=0.2), # Handles different lighting
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # 3. Load & Split Dataset
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Total Images: {len(dataset)} | Training: {train_size} | Validation: {val_size}")


    # 6. Training Loop
    history = {'train_loss': [], 'val_acc': []}

    print(f"\nStarting training on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        history['train_loss'].append(running_loss/len(train_loader))
        history['val_acc'].append(acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

    # 7. Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining Complete! Model saved to {MODEL_SAVE_PATH}")