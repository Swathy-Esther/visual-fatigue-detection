import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
from torch.optim import lr_scheduler

# 1. Custom Dataset for AFLW2000
class AFLWDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Labels: Pitch, Yaw, Roll
        labels_array = (self.data.iloc[idx, 1:4].values.astype('float32')) / 90.0
        labels = torch.tensor(labels_array, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

# 2. Hybrid Regression Model
class PoseRegressor(nn.Module):
    def __init__(self):
        super(PoseRegressor, self).__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze base layers
        for param in self.base.parameters():
            param.requires_grad = False

        # UNFREEZE the last 4 blocks of MobileNet for fine-tuning
        for param in self.base.features[-4:].parameters():
            param.requires_grad = True
            
        # Regression Head (Output: 3 continuous values)
        self.base.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3) # Pitch, Yaw, Roll
        )

    def forward(self, x):
        return self.base(x)

if __name__ == "__main__":
    # Config
    CSV_FILE = 'aflw_labels.csv'
    IMG_DIR = 'Data/Posture/AFLW2000/'
    BATCH_SIZE = 32
    EPOCHS = 25
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transform (Must match ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = AFLWDataset(CSV_FILE, IMG_DIR, transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PoseRegressor().to(device)
    criterion = nn.MSELoss() # Best for Regression
    #optimizer = optim.Adam(model.base.classifier.parameters(), lr=0.0001)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    print(f"Starting Regression Training on {device}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # --- BATCH LOOP (Processes images in groups of 32) ---
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # REMOVED scheduler.step() from here!

        # --- EPOCH END (Runs once after all batches are done) ---
        
        # 1. Update the learning rate now
        scheduler.step() 
        
        # 2. Print status
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - MSE: {avg_loss:.4f} - LR: {current_lr:.6f}")

    # --- FINAL SAVE ---
    torch.save(model.state_dict(), "models/pose_regressor.pth")
    print("Pose Model Saved!")
    
    '''for epoch in range(EPOCHS):
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

            scheduler.step() 
    
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} done. Current Learning Rate: {current_lr:.6f}")
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] Mean Squared Error: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "models/pose_regressor.pth")
    print("Pose Model Saved!")'''