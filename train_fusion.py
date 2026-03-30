import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset

# 1. THE FUSION BRAIN ARCHITECTURE
'''class FatigueFusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),   # Input: [PERCLOS, T_YAWN, T_POSE]
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),    # Prevents overfitting to specific room lighting
            nn.Linear(8, 1),
            nn.Sigmoid()        # Output probability: 0.0 (Alert) to 1.0 (Critical)
        )
    def forward(self, x): return self.net(x)'''

class FatigueFusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(8, 1) # Removed nn.Sigmoid() from here
        )
    def forward(self, x): 
        return self.net(x)

def train_model():
    # 2. CONFIGURATION
    FILE_PATH = 'fusion_training_data_refined.csv'
    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Fusion MLP on: {device}")

    # 3. LOAD & PREPARE DATA
    try:
        # We skip the header if you added one in Excel
        df = pd.read_csv(FILE_PATH, header=None)
        # Ensure we only have the 4 essential columns
        X_data = df.iloc[:, :3].values.astype('float32') # Features
        y_data = df.iloc[:, 3].values.astype('float32').reshape(-1, 1) # Target Score
    except Exception as e:
        print(f"Error loading CSV: {e}. Ensure you saved it as a standard CSV in Excel.")
        return

    # Convert to Tensors
    X = torch.tensor(X_data)
    y = torch.tensor(y_data)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4. INITIALIZE TRAINING
    model = FatigueFusionMLP().to(device)

    # Give 'Fatigued' (Label 1) 5x more importance to fix the low recall
    pos_weight = torch.tensor([3.0]).to(device) 

    # This combines Sigmoid + Binary Cross Entropy
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. TRAINING LOOP
    print("Starting Training (approx. 30 seconds)...")
    for epoch in range(300):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {epoch_loss/len(loader):.6f}")

    # 6. SAVE THE BRAIN
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fusion_mlp.pth"))
    print("\n✓ SUCCESS: fusion_mlp.pth saved in /models folder!")

if __name__ == "__main__":
    train_model()