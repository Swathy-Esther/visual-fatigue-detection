import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Define CNN
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Captures basic edges
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 2: Captures eye shapes (circle vs flat line)
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 3: Deeper features for fatigue detection
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Prevents the model from just "memorizing"
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
        # Load data
    X = np.load("Data/Eyes/X.npy")
    y = np.load("Data/Eyes/y.npy")

    X = X.astype('float32')/255.0

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = EyeCNN()
    weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # ----------------------------
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    for epoch in range(12):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    print("Training complete")
    torch.save(model.state_dict(), "models/eye_cnn.pth")
    print("Eye CNN model saved")


