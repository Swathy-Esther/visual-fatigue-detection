import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your Kochi dataset
df = pd.read_csv('validation_data_unseen.csv', header=None)
X = df.iloc[:, :3].values.astype('float32')
y = df.iloc[:, 3].values.astype('float32')

# Convert continuous scores to binary classes for the report (0=Alert, 1=Fatigued)
y_class = (y > 0.5).astype(int)

# 80-20 Split
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Load your trained model
from train_fusion import FatigueFusionMLP # Import your class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FatigueFusionMLP().to(device)
model.load_state_dict(torch.load("models/fusion_mlp.pth"))
model.eval()

# Run Prediction
X_test_tensor = torch.tensor(X_test).to(device)
with torch.no_grad():
    outputs = model(X_test_tensor).cpu().numpy()
    y_pred = (outputs > 0.3).astype(int)

# PRINT REPORT
print("--- FUSION MLP PERFORMANCE ---")
print(classification_report(y_test, y_pred, target_names=['Alert', 'Fatigued']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create the plot
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Alert', 'Fatigued'], yticklabels=['Alert', 'Fatigued'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Fusion MLP: Confusion Matrix')
plt.savefig('fusion_confusion_matrix.png') # Saves a nice image for your report
plt.show()