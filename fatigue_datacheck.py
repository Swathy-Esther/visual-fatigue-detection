import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('validation_data_unseen.csv', header=None)
df.columns = ['PERCLOS', 'T_YAWN', 'T_POSE', 'Ground_Truth']

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['PERCLOS'], label='PERCLOS (Eyes)', alpha=0.7)
plt.plot(df['T_YAWN'], label='T_YAWN (Mouth)', alpha=0.7)
plt.fill_between(range(len(df)), 0, 1, where=df['Ground_Truth']==1, 
                 color='red', alpha=0.1, label='Target: Fatigued Area')

plt.title("Feature Behavior During Validation")
plt.xlabel("Frame Number")
plt.ylabel("Normalized Score (0.0 - 1.0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()