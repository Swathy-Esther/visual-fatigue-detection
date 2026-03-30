import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. LOAD DATA
try:
    df = pd.read_csv('fusion_training_data_refined.csv', header=None)
    df.columns = ['PERCLOS', 'T_YAWN', 'T_POSE', 'Label']

    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0)
    df['Label'] = (df['Label'] > 0.5).astype(int)
    
    print("Cleaned Labels:", df['Label'].unique())
except Exception as e:
    print(f"Error: {e}")

print(f"Successfully loaded {len(df)} rows.")
print("Class Distribution:\n", df['Label'].value_counts())
print("Dataset Summary:")
print(df.groupby('Label').describe().T)

# 2. CORRELATION HEATMAP
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# 3. PAIR PLOT
# ✅ palette keys must match the actual values in the hue column
sns.pairplot(df, hue='Label', palette={0: 'green', 1: 'red'}, diag_kind='kde')
plt.suptitle("Feature Pair Plot: Alert (Green) vs Fatigued (Red)", y=1.02)
plt.savefig("pair_plot.png")
plt.show()

# 4. BOX PLOTS
plt.figure(figsize=(15, 5))
for i, col in enumerate(['PERCLOS', 'T_YAWN', 'T_POSE']):
    plt.subplot(1, 3, i+1)
    # ✅ Add hue='Label' and legend=False to fix both the warning and the ValueError
    sns.boxplot(x='Label', y=col, data=df,
                hue='Label', palette={0: 'green', 1: 'red'}, legend=False)
    plt.title(f"{col} Distribution")

plt.tight_layout()
plt.savefig("feature_boxplots.png")
plt.show()

print("\nAnalysis Complete! Check the saved PNG files.")