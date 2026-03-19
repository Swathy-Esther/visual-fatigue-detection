import os
import scipy.io as sio
import pandas as pd
import numpy as np

folder_path = 'Data/Posture/AFLW2000/' # Path to your extracted folder
data = []

# Loop through all files
for file in os.listdir(folder_path):
    if file.endswith('.mat'):
        # 1. Load the matlab file
        mat_contents = sio.loadmat(os.path.join(folder_path, file))
        
        # 2. Extract Pose_Para (First 3 are Pitch, Yaw, Roll)
        # Note: Values are in Radians, we convert to Degrees
        pose_para = mat_contents['Pose_Para'][0]
        pitch = pose_para[0] * (180 / np.pi)
        yaw = pose_para[1] * (180 / np.pi)
        roll = pose_para[2] * (180 / np.pi)
        
        # 3. Store image name and the 3 angles
        img_name = file.replace('.mat', '.jpg')
        data.append([img_name, pitch, yaw, roll])

# Save to CSV
df = pd.DataFrame(data, columns=['image', 'pitch', 'yaw', 'roll'])
df.to_csv('aflw_labels.csv', index=False)
print("Done! Labels saved to aflw_labels.csv")