import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CrosSimDataset(Dataset):
    def __init__(self, root_folder, text_folder):
        """
        Initialize the dataset by finding all .npy files and loading corresponding text and embedding files.
        Args:
            root_folder: Path to the root folder containing .npy files.
            text_folder: Path to the folder containing corresponding .txt files.
        """
        self.root_folder = root_folder
        self.text_folder = text_folder
        self.files = []
        self._gather_files()
    
    def _gather_files(self):
        """
        Gather all .npy files and find their corresponding .txt and .pt files.
        """
        for root, _, filenames in os.walk(self.root_folder):
            for filename in filenames:
                if filename.endswith('.npy'):
                    npy_path = os.path.join(root, filename)
                    # Derive the corresponding text and embedding paths
                    relative_path = os.path.relpath(npy_path, self.root_folder)  # Get the relative path from root_folder
                    text_path = os.path.join(self.text_folder, relative_path[:-4] + '.txt')  # Replace .npy with .txt
                    embedding_path = os.path.join(self.text_folder, relative_path[:-4] + '_gtr.pt')  # Replace .npy with .pt

                    # Check if the corresponding text and embedding files exist
                    if os.path.exists(text_path) and os.path.exists(embedding_path):
                        self.files.append({"npy": npy_path, "txt": text_path, "embedding": embedding_path})
                    else:
                        print(f"Missing text or embedding file for: {npy_path}")
                        print(npy_path,text_path,embedding_path)

    def __len__(self):
        """
        Return the total number of .npy files.
        """
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Load the .npy file, corresponding .txt, and .pt file.
        """
        #Load pose data
        file_data = self.files[idx]
        npy_data = np.load(file_data["npy"])

        num = '1'

        #Load IMU data
        imu_back = file_data["npy"].replace(".npy", "") + r"\back_"+num+"_grav.npz"
        imu_belt = file_data["npy"].replace(".npy", "") + r"\belt_"+num+"_grav.npz"
        imu_chest = file_data["npy"].replace(".npy", "") + r"\chest_"+num+"_grav.npz"
        imu_forehead = file_data["npy"].replace(".npy", "") + r"\forehead_"+num+"_grav.npz"
        imu_necklace = file_data["npy"].replace(".npy", "") + r"\necklace_"+num+"_grav.npz"
        imu_left_arm = file_data["npy"].replace(".npy", "") + r"\left_arm_"+num+"_grav.npz"
        imu_left_ear = file_data["npy"].replace(".npy", "") + r"\left_ear_"+num+"_grav.npz"
        imu_left_foot = file_data["npy"].replace(".npy", "") + r"\left_foot_"+num+"_grav.npz"
        imu_left_shin = file_data["npy"].replace(".npy", "") + r"\left_shin_"+num+"_grav.npz"
        imu_left_shirt_pocket = file_data["npy"].replace(".npy", "") + r"\left_shirt_pocket_"+num+"_grav.npz"
        imu_left_shoulder = file_data["npy"].replace(".npy", "") + r"\left_shoulder_"+num+"_grav.npz"
        imu_left_thigh = file_data["npy"].replace(".npy", "") + r"\left_thigh_"+num+"_grav.npz"
        imu_left_wrist = file_data["npy"].replace(".npy", "") + r"\left_wrist_"+num+"_grav.npz"
        imu_right_arm = file_data["npy"].replace(".npy", "") + r"\right_arm_"+num+"_grav.npz"
        imu_right_ear = file_data["npy"].replace(".npy", "") + r"\right_ear_"+num+"_grav.npz"
        imu_right_foot = file_data["npy"].replace(".npy", "") + r"\right_foot_"+num+"_grav.npz"
        imu_right_shin = file_data["npy"].replace(".npy", "") + r"\right_shin_"+num+"_grav.npz"
        imu_right_shirt_pocket = file_data["npy"].replace(".npy", "") + r"\right_shirt_pocket_"+num+"_grav.npz"
        imu_right_shoulder = file_data["npy"].replace(".npy", "") + r"\right_shoulder_"+num+"_grav.npz"
        imu_right_thigh = file_data["npy"].replace(".npy", "") + r"\right_thigh_"+num+"_grav.npz"
        imu_right_wrist = file_data["npy"].replace(".npy", "") + r"\right_wrist_"+num+"_grav.npz"

        acc = 'linear_acceleration_with_gravity'
        gyr = 'angular_velocity'

        imu_data = {
            "back": {
                "acc": np.load(imu_back)[acc],
                "gyr": np.load(imu_back)[gyr]
            },
            "belt": {
                "acc": np.load(imu_belt)[acc],
                "gyr": np.load(imu_belt)[gyr]
            },
            "chest": {
                "acc": np.load(imu_chest)[acc],
                "gyr": np.load(imu_chest)[gyr]
            },
            "forehead": {
                "acc": np.load(imu_forehead)[acc],
                "gyr": np.load(imu_forehead)[gyr]
            },
            "necklace": {
                "acc": np.load(imu_necklace)[acc],
                "gyr": np.load(imu_necklace)[gyr]
            },
            "left_arm": {
                "acc": np.load(imu_left_arm)[acc],
                "gyr": np.load(imu_left_arm)[gyr]
            },
            "left_ear": {
                "acc": np.load(imu_left_ear)[acc],
                "gyr": np.load(imu_left_ear)[gyr]
            },
            "left_foot": {
                "acc": np.load(imu_left_foot)[acc],
                "gyr": np.load(imu_left_foot)[gyr]
            },
            "left_shin": {
                "acc": np.load(imu_left_shin)[acc],
                "gyr": np.load(imu_left_shin)[gyr]
            },
            "left_shirt_pocket": {
                "acc": np.load(imu_left_shirt_pocket)[acc],
                "gyr": np.load(imu_left_shirt_pocket)[gyr]
            },
            "left_shoulder": {
                "acc": np.load(imu_left_shoulder)[acc],
                "gyr": np.load(imu_left_shoulder)[gyr]
            },
            "left_thigh": {
                "acc": np.load(imu_left_thigh)[acc],
                "gyr": np.load(imu_left_thigh)[gyr]
            },
            "left_shirt_wrist": {
                "acc": np.load(imu_left_wrist)[acc],
                "gyr": np.load(imu_left_wrist)[gyr]
            },
            "right_arm": {
                "acc": np.load(imu_right_arm)[acc],
                "gyr": np.load(imu_right_arm)[gyr]
            },
            "right_ear": {
                "acc": np.load(imu_right_ear)[acc],
                "gyr": np.load(imu_right_ear)[gyr]
            },
            "right_foot": {
                "acc": np.load(imu_right_foot)[acc],
                "gyr": np.load(imu_right_foot)[gyr]
            },
            "right_shin": {
                "acc": np.load(imu_right_shin)[acc],
                "gyr": np.load(imu_right_shin)[gyr]
            },
            "right_shirt_pocket": {
                "acc": np.load(imu_right_shirt_pocket)[acc],
                "gyr": np.load(imu_right_shirt_pocket)[gyr]
            },
            "right_shoulder": {
                "acc": np.load(imu_right_shoulder)[acc],
                "gyr": np.load(imu_right_shoulder)[gyr]
            },
            "right_thigh": {
                "acc": np.load(imu_right_thigh)[acc],
                "gyr": np.load(imu_right_thigh)[gyr]
            },
            "right_shirt_wrist": {
                "acc": np.load(imu_right_wrist)[acc],
                "gyr": np.load(imu_right_wrist)[gyr]
            }
        }

        # Load text data
        with open(file_data["txt"], 'r') as f:
            text_data = f.read()

        # Load embedding data
        embedding_data = torch.load(file_data["embedding"], weights_only=True)

        return npy_data, text_data, embedding_data, imu_data, file_data["npy"]

# Root folder containing the .npy files, text folder, and embedding folder
root_folder = r'test'
text_folder = r'test_label'

# Create the dataset and dataloader
dataset = CrosSimDataset(root_folder, text_folder)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Print loaded data
for npy_data, text_data, embedding_data,imu_data, npy_path in dataloader:
    file_name = os.path.basename(npy_path[0]).replace(".npy", "")
    print(file_name)
    print(f"Text: {text_data[0]}")
    print(f"Pose shape: {npy_data.shape}")
    print(f"Embedding shape: {embedding_data.shape}")
    # Iterate through the outer dictionary (for each position like "back", "belt")
    for position, data in imu_data.items():
        # Print the shape of the 'acc' and 'gyr' data for each position
        acc_data = data["acc"]
        gyr_data = data["gyr"]
        
        print(f"Position: {position}")
        print(f"  Accelerometer (acc) Shape: {acc_data.shape}")
        print(f"  Gyroscope (gyr) Shape: {gyr_data.shape}")
    break
