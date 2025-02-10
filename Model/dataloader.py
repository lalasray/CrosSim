import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Function to find all .pt files in a directory and subdirectories
def find_pt_files(root_dir):
    return glob.glob(os.path.join(root_dir, "**", "*.pt"), recursive=True)

# Custom Dataset
class DancePoseDataset(Dataset):
    def __init__(self, root_dir, pose_dir, imu_dir):
        self.root_dir = root_dir
        self.pose_dir = pose_dir
        self.imu_dir = imu_dir
        self.file_paths = self.find_valid_pairs()

    def find_valid_pairs(self):
        """Find all valid (.pt, .npy) file pairs."""
        pt_files = find_pt_files(self.root_dir)
        valid_pairs = []

        for pt_file in pt_files:
            base_name = os.path.basename(pt_file).replace("_gtr.pt", "")  # Remove _gtr.pt suffix
            npy_pattern = os.path.join(self.pose_dir, "**", f"{base_name}.npy")
            npy_files = glob.glob(npy_pattern, recursive=True)
            imu_back_pattern = os.path.join(self.imu_dir, "**", f"{base_name}/back_1_grav.npz")
            imu_back = glob.glob(imu_back_pattern, recursive=True)
            
            if npy_files and imu_back:
                valid_pairs.append((pt_file, npy_files[0], imu_back[0]))

        return valid_pairs

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        pt_path, npy_path, imu_back_path = self.file_paths[idx]

        # Load dance motion
        motion_data = torch.load(pt_path)

        # Load corresponding pose
        pose_data = np.load(npy_path)

        # Load IMU back sensor data
        imu_back_data = np.load(imu_back_path)
        imu_belt_data = imu_back_path.replace("back", "belt")
        imu_chest_data = imu_back_path.replace("back", "chest")
        imu_forehead_data = imu_back_path.replace("back", "forehead")
        imu_left_arm_data = imu_back_path.replace("back", "left_arm")
        imu_left_ear_data = imu_back_path.replace("back", "left_ear")
        imu_left_foot_data = imu_back_path.replace("back", "left_foot")
        imu_left_shin_data = imu_back_path.replace("back", "left_shin")
        imu_left_pocket_data = imu_back_path.replace("back", "left_pocket")
        imu_left_shoulder_data = imu_back_path.replace("back", "left_shoulder")
        imu_left_thigh_data = imu_back_path.replace("back", "left_thigh")
        imu_left_wrist_data = imu_back_path.replace("back", "left_wrist")
        imu_necklace_data = imu_back_path.replace("back", "necklace")
        imu_right_arm_data = imu_back_path.replace("back", "right_arm")
        imu_right_ear_data = imu_back_path.replace("back", "right_ear")
        imu_right_foot_data = imu_back_path.replace("back", "right_foot")
        imu_right_shin_data = imu_back_path.replace("back", "right_shin")
        imu_right_pocket_data = imu_back_path.replace("back", "right_pocket")
        imu_right_shoulder_data = imu_back_path.replace("back", "right_shoulder")
        imu_right_thigh_data = imu_back_path.replace("back", "right_thigh")
        imu_right_wrist_data = imu_back_path.replace("back", "right_wrist")

        return motion_data, pose_data, imu_back_data, imu_belt_data, imu_chest_data, imu_forehead_data, imu_left_arm_data, imu_left_ear_data, imu_left_foot_data, imu_left_shin_data, imu_left_pocket_data, imu_left_shoulder_data,imu_left_thigh_data, imu_left_wrist_data, imu_necklace_data, imu_right_arm_data, imu_right_ear_data, imu_right_foot_data, imu_right_shin_data, imu_right_pocket_data, imu_right_shoulder_data, imu_right_thigh_data, imu_right_wrist_data

# Custom collate function to handle variable-length sequences
def collate_fn(batch):
    # Simply return the batch as a list of tensors
    motion_data, pose_data, imu_back_data = zip(*batch)
    return list(motion_data), list(pose_data), list(imu_back_data)

# Example usage
root_dir = r"/media/lala/Seagate/CrosSim"
pose_dir = r"/media/lala/Seagate/CrosSim"
imu_dir = r"/media/lala/Seagate/CrosSim"
dataset = DancePoseDataset(root_dir, pose_dir, imu_dir)

dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

for motion_data, pose_data, imu_back_data, imu_belt_data, imu_chest_data, imu_forehead_data, imu_left_arm_data, imu_left_ear_data, imu_left_foot_data, imu_left_shin_data, imu_left_pocket_data, imu_left_shoulder_data,imu_left_thigh_data, imu_left_wrist_data, imu_necklace_data, imu_right_arm_data, imu_right_ear_data, imu_right_foot_data, imu_right_shin_data, imu_right_pocket_data, imu_right_shoulder_data, imu_right_thigh_data, imu_right_wrist_data in dataloader:
    print(f"Motion data shape (first sample): {motion_data[0].shape}")
    print(f"Pose data shape (first sample): {pose_data[0].shape}")
    imu_back = imu_back_data[0]
    print(f"IMU back gyro: {imu_back['angular_velocity'].shape}")
    print(f"IMU back acc with grav: {imu_back['linear_acceleration_with_gravity'].shape}")
    print(f"IMU back acc: {imu_back['linear_acceleration'].shape}")
    break
