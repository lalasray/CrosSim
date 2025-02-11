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
    motion_data, pose_data, imu_back_data, imu_belt_data, imu_chest_data, imu_forehead_data, imu_left_arm_data, imu_left_ear_data, imu_left_foot_data, imu_left_shin_data, imu_left_pocket_data, imu_left_shoulder_data, imu_left_thigh_data, imu_left_wrist_data, imu_necklace_data, imu_right_arm_data, imu_right_ear_data, imu_right_foot_data, imu_right_shin_data, imu_right_pocket_data, imu_right_shoulder_data, imu_right_thigh_data, imu_right_wrist_data = zip(*batch)

    return list(motion_data), list(pose_data), list(imu_back_data), list(imu_belt_data), list(imu_chest_data), list(imu_forehead_data), list(imu_left_arm_data), list(imu_left_ear_data), list(imu_left_foot_data), list(imu_left_shin_data), list(imu_left_pocket_data), list(imu_left_shoulder_data), list(imu_left_thigh_data), list(imu_left_wrist_data), list(imu_necklace_data), list(imu_right_arm_data), list(imu_right_ear_data), list(imu_right_foot_data), list(imu_right_shin_data), list(imu_right_pocket_data), list(imu_right_shoulder_data), list(imu_right_thigh_data), list(imu_right_wrist_data)



# Example usage
root_dir = r"/media/lala/Seagate/CrosSim"
pose_dir = r"/media/lala/Seagate/CrosSim"
imu_dir = r"/media/lala/Seagate/CrosSim"
target_dir = r"/media/lala/Seagate/temp"

dataset = DancePoseDataset(root_dir, pose_dir, imu_dir)

dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

for motion_data, pose_data, imu_back_data, imu_belt_data, imu_chest_data, imu_forehead_data, imu_left_arm_data, imu_left_ear_data, imu_left_foot_data, imu_left_shin_data, imu_left_pocket_data, imu_left_shoulder_data, imu_left_thigh_data, imu_left_wrist_data, imu_necklace_data, imu_right_arm_data, imu_right_ear_data, imu_right_foot_data, imu_right_shin_data, imu_right_pocket_data, imu_right_shoulder_data, imu_right_thigh_data, imu_right_wrist_data in dataloader:
    motion_data_tensor = torch.tensor(motion_data[0], dtype=torch.float32)
    pose_data_tensor = torch.tensor(pose_data[0], dtype=torch.float32)
    
    imu_back_gyro = torch.tensor(imu_back_data[0]['angular_velocity'], dtype=torch.float32)
    imu_back_acc = torch.tensor(imu_back_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_back_acc_g = torch.tensor(imu_back_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_belt_gyro = torch.tensor(imu_belt_data[0]['angular_velocity'], dtype=torch.float32)
    imu_belt_acc = torch.tensor(imu_belt_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_belt_acc_g = torch.tensor(imu_belt_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_chest_gyro = torch.tensor(imu_chest_data[0]['angular_velocity'], dtype=torch.float32)
    imu_chest_acc = torch.tensor(imu_chest_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_chest_acc_g = torch.tensor(imu_chest_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_forehead_gyro = torch.tensor(imu_forehead_data[0]['angular_velocity'], dtype=torch.float32)
    imu_forehead_acc = torch.tensor(imu_forehead_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_forehead_acc_g = torch.tensor(imu_forehead_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_left_arm_gyro = torch.tensor(imu_left_arm_data[0]['angular_velocity'], dtype=torch.float32)
    imu_left_arm_acc = torch.tensor(imu_left_arm_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_left_arm_acc_g = torch.tensor(imu_left_arm_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_left_ear_gyro = torch.tensor(imu_left_ear_data[0]['angular_velocity'], dtype=torch.float32)
    imu_left_ear_acc = torch.tensor(imu_left_ear_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_left_ear_acc_g = torch.tensor(imu_left_ear_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_left_foot_gyro = torch.tensor(imu_left_foot_data[0]['angular_velocity'], dtype=torch.float32)
    imu_left_foot_acc = torch.tensor(imu_left_foot_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_left_foot_acc_g = torch.tensor(imu_left_foot_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_left_shin_gyro = torch.tensor(imu_left_shin_data[0]['angular_velocity'], dtype=torch.float32)
    imu_left_shin_acc = torch.tensor(imu_left_shin_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_left_shin_acc_g = torch.tensor(imu_left_shin_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_left_pocket_gyro = torch.tensor(imu_left_pocket_data[0]['angular_velocity'], dtype=torch.float32)
    imu_left_pocket_acc = torch.tensor(imu_left_pocket_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_left_pocket_acc_g = torch.tensor(imu_left_pocket_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_left_shoulder_gyro = torch.tensor(imu_left_shoulder_data[0]['angular_velocity'], dtype=torch.float32)
    imu_left_shoulder_acc = torch.tensor(imu_left_shoulder_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_left_shoulder_acc_g = torch.tensor(imu_left_shoulder_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_left_thigh_gyro = torch.tensor(imu_left_thigh_data[0]['angular_velocity'], dtype=torch.float32)
    imu_left_thigh_acc = torch.tensor(imu_left_thigh_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_left_thigh_acc_g = torch.tensor(imu_left_thigh_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_left_wrist_gyro = torch.tensor(imu_left_wrist_data[0]['angular_velocity'], dtype=torch.float32)
    imu_left_wrist_acc = torch.tensor(imu_left_wrist_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_left_wrist_acc_g = torch.tensor(imu_left_wrist_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_necklace_gyro = torch.tensor(imu_necklace_data[0]['angular_velocity'], dtype=torch.float32)
    imu_necklace_acc = torch.tensor(imu_necklace_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_necklace_acc_g = torch.tensor(imu_necklace_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_right_arm_gyro = torch.tensor(imu_right_arm_data[0]['angular_velocity'], dtype=torch.float32)
    imu_right_arm_acc = torch.tensor(imu_right_arm_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_right_arm_acc_g = torch.tensor(imu_right_arm_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_right_ear_gyro = torch.tensor(imu_right_ear_data[0]['angular_velocity'], dtype=torch.float32)
    imu_right_ear_acc = torch.tensor(imu_right_ear_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_right_ear_acc_g = torch.tensor(imu_right_ear_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_right_foot_gyro = torch.tensor(imu_right_foot_data[0]['angular_velocity'], dtype=torch.float32)
    imu_right_foot_acc = torch.tensor(imu_right_foot_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_right_foot_acc_g = torch.tensor(imu_right_foot_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_right_shin_gyro = torch.tensor(imu_right_shin_data[0]['angular_velocity'], dtype=torch.float32)
    imu_right_shin_acc = torch.tensor(imu_right_shin_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_right_shin_acc_g = torch.tensor(imu_right_shin_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_right_pocket_gyro = torch.tensor(imu_right_pocket_data[0]['angular_velocity'], dtype=torch.float32)
    imu_right_pocket_acc = torch.tensor(imu_right_pocket_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_right_pocket_acc_g = torch.tensor(imu_right_pocket_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_right_thigh_gyro = torch.tensor(imu_right_thigh_data[0]['angular_velocity'], dtype=torch.float32)
    imu_right_thigh_acc = torch.tensor(imu_right_thigh_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_right_thigh_acc_g = torch.tensor(imu_right_thigh_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_right_shoulder_gyro = torch.tensor(imu_right_shoulder_data[0]['angular_velocity'], dtype=torch.float32)
    imu_right_shoulder_acc = torch.tensor(imu_right_shoulder_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_right_shoulder_acc_g = torch.tensor(imu_right_shoulder_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    imu_right_wrist_gyro = torch.tensor(imu_right_wrist_data[0]['angular_velocity'], dtype=torch.float32)
    imu_right_wrist_acc = torch.tensor(imu_right_wrist_data[0]['linear_acceleration'], dtype=torch.float32)
    imu_right_wrist_acc_g = torch.tensor(imu_right_wrist_data[0]['linear_acceleration_with_gravity'], dtype=torch.float32)

    filename = target_dir+f"/sensor_data_batch_{batch_idx}.pt"
    torch.save({
        'motion_data': motion_data_tensor,
        'pose_data': pose_data_tensor,
        'back': imu_back_data,
        'belt': imu_belt_data,
        'chest': imu_chest_data,
        'forehead': imu_forehead_data,
        'left_arm': imu_left_arm_data,
        'left_ear': imu_left_ear_data,
        'left_foot': imu_left_foot_data,
        'left_shin': imu_left_shin_data,
        'left_pocket': imu_left_pocket_data,
        'left_shoulder': imu_left_shoulder_data,
        'left_thigh': imu_left_thigh_data,
        'left_wrist': imu_left_wrist_data,
        'necklace': imu_necklace_data,
        'right_arm': imu_right_arm_data,
        'right_ear': imu_right_ear_data,
        'right_foot': imu_right_foot_data,
        'right_shin': imu_right_shin_data,
        'right_pocket': imu_right_pocket_data,
        'right_shoulder': imu_right_shoulder_data,
        'right_thigh': imu_right_thigh_data,
        'right_wrist': imu_right_wrist_data
    }, filename)

    print(f"Batch {batch_idx} saved to {filename}")
