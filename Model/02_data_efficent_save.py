import h5py
import torch
import os
import numpy as np
from tqdm import tqdm  # Import tqdm for progress tracking

# Define paths
data_dir = "../CrosSim_Data/UniMocap/processed"  # Update this path
save_path = "../CrosSim_Data/UniMocap/full_dataset.h5"  # Path to save the dataset

# Sensor positions
sensor_positions_acc = ["back.acc", "belt.acc", "chest.acc", "forehead.acc",
                        "left_arm.acc", "left_ear.acc", "left_foot.acc", "left_shin.acc",
                        "left_shirt_pocket.acc", "left_shoulder.acc", "left_thigh.acc", "left_wrist.acc",
                        "necklace.acc", "right_arm.acc", "right_ear.acc", "right_foot.acc",
                        "right_shin.acc", "right_shirt_pocket.acc", "right_shoulder.acc",
                        "right_thigh.acc", "right_wrist.acc"]

sensor_positions_gyro = [pos.replace(".acc", ".gyro") for pos in sensor_positions_acc]
sensor_positions_acc_g = [pos + "_g" for pos in sensor_positions_acc]

# Get list of all .pt files
file_list = [f for f in os.listdir(data_dir) if f.endswith(".pt")]

# Process files with tqdm progress bar
# Open the HDF5 file in append mode
with h5py.File(save_path, "a") as f:
    for file_name in tqdm(file_list, desc="Processing files", unit="file"):
        file_path = os.path.join(data_dir, file_name)
        data = torch.load(file_path, map_location="cpu")

        # Extract raw data
        motion = data.get("motion_data", None)
        pose_joint = data.get("pose_data", {}).get("joint", None)
        pose_body = data.get("pose_data", {}).get("body", None)
        pose_trans = data.get("pose_data", {}).get("trans", None)

        # Extract and structure IMU data
        imu_data = data.get("imu_data", {})
        imu_tensors = {f"{k}.{sk}": v for k, subdict in imu_data.items() for sk, v in subdict.items()}

        # Skip invalid samples
        if motion is None or pose_joint is None or pose_body is None or pose_trans is None:
            continue

        # --- Apply the processing ---
        text_data = motion.squeeze(1)
        pose = torch.cat([pose_trans, pose_body], dim=-1)
        full_Pose = pose.view(pose.shape[0], 24, 3)
        pose_data = torch.cat([full_Pose, pose_joint.squeeze(1)], dim=-1)

        # Combine IMU data
        try:
            combined_data_acc = torch.stack([imu_tensors[key] for key in sensor_positions_acc])
            combined_data_gyro = torch.stack([imu_tensors[key] for key in sensor_positions_gyro])
            imu_data = torch.cat((combined_data_acc, combined_data_gyro), dim=2)

            combined_data_acc_grav = torch.stack([imu_tensors[key] for key in sensor_positions_acc_g])
            imu_data_grav = torch.cat((combined_data_acc_grav, combined_data_gyro), dim=2)

            # Create a group in the HDF5 file for this specific data point (per file)
            group_name = f"data_{file_name.replace('.pt', '')}"
            group = f.create_group(group_name)

            # Save all the processed data within the group
            group.create_dataset("text_data", data=text_data.numpy(), compression="gzip")
            group.create_dataset("pose_data", data=pose_data.numpy(), compression="gzip")
            group.create_dataset("imu_data", data=imu_data.numpy(), compression="gzip")
            group.create_dataset("imu_data_grav", data=imu_data_grav.numpy(), compression="gzip")
        except KeyError as e:
            print(f"Skipping file {file_name} due to missing key: {e}")

print(f"\nProcessed dataset saved as {save_path}")
