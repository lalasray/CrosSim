import torch
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset, DataLoader

class UniMocapDataset(Dataset):
    def __init__(self, h5_file_path):
        # Open the HDF5 file for reading
        self.h5_file_path = h5_file_path
        self.h5_file = h5py.File(h5_file_path, 'r')
        
        # List all groups in the HDF5 file (each group is a datapoint)
        self.keys = list(self.h5_file.keys())
        
    def __len__(self):
        # Return the number of datapoints (files) in the dataset
        return len(self.keys)
    
    def __getitem__(self, idx):
        # Access the group for a specific datapoint (file)
        group_name = self.keys[idx]
        group = self.h5_file[group_name]
        
        # Load datasets (text_data, pose_data, imu_data, imu_data_grav)
        text_data = torch.tensor(group["text_data"][:], dtype=torch.float32)
        pose_data = torch.tensor(group["pose_data"][:], dtype=torch.float32)
        imu_data = torch.tensor(group["imu_data"][:], dtype=torch.float32)
        imu_data_grav = torch.tensor(group["imu_data_grav"][:], dtype=torch.float32)
        
        # Return the data as a tuple
        return text_data, pose_data, imu_data, imu_data_grav
    
    def close(self):
        # Close the HDF5 file when done
        self.h5_file.close()


# Custom collate function to handle padding of sequences
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def collate_fn(batch):
    """
    Custom collate function to handle variable sequence lengths across different data modalities.
    This function will pad the sequences to the length of the longest sequence in the batch,
    except for text_data, which will not be padded.
    """
    # Find the maximum length of sequences in the batch for each modality
    max_pose_len = max([item[1].size(0) for item in batch])  # pose_data: max by 0th dimension (time)
    max_imu_len = max([item[2].size(1) for item in batch])  # imu_data: max by 1st dimension (time)
    max_imu_grav_len = max([item[3].size(1) for item in batch])  # imu_data_grav: max by 1st dimension (time)

    print(f"Max Pose Length: {max_pose_len}")
    print(f"Max IMU Length: {max_imu_len}")
    print(f"Max IMU Grav Length: {max_imu_grav_len}")

    padded_text_data = []
    padded_pose_data = []
    padded_imu_data = []
    padded_imu_data_grav = []

    for text_data, pose_data, imu_data, imu_data_grav in batch:
        # Print original shapes
        print(f"Original text_data shape: {text_data.shape}")
        print(f"Original pose_data shape: {pose_data.shape}")
        print(f"Original imu_data shape: {imu_data.shape}")
        print(f"Original imu_data_grav shape: {imu_data_grav.shape}")

        # Pad the text data (text data is assumed to be fixed-size)
        padded_text_data.append(text_data)

        # Pad the pose data (pad the time dimension to max_pose_len)
        pose_padding = (0, 0, 0, max_pose_len - pose_data.size(0))  # Pad the time dimension
        padded_pose_data.append(F.pad(pose_data, pose_padding, value=0))

        # Pad the IMU data (pad the time dimension to max_imu_len)
        imu_padding = (0, 0, 0, max_imu_len - imu_data.size(1))  # Pad the time dimension
        padded_imu_data.append(F.pad(imu_data, imu_padding, value=0))

        # Pad the IMU data with gravity (pad the time dimension to max_imu_grav_len)
        imu_grav_padding = (0, 0, 0, max_imu_grav_len - imu_data_grav.size(1))  # Pad the time dimension
        padded_imu_data_grav.append(F.pad(imu_data_grav, imu_grav_padding, value=0))

    # Stack the padded sequences into batches
    padded_text_data = torch.stack(padded_text_data)
    padded_pose_data = torch.stack(padded_pose_data)
    padded_imu_data = torch.stack(padded_imu_data)
    padded_imu_data_grav = torch.stack(padded_imu_data_grav)

    # Print shapes of the final stacked arrays
    print(f"Final padded_text_data shape: {padded_text_data.shape}")
    print(f"Final padded_pose_data shape: {padded_pose_data.shape}")
    print(f"Final padded_imu_data shape: {padded_imu_data.shape}")
    print(f"Final padded_imu_data_grav shape: {padded_imu_data_grav.shape}")

    return padded_text_data, padded_pose_data, padded_imu_data, padded_imu_data_grav




# Define the path to the HDF5 file
h5_file_path = "../CrosSim_Data/UniMocap/full_dataset.h5"

# Instantiate the dataset
dataset = UniMocapDataset(h5_file_path)

# Create the DataLoader (use appropriate batch size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Example of iterating through the DataLoader
for text_data, pose_data, imu_data, imu_data_grav in dataloader:
    print(f"text_data shape: {text_data.shape}")
    print(f"pose_data shape: {pose_data.shape}")
    print(f"imu_data shape: {imu_data.shape}")
    print(f"imu_data_grav shape: {imu_data_grav.shape}")
    # You can use the data for training or other operations here

# Don't forget to close the HDF5 file when you're done
dataset.close()
