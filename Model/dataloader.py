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
    def __init__(self, root_dir, pose_dir):
        self.root_dir = root_dir
        self.pose_dir = pose_dir
        self.file_paths = self.find_valid_pairs()

    def find_valid_pairs(self):
        """Find valid (.pt, .npy) file pairs with required directory structure."""
        pt_files = find_pt_files(self.root_dir)
        valid_pairs = []

        for pt_file in pt_files:
            base_name = os.path.basename(pt_file).replace("_gtr.pt", "")  # Remove _gtr.pt suffix
            expected_dir = os.path.join(self.pose_dir, base_name)  # Expected folder path
            npz_file = os.path.join(expected_dir, "back_1_grav.npz")  # Expected .npz file

            # Check if the expected directory and file exist
            if os.path.isdir(expected_dir) and os.path.isfile(npz_file):
                valid_pairs.append((pt_file, npz_file))  # Store the matching pair

        return valid_pairs

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        pt_path, npz_path = self.file_paths[idx]

        # Load dance motion
        motion_data = torch.load(pt_path)

        # Load corresponding pose
        pose_data = np.load(npz_path)

        return motion_data, pose_data

# Example usage
root_dir = r"/media/lala/Seagate/CrosSim"
pose_dir = r"/media/lala/Seagate/CrosSim"
dataset = DancePoseDataset(root_dir, pose_dir)
print(len(dataset))

# Updated DataLoader with num_workers=4
#dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Iterate over the dataset
#for motion, pose in dataloader:
#    print("Motion Data:", motion.shape)
#    print("Pose Data:", pose.shape)
#    break
