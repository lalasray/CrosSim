import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader

class DancePoseDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = self.find_pt_files()

    def find_pt_files(self):
        """Finds all valid .pt files in the dataset directory."""
        return sorted(glob.glob(os.path.join(self.root_dir, "**", "*.pt"), recursive=True))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        pt_path = self.file_paths[idx]

        # Load the saved dictionary from the .pt file
        data = torch.load(pt_path)

        # Assuming each .pt file contains: {'motion': tensor, 'pose': tensor, 'imu': dict_of_tensors}
        motion_data = data['motion_data']
        pose_data = data['pose_data']
        imu_data = data['imu_data']  # Dict of IMU sensor tensors

        return motion_data, pose_data, imu_data

# Custom collate function for batch loading
def collate_fn(batch):
    motion_data, pose_data, imu_data_list = zip(*batch)

    # Convert lists to tensors where possible
    motion_data = [torch.tensor(md, dtype=torch.float32) for md in motion_data]
    pose_data = [torch.tensor(pd, dtype=torch.float32) for pd in pose_data]

    return motion_data, pose_data, imu_data_list  # IMU data remains a dict per sample

# Example usage
root_dir = "/media/lala/Seagate/temp"
dataset = DancePoseDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

for motion_data, pose_data, imu_data_list in dataloader:
    print("Motion Data Shape:", motion_data[0].shape)
    print("Pose Data Shape:", pose_data[0].shape)
    print("IMU Sensors:", list(imu_data_list[0].keys()))  # Prints IMU sensor names
