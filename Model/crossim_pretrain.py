import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from random_dataloader import RandomCrosSimDataset, collate_fn

# Parameters
num_samples = 30  # Number of samples in the dataset
num_positions = 20  # Number of IMU positions
imu_shape = (1200, 3)  # Shape of each IMU sensor's accelerometer and gyroscope data
embedding_dim = 768  # Dimension of the embedding vector
pose_shape = (300, 22, 3)  # Shape of the pose data (frames, joints, coordinates)
num_classes = 3000  # Number of classes for the one-hot embedding

# Create the random dataset and dataloader
random_dataset = RandomCrosSimDataset(num_samples, num_positions, imu_shape, embedding_dim, pose_shape, num_classes)
random_dataloader = DataLoader(random_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for npy_data, text_data, embedding_data, imu_data, one_hot_embedding, npy_paths in random_dataloader:
    print(f"Pose shape: {npy_data.shape}")
    print(f"Embedding shape: {embedding_data.shape}")
    print(f"One-hot embedding shape: {one_hot_embedding.shape}")
    print(f"Text data: {text_data}")
    print(f"IMU positions: {len(imu_data.keys())}")
    for position, data in imu_data.items():
        print(f"Position: {position}, Acc shape: {data['acc'].shape}, Gyr shape: {data['gyr'].shape}")
    break  # Print only one batch for demonstration
