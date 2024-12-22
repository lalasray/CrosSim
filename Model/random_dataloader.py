import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class RandomCrosSimDataset(Dataset):
    def __init__(self, num_samples, num_positions, imu_shape, embedding_dim, pose_shape, num_classes):
        """
        Initialize the random dataset.
        
        Args:
            num_samples (int): Number of samples in the dataset.
            num_positions (int): Number of IMU positions (e.g., back, belt, etc.).
            imu_shape (tuple): Shape of accelerometer and gyroscope data for each IMU.
            embedding_dim (int): Dimension of the embedding vector.
            pose_shape (tuple): Shape of the pose data.
            num_classes (int): Number of classes for the one-hot embedding.
        """
        self.num_samples = num_samples
        self.num_positions = num_positions
        self.imu_shape = imu_shape
        self.embedding_dim = embedding_dim
        self.pose_shape = pose_shape
        self.num_classes = num_classes  # Number of classes for one-hot encoding
        self.positions = [f"position_{i}" for i in range(num_positions)]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Generate random data for a single sample.
        """
        # Random pose data
        npy_data = np.random.rand(*self.pose_shape).astype(np.float32)
        
        # Random IMU data
        imu_data = {
            pos: {
                "acc": np.random.rand(*self.imu_shape).astype(np.float32),
                "gyr": np.random.rand(*self.imu_shape).astype(np.float32)
            }
            for pos in self.positions
        }
        
        # Random text data
        text_data = f"Random description for sample {idx}"
        
        # Random embedding data
        embedding_data = torch.rand(self.embedding_dim)
        
        # Random one-hot embedding for each sample (as a tensor)
        one_hot_embedding = torch.zeros(self.num_classes)
        one_hot_embedding[torch.randint(0, self.num_classes, (1,))] = 1
        
        return npy_data, text_data, embedding_data, imu_data, one_hot_embedding, f"random_sample_{idx}.npy"

def collate_fn(batch):
    """
    Custom collate function to handle nested dictionaries in the dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npy_data_batch = torch.stack([torch.from_numpy(sample[0]).to(device) for sample in batch])
    text_data_batch = [sample[1] for sample in batch]
    embedding_data_batch = torch.stack([sample[2].to(device) for sample in batch])
    one_hot_embedding_batch = torch.stack([sample[4].to(device) for sample in batch])
    imu_data_batch = defaultdict(lambda: {"acc": [], "gyr": []})
    npy_paths_batch = [sample[5] for sample in batch]

    for sample in batch:
        for position, data in sample[3].items():
            imu_data_batch[position]["acc"].append(torch.from_numpy(data["acc"]).to(device))
            imu_data_batch[position]["gyr"].append(torch.from_numpy(data["gyr"]).to(device))
    
    for position in imu_data_batch.keys():
        imu_data_batch[position]["acc"] = torch.stack(imu_data_batch[position]["acc"])
        imu_data_batch[position]["gyr"] = torch.stack(imu_data_batch[position]["gyr"])
    
    return npy_data_batch, text_data_batch, embedding_data_batch, imu_data_batch, one_hot_embedding_batch, npy_paths_batch

def main():
    # Parameters
    num_samples = 64  # Number of samples in the dataset
    num_positions = 20  # Number of IMU positions
    imu_shape = (1200, 3)  # Shape of each IMU sensor's accelerometer and gyroscope data
    embedding_dim = 768  # Dimension of the embedding vector
    pose_shape = (300, 66)  # Shape of the pose data (frames, joints, coordinates)
    num_classes = 10  # Number of classes for the one-hot embedding

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

if __name__ == "__main__":
    main()
