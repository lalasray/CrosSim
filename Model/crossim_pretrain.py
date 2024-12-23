import torch
from torch.utils.data import DataLoader
from random_dataloader import RandomCrosSimDataset, collate_fn
from class_encoder import ClassEncoder
from clip_text_encoder import TextEncoder
from graph_IMU_encoder import IMUGraphEncoderTemporal
from graph_pose_encoder import PoseGraphEncoderTemporal
from gtr_text_encoder import EmbeddingEncoder
from Single_IMU_with_pos_encoder import IMUSingleNodeEncoderWithClass

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
num_samples = 30  # Number of samples in the dataset
num_positions = 20  # Number of IMU positions
imu_shape = (1200, 3)  # Shape of each IMU sensor's accelerometer and gyroscope data
text_embedding_dim = 768  # Dimension of the embedding vector
pose_shape = (300, 22, 3)  # Shape of the pose data (frames, joints, coordinates)
num_classes = 300  # Example: 10 classes
batch_size = 16
embedding_dim = 512  # Size of the embedding space
imu_nodes = 20
imu_dim = 6
pose_joints = 22

# Create the random dataset and dataloader
random_dataset = RandomCrosSimDataset(num_samples, num_positions, imu_shape, text_embedding_dim, pose_shape, num_classes)
random_dataloader = DataLoader(random_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

#activity label
class_enc = ClassEncoder(num_classes, embedding_dim)
clip_enc = TextEncoder()
gtr_enc = EmbeddingEncoder()

#imu data
imu_enc = IMUGraphEncoderTemporal(num_nodes=imu_nodes, feature_dim=imu_dim, embedding_size=embedding_dim, max_hop=1, dilation=1, temporal_hidden_size=256)
single_imu_enc = IMUSingleNodeEncoderWithClass(feature_dim=imu_dim, embedding_size=embedding_dim, temporal_hidden_size=256, num_classes=num_positions)

#pose data
pose_enc = PoseGraphEncoderTemporal(num_nodes=pose_joints, feature_dim=imu_dim, embedding_size=embedding_dim, max_hop=1, dilation=1, temporal_hidden_size=256)



