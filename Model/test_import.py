import torch
import torch.nn as nn
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph, GraphPoseEncoderDown
from Loss.pretrain_loss import InfoNCELoss

pose_graph = PoseGraph(max_hop=1, dilation=1)
pose_edge_index = pose_graph.edge_index
pose_encoder = GraphPoseEncoderPre(num_nodes=24, feature_dim=6, hidden_dim=128, embedding_dim=64, window_size=1, stride=1)
sample_input = torch.randn(16, 25, 24, 6)
pose_data = torch.rand(16, 25, 24, 6)
output = pose_encoder(pose_data, pose_edge_index)

# Print the shapes of the outputs
#print("Text Embeddings Shape:", text_embeddings.shape)
#print("Pose Embeddings Shape:", pose_embeddings.shape)
#print("IMU Embeddings Shape:", imu_embeddings.shape)
#print("Single IMU Embeddings Shape:", single_imu_embeddings.shape)