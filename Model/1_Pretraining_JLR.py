import torch
import torch.nn as nn
from Encoder.Multi_IMU_Encoder import DeepConvGraphEncoderPre, IMUGraph, DeepConvGraphEncoderDownstream
from Encoder.Gtr_Text_Encoder import EmbeddingEncoder
from Encoder.Single_IMU_Encoder import IMUSingleNodeEncoderWithClass
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph, GraphPoseEncoderDown
from Loss.pretrain_loss import InfoNCELoss

pose_graph = PoseGraph(max_hop=1, dilation=1)
pose_edge_index = pose_graph.edge_index
IMU_graph = IMUGraph(max_hop=1, dilation=1)
IMU_edge_index = IMU_graph.edge_index

class MultiModalJLR(nn.Module):
    def __init__(self):
        super(MultiModalJLR, self).__init__()
        self.text_encoder = EmbeddingEncoder()
        self.pose_encoder = GraphPoseEncoderPre(16, 25, 24, 6)
        self.imu_encoder = DeepConvGraphEncoderPre(16, 100, 20, 6)
        self.single_imu_encoder = IMUSingleNodeEncoderWithClass()

    def forward(self, text, pose, imu):
        text_embeddings = self.text_encoder(text)
        pose_embeddings = self.pose_encoder(pose,pose_edge_index)
        imu_embeddings = self.imu_encoder(imu,IMU_edge_index)
        single_imu_embeddings = self.single_imu_encoder(imu)
        
        return text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings

def compute_total_loss(model, text, pose, imu):

    # Forward pass through the model to get embeddings
    text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings = model(text, pose, imu)
    
    # Initialize InfoNCE loss with hard negative mining
    info_nce_loss = InfoNCELoss(hard_negative_mining=True)
    
    # Compute InfoNCE loss for each pair of embeddings
    loss_text_pose = info_nce_loss(text_embeddings, pose_embeddings)
    loss_text_imu = info_nce_loss(text_embeddings, imu_embeddings)
    loss_pose_imu = info_nce_loss(pose_embeddings, imu_embeddings)
    loss_text_single_imu = info_nce_loss(text_embeddings, single_imu_embeddings)
    loss_imu_single_imu = info_nce_loss(imu_embeddings, single_imu_embeddings)
    
    # Calculate the total loss as the sum of all individual losses
    total_loss = (loss_text_pose + loss_text_imu + 
                  loss_pose_imu + loss_text_single_imu + 
                  loss_imu_single_imu)
    
    return total_loss

# Instantiate the model
model = MultiModalJLR()

# Generate dummy data
batch_size = 16
time_steps = 100
num_nodes = 20
imu_feature_dim = 3
pose_joints = 25
pose_joint_features = 24
pose_feature_dim = 6
text_feature_dim = 768
single_imu_feature_dim = 6
num_classes = 20

text_data = torch.rand(batch_size, text_feature_dim)
pose_data = torch.rand(batch_size, pose_joints, pose_joint_features, pose_feature_dim)
imu_data = torch.rand(batch_size, time_steps, num_nodes, imu_feature_dim)
single_imu_data = torch.rand(batch_size, time_steps, 1, single_imu_feature_dim)

# Perform a forward pass with dummy data
text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings = model(text_data, pose_data, imu_data)

# Print the shapes of the outputs
print("Text Embeddings Shape:", text_embeddings.shape)
print("Pose Embeddings Shape:", pose_embeddings.shape)
print("IMU Embeddings Shape:", imu_embeddings.shape)
print("Single IMU Embeddings Shape:", single_imu_embeddings.shape)