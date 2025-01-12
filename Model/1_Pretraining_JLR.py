import torch
import torch.nn as nn
from Encoder.Multi_IMU_Encoder import DeepConvGraphEncoderPre, IMUGraph
from Encoder.Gtr_Text_Encoder import EmbeddingEncoder
from Encoder.Single_IMU_Encoder import IMUSingleNodeEncoderWithClass
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph
from Loss.pretrain_loss import InfoNCELoss, predefined_infonce

batch_size = 16
Embedding_size = 512
window = 1
stride_size = 1
Pose_joints = 24
imu_positions = 20

pose_graph = PoseGraph(max_hop=1, dilation=1)
pose_edge_index = pose_graph.edge_index
IMU_graph = IMUGraph(max_hop=1, dilation=1)
IMU_edge_index = IMU_graph.edge_index

class MultiModalJLR(nn.Module):
    def __init__(self):
        super(MultiModalJLR, self).__init__()
        self.text_encoder = EmbeddingEncoder()
        self.pose_encoder = GraphPoseEncoderPre(num_nodes=Pose_joints, feature_dim=6, hidden_dim=128, embedding_dim=64, window_size=window, stride=stride_size, output_dim=Embedding_size)
        self.imu_encoder = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128, embedding_dim=64, window_size=window*4, stride=stride_size*4, output_dim=Embedding_size)
        self.single_imu_encoder = IMUSingleNodeEncoderWithClass(feature_dim=6, embedding_size=Embedding_size, temporal_hidden_size=256,num_classes=imu_positions)

    def forward(self, text, pose, imu, sing_imu, class_data):
        text_embeddings = self.text_encoder(text)
        pose_embeddings = self.pose_encoder(pose,pose_edge_index)
        imu_embeddings = self.imu_encoder(imu,IMU_edge_index)
        single_imu_embeddings = self.single_imu_encoder(sing_imu, class_data)
        
        return text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings

def compute_total_loss(model, text, pose, imu,):

    # Forward pass through the model to get embeddings
    text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings = model(text, pose, imu)
    
    # Initialize InfoNCE loss with hard negative mining
    #info_nce_loss = InfoNCELoss(hard_negative_mining=True)
    info_nce_loss = predefined_infonce()
    
    # Compute InfoNCE loss for each pair of embeddings
    loss_text_pose = info_nce_loss(text_embeddings, pose_embeddings)
    loss_text_imu = info_nce_loss(text_embeddings, imu_embeddings)
    loss_text_single_imu = info_nce_loss(text_embeddings, single_imu_embeddings)
    loss_pose_imu = info_nce_loss(pose_embeddings, imu_embeddings)
    loss_imu_single_imu = info_nce_loss(imu_embeddings, single_imu_embeddings)
    
    # Calculate the total loss as the sum of all individual losses
    total_loss = (loss_text_pose + loss_text_imu + 
                  loss_pose_imu
                  + loss_text_single_imu + loss_imu_single_imu)
    
    return total_loss

# Instantiate the model
model = MultiModalJLR()

text_data = torch.rand(batch_size, 768)
pose_data = torch.rand(batch_size, 25, Pose_joints, 6)
imu_data = torch.rand(batch_size, 100, imu_positions, 6)
single_imu_data = torch.rand(batch_size, 100, 1, 6)
class_data = torch.eye(imu_positions)[torch.randint(0, imu_positions, (batch_size,))]

# Perform a forward pass 
#text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings = model(text_data, pose_data, imu_data, single_imu_data, class_data)

# Print the shapes of the outputs
#print("Text Embeddings Shape:", text_embeddings.shape)
#print("Pose Embeddings Shape:", pose_embeddings.shape)
#print("IMU Embeddings Shape:", imu_embeddings.shape)
#print("Single IMU Embeddings Shape:", single_imu_embeddings.shape)

import torch.optim as optim

# Set device (e.g., GPU if available)
#device = "cpu"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the selected device
model = MultiModalJLR()#.to(device)

# Define optimizer (using Adam optimizer for this example)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define a number of epochs for training
epochs = 20

# Dummy data (replace this with your actual training dataset)
batch_size = 16
text_data = torch.rand(batch_size, 768)#.to(device)
pose_data = torch.rand(batch_size, 25, Pose_joints, 6)#.to(device)
imu_data = torch.rand(batch_size, 100, imu_positions, 6)#.to(device)
single_imu_data = torch.rand(batch_size, 100, 1, 6)#.to(device)
class_data = torch.eye(imu_positions)[torch.randint(0, imu_positions, (batch_size,))]#.to(device)

# Training Loop
for epoch in range(epochs):
    model.train()
    
    # Zero the gradients before running the backward pass
    optimizer.zero_grad()
    
    # Forward pass
    text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings = model(
        text_data, pose_data, imu_data, single_imu_data, class_data
    )
    
    # Compute loss
    total_loss = compute_total_loss(model, text_data, pose_data, imu_data)
    
    # Backward pass to compute gradients
    total_loss.backward()
    
    # Update parameters using optimizer
    optimizer.step()
    
    # Print loss every few epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item()}")

# After training, you can save the model's state_dict if needed
torch.save(model.state_dict(), 'multimodal_jlr_model.pth')
