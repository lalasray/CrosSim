import torch
import torch.nn as nn
from Encoder.Multi_IMU_Encoder import DeepConvGraphEncoderPre, IMUGraph
from Encoder.Gtr_Text_Encoder import EmbeddingEncoder
from Encoder.Single_IMU_Encoder import IMUSingleNodeEncoderWithClass
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph
from Loss.pretrain_loss import predefined_infonce
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
Embedding_size = 512
window = 1
stride_size = 1
Pose_joints = 24
imu_positions = 20

pose_graph = PoseGraph(max_hop=1, dilation=1)
pose_edge_index = pose_graph.edge_index.to(device)  # Move to GPU
IMU_graph = IMUGraph(max_hop=1, dilation=1)
IMU_edge_index = IMU_graph.edge_index.to(device)  # Move to GPU

class MultiModalJLR(nn.Module):
    def __init__(self):
        super(MultiModalJLR, self).__init__()
        self.text_encoder = EmbeddingEncoder().to(device)
        self.pose_encoder = GraphPoseEncoderPre(num_nodes=Pose_joints, feature_dim=6, hidden_dim=128, embedding_dim=64, window_size=window, stride=stride_size, output_dim=Embedding_size).to(device)
        self.imu_encoder = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128, embedding_dim=64, window_size=window*4, stride=stride_size*4, output_dim=Embedding_size).to(device)
        self.single_imu_encoder = IMUSingleNodeEncoderWithClass(feature_dim=6, embedding_size=Embedding_size, temporal_hidden_size=256, num_classes=imu_positions).to(device)

    def forward(self, text, pose, imu, sing_imu, class_data):
        text_embeddings = self.text_encoder(text)
        pose_embeddings = self.pose_encoder(pose, pose_edge_index)
        imu_embeddings = self.imu_encoder(imu, IMU_edge_index)
        single_imu_embeddings = self.single_imu_encoder(sing_imu, class_data)
        
        return text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings

def compute_total_loss(text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings):
    loss_text_pose = predefined_infonce(text_embeddings, pose_embeddings)
    loss_text_imu = predefined_infonce(text_embeddings, imu_embeddings)
    loss_text_single_imu = predefined_infonce(text_embeddings, single_imu_embeddings)
    loss_pose_imu = predefined_infonce(pose_embeddings, imu_embeddings)
    loss_imu_single_imu = predefined_infonce(imu_embeddings, single_imu_embeddings)

    total_loss = (loss_text_pose + loss_text_imu + loss_pose_imu + loss_text_single_imu + loss_imu_single_imu) / 5
    return total_loss

# Initialize model and optimizer
model = MultiModalJLR().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Early stopping parameters
early_stopping_patience = 20
best_loss = float('inf')
stopping_counter = 0

# Generate data and move to GPU
text_data = torch.rand(batch_size, 768).to(device)
pose_data = torch.rand(batch_size, 25, Pose_joints, 6).to(device)
imu_data = torch.rand(batch_size, 100, imu_positions, 6).to(device)
single_imu_data = torch.rand(batch_size, 100, 1, 6).to(device)
class_data = torch.eye(imu_positions)[torch.randint(0, imu_positions, (batch_size,))].to(device)

# Training loop
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings = model(text_data, pose_data, imu_data, single_imu_data, class_data)
    
    total_loss = compute_total_loss(text_embeddings, pose_embeddings, imu_embeddings, single_imu_embeddings)
    total_loss.backward()
    optimizer.step()
    
    # Scheduler step
    scheduler.step(total_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item()}")
    
    # Early stopping check
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        stopping_counter = 0
    else:
        stopping_counter += 1
        if stopping_counter >= early_stopping_patience:
            print("Early stopping triggered. Training stopped.")
            break

# Save model
torch.save(model.state_dict(), 'multimodal_jlr_model.pth')
