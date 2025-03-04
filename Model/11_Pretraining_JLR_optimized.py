import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

# Import necessary modules
from Encoder.Multi_IMU_Encoder import DeepConvGraphEncoderPre, IMUGraph, GATGraphEncoder
from Encoder.Gtr_Text_Encoder import EmbeddingEncoder
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph, GATPoseGraphEncoder
from Loss.pretrain_loss import predefined_infonce
from c_dataloader import UniMocapDataset, collate_fn
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 256
epochs = 100
Embedding_size = 256
window = 1
stride_size = 1
Pose_joints = 24
imu_positions = 21
h5_file_path = "../CrosSim_Data/UniMocap/full_dataset.h5"

# Load dataset with optimized DataLoader
dataset = UniMocapDataset(h5_file_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, 
                        num_workers=4, pin_memory=True, persistent_workers=True)

# Define MultiModal Model
class MultiModalJLR(nn.Module):
    def __init__(self):
        super(MultiModalJLR, self).__init__()
        self.text_encoder = EmbeddingEncoder(output_size=Embedding_size).to(device)
        self.pose_encoder = GraphPoseEncoderPre(num_nodes=Pose_joints, feature_dim=6, hidden_dim=128,
                                                embedding_dim=64, window_size=window, stride=stride_size,
                                                output_dim=Embedding_size).to(device)
        self.imu_encoder = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128,
                                                   embedding_dim=64, window_size=window*4, stride=stride_size*4,
                                                   output_dim=Embedding_size).to(device)
        self.imu_encoder_grav = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128,
                                                        embedding_dim=64, window_size=window*4, stride=stride_size*4,
                                                        output_dim=Embedding_size).to(device)
        self.pose_edge_index = PoseGraph(max_hop=1, dilation=1).edge_index.to(device)
        self.IMU_edge_index = IMUGraph(max_hop=1, dilation=1).edge_index.to(device)

    def forward(self, text, pose, imu, imu_grav):
        text_embeddings = self.text_encoder(text)
        pose_embeddings = self.pose_encoder(pose, self.pose_edge_index)
        imu_embeddings = self.imu_encoder(imu, self.IMU_edge_index)
        imu_embeddings_grav = self.imu_encoder_grav(imu_grav, self.IMU_edge_index)
        return text_embeddings, pose_embeddings, imu_embeddings, imu_embeddings_grav

# Optimized Loss Computation Function
def compute_total_loss(*embeddings):
    loss = sum(predefined_infonce(a, b) for i, a in enumerate(embeddings) for b in embeddings[i+1:])
    return loss / len(embeddings)

# Enable cuDNN Benchmark for better performance
torch.backends.cudnn.benchmark = True

# Initialize Model, Optimizer, AMP Scaler
model = torch.compile(MultiModalJLR().to(device))  
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scaler = torch.amp.GradScaler()  
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Early Stopping Parameters
patience = 15  # Stop training if no improvement in `patience` epochs
best_loss = float("inf")  # Track best loss
epochs_no_improve = 0  # Count epochs with no improvement
best_model_path = "best_multimodal_jlr_model.pth"

# Gradient Accumulation Setting
accumulation_steps = 4  # Adjust based on available memory

# Training Loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for i, (text_data, pose_data, imu_data, imu_data_grav) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
        optimizer.zero_grad()

        # Move Data to GPU Efficiently
        text_data = text_data.view(text_data.shape[0], 768).to(device, non_blocking=True)
        pose_data = pose_data.to(device, non_blocking=True)
        imu_data = imu_data.view(imu_data.shape[0], imu_data.shape[2], imu_data.shape[1], 6).to(device, non_blocking=True)
        imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, non_blocking=True)

        # Forward Pass with AMP
        with torch.amp.autocast(device_type='cuda'):  
            text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav = model(text_data, pose_data, imu_data, imu_data_grav)
            total_loss = compute_total_loss(text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav) / accumulation_steps

        # Backpropagation with Gradient Accumulation
        scaler.scale(total_loss).backward()
        
        if (i + 1) % accumulation_steps == 0:  # Accumulation steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += total_loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")

    # Save Best Model if Loss Improves
    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch+1} with loss: {best_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve}/{patience} epochs.")

    # Adjust Learning Rate
    scheduler.step(avg_loss)

    # Early Stopping Check
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered! Training stopped at epoch {epoch+1}.")
        break

# Final Save (in case training completes without early stopping)
if not os.path.exists(best_model_path):
    torch.save(model.state_dict(), best_model_path)
print(f"Training complete! Best model saved as '{best_model_path}'.")
