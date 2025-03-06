import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from Encoder.Multi_IMU_Encoder import DeepConvGraphEncoderPre, IMUGraph, GATGraphEncoder
from Encoder.Gtr_Text_Encoder import EmbeddingEncoder
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph, GATPoseGraphEncoder
from Loss.pretrain_loss import predefined_infonce
from c_dataloader import UniMocapDataset, collate_fn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open a log file to store training progress
log_file = open("training_log.txt", "w")

def log_message(message):
    print(message)  # Print to console
    log_file.write(message + "\n")  # Save to log file

log_message("Starting Training...")

# Hyperparameters
batch_size = 256
Embedding_size = 512
window = 1
stride_size = 1
Pose_joints = 24
imu_positions = 21
hof = 3
dilation = 1

h5_file_path = "../CrosSim_Data/UniMocap/full_dataset.h5"

epochs = 300
base_max_norm = 1.0  # Initial value
min_max_norm = 1.0   # Lower bound
max_max_norm = 2.0  # Upper bound
adjustment_factor = 0.9  # How much to reduce dynamically
best_loss = float('inf')
early_stop_patience = 10 
no_improvement_epochs = 0
patience = 10
patience_factor = 0.5
learning_rate = 0.001

# Instantiate the dataset
dataset = UniMocapDataset(h5_file_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)

# Define subset size
#subset_size = int(len(dataset) * 0.05)
#remaining_size = len(dataset) - subset_size

# Split the dataset
#subset_dataset, _ = random_split(dataset, [subset_size, remaining_size])

# Create DataLoader
#dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)


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
        self.pose_edge_index = PoseGraph(max_hop=hof, dilation=dilation).edge_index.to(device)
        self.IMU_edge_index = IMUGraph(max_hop=hof, dilation=dilation).edge_index.to(device)

    def forward(self, text, pose, imu, imu_grav):
        text_embeddings = self.text_encoder(text)
        pose_embeddings = self.pose_encoder(pose, self.pose_edge_index)
        imu_embeddings = self.imu_encoder(imu, self.IMU_edge_index)
        imu_embeddings_grav = self.imu_encoder_grav(imu_grav, self.IMU_edge_index)
        return text_embeddings, pose_embeddings, imu_embeddings, imu_embeddings_grav

# Loss Computation Function
def compute_total_loss(text_embeddings, pose_embeddings, imu_embeddings, imu_embeddings_grav):
    def safe_infonce(x, y, name_x, name_y):
        if torch.isnan(x).any():
            log_message(f"NaN detected in {name_x}: {x}")
        if torch.isnan(y).any():
            log_message(f"NaN detected in {name_y}: {y}")

        x = torch.clamp(x, min=1e-8)  # Ensure no zero values
        y = torch.clamp(y, min=1e-8)
        loss = predefined_infonce(x, y)
        return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=-1.0)  # Replace NaN/Inf

    loss_text_pose = safe_infonce(text_embeddings, pose_embeddings, "text_embeddings", "pose_embeddings")
    loss_text_imu = safe_infonce(text_embeddings, imu_embeddings, "text_embeddings", "imu_embeddings")
    loss_text_single_imu = safe_infonce(text_embeddings, imu_embeddings_grav, "text_embeddings", "imu_embeddings_grav")
    loss_pose_imu = safe_infonce(pose_embeddings, imu_embeddings, "pose_embeddings", "imu_embeddings")
    loss_imu_single_imu = safe_infonce(imu_embeddings, imu_embeddings_grav, "imu_embeddings", "imu_embeddings_grav")
    loss_pose_single_imu = safe_infonce(pose_embeddings, imu_embeddings_grav, "pose_embeddings", "imu_embeddings_grav")

    total_loss = torch.nanmean(torch.stack([
        loss_text_pose, loss_text_imu, loss_pose_imu, loss_text_single_imu, loss_imu_single_imu, loss_pose_single_imu
    ]))

    #log_message(f"Losses: {loss_text_pose.item()}, {loss_text_imu.item()}, {loss_text_single_imu.item()}, {loss_pose_imu.item()}, {loss_imu_single_imu.item()}, {loss_pose_single_imu.item()}, {total_loss.item()}")

    return total_loss

# Initialize Model, Optimizer, and Scheduler
model = MultiModalJLR().to(device) 
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=patience_factor, patience=patience, verbose=True)
scaler = torch.amp.GradScaler()
torch.backends.cudnn.benchmark = True

# Training Loop with Progress Bar
loss_values = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for text_data, pose_data, imu_data, imu_data_grav in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()
        
        text_data = text_data.view(text_data.shape[0], 768).to(device, non_blocking=True)
        pose_data = pose_data.to(device, non_blocking=True)
        imu_data = imu_data.view(imu_data.shape[0], imu_data.shape[2], imu_data.shape[1], 6).to(device, non_blocking=True)
        imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav = model(text_data, pose_data, imu_data, imu_data_grav)
            total_loss = compute_total_loss(text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=base_max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.step()
        
        epoch_loss += total_loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    loss_values.append(avg_loss)
    log_message(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= early_stop_patience:
        log_message(f"Early stopping triggered after {epoch+1} epochs.")
        break
    
    scheduler.step(avg_loss)

plt.plot(loss_values, marker='o', linestyle='-', label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_loss_plot.png")
plt.close()

torch.save(model.state_dict(), 'multimodal_jlr_model.pth')
log_message("Training complete! Model saved as 'multimodal_jlr_model.pth'.")
log_file.close()

