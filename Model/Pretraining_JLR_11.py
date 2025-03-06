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
base_max_norm = 1.0
best_loss = float('inf')
early_stop_patience = 10
no_improvement_epochs = 0
patience = 10
patience_factor = 0.5
learning_rate = 0.001

# Instantiate the dataset
dataset = UniMocapDataset(h5_file_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)

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
    def safe_infonce(x, y):
        x = torch.clamp(x, min=1e-8)
        y = torch.clamp(y, min=1e-8)
        loss = predefined_infonce(x, y)
        return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=-1.0)

    loss_text_pose = safe_infonce(text_embeddings, pose_embeddings)
    loss_text_imu = safe_infonce(text_embeddings, imu_embeddings)
    loss_text_imugrav = safe_infonce(text_embeddings, imu_embeddings_grav)
    loss_pose_imu = safe_infonce(pose_embeddings, imu_embeddings)
    loss_imu_imugrav = safe_infonce(imu_embeddings, imu_embeddings_grav)
    loss_pose_imugrav = safe_infonce(pose_embeddings, imu_embeddings_grav)

    total_loss = torch.nanmean(torch.stack([
        loss_text_pose, loss_text_imu, loss_pose_imu, loss_text_imugrav, loss_imu_imugrav, loss_pose_imugrav
    ]))

    return total_loss, loss_text_pose, loss_text_imu, loss_pose_imu, loss_text_imugrav, loss_imu_imugrav, loss_pose_imugrav
# Initialize Model, Optimizer, and Scheduler
model = MultiModalJLR().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=patience_factor, patience=patience, verbose=True)
torch.backends.cudnn.benchmark = True

# Training Loop with Float32 Only
loss_values = []
loss_values_text_pose = []
loss_values_text_imu = []
loss_values_pose_imu = []
loss_values_text_imugrav = []
loss_values_imu_imugrav = []
loss_values_pose_imugrav = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_loss_text_pose = 0
    epoch_loss_text_imu = 0
    epoch_loss_pose_imu = 0
    epoch_loss_text_imugrav = 0
    epoch_loss_imu_imugrav = 0
    epoch_loss_pose_imugrav = 0

    for text_data, pose_data, imu_data, imu_data_grav in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()

         # Check for NaN values before processing
        if torch.isnan(text_data).any():
            log_message("NaN detected in raw text_data")
        if torch.isnan(pose_data).any():
            log_message("NaN detected in raw pose_data")
        if torch.isnan(imu_data).any():
            log_message("NaN detected in raw imu_data")
        if torch.isnan(imu_data_grav).any():
            log_message("NaN detected in raw imu_data_grav")
        
        text_data = text_data.view(text_data.shape[0], 768).to(device, dtype=torch.float32, non_blocking=True)
        pose_data = pose_data.to(device, dtype=torch.float32, non_blocking=True)
        imu_data = imu_data.view(imu_data.shape[0], imu_data.shape[2], imu_data.shape[1], 6).to(device, dtype=torch.float32, non_blocking=True)
        imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, dtype=torch.float32, non_blocking=True)

        text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav = model(text_data, pose_data, imu_data, imu_data_grav)

         # Check for NaN values in embeddings
        if torch.isnan(text_embeddings).any():
            log_message("NaN detected in text_embeddings")
        if torch.isnan(pose_embeddings).any():
            log_message("NaN detected in pose_embeddings")
        if torch.isnan(imu_embeddings).any():
            log_message("NaN detected in imu_embeddings")
        if torch.isnan(imu_emb_grav).any():
            log_message("NaN detected in imu_embeddings_grav")

        total_loss, loss_text_pose, loss_text_imu, loss_pose_imu, loss_text_imugrav, loss_imu_imugrav, loss_pose_imugrav = compute_total_loss(text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=base_max_norm)
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_loss_text_pose += loss_text_pose.item()
        epoch_loss_text_imu += loss_text_imu.item()
        epoch_loss_pose_imu += loss_pose_imu.item()
        epoch_loss_text_imugrav += loss_text_imugrav.item()
        epoch_loss_imu_imugrav += loss_imu_imugrav.item()
        epoch_loss_pose_imugrav += loss_pose_imugrav.item()

    avg_loss = epoch_loss / len(dataloader)
    avg_loss_text_pose = epoch_loss_text_pose / len(dataloader)
    avg_loss_text_imu = epoch_loss_text_imu / len(dataloader)
    avg_loss_pose_imu = epoch_loss_pose_imu / len(dataloader)
    avg_loss_text_imugrav = epoch_loss_text_imugrav / len(dataloader)
    avg_loss_imu_imugrav = epoch_loss_imu_imugrav / len(dataloader)
    avg_loss_pose_imugrav = epoch_loss_pose_imugrav / len(dataloader)

    loss_values.append(avg_loss)
    loss_values_text_pose.append(avg_loss_text_pose)
    loss_values_text_imu.append(avg_loss_text_imu)
    loss_values_pose_imu.append(avg_loss_pose_imu)
    loss_values_text_imugrav.append(avg_loss_text_imugrav)
    loss_values_imu_imugrav.append(avg_loss_imu_imugrav)
    loss_values_pose_imugrav.append(avg_loss_pose_imugrav)

    log_message(f"Epoch [{epoch+1}/{epochs}], avg_loss: {avg_loss:.4f},  avg_loss_pose_imugrav: {avg_loss_pose_imugrav:.4f},  avg_loss_text_pose: {avg_loss_text_pose:.4f}, avg_loss_text_imu: {avg_loss_text_imu:.4f} , avg_loss_pose_imu: {avg_loss_pose_imu:.4f},  avg_loss_text_imugrav: {avg_loss_text_imugrav:.4f},  avg_loss_imu_imugrav: {avg_loss_imu_imugrav:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= early_stop_patience:
        log_message(f"Early stopping triggered after {epoch+1} epochs.")
        break

    scheduler.step(avg_loss)

for epoch in range(epochs):
    model.train()
    
    for text_data, pose_data, imu_data, imu_data_grav in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()
        
       
        
        text_data = text_data.view(text_data.shape[0], 768).to(device, non_blocking=True)
        pose_data = pose_data.to(device, non_blocking=True)
        imu_data = imu_data.view(imu_data.shape[0], imu_data.shape[2], imu_data.shape[1], 6).to(device, non_blocking=True)
        imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, non_blocking=True)
        
        # Forward Pass
        text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav = model(text_data, pose_data, imu_data, imu_data_grav)
        
       
        
log_file.close()

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
