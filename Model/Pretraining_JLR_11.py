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

# Hyperparameters
batch_size = 256
Embedding_size = 512
window = 1
stride_size = 1
Pose_joints = 24
imu_positions = 21
hof = 1
dilation = 1

h5_file_path = "../CrosSim_Data/UniMocap/full_dataset.h5"

epochs = 100
base_max_norm = 5.0  # Initial value
min_max_norm = 1.0   # Lower bound
max_max_norm = 10.0  # Upper bound
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

# Define MultiModal Model
class MultiModalJLR(nn.Module):
    def __init__(self):
        super(MultiModalJLR, self).__init__()
        self.text_encoder = EmbeddingEncoder(output_size=Embedding_size).to(device)
        self.pose_encoder = GraphPoseEncoderPre(num_nodes=Pose_joints, feature_dim=6, hidden_dim=128,
                                                embedding_dim=64, window_size=window, stride=stride_size,
                                                output_dim=Embedding_size).to(device)
        #self.pose_encoder = GATPoseGraphEncoder(num_nodes=24, feature_dim=6, hidden_dim=128, window_size=1, stride=1)
        self.imu_encoder = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128,
                                                   embedding_dim=64, window_size=window*4, stride=stride_size*4,
                                                   output_dim=Embedding_size).to(device)
        #self.imu_encoder = GATGraphEncoder(num_nodes=21, feature_dim=6, hidden_dim=32, window_size=1, stride=1)
        self.imu_encoder_grav = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128,
                                                        embedding_dim=64, window_size=window*4, stride=stride_size*4,
                                                        output_dim=Embedding_size).to(device)
        #self.imu_encoder_grav = GATGraphEncoder(num_nodes=21, feature_dim=6, hidden_dim=32, window_size=1, stride=1)
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
    loss_text_pose = predefined_infonce(text_embeddings, pose_embeddings)
    loss_text_imu = predefined_infonce(text_embeddings, imu_embeddings)
    loss_text_single_imu = predefined_infonce(text_embeddings, imu_embeddings_grav)
    loss_pose_imu = predefined_infonce(pose_embeddings, imu_embeddings)
    loss_imu_single_imu = predefined_infonce(imu_embeddings, imu_embeddings_grav)
    total_loss = (loss_text_pose + loss_text_imu + loss_pose_imu + loss_text_single_imu + loss_imu_single_imu) / 5
    return total_loss

# Initialize Model, Optimizer, and Scheduler
model = MultiModalJLR().to(device) 
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=patience_factor, patience=patience, verbose=True)
scaler = torch.amp.GradScaler()
torch.backends.cudnn.benchmark = True

# Training Loop with Progress Bar
loss_values = []  # Store loss per epoch

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for text_data, pose_data, imu_data, imu_data_grav in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()

        # Prepare Data
        text_data = text_data.view(text_data.shape[0], 768).to(device, non_blocking=True)
        pose_data = pose_data.to(device, non_blocking=True)
        imu_data = imu_data.view(imu_data.shape[0], imu_data.shape[2], imu_data.shape[1], 6).to(device, non_blocking=True)
        imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, non_blocking=True)

        # Forward Pass with AMP
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
        #with torch.cuda.amp.autocast(enabled=False):
            text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav = model(text_data, pose_data, imu_data, imu_data_grav)
            total_loss = compute_total_loss(text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav)

        # Backward Pass with AMP
        scaler.scale(total_loss).backward()
        
        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=base_max_norm)

        scaler.step(optimizer)
        scaler.update()

        optimizer.step()

        # Track Loss
        epoch_loss += total_loss.item()
    
    # Compute Average Loss Per Epoch
    avg_loss = epoch_loss / len(dataloader)
    loss_values.append(avg_loss)  # Store loss for plotting
    print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")
    
    # Check for Early Stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improvement_epochs = 0  # Reset no improvement counter
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= early_stop_patience:
        print(f"Early stopping triggered after {epoch+1} epochs with no improvement in loss.")
        break
    
    # Adjust Learning Rate
    scheduler.step(avg_loss)

# Plot Training Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(loss_values, marker='o', linestyle='-', color='b', label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Trend Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Save Model
torch.save(model.state_dict(), 'multimodal_jlr_model.pth')
print("Training complete! Model saved as 'multimodal_jlr_model.pth'.")




'''
openpack = MotionDataset(data_dir, "openpack")
alshar = MotionDataset(data_dir, "alshar")
opportunity = MotionDataset(data_dir, "opportunity")
ucihar = MotionDataset(data_dir, "ucihar")
wHAR = MotionDataset(data_dir, "wHAR")
shoaib = MotionDataset(data_dir, "shoaib")
har70 = MotionDataset(data_dir, "har70")
realworld = MotionDataset(data_dir, "realworld")
pamap2 = MotionDataset(data_dir, "pamap2")
uschad = MotionDataset(data_dir, "uschad")
mhealth = MotionDataset(data_dir, "mhealth")
harth = MotionDataset(data_dir, "harth")
wharf = MotionDataset(data_dir, "wharf")
dsads = MotionDataset(data_dir, "dsads")
wisdm = MotionDataset(data_dir, "wisdm")
utdmhad = MotionDataset(data_dir, "utdmhad")
mmact = MotionDataset(data_dir, "mmact")
mmfit = MotionDataset(data_dir, "mmfit")
dip = MotionDataset(data_dir, "dip")
totalcapture = MotionDataset(data_dir, "totalcapture")
datasets = [
    OGdataset, openpack, alshar, opportunity, utdmhad, ucihar, wHAR, shoaib,
    har70, realworld, pamap2, uschad, mhealth, harth, wharf, wisdm, dsads,
    mmact, mmfit, dip, totalcapture
]

# Sensor Positions
#sensor_positions_acc = ["back.acc", "belt.acc", "chest.acc", "forehead.acc", "left_arm.acc", "left_ear.acc", "left_foot.acc", "left_shin.acc", "left_shirt_pocket.acc", "left_shoulder.acc", "left_thigh.acc", "left_wrist.acc", "necklace.acc", "right_arm.acc", "right_ear.acc", "right_foot.acc", "right_shin.acc", "right_shirt_pocket.acc", "right_shoulder.acc", "right_thigh.acc", "right_wrist.acc"]
'''