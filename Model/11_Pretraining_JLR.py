import torch
torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for progress bar

from Encoder.Multi_IMU_Encoder import DeepConvGraphEncoderPre, IMUGraph
from Encoder.Gtr_Text_Encoder import EmbeddingEncoder
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph
from Loss.pretrain_loss import predefined_infonce
from dataloader_var import MotionDataset, OGMotionDataset, collate_fn
from torch.utils.data import DataLoader
#

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 8
epochs = 100
Embedding_size = 256
window = 1
stride_size = 1
Pose_joints = 24
imu_positions = 21

# Load Dataset
data_dir = "/home/lala/Documents/GitHub/CrosSim/CrosSim_Data/UniMocap/processed"
OGdataset = OGMotionDataset(data_dir)
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
'''
dataloader = DataLoader(OGdataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)#, num_workers = 4)# * torch.cuda.device_count())
#combined_dataset = ConcatDataset(datasets)
#dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Sensor Positions
sensor_positions_acc = ["back.acc", "belt.acc", "chest.acc", "forehead.acc",
                        "left_arm.acc", "left_ear.acc", "left_foot.acc", "left_shin.acc",
                        "left_shirt_pocket.acc", "left_shoulder.acc", "left_thigh.acc", "left_wrist.acc",
                        "necklace.acc", "right_arm.acc", "right_ear.acc", "right_foot.acc",
                        "right_shin.acc", "right_shirt_pocket.acc", "right_shoulder.acc",
                        "right_thigh.acc", "right_wrist.acc"]

sensor_positions_gyro = [pos.replace(".acc", ".gyro") for pos in sensor_positions_acc]
sensor_positions_acc_g = [pos + "_g" for pos in sensor_positions_acc]

# Graph Initialization
pose_graph = PoseGraph(max_hop=1, dilation=1)
pose_edge_index = pose_graph.edge_index.to(device)
IMU_graph = IMUGraph(max_hop=1, dilation=1)
IMU_edge_index = IMU_graph.edge_index.to(device)

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

    def forward(self, text, pose, imu, imu_grav):
        text_embeddings = self.text_encoder(text)
        pose_embeddings = self.pose_encoder(pose, pose_edge_index)
        imu_embeddings = self.imu_encoder(imu, IMU_edge_index)
        imu_embeddings_grav = self.imu_encoder_grav(imu_grav, IMU_edge_index)
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
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Training Loop with Progress Bar
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")  # Initialize tqdm
    
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        #batch = batch.to(device)
        # Prepare Data
        text_data = batch["motion"].squeeze(1).to(device)
        pose = torch.cat([batch["pose_trans"], batch["pose_body"]], dim=-1)
        full_Pose = pose.view(pose.shape[0], pose.shape[1], 24, 3)
        pose_data = torch.cat([full_Pose, batch["pose_joint"].squeeze(2)], dim=-1).to(device)

        combined_data_acc = torch.stack([batch[key] for key in sensor_positions_acc], dim=2)
        combined_data_gyro = torch.stack([batch[key] for key in sensor_positions_gyro], dim=2)
        imu_data = torch.cat((combined_data_acc, combined_data_gyro), dim=3).to(device)

        combined_data_acc_grav = torch.stack([batch[key] for key in sensor_positions_acc_g], dim=2)
        imu_data_grav = torch.cat((combined_data_acc_grav, combined_data_gyro), dim=3).to(device)

        # Forward Pass
        text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav = model(
            text_data, pose_data, imu_data, imu_data_grav
        )

        # Compute Loss
        total_loss = compute_total_loss(text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav)
        total_loss.backward()
        optimizer.step()

        # Update Progress Bar
        batch_loss = total_loss.item()
        epoch_loss += batch_loss
        progress_bar.set_postfix({"Batch Loss": batch_loss})  # Show batch loss in tqdm

    # Compute average loss per epoch
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")

    # Adjust learning rate if using ReduceLROnPlateau
    scheduler.step(avg_loss if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

# Save Model
torch.save(model.state_dict(), 'multimodal_jlr_model.pth')
print("Training complete! Model saved as 'multimodal_jlr_model.pth'.")
