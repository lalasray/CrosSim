import torch
import torch.nn as nn
from Encoder.Multi_IMU_Encoder import DeepConvGraphEncoderPre, IMUGraph
from Encoder.Gtr_Text_Encoder import EmbeddingEncoder
from Encoder.Single_IMU_Encoder import IMUSingleNodeEncoderWithClass
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph
from Loss.pretrain_loss import predefined_infonce
import torch.optim as optim
from dataloader_var import MotionDataset,OGMotionDataset,collate_fn
from torch.utils.data import DataLoader, ConcatDataset


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32


data_dir = "/home/lala/Documents/GitHub/CrosSim/CrosSim_Data/UniMocap/processed"  # Update path
OGdataset = OGMotionDataset(data_dir)
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
sensor_positions_acc = [
    "back.acc", "belt.acc", "chest.acc", "forehead.acc",
    "left_arm.acc", "left_ear.acc", "left_foot.acc", "left_shin.acc",
    "left_shirt_pocket.acc", "left_shoulder.acc", "left_thigh.acc", "left_wrist.acc",
    "necklace.acc", "right_arm.acc", "right_ear.acc", "right_foot.acc", 
    "right_shin.acc", "right_shirt_pocket.acc", "right_shoulder.acc",
    "right_thigh.acc", "right_wrist.acc"
]

sensor_positions_gyro = [
    "back.gyro", "belt.gyro", "chest.gyro", "forehead.gyro",
    "left_arm.gyro", "left_ear.gyro", "left_foot.gyro", "left_shin.gyro",
    "left_shirt_pocket.gyro", "left_shoulder.gyro", "left_thigh.gyro", "left_wrist.gyro",
    "necklace.gyro", "right_arm.gyro", "right_ear.gyro", "right_foot.gyro", 
    "right_shin.gyro", "right_shirt_pocket.gyro", "right_shoulder.gyro",
    "right_thigh.gyro", "right_wrist.gyro"
]

sensor_positions_acc_g = [
    "back.acc_g", "belt.acc_g", "chest.acc_g", "forehead.acc_g",
    "left_arm.acc_g", "left_ear.acc_g", "left_foot.acc_g", "left_shin.acc_g",
    "left_shirt_pocket.acc_g", "left_shoulder.acc_g", "left_thigh.acc_g", "left_wrist.acc_g",
    "necklace.acc_g", "right_arm.acc_g", "right_ear.acc_g", "right_foot.acc_g", 
    "right_shin.acc_g", "right_shirt_pocket.acc_g", "right_shoulder.acc_g",
    "right_thigh.acc_g", "right_wrist.acc_g"
]

combined_dataset = ConcatDataset(datasets)
dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
'''
for batch in dataloader:
    print("Batch keys:", batch.keys())
    print("Motion shape:", batch["motion"].squeeze(1).shape )
    pose = torch.cat([batch["pose_trans"], batch["pose_body"]], dim=-1)
    full_Pose = pose.view(pose.shape[0], pose.shape[1], 24, 3)
    pose_with_angle = torch.cat([full_Pose, batch["pose_joint"].squeeze(2)], dim=-1)
    print("Pose Joint shape:", pose_with_angle.shape) 
    # Stack data along the new axis (joints dimension)
    combined_data_acc = torch.stack([batch[key] for key in sensor_positions_acc], dim=2)
    combined_data_gyro = torch.stack([batch[key] for key in sensor_positions_gyro], dim=2)
    combined_imu = torch.cat((combined_data_acc, combined_data_gyro), dim=3)

    combined_data_acc_grav = torch.stack([batch[key] for key in sensor_positions_acc_g], dim=2)
    combined_imu_grav =  torch.cat((combined_data_acc_grav, combined_data_gyro), dim=3)

    print("imu_acc shape:", combined_imu.shape)  # Expected output: (batch, 800, 21, 3)
    print("imu_acc_g shape:", combined_imu_grav.shape)
    break
'''
Embedding_size = 768
window = 1
stride_size = 1
Pose_joints = 24
imu_positions = 21

pose_graph = PoseGraph(max_hop=1, dilation=1)
pose_edge_index = pose_graph.edge_index.to(device)  # Move to GPU
IMU_graph = IMUGraph(max_hop=1, dilation=1)
IMU_edge_index = IMU_graph.edge_index.to(device)  # Move to GPU

class MultiModalJLR(nn.Module):
    def __init__(self):
        super(MultiModalJLR, self).__init__()
        self.text_encoder = EmbeddingEncoder(output_size=Embedding_size).to(device)
        self.pose_encoder = GraphPoseEncoderPre(num_nodes=Pose_joints, feature_dim=6, hidden_dim=128, embedding_dim=64, window_size=window, stride=stride_size, output_dim=Embedding_size).to(device)
        self.imu_encoder = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128, embedding_dim=64, window_size=window*4, stride=stride_size*4, output_dim=Embedding_size).to(device)
        self.imu_encoder_grav = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128, embedding_dim=64, window_size=window*4, stride=stride_size*4, output_dim=Embedding_size).to(device)

    def forward(self, text, pose, imu, imu_grav):
        text_embeddings = self.text_encoder(text)
        pose_embeddings = self.pose_encoder(pose, pose_edge_index)
        imu_embeddings = self.imu_encoder(imu, IMU_edge_index)
        imu_embeddings_grav = self.imu_encoder_grav(imu_grav, IMU_edge_index)
        
        return text_embeddings, pose_embeddings, imu_embeddings, imu_embeddings_grav

def compute_total_loss(text_embeddings, pose_embeddings, imu_embeddings, imu_embeddings_grav):
    loss_text_pose = predefined_infonce(text_embeddings, pose_embeddings)
    loss_text_imu = predefined_infonce(text_embeddings, imu_embeddings)
    loss_text_single_imu = predefined_infonce(text_embeddings, imu_embeddings_grav)
    loss_pose_imu = predefined_infonce(pose_embeddings, imu_embeddings)
    loss_imu_single_imu = predefined_infonce(imu_embeddings, imu_embeddings_grav)

    total_loss = (loss_text_pose + loss_text_imu + loss_pose_imu + loss_text_single_imu + loss_imu_single_imu) / 5
    return total_loss

# Initialize model and optimizer
model = MultiModalJLR().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Early stopping parameters
early_stopping_patience = 200
best_loss = float('inf')
stopping_counter = 0
epochs = 2000

# Training loop

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    for batch in dataloader:
        
        text_data = batch["motion"].squeeze(1)
        
        pose = torch.cat([batch["pose_trans"], batch["pose_body"]], dim=-1)
        full_Pose = pose.view(pose.shape[0], pose.shape[1], 24, 3)
        pose_data = torch.cat([full_Pose, batch["pose_joint"].squeeze(2)], dim=-1)
        
        combined_data_acc = torch.stack([batch[key] for key in sensor_positions_acc], dim=2)
        combined_data_gyro = torch.stack([batch[key] for key in sensor_positions_gyro], dim=2)
        imu_data = torch.cat((combined_data_acc, combined_data_gyro), dim=3)

        combined_data_acc_grav = torch.stack([batch[key] for key in sensor_positions_acc_g], dim=2)
        imu_data_grav =  torch.cat((combined_data_acc_grav, combined_data_gyro), dim=3)

        text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav = model(text_data, pose_data, imu_data, imu_data_grav)
    
        total_loss = compute_total_loss(text_embeddings, pose_embeddings, imu_embeddings, imu_emb_grav)
        total_loss.backward()
        optimizer.step()
    
        # Scheduler step
        scheduler.step(total_loss)
    
        if (epoch + 1) % 2 == 0:
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