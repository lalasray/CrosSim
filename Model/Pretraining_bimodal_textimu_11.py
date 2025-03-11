import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from Encoder.Multi_IMU_Encoder import DeepConvGraphEncoderPre, IMUGraph
from Encoder.Gtr_Text_Encoder import EmbeddingEncoder
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph
from Loss.pretrain_loss import contra_loss
from Loss.to_text_loss import loss_fn
from c_dataloader import UniMocapDataset, collate_fn
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging Function
def log_message(log_file, message):
    print(message)
    log_file.write(message + "\n")

# Function to count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Model Definition
class BiModalIMU(nn.Module):
    def __init__(self, embedding_size=768, pose_joints=24, imu_positions=21, window=1, stride_size=1, hof=3, dilation=1):
        super(BiModalIMU, self).__init__()
        self.text_encoder = EmbeddingEncoder(output_size=embedding_size).to(device)
        self.imu_encoder_grav = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128,
                                                        embedding_dim=64, window_size=window * 4, stride=stride_size * 4,
                                                        output_dim=embedding_size).to(device)
        self.IMU_edge_index = IMUGraph(max_hop=hof, dilation=dilation).edge_index.to(device)

    def forward(self, text, imu_grav):
        text_embeddings = self.text_encoder(text)
        imu_embeddings_grav, imuint = self.imu_encoder_grav(imu_grav, self.IMU_edge_index)
        return text_embeddings, imu_embeddings_grav

# Training Function
def train_bimodel(epochs=300, batch_size=128, learning_rate=0.001, early_stop_patience=10, patience=7, patience_factor=0.5, h5_file_path = "../CrosSim_Data/UniMocap/full_dataset.h5"):
    log_file = open("training_log_textimu_downtext.txt", "w")
    log_message(log_file, "Starting Training...")

    # Load dataset
    h5_file_path = h5_file_path
    dataset = UniMocapDataset(h5_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    # Initialize model, optimizer, scheduler
    model = BiModalIMU().to(device)
    print(f"Total Trainable Parameters: {count_parameters(model):,}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=patience_factor, patience=patience, verbose=True)

    best_loss = float('inf')
    no_improvement_epochs = 0

    loss_values = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for text_data, pose_data, imu_data, imu_data_grav in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            text_data = text_data.view(text_data.shape[0], 768).to(device, dtype=torch.float32, non_blocking=True)
            imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, dtype=torch.float32, non_blocking=True)

            text_embeddings, imu_emb_grav = model(text_data, imu_data_grav)


           # Compute contra loss
            contra_loss_val = contra_loss(text_embeddings, imu_emb_grav)

            # Combine both losses
            total_loss = contra_loss_val 

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_values.append(avg_loss)

        log_message(log_file, f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stop_patience:
            log_message(log_file, f"Early stopping triggered at epoch {epoch+1}.")
            break

        scheduler.step(avg_loss)

    torch.save(model.state_dict(), "textimu_downtext.pth")
    log_message(log_file, "Training complete! Model saved.")

    log_file.close()

if __name__ == "__main__":
    train_bimodel()
