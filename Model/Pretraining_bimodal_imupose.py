import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from architectures import BiModalPose, count_parameters
from Loss.pretrain_loss import contra_loss
from c_dataloader import UniMocapDataset, collate_fn
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Function
def train_bipose(epochs=500, batch_size=128, learning_rate=0.001, early_stop_patience=20, patience=15, patience_factor=0.5, h5_file_path = "../CrosSim_Data/UniMocap/full_dataset.h5"):
    # Load dataset
    h5_file_path = h5_file_path
    dataset = UniMocapDataset(h5_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    # Initialize model, optimizer, scheduler
    model = BiModalPose().to(device)
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
            pose_data = pose_data.to(device, dtype=torch.float32, non_blocking=True)
            imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, dtype=torch.float32, non_blocking=True)

            pose_embeddings,imu_emb_grav = model(pose_data, imu_data_grav)

            total_loss = contra_loss(pose_embeddings, imu_emb_grav)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_values.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stop_patience:
            break

        scheduler.step(avg_loss)

    torch.save(model.state_dict(), "imupose.pth")

    log_file.close()

if __name__ == "__main__":
    train_bipose()
