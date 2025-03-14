import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import os
from architectures import BiModalIMUDownPose, count_parameters
from Loss.pretrain_loss import contra_loss
from Loss.to_pose_loss import pose_loss
from c_dataloader import UniMocapDataset, collate_fn
from torch.utils.data import DataLoader, Subset, random_split

import random
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Function
def train_bimodeldown(epochs=500, batch_size=128, learning_rate=0.001, early_stop_patience=20, patience=15, patience_factor=0.5, h5_file_path="../CrosSim_Data/UniMocap/full_dataset.h5"):
    # Initialize WandB
    wandb.init(project="CrosSim", name="TextIMU-DownPose", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "early_stop_patience": early_stop_patience,
        "patience": patience,
        "patience_factor": patience_factor
    })

    # Load dataset and split into train/validation
    dataset = UniMocapDataset(h5_file_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True)

    # Initialize model, optimizer, scheduler
    model = BiModalIMUDownPose().to(device)
    print(f"Total Trainable Parameters: {count_parameters(model):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=patience_factor, patience=patience, verbose=True)

    best_loss = float('inf')
    no_improvement_epochs = 0
    model_path = "best_textimu_downpose.pth"

    # Load existing checkpoint (if available)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-existing model checkpoint.")

    first_epoch_contra_loss = None
    first_epoch_pose_loss = None

    for epoch in range(epochs):
        model.train()
        train_loss, train_contra_loss, train_pose_loss = 0.0, 0.0, 0.0

        for text_data, pose_data, imu_data, imu_data_grav in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            optimizer.zero_grad()
            text_data = text_data.view(text_data.shape[0], 768).to(device, dtype=torch.float32, non_blocking=True)
            imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, dtype=torch.float32, non_blocking=True)
            pose_data = pose_data.to(device, dtype=torch.float32, non_blocking=True)

            text_embeddings, imu_emb_grav, pose_emb = model(text_data, imu_data_grav)

            # Compute the contrastive loss and pose loss
            contra_loss_val = contra_loss(text_embeddings, imu_emb_grav)
            pose_loss_val = pose_loss(pose_emb, pose_data)

            if first_epoch_contra_loss is None and first_epoch_pose_loss is None:
                first_epoch_contra_loss = contra_loss_val.item()
                first_epoch_pose_loss = pose_loss_val.item()

            # Use first epoch losses for normalization
            contrastive_loss_weight = first_epoch_pose_loss / first_epoch_contra_loss if first_epoch_contra_loss != 0 else 1.0
            pose_loss_weight = first_epoch_contra_loss / first_epoch_pose_loss if first_epoch_pose_loss != 0 else 1.0

            # Total loss with fixed weights
            total_loss = contrastive_loss_weight * contra_loss_val + pose_loss_weight * pose_loss_val
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track losses
            train_loss += total_loss.item()
            train_contra_loss += contra_loss_val.item()
            train_pose_loss += pose_loss_val.item()

        # Compute average training losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_contra_loss = train_contra_loss / len(train_loader)
        avg_train_pose_loss = train_pose_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss, val_contra_loss, val_pose_loss = 0.0, 0.0, 0.0

        with torch.no_grad():
            for text_data, pose_data, imu_data, imu_data_grav in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                text_data = text_data.view(text_data.shape[0], 768).to(device, dtype=torch.float32, non_blocking=True)
                imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, dtype=torch.float32, non_blocking=True)
                pose_data = pose_data.to(device, dtype=torch.float32, non_blocking=True)

                text_embeddings, imu_emb_grav, pose_emb = model(text_data, imu_data_grav)

                contra_loss_val = contra_loss(text_embeddings, imu_emb_grav)
                pose_loss_val = pose_loss(pose_emb, pose_data)

                total_loss = contrastive_loss_weight * contra_loss_val + pose_loss_weight * pose_loss_val
                val_loss += total_loss.item()
                val_contra_loss += contra_loss_val.item()
                val_pose_loss += pose_loss_val.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_contra_loss = val_contra_loss / len(val_loader)
        avg_val_pose_loss = val_pose_loss / len(val_loader)

        wandb.log({
            "Epoch": epoch+1,
            "Train Loss": avg_train_loss,
            "Train Contrastive Loss": avg_train_contra_loss,
            "Train Pose Loss": avg_train_pose_loss,
            "Validation Loss": avg_val_loss,
            "Validation Contrastive Loss": avg_val_contra_loss,
            "Validation Pose Loss": avg_val_pose_loss,
            "Best Validation Loss": best_loss
        })

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

        scheduler.step(avg_val_loss)

    wandb.finish()

if __name__ == "__main__":
    train_bimodeldown()
