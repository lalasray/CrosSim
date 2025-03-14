import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import os
from architectures import BiModalIMUDownText, count_parameters
from Loss.pretrain_loss import contra_loss
from Loss.to_text_loss import loss_fn
from c_dataloader import UniMocapDataset, collate_fn
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_bimodeldown(epochs=500, batch_size=128, learning_rate=0.001, early_stop_patience=20, patience=15, patience_factor=0.5, h5_file_path="../CrosSim_Data/UniMocap/full_dataset.h5"):
    wandb.init(project="CrosSim", name="TextIMU-DownText", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "early_stop_patience": early_stop_patience,
        "patience": patience,
        "patience_factor": patience_factor
    })

    dataset = UniMocapDataset(h5_file_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True)

    model = BiModalIMUDownText().to(device)
    print(f"Total Trainable Parameters: {count_parameters(model):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=patience_factor, patience=patience, verbose=True)

    best_loss = float('inf')
    no_improvement_epochs = 0
    model_path = "best_textimu_downtext.pth"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-existing model checkpoint.")

    first_epoch_contra_loss = None
    first_epoch_text_loss = None

    for epoch in range(epochs):
        model.train()
        train_loss, train_contra_loss, train_text_loss = 0.0, 0.0, 0.0

        for text_data, pose_data, imu_data, imu_data_grav in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            optimizer.zero_grad()
            text_data = text_data.view(text_data.shape[0], 768).to(device, dtype=torch.float32, non_blocking=True)
            imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, dtype=torch.float32, non_blocking=True)

            text_embeddings, imu_emb_grav, gtr = model(text_data, imu_data_grav)

            contra_loss_val = contra_loss(text_embeddings, imu_emb_grav)
            text_loss_val = loss_fn(gtr, text_data)

            if first_epoch_contra_loss is None:
                first_epoch_contra_loss = contra_loss_val.item()
                first_epoch_text_loss = text_loss_val.item()

            contrastive_loss_weight = first_epoch_text_loss / first_epoch_contra_loss
            text_loss_weight = first_epoch_contra_loss / first_epoch_text_loss

            total_loss = contrastive_loss_weight * contra_loss_val + text_loss_weight * text_loss_val
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_contra_loss += contra_loss_val.item()
            train_text_loss += text_loss_val.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_contra_loss = train_contra_loss / len(train_loader)
        avg_train_text_loss = train_text_loss / len(train_loader)

        model.eval()
        val_loss, val_contra_loss, val_text_loss = 0.0, 0.0, 0.0

        with torch.no_grad():
            for text_data, pose_data, imu_data, imu_data_grav in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                text_data = text_data.view(text_data.shape[0], 768).to(device, dtype=torch.float32, non_blocking=True)
                imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, dtype=torch.float32, non_blocking=True)

                text_embeddings, imu_emb_grav, gtr = model(text_data, imu_data_grav)

                contra_loss_val = contra_loss(text_embeddings, imu_emb_grav)
                text_loss_val = loss_fn(gtr, text_data)

                total_loss = contrastive_loss_weight * contra_loss_val + text_loss_weight * text_loss_val

                val_loss += total_loss.item()
                val_contra_loss += contra_loss_val.item()
                val_text_loss += text_loss_val.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_contra_loss = val_contra_loss / len(val_loader)
        avg_val_text_loss = val_text_loss / len(val_loader)

        wandb.log({
            "Epoch": epoch+1,
            "Train Loss": avg_train_loss,
            "Train Contrastive Loss": avg_train_contra_loss,
            "Train Text Loss": avg_train_text_loss,
            "Validation Loss": avg_val_loss,
            "Validation Contrastive Loss": avg_val_contra_loss,
            "Validation Text Loss": avg_val_text_loss,
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
