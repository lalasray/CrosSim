import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from architectures import BiModalIMUDown,count_parameters

from Loss.pretrain_loss import contra_loss
from Loss.to_text_loss import loss_fn
from c_dataloader import UniMocapDataset, collate_fn
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Function
def train_bimodeldown(epochs=300, batch_size=128, learning_rate=0.001, early_stop_patience=20, patience=15, patience_factor=0.5, h5_file_path = "../CrosSim_Data/UniMocap/full_dataset.h5"):
    # Load dataset
    dataset = UniMocapDataset(h5_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    # Initialize model, optimizer, scheduler
    model = BiModalIMUDown().to(device)
    print(f"Total Trainable Parameters: {count_parameters(model):,}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=patience_factor, patience=patience, verbose=True)

    best_loss = float('inf')
    no_improvement_epochs = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_contra_loss = 0
        epoch_text_loss = 0

        for text_data, pose_data, imu_data, imu_data_grav in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            text_data = text_data.view(text_data.shape[0], 768).to(device, dtype=torch.float32, non_blocking=True)
            imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, dtype=torch.float32, non_blocking=True)

            text_embeddings, imu_emb_grav, gtr = model(text_data, imu_data_grav)

            # Compute contra loss
            contra_loss_val = contra_loss(text_embeddings, imu_emb_grav)
            epoch_contra_loss += contra_loss_val.item()

            len_seq = imu_data_grav.shape[1]
            text_loss_val = loss_fn(gtr, text_data.unsqueeze(1).expand(-1, int(len_seq/4), -1))
            epoch_text_loss += text_loss_val.item()

            # Total loss
            total_loss = contra_loss_val + text_loss_val
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        avg_contra_loss = epoch_contra_loss / len(dataloader)
        avg_text_loss = epoch_text_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Contrastive Loss: {avg_contra_loss:.4f}, Text Loss: {avg_text_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

        scheduler.step(avg_loss)

    torch.save(model.state_dict(), "textimu_downtext.pth")

if __name__ == "__main__":
    train_bimodeldown()
