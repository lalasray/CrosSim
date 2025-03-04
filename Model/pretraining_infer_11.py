import torch
import os
from tqdm import tqdm

# Import necessary modules from the training script
from Encoder.Multi_IMU_Encoder import DeepConvGraphEncoderPre, IMUGraph
from Encoder.Gtr_Text_Encoder import EmbeddingEncoder
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph
from c_dataloader import UniMocapDataset, collate_fn
from torch.utils.data import DataLoader
from Pretraining_JLR_optimized import MultiModalJLR, device, h5_file_path, batch_size, best_model_path

# Load dataset for inference
dataset = UniMocapDataset(h5_file_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=4, pin_memory=True, persistent_workers=True)

# Load trained model
model = MultiModalJLR().to(device)
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded model from {best_model_path}")
else:
    raise FileNotFoundError(f"Trained model not found at {best_model_path}")

model.eval()  # Set model to evaluation mode

# Inference Loop
results = []
with torch.no_grad():
    for text_data, pose_data, imu_data, imu_data_grav in tqdm(dataloader, desc="Running Inference"):
        # Move data to GPU
        text_data = text_data.view(text_data.shape[0], 768).to(device, non_blocking=True)
        pose_data = pose_data.to(device, non_blocking=True)
        imu_data = imu_data.view(imu_data.shape[0], imu_data.shape[2], imu_data.shape[1], 6).to(device, non_blocking=True)
        imu_data_grav = imu_data_grav.view(imu_data_grav.shape[0], imu_data_grav.shape[2], imu_data_grav.shape[1], 6).to(device, non_blocking=True)
        
        # Forward pass
        text_embeddings, pose_embeddings, imu_embeddings, imu_embeddings_grav = model(text_data, pose_data, imu_data, imu_data_grav)
        
        # Store results
        results.append({
            "text_embeddings": text_embeddings.cpu(),
            "pose_embeddings": pose_embeddings.cpu(),
            "imu_embeddings": imu_embeddings.cpu(),
            "imu_embeddings_grav": imu_embeddings_grav.cpu()
        })

# Save results
torch.save(results, "inference_results.pth")
print("Inference complete. Results saved as 'inference_results.pth'")
