import torch
from torch.utils.data import Dataset, DataLoader
import os


sensor_positions_acc = ["back.acc", "belt.acc", "chest.acc", "forehead.acc",
                        "left_arm.acc", "left_ear.acc", "left_foot.acc", "left_shin.acc",
                        "left_shirt_pocket.acc", "left_shoulder.acc", "left_thigh.acc", "left_wrist.acc",
                        "necklace.acc", "right_arm.acc", "right_ear.acc", "right_foot.acc",
                        "right_shin.acc", "right_shirt_pocket.acc", "right_shoulder.acc",
                        "right_thigh.acc", "right_wrist.acc"]

sensor_positions_gyro = [pos.replace(".acc", ".gyro") for pos in sensor_positions_acc]
sensor_positions_acc_g = [pos + "_g" for pos in sensor_positions_acc]

class MotionDataset(Dataset):
    def __init__(self, data_dir):
        """Initialize dataset by listing all .pt files in the directory."""
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        
    def __len__(self):
        """Return total number of samples."""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Load and return one data sample."""
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_path, map_location="cpu")  # Load directly on CPU
        
        # Extract relevant tensors
        motion_data = data.get('motion_data', None)
        pose_data = data.get('pose_data', {})
        imu_data = data.get('imu_data', {})
        
        # Extract pose data components
        joint = pose_data.get('joint', None)
        body = pose_data.get('body', None)
        trans = pose_data.get('trans', None)
        
        # Extract IMU data (flatten nested dict)
        imu_tensors = {f"{k}.{sk}": v for k, subdict in imu_data.items() for sk, v in subdict.items()}
        
        sample = {
            "motion": motion_data,
            "pose_joint": joint,
            "pose_body": body,
            "pose_trans": trans,
            **imu_tensors
        }
        
        return sample

def collate_fn(batch):
    """Custom collate function to handle variable-length tensors."""
    batch_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for key in batch[0]:
        tensors = [item[key] for item in batch if item[key] is not None]
        
        if tensors and isinstance(tensors[0], torch.Tensor):
            # Determine max shape for padding
            max_shape = list(torch.tensor([list(t.shape) for t in tensors]).max(dim=0).values)
            
            padded_tensors = []
            for t in tensors:
                pad_size = [(0, max_dim - t_dim) for max_dim, t_dim in zip(max_shape, t.shape)]
                pad_size = [p for pair in reversed(pad_size) for p in pair]  # Flatten for torch.nn.functional.pad
                padded_tensors.append(torch.nn.functional.pad(t, pad_size))
            
            batch_dict[key] = torch.stack(padded_tensors).to(device, non_blocking=True)
        else:
            batch_dict[key] = None  # Maintain key consistency
    
    return batch_dict

# Example usage
data_dir = "../CrosSim_Data/UniMocap/processed"  # Update this path
dataset = MotionDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

# Iterate through the DataLoader
for batch in dataloader:

    text_data = batch["motion"].squeeze(1)
    pose = torch.cat([batch["pose_trans"], batch["pose_body"]], dim=-1)
    full_Pose = pose.view(pose.shape[0], pose.shape[1], 24, 3)
    pose_data = torch.cat([full_Pose, batch["pose_joint"].squeeze(2)], dim=-1)

    combined_data_acc = torch.stack([batch[key] for key in sensor_positions_acc], dim=2)
    combined_data_gyro = torch.stack([batch[key] for key in sensor_positions_gyro], dim=2)
    imu_data = torch.cat((combined_data_acc, combined_data_gyro), dim=3)

    combined_data_acc_grav = torch.stack([batch[key] for key in sensor_positions_acc_g], dim=2)
    imu_data_grav = torch.cat((combined_data_acc_grav, combined_data_gyro), dim=3)
    print(text_data.shape)
