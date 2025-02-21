import torch
from torch.utils.data import Dataset, DataLoader
import os

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
data_dir = "/home/lala/Documents/GitHub/CrosSim_Data/UniMocap/processed/"  # Update this path
dataset = MotionDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Iterate through the DataLoader
for batch in dataloader:
    print("Batch keys:", batch.keys())
    print("Motion shape:", batch["motion"].shape if batch["motion"] is not None else "None")
    print("Pose Joint shape:", batch["pose_joint"].shape if batch["pose_joint"] is not None else "None")
    #break
