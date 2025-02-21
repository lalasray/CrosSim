import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os

# Define dataset-specific IMU sensor placements
dataset_sensors = {
    "openpack": {"left_wrist", "right_wrist"},
    "alshar": {"right_wrist"},
    "opportunity": {"left_foot", "right_foot", "back", "left_shoulder", "right_shoulder", "left_arm", "right_arm"},
    "ucihar": {"belt"},
    "motionsense": {"chest"},
    "wHAR": {"right_shin"},
    "shoaib": {"right_wrist", "right_arm", "belt", "right_thigh", "left_thigh"},
    "har70": {"right_thigh", "back"},
    "realworld": {"chest", "left_shoulder", "head", "left_shin", "left_thigh", "left_arm", "waist"},
    "pamap2": {"right_arm", "right_shin", "chest"},
    "uschad": {"right_thigh"},
    "mhealth": {"chest", "right_wrist", "left_shin"},
    "harth": {"right_thigh", "back", "belt"},
    "wharf": {"right_wrist"},
    "wisdm": {"right_wrist", "right_pocket"},
    "dsads": {"left_thigh", "right_thigh", "left_wrist", "right_wrist", "chest"},
    "utdmhad": {"right_wrist", "right_thigh"},
    "mmact": {"right_wrist", "right_thigh"},
    "mmfit": {"left_ear", "right_ear", "left_thigh", "right_thigh", "left_wrist", "right_wrist"},
    "dip": {"left_shin", "right_shin", "back", "head", "left_wrist", "right_wrist"},
    "totalcapture": {"left_shoulder", "left_arm", "right_shoulder", "right_arm", "left_foot", "right_foot", 
                     "left_shin", "right_shin", "left_thigh", "right_thigh", "head", "chest", "belt"}
}

class MotionDataset(Dataset):
    def __init__(self, data_dir, dataset_variant):
        """
        Initialize dataset with a specific variant.
        Args:
            data_dir (str): Directory containing .pt files.
            dataset_variant (str): Name of dataset variant (e.g., "openpack").
        """
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.dataset_variant = dataset_variant

        if dataset_variant not in dataset_sensors:
            raise ValueError(f"Dataset variant '{dataset_variant}' not found. Choose from: {list(dataset_sensors.keys())}")

    def __len__(self):
        """Return total number of samples."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """Load and return only the dataset-specific variant."""
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_path, map_location="cpu")  # Load on CPU

        # Extract motion and pose data
        motion_data = data.get("motion_data", None)
        pose_data = data.get("pose_data", {})
        imu_data = data.get("imu_data", {})

        joint = pose_data.get("joint", None)
        body = pose_data.get("body", None)
        trans = pose_data.get("trans", None)

        # Apply dataset-specific IMU filtering
        allowed_sensors = dataset_sensors[self.dataset_variant]
        imu_tensors = {}
        for sensor, subdict in imu_data.items():
            if sensor in allowed_sensors:
                for sub_key, value in subdict.items():
                    imu_tensors[f"{sensor}.{sub_key}"] = value
            else:
                # Zero out non-allowed sensors while maintaining shape
                for sub_key, value in subdict.items():
                    imu_tensors[f"{sensor}.{sub_key}"] = torch.zeros_like(value) if isinstance(value, torch.Tensor) else None

        # Return only the modified variant
        return {
            "motion": motion_data,
            "pose_joint": joint,
            "pose_body": body,
            "pose_trans": trans,
            **imu_tensors
        }

class OGMotionDataset(Dataset):
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
            max_shape = list(torch.tensor([list(t.shape) for t in tensors]).max(dim=0).values)

            padded_tensors = []
            for t in tensors:
                pad_size = [(0, max_dim - t_dim) for max_dim, t_dim in zip(max_shape, t.shape)]
                pad_size = [p for pair in reversed(pad_size) for p in pair]
                padded_tensors.append(torch.nn.functional.pad(t, pad_size))

            batch_dict[key] = torch.stack(padded_tensors).to(device, non_blocking=True)
        else:
            batch_dict[key] = None

    return batch_dict

# Example Usage
data_dir = "/home/lala/Documents/GitHub/CrosSim_Data/UniMocap/processed/"  # Update path
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

combined_dataset = ConcatDataset(datasets)
# Combine datasets using ConcatDataset
#combined_dataset = ConcatDataset(datasets)
dataloader = DataLoader(combined_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
for batch in dataloader:
    print(batch)
    break
