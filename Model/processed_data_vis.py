import torch

# Load the .pt file
file_path = "/home/lala/Documents/GitHub/CrosSim_Data/UniMocap/processed/Datapoint_1_1.pt"  # Change this to your actual file path
#file_path = "/home/lala/Documents/GitHub/CrosSim_Data/UniMocap/variations/Datapoint_0_1_opportunity.pt"
data = torch.load(file_path)

# Function to print tensor shapes
def print_shapes(data, prefix=""):
    if isinstance(data, dict):
        for key, value in data.items():
            print_shapes(value, prefix + f"{key}.")
    elif isinstance(data, torch.Tensor):
        print(f"{prefix} shape: {data.shape}")
    elif isinstance(data, list):
        print(f"{prefix} list of {len(data)} elements")
    elif isinstance(data, tuple):
        print(f"{prefix} tuple of {len(data)} elements")
    else:
        print(f"{prefix} type: {type(data)}")

# Print shapes of all tensors in the file
print_shapes(data)
#print(data)