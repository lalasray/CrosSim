import torch
import os

# Define the dataset-specific IMU sensor placements
dataset_sensors = {
    "openpack": {"left_wrist", "right_wrist"},
    "alshar": {"right_wrist"},
    "opportunity": {"left_foot", "right_foot", "back", "left_shoulder", "right_shoulder", "left_arm", "right_arm"},
    "utdmhad": {"right_wrist", "right_thigh"},
    "ucihar": {"belt"},
    "motionsense": {"chest"},
    "w-HAR": {"right_shin"},
    "shoaib": {"right_wrist", "right_arm", "belt", "right_thigh", "left_thigh"},
    "har70+": {"right_thigh", "back"},
    "realworld": {"chest", "left_shoulder", "head", "left_shin", "left_thigh", "left_arm", "waist"},
    "pamap2": {"right_arm", "right_shin", "chest"},
    "usc-had": {"right_thigh"},
    "mhealth": {"chest", "right_wrist", "left_shin"},
    "harth": {"right_thigh", "back", "belt"},
    "wharf": {"right_wrist"},
    "wisdm": {"right_wrist", "right_pocket"},
    "dsads": {"left_thigh", "right_thigh", "left_wrist", "right_wrist", "chest"},
    "utd-mhad": {"right_wrist", "right_thigh"},
    "mmact": {"right_wrist", "right_thigh"},
    "mmfit": {"left_ear", "right_ear", "left_thigh", "right_thigh", "left_wrist", "right_wrist"},
    "dip": {"left_shin", "right_shin", "back", "head", "left_wrist", "right_wrist"},
    "totalcapture": {"left_shoulder", "left_arm", "right_shoulder", "right_arm", "left_foot", "right_foot", "left_shin", "right_shin", "left_thigh", "right_thigh", "head", "chest", "belt"}
}

# Input file path
file_path = "/home/lala/Documents/GitHub/CrosSim_Data/UniMocap/processed/Datapoint_0_1.pt"
output_dir = "/home/lala/Documents/GitHub/CrosSim_Data/UniMocap/variations/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_file_for_all_datasets(file_path, output_dir):
    # Load the file
    data = torch.load(file_path)
    
    for dataset_name, allowed_sensors in dataset_sensors.items():
        modified_data = data.copy()
        
        # Modify only the IMU data
        if "imu_data" in modified_data:
            imu_data = modified_data["imu_data"].copy()
            for sensor in imu_data.keys():
                if sensor not in allowed_sensors:
                    imu_data[sensor] = torch.zeros_like(imu_data[sensor])  # Zero out
            modified_data["imu_data"] = imu_data
        
        # Construct output filename
        base_name = os.path.basename(file_path).replace(".pt", f"_{dataset_name}.pt")
        output_path = os.path.join(output_dir, base_name)
        
        # Save the modified file
        torch.save(modified_data, output_path)
        print(f"Processed for {dataset_name}: {output_path}")

# Process the file for all datasets
process_file_for_all_datasets(file_path, output_dir)
