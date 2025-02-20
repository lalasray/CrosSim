import torch
import os
import copy

# Define dataset-specific IMU sensor placements
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
    "totalcapture": {"left_shoulder", "left_arm", "right_shoulder", "right_arm", "left_foot", "right_foot", 
                     "left_shin", "right_shin", "left_thigh", "right_thigh", "head", "chest", "belt"}
}

# Input directory path (to process all .pt files)
input_dir = "/home/lala/Documents/GitHub/CrosSim_Data/UniMocap/processed/"
output_dir = "/home/lala/Documents/GitHub/CrosSim_Data/UniMocap/variations/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get list of files in the input directory
files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
total_files = len(files)

# Track overall progress
for i, filename in enumerate(files, start=1):
    file_path = os.path.join(input_dir, filename)

    # Load data safely
    try:
        data = torch.load(file_path)  # Removed 'weights_only=True'
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        continue  # Skip the current file and proceed to the next

    # Track progress for this file
    print(f"\nProcessing file {i}/{total_files}: {filename}")

    # Process each dataset
    for dataset_name, allowed_sensors in dataset_sensors.items():
        modified_data = copy.deepcopy(data)  # Ensures deep copy of nested structures

        imu_data = modified_data.get("imu_data", {})  # Avoids KeyError
        if isinstance(imu_data, dict):
            for sensor in list(imu_data.keys()):  # Iterate over a copy of keys
                if sensor not in allowed_sensors:
                    if isinstance(imu_data[sensor], torch.Tensor):
                        imu_data[sensor] = torch.zeros_like(imu_data[sensor])  # Keep original shape but zero values
                    elif isinstance(imu_data[sensor], dict):
                        for sub_key in imu_data[sensor]:  # Handle nested structures like "gyro", "acc"
                            if isinstance(imu_data[sensor][sub_key], torch.Tensor):
                                imu_data[sensor][sub_key] = torch.zeros_like(imu_data[sensor][sub_key])
                            else:
                                imu_data[sensor][sub_key] = None  # Assign None for unknown cases

        # Save modified data
        output_filename = filename.replace(".pt", f"_{dataset_name}.pt")
        output_path = os.path.join(output_dir, output_filename)
        torch.save(modified_data, output_path)
        print(f"  Processed for {dataset_name}: {output_path}")

    # Report overall progress
    print(f"Completed processing {i}/{total_files} files.")
