import torch
import os
import glob

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
    "harth": {"right_thigh", "back", "back2"},
    "wharf": {"right_wrist"},
    "wisdm": {"right_wrist", "right_pocket"},
    "dsads": {"left_thigh", "right_thigh", "left_wrist", "right_wrist", "chest"},
    "utd-mhad": {"right_wrist", "right_thigh"},
    "mmact": {"right_wrist", "right_thigh"},
    "mmfit": {"left_ear", "right_ear", "left_thigh", "right_thigh", "left_wrist", "right_wrist"},
    "dip": {"left_shin", "right_shin", "back", "head", "left_wrist", "right_wrist"},
    "totalcapture": {"left_shoulder", "left_arm", "right_shoulder", "right_arm", "left_foot", "right_foot", "left_shin", "right_shin", "left_thigh", "right_thigh", "head", "chest", "belt"}
}

def process_dataset(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name, allowed_sensors in dataset_sensors.items():
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Find all .pt files in the root directory
        pt_files = glob.glob(os.path.join(root_dir, "**", "*.pt"), recursive=True)
        
        for pt_file in pt_files:
            data = torch.load(pt_file)
            imu_data = data.get("imu_data", {})
            
            # Modify IMU data based on allowed sensors
            modified_imu_data = {}
            for sensor, tensor in imu_data.items():
                if sensor in allowed_sensors:
                    modified_imu_data[sensor] = tensor  # Keep original
                else:
                    modified_imu_data[sensor] = torch.zeros_like(tensor)  # Zero out
            
            data["imu_data"] = modified_imu_data
            
            # Save the modified file
            filename = os.path.basename(pt_file)
            torch.save(data, os.path.join(dataset_output_dir, filename))
            
            print(f"Processed {filename} for {dataset_name}")

# Example usage
root_dir = "/media/lala/Seagate/temp"  # Change this to your dataset directory
output_dir = "/media/lala/Seagate/filtered_imu"  # Change this to the desired output directory
process_dataset(root_dir, output_dir)
