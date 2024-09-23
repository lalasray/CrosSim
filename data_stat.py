import os
import numpy as np

# Directory path
base_dir = "motion_IMU" 

largest_file = None
largest_size = 0

def process_file(file_path):
    global largest_file, largest_size
    
    if os.path.exists(file_path):
        with np.load(file_path, allow_pickle=True) as data:
            # Check if key 'linear_acceleration_with_gravity' is present
            if 'linear_acceleration_with_gravity' in data:
                positions = data['positions']
                orientations = data['orientations']
                linear_acceleration = data['linear_acceleration']
                linear_acceleration_g = data['linear_acceleration_with_gravity']
                angular_velocity = data['angular_velocity']
                                
                total_size = linear_acceleration.shape[0]               
                # Update largest file if this file has more data
                if total_size > largest_size:
                    largest_size = total_size
                    largest_file = file_path

def traverse_and_process(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".npz"):
                file_path = os.path.join(root, file)
                process_file(file_path)

    # Print the file with the largest data size
    if largest_file:
        print(f"The file with the largest data size is: {largest_file}")
        print(f"Total size: {largest_size} elements")

traverse_and_process(base_dir)
