import os
import numpy as np
from scipy.interpolate import interp1d
from gravity_function import apply_gravity_correction

# Directory path
base_dir = "motion_IMU"  # Replace with the actual base directory path

def process_file(file_path):
    """Process a single .npz file by applying gravity correction and interpolation."""
    try:
        if os.path.exists(file_path):
            # Load the data from the .npz file
            with np.load(file_path, allow_pickle=True) as data:
                if 'linear_acceleration_with_gravity' in data:
                    print(f"Skipping {file_path}: 'linear_acceleration_with_gravity' already exists.")
                    return

                # Extract data arrays
                positions = data['positions']
                orientations = data['orientations']  # Assuming orientations are stored as Euler angles
                linear_acceleration = data['linear_acceleration']
                angular_velocity = data['angular_velocity']

                # Spline interpolation function
                def spline_interpolate(data, factor):
                    x = np.arange(len(data))
                    x_new = np.linspace(0, len(data) - 1, len(data) * factor)
                    interpolated_data = interp1d(x, data, kind='cubic', axis=0)(x_new)
                    return interpolated_data
                
                # Apply gravity correction
                corrected_linear_acceleration = apply_gravity_correction(linear_acceleration, orientations[:-2])

                # Interpolate linear_acceleration and angular_velocity by a factor of 8
                linear_acceleration_interp = spline_interpolate(corrected_linear_acceleration, 8)
                angular_velocity_interp = spline_interpolate(angular_velocity, 8)
                linear_acceleration_i = spline_interpolate(linear_acceleration, 8)

                # Save the updated data to the .npz file
                np.savez(file_path,
                         positions=positions,
                         orientations=orientations,
                         linear_acceleration=linear_acceleration_i,
                         linear_acceleration_with_gravity=linear_acceleration_interp,
                         angular_velocity=angular_velocity_interp)
                
                print(f"Processed and updated {file_path}")
        else:
            print(f"File does not exist: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def traverse_and_process(base_dir):
    """Traverse all subdirectories and process each .npz file."""
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".npz"):
                file_path = os.path.join(root, file)
                process_file(file_path)

# Start processing
traverse_and_process(base_dir)
