import os
import numpy as np
from scipy.interpolate import interp1d
from gravity_function import apply_gravity_correction  # Ensure this is implemented correctly
from with_grav_test import reshape

# Directory path
base_dir = r"Data\MotionX\motionx_smplx\fitness"  # Replace with the actual directory path
error_log_path = os.path.join(base_dir, "error_files.txt")  # File to store names of problematic files

# Spline interpolation function
def spline_interpolate(data, factor):
    x = np.arange(len(data))
    x_new = np.linspace(0, len(data) - 1, len(data) * factor)
    return interp1d(x, data, kind='cubic', axis=0)(x_new)

# Function to process a single .npz file
def process_file(file_path):
    name = os.path.splitext(os.path.basename(file_path))[0]
    try:
        # Load the .npz file
        data = np.load(file_path)
        positions = data['positions']
        orientations = data['orientations']  # Assuming orientations are stored as Euler angles
        linear_acceleration = data['linear_acceleration']
        angular_velocity = data['angular_velocity']
        
        # Apply gravity correction
        corrected_linear_acceleration = apply_gravity_correction(linear_acceleration, orientations[:-2])
        
        # Interpolate data by a factor
        factor = 4
        linear_acceleration_interp = reshape(spline_interpolate(corrected_linear_acceleration, factor), positions.shape[0] * factor)
        angular_velocity_interp = reshape(spline_interpolate(angular_velocity, factor), positions.shape[0] * factor)
        linear_acceleration_original_interp = reshape(spline_interpolate(linear_acceleration, factor), positions.shape[0] * factor)

        # Save updated data into a new file
        output_file_path = os.path.join(os.path.dirname(file_path), f'{name}_grav.npz')
        np.savez(output_file_path,
                 positions=positions,
                 orientations=orientations,
                 linear_acceleration=linear_acceleration_original_interp,
                 linear_acceleration_with_gravity=linear_acceleration_interp,
                 angular_velocity=angular_velocity_interp)
        print(f"Processed and saved: {output_file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        with open(error_log_path, "a") as error_file:
            error_file.write(f"{file_path}\n")

# Walk through the directory and process each .npz file
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.npz') and not file.endswith('_grav.npz'):
            process_file(os.path.join(root, file))
