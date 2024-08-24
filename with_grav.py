import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from gravity_function import apply_gravity_correction

# Directory path
save_dir = "motion_IMU/humman/subset_0000/All_Squat"  # Replace with the actual directory path

# Body part to visualize
body_part = "right_wrist"  # Example body part
file_path = os.path.join(save_dir, f'{body_part}_v2.npz')

if os.path.exists(file_path):
    # Load the data from the .npz file
    data = np.load(file_path)
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

    # Interpolate linear_acceleration and angular_velocity by a factor of 4
    linear_acceleration_interp = spline_interpolate(corrected_linear_acceleration, 8)
    angular_velocity_interp = spline_interpolate(angular_velocity, 8)
    linear_acceleration_i = spline_interpolate(linear_acceleration, 8)

    # Update the existing data with new interpolations
    with np.load(file_path, allow_pickle=True) as data:
        np.savez(file_path,
                 positions=positions,
                 orientations=orientations,
                 linear_acceleration=linear_acceleration_i,
                 linear_acceleration_with_gravity=linear_acceleration_interp,
                 angular_velocity=angular_velocity_interp)
    
    '''
    # Create a figure with subplots
    fig = plt.figure(figsize=(30, 5))

    # Linear Acceleration Subplot (Original Data)
    ax3 = fig.add_subplot(2, 2, 1)
    ax3.plot(linear_acceleration_i[:, 0], label='X')
    ax3.plot(linear_acceleration_i[:, 1], label='Y')
    ax3.plot(linear_acceleration_i[:, 2], label='Z')
    ax3.set_title(f'{body_part} - Linear Acceleration (Interpolated)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Acceleration')
    ax3.legend()

    # Angular Velocity Subplot (Original Data)
    ax4 = fig.add_subplot(2, 2, 2)
    ax4.plot(angular_velocity[:, 0], label='X')
    ax4.plot(angular_velocity[:, 1], label='Y')
    ax4.plot(angular_velocity[:, 2], label='Z')
    ax4.set_title(f'{body_part} - Angular Velocity (Interpolated)')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Velocity')
    ax4.legend()


    # Linear Acceleration Subplot (Interpolated Data)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(linear_acceleration_interp[:, 0], label='X')
    ax3.plot(linear_acceleration_interp[:, 1], label='Y')
    ax3.plot(linear_acceleration_interp[:, 2], label='Z')
    ax3.set_title(f'{body_part} - Linear Acceleration (Interpolated with Gravity)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Acceleration')
    ax3.legend()

    # Angular Velocity Subplot (Interpolated Data)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(angular_velocity_interp[:, 0], label='X')
    ax4.plot(angular_velocity_interp[:, 1], label='Y')
    ax4.plot(angular_velocity_interp[:, 2], label='Z')
    ax4.set_title(f'{body_part} - Angular Velocity (Interpolated)')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Velocity')
    ax4.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    '''
else:
    print(f"File for {body_part} does not exist at {file_path}")
