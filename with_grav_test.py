import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from gravity_function import apply_gravity_correction  # Ensure this is implemented correctly

# Directory path
save_dir = "sample"  # Replace with the actual directory path

# Body part to visualize
body_part = "left_wrist"  # Example body part
name = body_part+'_1'
file_path = os.path.join(save_dir, f'{name}.npz')

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

    # Interpolate linear_acceleration and angular_velocity by a factor
    linear_acceleration_interp = spline_interpolate(corrected_linear_acceleration, 4)
    angular_velocity_interp = spline_interpolate(angular_velocity, 4)
    linear_acceleration_original_interp = spline_interpolate(linear_acceleration, 4)

    # Save updated data into a new file
    output_file_path = os.path.join(save_dir, f'{name}_grav.npz')
    np.savez(output_file_path,
             positions=positions,
             orientations=orientations,
             linear_acceleration=linear_acceleration_original_interp,
             linear_acceleration_with_gravity=linear_acceleration_interp,
             angular_velocity=angular_velocity_interp)

    # Visualization
    fig = plt.figure(figsize=(30, 10))

    # Original Linear Acceleration
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(linear_acceleration[:, 0], 'r-', label='X')
    ax1.plot(linear_acceleration[:, 1], 'g-', label='Y')
    ax1.plot(linear_acceleration[:, 2], 'b-', label='Z')
    ax1.set_title(f'{body_part} - Linear Acceleration Original')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Acceleration')
    ax1.legend()

    # Corrected and Interpolated Linear Acceleration
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(linear_acceleration_interp[:, 0], 'r-', label='X')
    ax2.plot(linear_acceleration_interp[:, 1], 'g-', label='Y')
    ax2.plot(linear_acceleration_interp[:, 2], 'b-', label='Z')
    ax2.set_title(f'{body_part} - Linear Acceleration SimIntrepolation & gravity)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Acceleration')
    ax2.legend()

    # Original Angular Velocity
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(angular_velocity[:, 0], 'r-', label='X')
    ax3.plot(angular_velocity[:, 1], 'g-', label='Y')
    ax3.plot(angular_velocity[:, 2], 'b-', label='Z')
    ax3.set_title(f'{body_part} - Angular Velocity Original')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Velocity')
    ax3.legend()

    # Interpolated Angular Velocity
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(angular_velocity_interp[:, 0], 'r-', label='X')
    ax4.plot(angular_velocity_interp[:, 1], 'g-', label='Y')
    ax4.plot(angular_velocity_interp[:, 2], 'b-', label='Z')
    ax4.set_title(f'{body_part} - Angular Velocity SimInterpolation')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Velocity')
    ax4.legend()

     # Corrected and Interpolated Linear Acceleration
    ax5 = fig.add_subplot(3, 1, 3)
    ax5.plot(linear_acceleration_original_interp[:, 0], 'r-', label='X')
    ax5.plot(linear_acceleration_original_interp[:, 1], 'g-', label='Y')
    ax5.plot(linear_acceleration_original_interp[:, 2], 'b-', label='Z')
    ax5.set_title(f'{body_part} - Linear Acceleration SimInterpolation')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Acceleration')
    ax5.legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

else:
    print(f"File for {body_part} does not exist at {file_path}")
