import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

body_parts = [
    'right_wrist', 'left_wrist', 'right_shin', 'left_shin', 
    'right_thigh', 'left_thigh', 'right_arm', 'left_arm', 
    'right_shoulder', 'left_shoulder', 'forehead', 'right_foot', 
    'left_foot', 'back', 'right_shirt_pocket', 'left_shirt_pocket', 
    'chest', 'Necklace', 'belt', 'left_ear', 'right_ear'
]
# Define the path to the saved .npz file
body_part = "left_wrist"  # Replace with the actual body part name
save_dir = r"Data\MotionX\motionx_smplx\aist\subset_0000\Dance_Break"  # Replace with the actual save directory path
file_path = os.path.join(save_dir, f'{body_part}_1.npz')

# Load the data from the .npz file
data = np.load(file_path)

# Function to convert Euler angles to direction vector
def euler_to_vector(euler_angles):
    rotation = R.from_radians('xyz', euler_angles)  # 'xyz' means rotations are applied around x, y, and z axes respectively
    return rotation.apply([1, 0, 0])  # Assuming a unit vector along the X-axis to show orientation

# Extract data arrays
positions = data['positions']
orientations = data['orientations']
linear_acceleration = data['linear_acceleration']
angular_velocity = data['angular_velocity']

# Visualization
fig = plt.figure(figsize=(15, 15))
# Positions Subplot
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Positions')
ax1.set_title(f'{body_part} - Positions')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Orientation Subplot with Color Gradient
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
norm = Normalize(vmin=np.min(orientations[:, 0]), vmax=np.max(orientations[:, 0]))
colors = plt.cm.viridis(norm(orientations[:, 0]))  # Using the X component of orientation for color mapping

for i in range(len(positions) - 1):
    ax2.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2], color=colors[i])

ax2.set_title(f'{body_part} - Orientations with Color Gradient')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Linear Acceleration Subplot
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(linear_acceleration[:, 0], label='X')
ax3.plot(linear_acceleration[:, 1], label='Y')
ax3.plot(linear_acceleration[:, 2], label='Z')
ax3.set_title(f'{body_part} - Linear Acceleration')
ax3.set_xlabel('Time')
ax3.set_ylabel('Acceleration')
ax3.legend()

# Angular Velocity Subplot
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(angular_velocity[:, 0], label='X')
ax4.plot(angular_velocity[:, 1], label='Y')
ax4.plot(angular_velocity[:, 2], label='Z')
ax4.set_title(f'{body_part} - Angular Velocity')
ax4.set_xlabel('Time')
ax4.set_ylabel('Velocity')
ax4.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
