import numpy as np

# Convert Euler angles to a rotation matrix
def euler_to_rotation_matrix(euler_angles):
    roll, pitch, yaw = np.deg2rad(euler_angles)  # Convert degrees to radians

    # Rotation matrices around x, y, and z axes
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def radian_to_rotation_matrix(radian_angles):
    roll, pitch, yaw = radian_angles  # Assume euler_angles are already in radians

    # Rotation matrices around x, y, and z axes
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

# Function to apply gravity correction to linear acceleration
def apply_gravity_correction(linear_acceleration, orientations):
    gravity_global = np.array([0, 0, -9.81])  # Gravity vector in global coordinates (m/s²)
    corrected_acceleration = np.zeros_like(linear_acceleration)

    for i in range(len(orientations)):
        # Convert Euler angles to rotation matrix
        R = radian_to_rotation_matrix(orientations[i])
        
        # Transform gravity vector from global to local frame
        gravity_local = np.dot(R, gravity_global)
        
        # Add local gravity to linear acceleration
        corrected_acceleration[i] = linear_acceleration[i] + gravity_local

    return corrected_acceleration

