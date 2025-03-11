import torch
import torch.nn as nn
import torch.nn.functional as F

def pose_loss(pred, gt, lambda_root=1.0, lambda_rot=1.0, lambda_vel=0.1, lambda_acc=0.05):
    """
    Compute total pose loss for SMPL-based pose estimation.

    Args:
        pred: (batch_size, timesteps, joints, 6) Predicted pose data
        gt: (batch_size, timesteps, joints, 6) Ground truth pose data
        lambda_root: Weight for root translation loss
        lambda_rot: Weight for rotation loss
        lambda_vel: Weight for velocity loss
        lambda_acc: Weight for acceleration loss

    Returns:
        Total loss (scalar)
    """

    # **1. Root translation loss (first 3 values)**
    loss_root = torch.nn.functional.mse_loss(pred[..., :3], gt[..., :3])

    # **2. Rotation loss (last 3 values - axis-angle)**
    loss_rot = torch.nn.functional.mse_loss(pred[..., 3:], gt[..., 3:])

    # **3. Temporal smoothness losses (only on root translation)**
    loss_vel = torch.mean(torch.norm(pred[:, 1:, :, :3] - pred[:, :-1, :, :3], dim=-1))  # Velocity loss
    loss_acc = torch.mean(torch.norm(pred[:, 2:, :, :3] - 2 * pred[:, 1:-1, :, :3] + pred[:, :-2, :, :3], dim=-1))  # Acceleration loss

    # **Total weighted loss**
    total_loss = (
        lambda_root * loss_root +
        lambda_rot * loss_rot +
        lambda_vel * loss_vel +
        lambda_acc * loss_acc
    )

    return total_loss