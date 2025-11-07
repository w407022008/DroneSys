import torch.nn as nn
import torch as th
import torch.nn.functional as F
from config.config import cfg


class GuidanceLoss(nn.Module):
    def __init__(self):
        super(GuidanceLoss, self).__init__()
        self.goal_length = cfg['goal_length']

    def forward(self, Df, Dp, goal):
        """
        Args:
            Dp: decision parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            goal: (batch_size, 3)
        Returns:
            guidance_loss: (batch_size) → guidance loss

        GuidanceLoss: distance_loss (for straighter flight) or similarity_loss (for faster flight in large scenario)
        """
        cur_pos = Df[:, :, 0]
        end_pos = Dp[:, :, 0]

        traj_dir = end_pos - cur_pos  # [B, 3]
        goal_dir = goal - cur_pos  # [B, 3]

        guidance_loss = self.distance_loss(traj_dir, goal_dir)
        # guidance_loss = self.similarity_loss(traj_dir, goal_dir)
        return guidance_loss

    def distance_loss(self, traj_dir, goal_dir):
        """
        Returns:
            l1_distance: (batch_size) → guidance loss

        L1Loss: L1 distance (same scale as the similarity loss) to the normalized goal (for numerical stability).
                closer to the goal is preferred.
        Straighter flight, but slightly inferior to the similarity cost in flight speed.
        """
        l1_distance = F.smooth_l1_loss(traj_dir, goal_dir, reduction='none')  # shape: (B, 3)
        l1_distance = l1_distance.sum(dim=1)  # (B)
        return l1_distance

    def similarity_loss(self, traj_dir, goal_dir):
        """
        Returns:
            similarity: (batch_size) → guidance loss

        SimilarityLoss: Projection length of the trajectory onto the goal direction:
                        higher cosine similarity and longer trajectory are preferred.
        Faster flight in large-scale scenario by allowing longer lateral avoidance without slowing down, but less precise near the goal.
        """
        goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)  # [B, 3]

        # projection length of trajectory on goal direction
        traj_along = (traj_dir * goal_dir_norm).sum(dim=1)  # [B]
        goal_length = goal_dir.norm(dim=1)  # [B]

        # length difference along goal direction (cosine similarity)
        parallel_diff = (goal_length - traj_along).abs()  # [B]

        # length perpendicular to goal direction
        traj_perp = traj_dir - traj_along.unsqueeze(1) * goal_dir_norm  # [B, 3]
        perp_diff = traj_perp.norm(dim=1)  # [B]

        # distance weighting (reduce perpendicular constraint, allow lateral exploration)
        perp_weight = 0.5   # the given weight is trained with perp_weight = 0, for higher speed in large-scale scenario
        similarity_loss = parallel_diff + perp_weight * perp_diff
        return similarity_loss