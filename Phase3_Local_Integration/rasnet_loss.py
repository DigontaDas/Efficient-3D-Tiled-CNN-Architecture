# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\rasnet_loss.py
"""
Custom Compound Loss Function for Coronary Artery Segmentation.
Combines MONAI's DiceLoss and FocalLoss. Focal loss penalizes easy background 
voxels, forcing the model to focus on thin, highly curvature-constricted vessel regions.
"""
import torch
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss
import numpy as np
from scipy.ndimage import distance_transform_edt

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (Tensor): Model logits of shape (B, 2, D, H, W)
            target (Tensor): Ground truth labels of shape (B, 1, D, H, W)
        """
        # Get foreground probability channel (index 1) after softmax
        probs = torch.softmax(pred, dim=1)
        fg_prob = probs[:, 1]  # (B, D, H, W)

        # Compute distance transform of the target mask (inverted)
        target_np = target.detach().cpu().numpy().astype(np.uint8)
        
        dist_maps = []
        for b in range(target_np.shape[0]):
            gt_mask = target_np[b, 0]  # (D, H, W)
            if np.any(gt_mask):
                # distance to closest foreground voxel (nearest 1-voxel in gt_mask)
                # distance_transform_edt computes distances to 0s, so we invert gt_mask
                dist_map = distance_transform_edt(1 - gt_mask)
            else:
                dist_map = np.zeros_like(gt_mask, dtype=np.float32)
            dist_maps.append(dist_map)
            
        dist_maps = np.stack(dist_maps, axis=0)
        dist_tensor = torch.from_numpy(dist_maps).to(pred.device, dtype=torch.float32)
        
        # Inner product penalty for predictions leaking beyond boundaries
        return torch.mean(fg_prob * dist_tensor)

class StenosisAwareLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, delta=0.2, gamma=2.5):
        """
        Args:
            alpha (float): Weight for Dice loss (default: 0.5)
            beta (float): Weight for Focal loss (default: 0.3)
            delta (float): Weight for Boundary loss (default: 0.2)
            gamma (float): Focal loss focusing parameter (default: 2.5)
        """
        super().__init__()
        self.dice_loss = DiceLoss(
            smooth_nr=0.0, 
            smooth_dr=1e-5, 
            to_onehot_y=True, 
            softmax=True
        )
        self.focal_loss = FocalLoss(
            gamma=gamma,
            to_onehot_y=True
        )
        self.boundary_loss = BoundaryLoss()
        
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (Tensor): Model prediction logits of shape (B, C, D, H, W)
            target (Tensor): Ground truth labels of shape (B, 1, D, H, W)
        """
        d_loss = self.dice_loss(pred, target)
        f_loss = self.focal_loss(pred, target)
        b_loss = self.boundary_loss(pred, target)
        return self.alpha * d_loss + self.beta * f_loss + self.delta * b_loss

if __name__ == "__main__":
    # Smoke test to verify forward and backward passes
    print("Testing StenosisAwareLoss...")
    loss_fn = StenosisAwareLoss()
    
    # Mock model prediction (logits for 2 classes: background and vessel)
    mock_pred = torch.randn(2, 2, 32, 32, 32, requires_grad=True)
    # Mock ground-truth labels (values 0 or 1)
    mock_target = torch.randint(0, 2, (2, 1, 32, 32, 32), dtype=torch.float32)
    
    loss = loss_fn(mock_pred, mock_target)
    print(f"Computed loss: {loss.item():.4f}")
    loss.backward()
    print("Backward pass successfully completed.")
