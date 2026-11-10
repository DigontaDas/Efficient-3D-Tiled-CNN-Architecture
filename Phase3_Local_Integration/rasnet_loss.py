# c:\Thesis_RASNET\Phase3_Local_Integration\rasnet_loss.py
"""
Custom Compound Loss Function for Coronary Artery Segmentation.

Implemented Formulation (used during 200-epoch RASNet training in train_rasnet_200ep.py):
    L_total = α · L_Dice + (1 - α) · L_Focal

Calibration Parameters:
    α = 0.4  (Dice weight — MONAI DiceLoss with smooth_nr=0.0, smooth_dr=1e-5,
              to_onehot_y=True, softmax=True, include_background=True [default])
    γ = 2.5  (Focal focusing parameter — MONAI FocalLoss with gamma=2.5,
              to_onehot_y=True, use_softmax=False [default sigmoid per channel],
              include_background=True [default])

Note on Deep Supervision & Historical Provenance:
    - Multi-scale deep supervision weights are applied in train_rasnet_200ep.py:215 as:
      total_loss = loss_main + 0.4 * aux2_loss + 0.2 * aux3_loss
      (where aux2 is Decoder Level 2 at 1/4 resolution and aux3 is Decoder Level 1 at 1/2 resolution).
    - BoundaryLoss defined below is a standalone class and is NOT called by StenosisAwareLoss.
    - Because historical calibration (α=0.4, γ=2.5) and threshold sweeps previously inspected
      the 150-case ImageCAS test split (IDs 851-1000), evaluations on IDs 851-1000 are reported
      as retrospective reassessments (see BENCHMARK_PREREGISTRATION.md and LAB_CHECKS.md).
"""
import torch
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss


class StenosisAwareLoss(nn.Module):
    """
    Combined two-component Dice + Focal loss for 3D coronary artery segmentation.

    L_total = α · L_Dice + (1 - α) · L_Focal

    Hardcoded calibration:
        α = 0.4  → DiceLoss(smooth_nr=0.0, smooth_dr=1e-5, to_onehot_y=True,
                            softmax=True, include_background=True)
        γ = 2.5  → FocalLoss(gamma=2.5, to_onehot_y=True, use_softmax=False,
                             include_background=True)
    """

    # ── Hardcoded calibration parameters ────────────────────────────────
    ALPHA = 0.4   # Dice weight
    GAMMA = 2.5   # Focal focusing parameter

    def __init__(self, alpha: float = None, gamma: float = None):
        """
        Args:
            alpha (float | None): Weight for Dice loss.
                Defaults to the hardcoded calibration value (0.4).
            gamma (float | None): Focal loss focusing parameter.
                Defaults to the hardcoded calibration value (2.5).
        """
        super().__init__()

        self.alpha = alpha if alpha is not None else self.ALPHA
        self.gamma = gamma if gamma is not None else self.GAMMA

        self.dice_loss = DiceLoss(
            smooth_nr=0.0,
            smooth_dr=1e-5,
            to_onehot_y=True,
            softmax=True,
        )
        self.focal_loss = FocalLoss(
            gamma=self.gamma,
            to_onehot_y=True,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined Stenosis-Aware Loss.

        Args:
            pred   (Tensor): Model prediction logits, shape (B, C, D, H, W).
            target (Tensor): Ground truth labels,     shape (B, 1, D, H, W).

        Returns:
            Tensor: Scalar loss value.
        """
        d_loss = self.dice_loss(pred, target)
        f_loss = self.focal_loss(pred, target)

        return self.alpha * d_loss + (1.0 - self.alpha) * f_loss


# ── Standalone BoundaryLoss (kept for potential future experiments) ──────────
# Not part of the verified StenosisAwareLoss formulation.  Import directly
# if needed in curriculum or experimental scripts.

import numpy as np
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(nn.Module):
    """
    Penalizes predictions that leak beyond ground-truth vessel boundaries
    by computing the inner product of foreground probability with a
    Euclidean distance map from the GT surface.

    NOTE: This is NOT included in StenosisAwareLoss.  It is preserved
    here as a standalone module for experimental use.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): Model logits,         shape (B, 2, D, H, W).
            target (Tensor): Ground truth labels,   shape (B, 1, D, H, W).
        """
        probs = torch.softmax(pred, dim=1)
        fg_prob = probs[:, 1]  # (B, D, H, W)

        target_np = target.detach().cpu().numpy().astype(np.uint8)

        dist_maps = []
        for b in range(target_np.shape[0]):
            gt_mask = target_np[b, 0]  # (D, H, W)
            if np.any(gt_mask):
                dist_map = distance_transform_edt(1 - gt_mask)
            else:
                dist_map = np.zeros_like(gt_mask, dtype=np.float32)
            dist_maps.append(dist_map)

        dist_maps = np.stack(dist_maps, axis=0)
        dist_tensor = torch.from_numpy(dist_maps).to(pred.device, dtype=torch.float32)

        return torch.mean(fg_prob * dist_tensor)


# ── Smoke Test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing StenosisAwareLoss (α=0.4, γ=2.5)...")
    loss_fn = StenosisAwareLoss()

    # Verify calibration
    assert loss_fn.alpha == 0.4, f"Expected α=0.4, got {loss_fn.alpha}"
    assert loss_fn.gamma == 2.5, f"Expected γ=2.5, got {loss_fn.gamma}"
    print(f"  α = {loss_fn.alpha}  (Dice weight)")
    print(f"  γ = {loss_fn.gamma}  (Focal focusing parameter)")

    # Mock model prediction (logits for 2 classes: background and vessel)
    mock_pred = torch.randn(2, 2, 32, 32, 32, requires_grad=True)
    # Mock ground-truth labels (values 0 or 1)
    mock_target = torch.randint(0, 2, (2, 1, 32, 32, 32), dtype=torch.float32)

    loss = loss_fn(mock_pred, mock_target)
    print(f"  L_total = {loss.item():.4f}")
    loss.backward()
    print("  Backward pass OK.")

    # Verify the formula: α * Dice + (1-α) * Focal
    with torch.no_grad():
        d = loss_fn.dice_loss(mock_pred, mock_target)
        f = loss_fn.focal_loss(mock_pred, mock_target)
        expected = 0.4 * d + 0.6 * f
        print(f"  Dice component:  {d.item():.4f}")
        print(f"  Focal component: {f.item():.4f}")
        print(f"  Manual check:    0.4*{d.item():.4f} + 0.6*{f.item():.4f} = {expected.item():.4f}")
    print("All checks passed.")
