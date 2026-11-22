"""Strategy A matched DiceCE(batch=True) loss with fixed downsampled-GT deep supervision.

Implements:
  - Base loss: MONAI DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, batch=True)
    computed strictly in FP32 (matching nnU-Net's BatchDice + CE).
  - Deep supervision weighting (non-inverted):
      L_total = (1.0 * L_main + 0.50 * L_half(1/2) + 0.25 * L_quarter(1/4)) / 1.75
    where ds_outputs[0] is 1/4 resolution (weight 0.25) and ds_outputs[1] is 1/2 resolution (weight 0.50).
  - Downsamples ground-truth labels via max_pool3d on integer stride ratios (preserving thin 1-2 voxel
    distal coronary arteries) instead of trilinearly upsampling coarse logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


def downsample_binary_target(
    target: torch.Tensor,
    spatial_size: tuple[int, int, int],
) -> torch.Tensor:
    """Downsample (B, 1, D, H, W) integer binary GT to `spatial_size` preserving thin vessels."""
    if tuple(target.shape[2:]) == tuple(spatial_size):
        return target.long()
    src = tuple(int(s) for s in target.shape[2:])
    dst = tuple(int(s) for s in spatial_size)
    if all(s >= d and d > 0 and (s % d) == 0 for s, d in zip(src, dst)):
        stride = tuple(s // d for s, d in zip(src, dst))
        pooled = F.max_pool3d(target.float(), kernel_size=stride, stride=stride)
        return (pooled > 0.5).long()
    resized = F.interpolate(target.float(), size=dst, mode="nearest")
    return (resized > 0.5).long()


class StrategyADiceCEDSLoss(nn.Module):
    """Unified FP32 DiceCELoss(batch=True) supporting single-output and (main, [ds_1/4, ds_1/2])."""

    def __init__(
        self,
        main_weight: float = 1.0,
        ds_weights_quarter_half: tuple[float, float] = (0.25, 0.50),
        normalize_by_sum: bool = True,
    ):
        super().__init__()
        self.main_weight = float(main_weight)
        self.ds_weights_quarter_half = (float(ds_weights_quarter_half[0]), float(ds_weights_quarter_half[1]))
        self.normalize_by_sum = bool(normalize_by_sum)
        self.base_loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            batch=True,
        )

    def forward(
        self,
        outputs: torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
        target: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            tgt = target.long()
            if isinstance(outputs, (tuple, list)):
                main_pred, ds_preds = outputs[0], list(outputs[1])
            else:
                main_pred, ds_preds = outputs, []

            total = self.main_weight * self.base_loss(main_pred.float(), tgt)
            weight_sum = self.main_weight

            if ds_preds:
                if len(ds_preds) != len(self.ds_weights_quarter_half):
                    raise ValueError(
                        f"Expected {len(self.ds_weights_quarter_half)} DS heads, got {len(ds_preds)}"
                    )
                for w, aux_pred in zip(self.ds_weights_quarter_half, ds_preds):
                    if w <= 0.0:
                        continue
                    aux_tgt = downsample_binary_target(tgt, tuple(aux_pred.shape[2:]))
                    total = total + w * self.base_loss(aux_pred.float(), aux_tgt)
                    weight_sum += w

            if self.normalize_by_sum and weight_sum > 0.0:
                total = total / weight_sum
            return total
