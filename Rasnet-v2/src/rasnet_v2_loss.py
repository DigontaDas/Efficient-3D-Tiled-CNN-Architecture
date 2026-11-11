"""
RASNet-v2 Compound Loss Module.
Fused FP32 implementation matching exact 2-class softmax FocalLoss(gamma=1.5) +
DiceLoss(include_background=False) + SoftclDiceLoss(include_background=False) while
computing shared log_softmax / probs / foreground targets once per forward pass.

Avoids MONAI 1.6.1's FocalLoss(include_background=False, use_softmax=True) bug,
where slicing input[:, 1:] prior to log_softmax yields 0.0 loss and 0.0 gradients.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.cldice import soft_skel


class SingleHeadCompoundLoss(nn.Module):
    """Memory-efficient fused Dice + 2-class softmax Focal (+ SoftclDice) computed strictly in FP32."""

    def __init__(
        self,
        w_dice: float = 0.5,
        w_focal: float = 0.3,
        w_cldice: float = 0.2,
        gamma: float = 1.5,
        cl_iter: int = 3,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
    ):
        super().__init__()
        self.w_dice = float(w_dice)
        self.w_focal = float(w_focal)
        self.w_cldice = float(w_cldice)
        self.gamma = float(gamma)
        self.cl_iter = int(cl_iter)
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim < 3 or pred.shape[1] < 2:
            raise ValueError(f"Expected logits (B, C>=2, *spatial), got {tuple(pred.shape)}")
        if target.shape[0] != pred.shape[0] or target.shape[2:] != pred.shape[2:]:
            raise ValueError(f"Target shape {tuple(target.shape)} incompatible with pred shape {tuple(pred.shape)}")
        n_ch = int(pred.shape[1])
        if target.shape[1] not in (1, n_ch):
            raise ValueError(f"Expected target channel dim 1 or {n_ch}, got {target.shape[1]}")

        with torch.amp.autocast("cuda", enabled=False):
            p32 = pred.float()
            loss = p32.new_zeros(())
            if self.w_dice <= 0.0 and self.w_focal <= 0.0 and self.w_cldice <= 0.0:
                return loss

            log_probs = F.log_softmax(p32, dim=1)
            probs = log_probs.exp()

            need_fg = (self.w_dice > 0.0) or (self.w_cldice > 0.0)
            if target.shape[1] == 1:
                idx = target.long()
                if self.w_focal > 0.0:
                    log_pt = torch.gather(log_probs, dim=1, index=idx)
                    pt = torch.gather(probs, dim=1, index=idx)
                    focal = (-(1.0 - pt).clamp(min=0.0).pow(self.gamma) * log_pt).mean() / float(n_ch)
                    loss = loss + self.w_focal * focal
                if need_fg:
                    probs_fg = probs[:, 1:]
                    if n_ch == 2:
                        target_fg = (idx == 1).float()
                    else:
                        target_fg = torch.zeros_like(probs).scatter_(1, idx, 1.0)[:, 1:]
            else:
                target_oh = target.float()
                if self.w_focal > 0.0:
                    focal = (-(1.0 - probs).clamp(min=0.0).pow(self.gamma) * log_probs * target_oh).mean()
                    loss = loss + self.w_focal * focal
                if need_fg:
                    probs_fg = probs[:, 1:]
                    target_fg = target_oh[:, 1:]

            if self.w_dice > 0.0:
                reduce_axis = list(range(2, p32.ndim))
                tp = (probs_fg * target_fg).sum(dim=reduce_axis)
                denom = probs_fg.sum(dim=reduce_axis) + target_fg.sum(dim=reduce_axis)
                dice = (1.0 - (2.0 * tp + self.smooth_nr) / (denom + self.smooth_dr)).mean()
                loss = loss + self.w_dice * dice

            if self.w_cldice > 0.0:
                skel_pred = soft_skel(probs_fg, self.cl_iter)
                with torch.no_grad():
                    skel_true = soft_skel(target_fg, self.cl_iter)
                cl_reduce = list(range(1, probs_fg.ndim))
                tprec = ((skel_pred * target_fg).sum(dim=cl_reduce) + 1.0) / (skel_pred.sum(dim=cl_reduce) + 1.0)
                tsens = ((skel_true * probs_fg).sum(dim=cl_reduce) + 1.0) / (skel_true.sum(dim=cl_reduce) + 1.0)
                cldice = (1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + 1e-4)).mean()
                loss = loss + self.w_cldice * cldice

        return loss


class RASNetV2Loss(nn.Module):
    """
    Unified loss supporting both single-tensor and (main, ds_outputs) predictions:
      - Main head: w_dice * Dice + w_focal * Focal + w_cldice * SoftclDice
      - Aux DS heads:
        ds_outputs[0] = 1/4 resolution -> ds_weights[0] (default 0.25)
        ds_outputs[1] = 1/2 resolution -> ds_weights[1] (default 0.50)
        Dice + Focal only (unless ds_include_cldice=True), trilinear align_corners=False.
    """

    def __init__(
        self,
        w_dice: float = 0.5,
        w_focal: float = 0.3,
        w_cldice: float = 0.2,
        gamma: float = 1.5,
        cl_iter: int = 3,
        ds_weights: tuple[float, float] = (0.25, 0.5),
        ds_include_cldice: bool = False,
    ):
        super().__init__()
        self.main_loss = SingleHeadCompoundLoss(
            w_dice=w_dice,
            w_focal=w_focal,
            w_cldice=w_cldice,
            gamma=gamma,
            cl_iter=cl_iter,
        )
        # Keep auxiliary DS loss balanced over Dice+Focal without heavy skeleton iterations
        aux_sum = max(float(w_dice) + float(w_focal), 1e-8)
        self.aux_loss = SingleHeadCompoundLoss(
            w_dice=float(w_dice) / aux_sum if not ds_include_cldice else float(w_dice),
            w_focal=float(w_focal) / aux_sum if not ds_include_cldice else float(w_focal),
            w_cldice=float(w_cldice) if ds_include_cldice else 0.0,
            gamma=gamma,
            cl_iter=max(1, int(cl_iter) - 1),
        )
        self.ds_weights = tuple(float(w) for w in ds_weights)

    def forward(
        self,
        outputs: torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            main_pred, ds_preds = outputs[0], outputs[1]
        else:
            main_pred, ds_preds = outputs, []

        total = self.main_loss(main_pred, target)
        for weight, aux in zip(self.ds_weights, ds_preds):
            if weight <= 0.0:
                continue
            with torch.amp.autocast("cuda", enabled=False):
                aux_up = F.interpolate(
                    aux.float(),
                    size=target.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            total = total + weight * self.aux_loss(aux_up, target)
        return total


class BaseDiceFocalLoss(RASNetV2Loss):
    """Matched base-loss control (w_cldice = 0, w_dice = 0.5, w_focal = 0.5, gamma = 1.5)."""

    def __init__(self, gamma: float = 1.5):
        super().__init__(
            w_dice=0.5,
            w_focal=0.5,
            w_cldice=0.0,
            gamma=gamma,
            cl_iter=0,
            ds_weights=(0.0, 0.0),
            ds_include_cldice=False,
        )
