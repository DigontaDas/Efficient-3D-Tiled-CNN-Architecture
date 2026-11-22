"""Strategy A (retrain_v1_1) fixed 3D segmentation architectures and unified model factory.

Fixes baked-in v1 defects without modifying frozen Phase3_Local_Integration/rasnet_model.py:
  1. AttentionGate3DV11 replaces GroupNorm(1, 1) in psi with biased Conv3d + Sigmoid so gates
     can genuinely close on empty/background patches.
  2. Removes redundant dual bias in W_g + W_x (single post-sum GroupNorm before ReLU).
  3. Guards GroupNorm groups via max divisor <= min(4, F_int), avoiding 1-ch InstanceNorm
     degeneration at Gate 2 (F_int = 8 -> GroupNorm(4, 8)).
  4. Derives all attention gate [4F, 2F, F] and DS head [4F, 2F] widths dynamically from
     init_filters, with clean enable_attention / enable_ds toggles for controlled ablations.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, VNet


def _safe_num_groups(channels: int, max_groups: int = 4) -> int:
    """Return the largest divisor of `channels` that is <= min(max_groups, channels)."""
    ng = min(int(max_groups), max(int(channels), 1))
    while ng > 1 and (int(channels) % ng) != 0:
        ng -= 1
    return max(ng, 1)


class AttentionGate3DV11(nn.Module):
    """Fixed 3D additive attention gate (`conv_bias` gating head, single post-sum GroupNorm)."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        f_int = max(int(F_int), 1)
        num_groups = _safe_num_groups(f_int, max_groups=4)
        self.W_g = nn.Conv3d(F_g, f_int, kernel_size=1, stride=1, padding=0, bias=False)
        self.W_x = nn.Conv3d(F_l, f_int, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.GroupNorm(num_groups, f_int)
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        combined = self.relu(self.norm(self.W_g(g) + self.W_x(x)))
        alpha = self.psi(combined)
        return x * alpha


class RASNetV11(SegResNet):
    """Fixed Strategy A RASNet (dynamic width, conv_bias gates, toggleable attention & DS)."""

    def __init__(
        self,
        enable_attention: bool = True,
        enable_ds: bool = True,
        **kwargs: Any,
    ):
        kwargs.setdefault("spatial_dims", 3)
        kwargs.setdefault("in_channels", 1)
        kwargs.setdefault("out_channels", 2)
        kwargs.setdefault("init_filters", 16)
        kwargs.setdefault("dropout_prob", 0.1)

        self.out_channels = int(kwargs["out_channels"])
        self.enable_attention = bool(enable_attention)
        self.enable_ds = bool(enable_ds)
        super().__init__(**kwargs)

        n_up = len(self.up_samples)
        if n_up < 2:
            raise ValueError(f"Expected >= 2 decoder stages, got {n_up}")

        # Skip feature channels across decoder stages: [4F, 2F, F] for default 4-stage SegResNet
        skip_channels = [int(self.init_filters) * (2 ** (n_up - 1 - i)) for i in range(n_up)]

        if self.enable_attention:
            self.attention_gates = nn.ModuleList([
                AttentionGate3DV11(F_g=ch, F_l=ch, F_int=max(ch // 2, 1))
                for ch in skip_channels
            ])
        else:
            self.attention_gates = nn.ModuleList()

        if self.enable_ds:
            # Level 0 (1/4 spatial res): skip_channels[0] (4F)
            # Level 1 (1/2 spatial res): skip_channels[1] (2F)
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(skip_channels[0], self.out_channels, kernel_size=1, bias=True),
                nn.Conv3d(skip_channels[1], self.out_channels, kernel_size=1, bias=True),
            ])
        else:
            self.ds_heads = nn.ModuleList()

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]):
        ds_outputs: list[torch.Tensor] = []
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            g = up(x)
            skip = down_x[i + 1]
            if self.enable_attention:
                skip = self.attention_gates[i](g=g, x=skip)
            x = upl(g + skip)
            if self.enable_ds and i < 2:
                ds_outputs.append(self.ds_heads[i](x))

        if self.use_conv_final:
            x = self.conv_final(x)

        if self.training and self.enable_ds:
            return x, ds_outputs
        return x


def build_strategy_a_model(
    arch: str,
    init_filters: int = 16,
    dropout_prob: float = 0.1,
    enable_attention: bool = True,
    enable_ds: bool = True,
) -> nn.Module:
    """Instantiate any Strategy A trainable 2-channel 3D segmentation model."""
    arch_l = str(arch).lower()
    if arch_l == "rasnet_v1_1":
        return RASNetV11(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            init_filters=int(init_filters),
            dropout_prob=float(dropout_prob),
            enable_attention=bool(enable_attention),
            enable_ds=bool(enable_ds),
        )
    if arch_l == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            init_filters=int(init_filters),
            dropout_prob=float(dropout_prob),
        )
    if arch_l == "vnet":
        return VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            act="elu",
            dropout_prob_down=float(dropout_prob),
            dropout_prob_up=(float(dropout_prob), float(dropout_prob)),
            bias=True,
        )
    raise ValueError(f"Unsupported Strategy A trainable architecture: {arch!r}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
