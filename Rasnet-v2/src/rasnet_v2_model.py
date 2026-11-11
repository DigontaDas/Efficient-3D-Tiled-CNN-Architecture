"""
RASNet-v2 Model Architectures and Unified MONAI Model Factory.
Copied/evolved inside Rasnet-v2 from Phase3_Local_Integration/rasnet_model.py.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, UNet, VNet


class AttentionGate3D(nn.Module):
    """
    3D Additive Attention Gate with safe dynamic GroupNorm divisor and selectable gating head:
      - gate_mode='conv_bias'      : Plain biased 1x1x1 Conv3d + Sigmoid (lead v2 candidate).
      - gate_mode='groupnorm_orig' : Unbiased Conv3d + GroupNorm(1, 1) + Sigmoid (v1 ablation).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int, gate_mode: str = "conv_bias"):
        super().__init__()
        if gate_mode not in ("conv_bias", "groupnorm_orig"):
            raise ValueError(f"Unsupported gate_mode: {gate_mode!r}")
        self.gate_mode = gate_mode
        ng = max(1, math.gcd(int(F_int), 8))

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(ng, F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(ng, F_int),
        )
        if gate_mode == "conv_bias":
            self.psi = nn.Sequential(
                nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.psi = nn.Sequential(
                nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(1, 1),
                nn.Sigmoid(),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class RASNetV2(SegResNet):
    """
    Width-scalable 3D SegResNet with skip-level AttentionGate3D modules and optional
    multi-scale deep supervision heads. Supports F=16, F=24, F=32 without GroupNorm crashes.
    """

    def __init__(
        self,
        gate_mode: str = "conv_bias",
        enable_ds: bool = True,
        **kwargs: Any,
    ):
        kwargs.setdefault("spatial_dims", 3)
        kwargs.setdefault("in_channels", 1)
        kwargs.setdefault("out_channels", 2)
        kwargs.setdefault("init_filters", 16)
        kwargs.setdefault("dropout_prob", 0.1)

        self.out_channels = int(kwargs["out_channels"])
        self.gate_mode = gate_mode
        self.enable_ds = bool(enable_ds)
        super().__init__(**kwargs)

        n_up = len(self.up_samples)
        if n_up < 2:
            raise ValueError(f"Expected >= 2 upsampling stages, got {n_up}")

        # Skip feature channels at decoder stages: [4F, 2F, F] for default 4-stage SegResNet
        skip_channels = [int(self.init_filters) * (2 ** (n_up - 1 - i)) for i in range(n_up)]
        self.attention_gates = nn.ModuleList([
            AttentionGate3D(
                F_g=ch,
                F_l=ch,
                F_int=max(ch // 2, 1),
                gate_mode=gate_mode,
            )
            for ch in skip_channels
        ])

        if self.enable_ds:
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
            attended_skip = self.attention_gates[i](g=g, x=skip)
            x = g + attended_skip
            x = upl(x)
            if self.enable_ds and i < 2:
                ds_outputs.append(self.ds_heads[i](x))

        if self.use_conv_final:
            x = self.conv_final(x)

        if self.training and self.enable_ds:
            return x, ds_outputs
        return x


def build_monai_model(
    arch: str,
    init_filters: int = 16,
    gate_mode: str = "conv_bias",
    enable_ds: bool = True,
    dropout_prob: float = 0.1,
) -> nn.Module:
    """Unified 2-channel 3D model builder across Tier 1 MONAI architectures."""
    arch_l = arch.lower()
    if arch_l == "rasnet_v2":
        return RASNetV2(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            init_filters=int(init_filters),
            dropout_prob=float(dropout_prob),
            gate_mode=gate_mode,
            enable_ds=enable_ds,
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
    if arch_l == "unet3d":
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 320),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance",
            dropout=0.15,
        )
    raise ValueError(f"Unknown MONAI architecture: {arch!r}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))

