# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\rasnet_model.py
"""
Residual Attention Segmentation Net (RASNet) Architecture.
Subclasses MONAI's SegResNet to integrate:
  1. AttentionGate3D modules on encoder-decoder skip paths.
  2. Auxiliary Deep Supervision heads for multi-scale intermediate loss calculations.
Supports strict=False parameter loading to transfer public ImageCAS SegResNet weights.
"""
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

class AttentionGate3D(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Gating signal (g) from decoder and skip features (x) from encoder
        are mapped, added, activated, and multiplied to filter spatial noise.
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Combine gating and skip signals
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class RASNet(SegResNet):
    def __init__(self, *args, **kwargs):
        """
        Passes arguments to SegResNet and instantiates custom AttentionGates
        and Deep Supervision heads.
        """
        # Ensure default filters and layers match your baseline configuration
        if "init_filters" not in kwargs:
            kwargs["init_filters"] = 16
        if "dropout_prob" not in kwargs:
            kwargs["dropout_prob"] = 0.1
        if "in_channels" not in kwargs:
            kwargs["in_channels"] = 1
        if "out_channels" not in kwargs:
            kwargs["out_channels"] = 2
        if "spatial_dims" not in kwargs:
            kwargs["spatial_dims"] = 3
            
        self.out_channels = kwargs["out_channels"]
        super().__init__(*args, **kwargs)

        # 3 Attention Gates matching the channels at each skip connection level
        # up_samples output channels for SegResNet (init_filters=16) are: [64, 32, 16]
        self.attention_gates = nn.ModuleList([
            AttentionGate3D(F_g=64, F_l=64, F_int=32),
            AttentionGate3D(F_g=32, F_l=32, F_int=16),
            AttentionGate3D(F_g=16, F_l=16, F_int=8)
        ])

        # Deep Supervision heads to predict output channels at intermediate resolutions
        # Level 0 (1/2 size): 64 channels, Level 1 (1/4 size): 32 channels
        self.ds_heads = nn.ModuleList([
            nn.Conv3d(64, self.out_channels, kernel_size=1, bias=True),
            nn.Conv3d(32, self.out_channels, kernel_size=1, bias=True)
        ])

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]] or torch.Tensor:
        """
        Overrides SegResNet's decode to apply Attention Gates on skip connections
        and return intermediate outputs when training (for Deep Supervision).
        """
        ds_outputs = []
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            g = up(x)
            skip = down_x[i + 1]
            
            # Apply attention mapping to filter skip connection features
            attended_skip = self.attention_gates[i](g=g, x=skip)
            x = g + attended_skip
            x = upl(x)
            
            # Save deep supervision intermediate outputs (i = 0, 1) during training
            if i < 2:
                ds_outputs.append(self.ds_heads[i](x))

        if self.use_conv_final:
            x = self.conv_final(x)

        if self.training:
            return x, ds_outputs
        return x

if __name__ == "__main__":
    print("Testing RASNet Architecture compilation...")
    model = RASNet()
    model.train()
    
    # Mock forward pass
    mock_input = torch.randn(2, 1, 96, 96, 96)
    out_final, out_ds = model(mock_input)
    
    print("Successful compilation.")
    print(f"Final output shape: {out_final.shape}")
    print(f"Deep supervision levels: {len(out_ds)}")
    for idx, ds in enumerate(out_ds):
        print(f"  Level {idx} shape: {ds.shape}")
