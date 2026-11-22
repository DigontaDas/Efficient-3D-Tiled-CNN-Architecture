import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import pytest
from monai.networks.nets import SegResNet
from benchmark.adapters.monai_models import build_random_model
from Phase3_Local_Integration.rasnet_model import RASNet

REPO = Path(__file__).resolve().parents[1]
FROZEN_V1_LOGIT_SHA256 = "872b83d88efd991a3333f54a2f27ca696902c6af0b4dad63890453eec9c28d67"


@pytest.mark.parametrize('name',['rasnet','segresnet','vnet','3dunet'])
def test_random_architecture_output_grid(name):
    net=build_random_model(name).eval()
    with torch.no_grad(): out=net(torch.zeros((1,1,32,32,32)))
    if isinstance(out,(tuple,list)): out=out[0]
    assert out.shape[2:]==(32,32,32)


def test_rasnet_f16_strict_load_and_bit_identical_logits():
    """RASNet(init_filters=16) must stay 100% state-dict and bit-for-bit logit identical to frozen v1."""
    torch.set_num_threads(1)
    net = RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1).eval()
    assert len(net.state_dict()) == 114
    assert sum(p.numel() for p in net.parameters()) == 4_706_772
    for gate in net.attention_gates:
        assert gate.W_g[1].num_groups == 8
        assert gate.W_x[1].num_groups == 8
        assert isinstance(gate.psi[1], nn.GroupNorm)
        assert gate.psi[1].num_groups == 1 and gate.psi[1].num_channels == 1

    # Independent literal frozen v1 implementation (does NOT import AttentionGate3D from rasnet_model)
    class _LiteralFrozenV1Gate(nn.Module):
        def __init__(self, F_g: int, F_l: int, F_int: int):
            super().__init__()
            self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1, 1, 0, bias=False), nn.GroupNorm(8, F_int))
            self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1, 1, 0, bias=False), nn.GroupNorm(8, F_int))
            self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1, 1, 0, bias=False), nn.GroupNorm(1, 1), nn.Sigmoid())
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

    class _FrozenV1Ref(SegResNet):
        def __init__(self):
            super().__init__(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)
            self.attention_gates = nn.ModuleList([
                _LiteralFrozenV1Gate(F_g=64, F_l=64, F_int=32),
                _LiteralFrozenV1Gate(F_g=32, F_l=32, F_int=16),
                _LiteralFrozenV1Gate(F_g=16, F_l=16, F_int=8),
            ])
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(64, 2, kernel_size=1, bias=True),
                nn.Conv3d(32, 2, kernel_size=1, bias=True),
            ])

        def decode(self, x, down_x):
            for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
                g = up(x)
                x = upl(g + self.attention_gates[i](g=g, x=down_x[i + 1]))
            return self.conv_final(x)

    ref = _FrozenV1Ref().eval()
    ckpt = REPO / "Q1_Publication_Package" / "matched_200ep_benchmark" / "checkpoints" / "rasnet_best.pth"
    has_ckpt = ckpt.is_file()
    if has_ckpt:
        assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == (
            "8d4c25cd12d07c40423062b27b1d36836e5fd6cecb30c20c34055eeac976d1b8"
        )
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        net.load_state_dict(sd, strict=True)
        ref.load_state_dict(sd, strict=True)
    else:
        ref.load_state_dict(net.state_dict(), strict=True)

    x = torch.linspace(0.0, 1.0, 32 * 32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32, 32)
    with torch.no_grad():
        out_net = net(x)
        out_ref = ref(x)
    assert torch.equal(out_net, out_ref)
    if has_ckpt:
        assert hashlib.sha256(out_net.numpy().tobytes()).hexdigest() == FROZEN_V1_LOGIT_SHA256


def test_rasnet_dynamic_init_filters_32_and_8_forward():
    """RASNet must derive gate/DS channels from init_filters and guard GroupNorm when F_int < 8."""
    torch.set_num_threads(1)
    x = torch.zeros((1, 1, 32, 32, 32))

    # init_filters=32: train + eval, plus verify ds_outputs[0] is 1/4 res and ds_outputs[1] is 1/2 res
    net32 = RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=32, dropout_prob=0.1)
    net32.train()
    with torch.no_grad():
        main_out, ds_out = net32(x)
    assert main_out.shape == (1, 2, 32, 32, 32)
    assert len(ds_out) == 2
    assert ds_out[0].shape == (1, 2, 8, 8, 8)    # Level 0: 1/4 res
    assert ds_out[1].shape == (1, 2, 16, 16, 16) # Level 1: 1/2 res
    net32.eval()
    with torch.no_grad():
        eval_out = net32(x)
    assert eval_out.shape == (1, 2, 32, 32, 32)

    # init_filters=8: Gate 2 has F_int = 4 < 8, exercising safe GroupNorm guard
    net8 = RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=8, dropout_prob=0.1).eval()
    with torch.no_grad():
        out8 = net8(x)
    assert out8.shape == (1, 2, 32, 32, 32)
