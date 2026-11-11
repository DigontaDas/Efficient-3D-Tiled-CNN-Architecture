"""Gate 1 Verification Suite: Architecture scaling, GroupNorm safety, fused FP32 loss, DS weights, mid-epoch resume, SpatialPadd uint8 safety, and surface-v1 parity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from monai.losses import DiceLoss, FocalLoss, SoftclDiceLoss

ROOT = Path(r"C:\Thesis_RASNET")
V2_DIR = ROOT / "Rasnet-v2"
V2_SRC = V2_DIR / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V2_SRC))

from benchmark import metrics as v1_metrics
from benchmark import postprocess as v1_postprocess
from augmentations import (
    PAD_IMAGE_VAL,
    apply_shared_patch_augmentations,
    build_per_key_spatial_pad,
    sample_pos_neg_patch,
)
from cache_builder import build_cache
import eval_fair_v2
from rasnet_v2_loss import BaseDiceFocalLoss, RASNetV2Loss, SingleHeadCompoundLoss
from rasnet_v2_model import RASNetV2, build_monai_model, count_parameters
from train_fair_v2 import run_training


@pytest.mark.parametrize("init_filters", [16, 24, 32])
@pytest.mark.parametrize("gate_mode", ["conv_bias", "groupnorm_orig"])
@pytest.mark.parametrize("enable_ds", [True, False])
def test_rasnet_v2_width_gate_ds_shapes(init_filters: int, gate_mode: str, enable_ds: bool):
    net = RASNetV2(
        init_filters=init_filters,
        gate_mode=gate_mode,
        enable_ds=enable_ds,
        out_channels=2,
    )
    x = torch.randn(1, 1, 32, 32, 16)
    net.train()
    out_tr = net(x)
    if enable_ds:
        assert isinstance(out_tr, tuple) and len(out_tr) == 2
        main_p, ds_p = out_tr
        assert main_p.shape == (1, 2, 32, 32, 16)
        assert len(ds_p) == 2
        assert ds_p[0].shape == (1, 2, 8, 8, 4)
        assert ds_p[1].shape == (1, 2, 16, 16, 8)
    else:
        assert torch.is_tensor(out_tr) and out_tr.shape == (1, 2, 32, 32, 16)

    net.eval()
    with torch.no_grad():
        out_ev = net(x)
    assert torch.is_tensor(out_ev) and out_ev.shape == (1, 2, 32, 32, 16)


@pytest.mark.parametrize("arch", ["rasnet_v2", "segresnet", "vnet", "unet3d"])
def test_unified_monai_builders_and_amp_loss(arch: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_monai_model(arch=arch, init_filters=16).to(device)
    assert count_parameters(net) > 100_000
    net.train()
    x = torch.randn(2, 1, 32, 32, 32, device=device)
    y = (torch.rand(2, 1, 32, 32, 32, device=device) > 0.90).long()

    crit = RASNetV2Loss(w_dice=0.5, w_focal=0.3, w_cldice=0.2, gamma=1.5, cl_iter=2).to(device)
    crit_base = BaseDiceFocalLoss(gamma=1.5).to(device)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = net(x)
    loss = crit(out, y)
    loss_base = crit_base(out[0] if isinstance(out, tuple) else out, y)
    assert torch.isfinite(loss) and float(loss.item()) > 0.0
    assert torch.isfinite(loss_base) and float(loss_base.item()) > 0.0
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_fused_focal_nonzero_grad_and_monai_parity():
    torch.manual_seed(7)
    x_ref = torch.randn(2, 2, 20, 20, 20, requires_grad=True)
    x_fus = x_ref.detach().clone().requires_grad_(True)
    y = (torch.rand(2, 1, 20, 20, 20) > 0.85).long()

    # Pure Focal component must be strictly positive with non-zero gradients (fixing MONAI's 0.0 bug)
    focal_only = SingleHeadCompoundLoss(w_dice=0.0, w_focal=1.0, w_cldice=0.0, gamma=1.5)
    f_val = focal_only(x_fus, y)
    assert float(f_val.item()) > 0.01
    f_val.backward()
    assert x_fus.grad is not None and float(x_fus.grad.abs().max().item()) > 1e-6
    assert float(x_fus.grad.norm().item()) > 1e-4

    # Exact FocalLoss(include_background=True, use_softmax=True) parity including gradients
    x_f_ref = x_ref.detach().clone().requires_grad_(True)
    f_monai = FocalLoss(include_background=True, to_onehot_y=True, use_softmax=True, gamma=1.5)(x_f_ref, y)
    f_monai.backward()
    assert abs(float(f_val.item()) - float(f_monai.item())) <= 1e-6
    assert float((x_fus.grad - x_f_ref.grad).abs().max().item()) <= 1e-7

    # Also test (B, 2, ...) one-hot target path & C=3 multi-class path
    y_oh = torch.cat([(y == 0).float(), (y == 1).float()], dim=1)
    assert abs(float(focal_only(x_fus.detach(), y_oh).item()) - float(f_val.item())) <= 1e-6
    x_3ch = torch.randn(1, 3, 12, 12, 12, requires_grad=True)
    y_3ch = torch.randint(0, 3, (1, 1, 12, 12, 12))
    l_3ch = SingleHeadCompoundLoss(w_dice=0.5, w_focal=0.3, w_cldice=0.2, gamma=1.5, cl_iter=2)(x_3ch, y_3ch)
    assert torch.isfinite(l_3ch) and float(l_3ch.item()) > 0.0
    l_3ch.backward()
    assert float(x_3ch.grad.abs().max().item()) > 0.0

    # Exact numerical & gradient parity vs 2-class MONAI Dice + Focal(include_background=True, use_softmax=True) + SoftclDice
    x_ref2 = x_ref.detach().clone().requires_grad_(True)
    x_fus2 = x_ref.detach().clone().requires_grad_(True)
    ref_dice = DiceLoss(include_background=False, to_onehot_y=True, softmax=True, smooth_nr=1e-5, smooth_dr=1e-5)
    ref_focal = FocalLoss(include_background=True, to_onehot_y=True, use_softmax=True, gamma=1.5)
    ref_cl = SoftclDiceLoss(iter_=3, include_background=False, to_onehot_y=True, softmax=True)

    l_ref = 0.5 * ref_dice(x_ref2, y) + 0.3 * ref_focal(x_ref2, y) + 0.2 * ref_cl(x_ref2, y)
    l_fus = SingleHeadCompoundLoss(w_dice=0.5, w_focal=0.3, w_cldice=0.2, gamma=1.5, cl_iter=3)(x_fus2, y)
    assert abs(float(l_ref.item()) - float(l_fus.item())) <= 1e-6

    l_ref.backward()
    l_fus.backward()
    grad_diff = float((x_ref2.grad - x_fus2.grad).abs().max().item())
    assert grad_diff <= 1e-6


def test_ds_weights_ordering_and_yaml_alignment():
    cfg = yaml.safe_load((V2_DIR / "configs" / "fair_v2.yaml").read_text(encoding="utf-8"))
    assert cfg["loss"]["rasnet_v2"]["ds_weights"] == [0.25, 0.5]

    crit = RASNetV2Loss(w_dice=0.5, w_focal=0.3, w_cldice=0.2, gamma=1.5, cl_iter=2)
    assert crit.ds_weights == (0.25, 0.5)

    net = RASNetV2(init_filters=16, gate_mode="conv_bias", enable_ds=True, out_channels=2)
    net.train()
    x = torch.randn(1, 1, 32, 32, 16)
    y = (torch.rand(1, 1, 32, 32, 16) > 0.85).long()
    main_p, ds_p = net(x)
    assert ds_p[0].shape == (1, 2, 8, 8, 4)   # 1/4 resolution -> weight 0.25
    assert ds_p[1].shape == (1, 2, 16, 16, 8)  # 1/2 resolution -> weight 0.50

    loss_total = crit((main_p, ds_p), y)
    up_q = torch.nn.functional.interpolate(ds_p[0].float(), size=y.shape[2:], mode="trilinear", align_corners=False)
    up_h = torch.nn.functional.interpolate(ds_p[1].float(), size=y.shape[2:], mode="trilinear", align_corners=False)
    expected = crit.main_loss(main_p, y) + 0.25 * crit.aux_loss(up_q, y) + 0.50 * crit.aux_loss(up_h, y)
    assert abs(float(loss_total.item()) - float(expected.item())) <= 1e-6

    # Verify ordering is non-degenerate vs inverted weights
    crit_inv = RASNetV2Loss(w_dice=0.5, w_focal=0.3, w_cldice=0.2, gamma=1.5, cl_iter=2, ds_weights=(0.5, 0.25))
    assert abs(float(loss_total.item()) - float(crit_inv((main_p, ds_p), y).item())) > 1e-5


def test_per_key_spatial_padd_no_uint8_255_corruption():
    pad = build_per_key_spatial_pad(patch_size_xyz=(160, 160, 96))
    d = {
        "image": torch.zeros(1, 120, 120, 72, dtype=torch.float32),
        "label": torch.randint(0, 2, (1, 120, 120, 72), dtype=torch.uint8),
    }
    out = pad(d)
    assert out["image"].shape == (1, 160, 160, 96)
    assert out["label"].shape == (1, 160, 160, 96)
    assert out["label"].dtype == torch.uint8
    assert set(out["label"].unique().tolist()) <= {0, 1}
    assert np.isclose(float(out["image"].min().item()), PAD_IMAGE_VAL, atol=1e-5)

    # Also test fast cached volume sampler on short Z volume (<96 slices)
    img_np = np.zeros((1, 120, 120, 72), dtype=np.float16)
    lbl_np = np.zeros((1, 120, 120, 72), dtype=np.uint8)
    lbl_np[0, 50:70, 50:70, 30:45] = 1
    fg = np.argwhere(lbl_np[0] > 0).astype(np.int16)
    bg = np.argwhere(lbl_np[0] == 0)[:1000].astype(np.int16)
    rng = np.random.default_rng(42)
    p_img, p_lbl = sample_pos_neg_patch(img_np, lbl_np, fg, bg, rng, patch_size_xyz=(160, 160, 96))
    p_img, p_lbl = apply_shared_patch_augmentations(p_img, p_lbl, rng)
    assert p_img.shape == (1, 160, 160, 96)
    assert set(np.unique(p_lbl).tolist()) <= {0, 1}
    assert np.isclose(float(p_img.min()), PAD_IMAGE_VAL, atol=2e-3)


def test_surface_v1_and_postprocess_exact_parity_with_round1():
    rng = np.random.default_rng(123)
    gt = np.zeros((32, 32, 32), dtype=bool)
    gt[4:28, 14:17, 14:17] = True
    gt[14:17, 14:26, 14:17] = True
    pred = np.zeros((32, 32, 32), dtype=bool)
    pred[4:28, 13:17, 14:17] = True
    pred[14:17, 14:25, 14:17] = True
    sp = (0.50, 0.35, 0.35)

    m_v1 = v1_metrics.compute_metrics(pred, gt, sp)
    m_v2 = eval_fair_v2.compute_metrics(pred, gt, sp)
    assert m_v1["status"] == m_v2["status"] == "ok"
    for k in ("dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall"):
        assert np.isfinite(m_v1[k])
        assert abs(float(m_v1[k]) - float(m_v2[k])) == 0.0

    # Alias and postprocess parity
    prob = rng.uniform(0.0, 0.4, size=(28, 32, 32)).astype(np.float32)
    prob[6:22, 14:18, 14:18] = 0.80
    prob[2:5, 2:5, 2:5] = 0.60  # small secondary island (~27 vox)
    m1, s1 = v1_postprocess.prune_components(prob >= 0.5, sp, keep_top_k=1)
    m2, s2 = eval_fair_v2.prune_connected_components_3d(prob >= 0.5, sp, keep_top_k=1)
    assert np.array_equal(m1, m2)
    assert s1 == s2

    # Seeded dust filter retains disconnected 3rd branch exceeding min_seed_voxels or min_comp_vol_mm3
    prob_branch = prob.copy()
    prob_branch[20:26, 2:6, 2:6] = 0.70  # secondary artery branch (~96 voxels >= 50 seed voxels)
    m_seed, s_seed = eval_fair_v2.seeded_component_filter_3d(
        prob_branch, sp, tau=0.45, tau_seed=0.55, min_seed_voxels=50, min_comp_vol_mm3=999.0
    )
    assert s_seed["components_retained"] == 2
    assert np.any(m_seed[20:26, 2:6, 2:6])


def test_mid_epoch_state_step_resume_parity_cpu(tmp_path: Path):
    # Create tiny synthetic dataset, splits, val20, and config inside tmp_path
    dset = tmp_path / "dset"
    dset.mkdir()
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    train_ids = [1, 2]
    val_ids = [3]
    test_ids = [851]
    for cid in train_ids + val_ids + test_ids:
        vol = np.full((24, 24, 24), -50.0, dtype=np.float32)
        vol[6:18, 6:18, 6:18] = 250.0
        lbl = np.zeros((24, 24, 24), dtype=np.uint8)
        lbl[8:16, 8:16, 8:16] = 1
        nib.save(nib.Nifti1Image(vol, affine), dset / f"{cid}.img.nii.gz")
        nib.save(nib.Nifti1Image(lbl, affine), dset / f"{cid}.label.nii.gz")

    splits_p = tmp_path / "splits.json"
    splits_p.write_text(json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}), encoding="utf-8")
    val20_p = tmp_path / "val20.json"
    val20_p.write_text(json.dumps({"case_ids": val_ids}), encoding="utf-8")

    v2_root = tmp_path / "v2_root"
    cfg = yaml.safe_load((V2_DIR / "configs" / "fair_v2.yaml").read_text(encoding="utf-8"))
    cfg["study"]["root_dir"] = str(v2_root)
    cfg["study"]["dataset_root"] = str(dset)
    cfg["study"]["splits_file"] = str(splits_p)
    cfg["study"]["val20_ids_file"] = str(val20_p)
    cfg["preprocessing"]["pixdim_xyz"] = [0.5, 0.5, 0.5]
    cfg_p = tmp_path / "fair_test.yaml"
    cfg_p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    build_cache(cfg_p, case_ids_override=train_ids + val_ids, max_workers=1)

    run_training(
        cfg_p,
        run_name="unint",
        arch="rasnet_v2",
        init_filters=16,
        lr=3e-4,
        seed=42,
        epochs=2,
        steps_per_epoch=4,
        val_interval=99,
        num_workers=0,
        mid_epoch_ckpt_every_steps=2,
        train_ids_override=train_ids,
        val_ids_override=val_ids,
        force_device="cpu",
        patch_size_override=(32, 32, 32),
        eval_at_end=False,
    )
    res_int = run_training(
        cfg_p,
        run_name="res",
        arch="rasnet_v2",
        init_filters=16,
        lr=3e-4,
        seed=42,
        epochs=2,
        steps_per_epoch=4,
        val_interval=99,
        num_workers=0,
        mid_epoch_ckpt_every_steps=2,
        interrupt_after_step=2,
        train_ids_override=train_ids,
        val_ids_override=val_ids,
        force_device="cpu",
        patch_size_override=(32, 32, 32),
        eval_at_end=False,
    )
    assert res_int["status"] == "interrupted" and res_int["global_step"] == 2
    mid_p = v2_root / "checkpoints" / "res" / "mid_epoch_state.pt"
    assert mid_p.is_file()
    mid_st = torch.load(mid_p, map_location="cpu", weights_only=False)
    assert mid_st["next_epoch"] == 1 and mid_st["next_step_in_epoch"] == 2

    res_done = run_training(
        cfg_p,
        run_name="res",
        arch="rasnet_v2",
        init_filters=16,
        lr=3e-4,
        seed=42,
        epochs=2,
        steps_per_epoch=4,
        val_interval=99,
        num_workers=0,
        mid_epoch_ckpt_every_steps=2,
        train_ids_override=train_ids,
        val_ids_override=val_ids,
        force_device="cpu",
        patch_size_override=(32, 32, 32),
        eval_at_end=False,
    )
    assert res_done["status"] == "complete"

    st_u = torch.load(v2_root / "checkpoints" / "unint" / "latest_state.pt", map_location="cpu", weights_only=False)
    st_r = torch.load(v2_root / "checkpoints" / "res" / "latest_state.pt", map_location="cpu", weights_only=False)
    max_w = max(float((st_u["model_state_dict"][k].float() - st_r["model_state_dict"][k].float()).abs().max()) for k in st_u["model_state_dict"])
    max_exp = max(
        float((st_u["optimizer_state_dict"]["state"][k]["exp_avg"].float() - st_r["optimizer_state_dict"]["state"][k]["exp_avg"].float()).abs().max())
        for k in st_u["optimizer_state_dict"]["state"]
    )
    max_loss = max(abs(float(a["train_loss"]) - float(b["train_loss"])) for a, b in zip(st_u["history"], st_r["history"]))
    assert max_w <= 1e-6
    assert max_exp <= 1e-6
    assert max_loss <= 1e-6
    assert st_u["global_opt_step"] == st_r["global_opt_step"] == 8
    assert st_u["scheduler_state_dict"]["_last_lr"] == st_r["scheduler_state_dict"]["_last_lr"]


def test_bf16_amp_selection_and_segresnet_large_activation_stability():
    from train_fair_v2 import resolve_amp_dtype

    assert resolve_amp_dtype(torch.device("cpu")) == torch.float16
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return
    dev = torch.device("cuda")
    assert resolve_amp_dtype(dev) == torch.bfloat16

    net = build_monai_model("segresnet", init_filters=16).to(dev)
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(5.0)
    x = torch.randn(1, 1, 64, 64, 64, device=dev) * 20.0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_bf = net(x)
    assert torch.isfinite(out_bf).all(), "BF16 SegResNet overflowed on large activations"
