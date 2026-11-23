"""Tests-first (Rule 6) unit & synthetic CPU integration tests for PART II Strategy A (retrain_v1_1/)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[2]
STUDY_DIR = REPO / "retrain_v1_1"
CFG_PATH = STUDY_DIR / "configs" / "strategy_a.yaml"


# ---------------------------------------------------------------------------
# 1. Config, Isolation & Pre-Registration Contracts
# ---------------------------------------------------------------------------
def test_strategy_a_config_isolation_and_schema():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    assert cfg["study"]["study_dir"] == "retrain_v1_1"
    assert cfg["study"]["seed"] == 42
    assert cfg["training"]["epochs"] == 200
    assert cfg["training"]["grad_clip_norm"] == pytest.approx(1.0)
    assert cfg["preprocessing"]["crop_foreground_hu_gt_0"] is False
    assert cfg["preprocessing"]["pixdim_xyz"] == [0.349609375, 0.349609375, 0.5]
    assert cfg["preprocessing"]["patch_size_xyz"] == [160, 160, 96]
    assert cfg["loss"]["name"] == "DiceCELoss"
    assert cfg["loss"]["include_background"] is False
    assert cfg["loss"]["softmax"] is True
    assert cfg["loss"]["batch"] is True
    assert cfg["loss"]["ds_weights_quarter_half"] == [0.25, 0.50]

    tau_grid = cfg["evaluation"]["tau_grid"]
    assert tau_grid[0] == pytest.approx(0.10)
    assert tau_grid[-1] == pytest.approx(0.70)
    assert len(tau_grid) == 13
    assert cfg["evaluation"]["bootstrap_resamples"] == 10000
    assert cfg["evaluation"]["ni_margin"] == pytest.approx(-0.005)

    model_names = [m["name"] for m in cfg["models"]]
    assert model_names == [
        "rasnet_v1_1",
        "segresnet_v1_1",
        "vnet_v1_1",
        "ablation_no_attn_v1_1",
        "ablation_no_ds_v1_1",
        "nnunet_v2",
    ]
    # Rule 1: dataset_root must not be hardcoded in YAML/code of strategy_a
    raw_txt = CFG_PATH.read_text(encoding="utf-8")
    assert "secondary-imagcas-dataset" not in raw_txt
    assert "C:\\" not in raw_txt and "C:/" not in raw_txt


def test_preregistration_amendment_section7_present():
    text = (REPO / "BENCHMARK_PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "## 7. Amendment: Strategy A" in text
    for required in (
        "retrain_v1_1",
        "conv_bias",
        "DiceCELoss",
        "0.349609375",
        "160",
        "10,000",
        "0.10",
        "0.70",
        "-0.005",
        "retrospective",
    ):
        assert required in text, f"Missing {required!r} in BENCHMARK_PREREGISTRATION.md Section 7"


# ---------------------------------------------------------------------------
# 2. Architecture Fixes (Part II §1.1)
# ---------------------------------------------------------------------------
def test_attention_gate_v11_closes_on_background_and_guards_groupnorm():
    from retrain_v1_1.src.models import AttentionGate3DV11

    # Gate 2 for init_filters=16 has F_g=16, F_l=16, F_int=8 -> num_groups must be min(4, 8) = 4 (not 8)
    gate = AttentionGate3DV11(F_g=16, F_l=16, F_int=8)
    assert isinstance(gate.norm, nn.GroupNorm)
    assert gate.norm.num_groups == 4
    assert gate.norm.num_channels == 8
    # Redundant dual bias removed from W_g and W_x
    assert gate.W_g.bias is None
    assert gate.W_x.bias is None
    # psi uses biased Conv3d + Sigmoid (NO GroupNorm(1, 1))
    assert not any(isinstance(m, nn.GroupNorm) for m in gate.psi.modules())
    assert isinstance(gate.psi[0], nn.Conv3d)
    assert gate.psi[0].bias is not None
    assert isinstance(gate.psi[1], nn.Sigmoid)

    # Prove gate can genuinely close on constant/empty background patches when bias is negative
    with torch.no_grad():
        gate.psi[0].weight.zero_()
        gate.psi[0].bias.fill_(-5.0)
        g = torch.zeros(1, 16, 8, 8, 8)
        x = torch.ones(1, 16, 8, 8, 8)
        out = gate(g, x)
        assert float(out.max().item()) < 0.01

    # Guard works for small F_int (e.g., F_int=2 -> num_groups=2; F_int=1 -> num_groups=1)
    assert AttentionGate3DV11(4, 4, 2).norm.num_groups == 2
    assert AttentionGate3DV11(2, 2, 1).norm.num_groups == 1


def test_models_factory_all_five_trainable_variants_and_dynamic_f32():
    from retrain_v1_1.src.models import build_strategy_a_model, count_parameters

    x = torch.zeros((1, 1, 32, 32, 16))
    m_full = build_strategy_a_model("rasnet_v1_1", init_filters=16, enable_attention=True, enable_ds=True)
    m_full.train()
    main_out, ds_out = m_full(x)
    assert main_out.shape == (1, 2, 32, 32, 16)
    assert len(ds_out) == 2
    assert ds_out[0].shape == (1, 2, 8, 8, 4)    # 1/4 res
    assert ds_out[1].shape == (1, 2, 16, 16, 8)  # 1/2 res
    m_full.eval()
    assert m_full(x).shape == (1, 2, 32, 32, 16)

    # Ablation No-Attn (enable_attention=False, enable_ds=True)
    m_no_attn = build_strategy_a_model("rasnet_v1_1", init_filters=16, enable_attention=False, enable_ds=True)
    m_no_attn.train()
    ma_out, ma_ds = m_no_attn(x)
    assert ma_out.shape == (1, 2, 32, 32, 16) and len(ma_ds) == 2
    assert len(m_no_attn.attention_gates) == 0

    # Ablation No-DS (enable_attention=True, enable_ds=False)
    m_no_ds = build_strategy_a_model("rasnet_v1_1", init_filters=16, enable_attention=True, enable_ds=False)
    m_no_ds.train()
    nd_out = m_no_ds(x)
    assert isinstance(nd_out, torch.Tensor) and nd_out.shape == (1, 2, 32, 32, 16)
    assert len(m_no_ds.ds_heads) == 0

    # SegResNet & V-Net
    m_seg = build_strategy_a_model("segresnet", init_filters=16)
    m_vnet = build_strategy_a_model("vnet", init_filters=16)
    assert m_seg(x).shape == (1, 2, 32, 32, 16)
    assert m_vnet(x).shape == (1, 2, 32, 32, 16)
    assert count_parameters(m_seg) == 4_700_914
    assert count_parameters(m_full) > count_parameters(m_seg)

    # Dynamic F=32
    m_f32 = build_strategy_a_model("rasnet_v1_1", init_filters=32, enable_attention=True, enable_ds=True).eval()
    assert m_f32(x).shape == (1, 2, 32, 32, 16)


# ---------------------------------------------------------------------------
# 3. Fixed Deep Supervision & DiceCE(batch=True) Loss (Part II §1.2-§1.3)
# ---------------------------------------------------------------------------
def test_fixed_ds_loss_weighting_and_downsampled_gt_preservation():
    from retrain_v1_1.src.losses import StrategyADiceCEDSLoss, downsample_binary_target

    # Thin 1-voxel vessel must survive 1/2 and 1/4 max-pool downsampling
    tgt = torch.zeros((2, 1, 16, 16, 16), dtype=torch.long)
    tgt[0, 0, 5, 5, 5] = 1
    tgt[1, 0, 10, 10, 10] = 1
    tgt_half = downsample_binary_target(tgt, (8, 8, 8))
    tgt_quarter = downsample_binary_target(tgt, (4, 4, 4))
    assert int(tgt_half.sum().item()) == 2
    assert int(tgt_quarter.sum().item()) == 2

    crit = StrategyADiceCEDSLoss(ds_weights_quarter_half=(0.25, 0.50), normalize_by_sum=True)
    main = torch.zeros((2, 2, 16, 16, 16), requires_grad=True)
    ds_q = torch.zeros((2, 2, 4, 4, 4), requires_grad=True)   # 1/4 res -> weight 0.25/1.75
    ds_h = torch.zeros((2, 2, 8, 8, 8), requires_grad=True)   # 1/2 res -> weight 0.50/1.75

    # Verify exact weighted average formula against single-head calls
    l_main = crit.base_loss(main, tgt)
    l_q = crit.base_loss(ds_q, tgt_quarter)
    l_h = crit.base_loss(ds_h, tgt_half)
    expected = (1.0 * l_main + 0.50 * l_h + 0.25 * l_q) / 1.75
    got = crit((main, [ds_q, ds_h]), tgt)
    assert torch.allclose(got, expected, atol=1e-6)

    # Verify single-tensor (no DS) equals base_loss directly
    assert torch.allclose(crit(main, tgt), l_main, atol=1e-6)

    # Prove 1/2-res head gets strictly larger weight than 1/4-res head (not inverted!)
    assert crit.ds_weights_quarter_half[1] == pytest.approx(0.50)
    assert crit.ds_weights_quarter_half[0] == pytest.approx(0.25)
    assert crit.ds_weights_quarter_half[1] > crit.ds_weights_quarter_half[0]


# ---------------------------------------------------------------------------
# 4. Preprocessing, Full 3D Augmentations & Transform-Hashed Cache (Part II §1.4-§1.5)
# ---------------------------------------------------------------------------
def test_transforms_no_crop_foreground_and_full_3d_augs_preserve_binary_label():
    from retrain_v1_1.src.transforms import (
        apply_3d_train_augmentations,
        build_deterministic_preprocess,
        compute_transform_hash,
    )

    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    tf = build_deterministic_preprocess(cfg["preprocessing"], with_label=True)
    tf_names = [type(op).__name__ for op in tf.transforms]
    assert "CropForegroundd" not in tf_names
    assert "Spacingd" in tf_names
    assert "CTStandardizeD" in tf_names

    # Transform hash changes if ANY preprocessing parameter changes (prevents LAS/RAS cache reuse bug)
    h1 = compute_transform_hash(cfg["preprocessing"])
    p_mod = dict(cfg["preprocessing"], pixdim_xyz=[0.5, 0.5, 0.5])
    h2 = compute_transform_hash(p_mod)
    assert h1 != h2 and len(h1) == 64

    # Augmentations work safely on negative z-scored CT values and preserve binary {0, 1} labels
    rng = np.random.default_rng(42)
    img = rng.uniform(-1.747, 3.2, size=(1, 32, 32, 16)).astype(np.float32)
    lbl = np.zeros((1, 32, 32, 16), dtype=np.uint8)
    lbl[0, 10:22, 10:22, 4:12] = 1
    aug_img, aug_lbl = apply_3d_train_augmentations(img, lbl, cfg["augmentations"], seed=12345)
    assert aug_img.shape == img.shape and aug_lbl.shape == lbl.shape
    assert np.all(np.isfinite(aug_img))
    assert set(np.unique(aug_lbl).tolist()) <= {0, 1}


def test_transform_hashed_blosc2_cache_roundtrip_and_isolation(tmp_path: Path):
    from retrain_v1_1.src.cache import (
        case_cache_dir,
        load_cached_case,
        save_cached_case_atomic,
    )
    from retrain_v1_1.src.transforms import compute_transform_hash

    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    h1 = compute_transform_hash(cfg["preprocessing"])
    h2 = compute_transform_hash(dict(cfg["preprocessing"], ct_mean=150.0))
    d1 = case_cache_dir(tmp_path, h1)
    d2 = case_cache_dir(tmp_path, h2)
    assert d1 != d2

    img = np.linspace(-1.74, 3.0, 1 * 24 * 24 * 16, dtype=np.float16).reshape(1, 24, 24, 16)
    lbl = np.zeros((1, 24, 24, 16), dtype=np.uint8)
    lbl[0, 8:16, 8:16, 4:12] = 1
    fg = np.argwhere(lbl[0] > 0).astype(np.int16)
    bg = np.argwhere(lbl[0] == 0)[:100].astype(np.int16)

    save_cached_case_atomic(d1, 101, img, lbl, fg, bg, meta={"case_id": 101, "transform_sha256": h1})
    got_img, got_lbl, got_fg, got_bg, got_meta = load_cached_case(d1, 101)
    assert np.array_equal(got_img, img)
    assert np.array_equal(got_lbl, lbl)
    assert np.array_equal(got_fg, fg)
    assert np.array_equal(got_bg, bg)
    assert got_meta["transform_sha256"] == h1


# ---------------------------------------------------------------------------
# 5. Trainer Full Resume Checkpoint, RNG Parity & Manifest (Part II §1.5)
# ---------------------------------------------------------------------------
def test_trainer_full_resume_checkpoint_and_manifest_cpu(tmp_path: Path):
    from retrain_v1_1.src.cache import case_cache_dir, save_cached_case_atomic
    from retrain_v1_1.src.train import train_model
    from retrain_v1_1.src.transforms import compute_transform_hash

    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    cfg["training"]["epochs"] = 4
    cfg["training"]["steps_per_epoch"] = 2
    cfg["training"]["batch_size"] = 2
    cfg["training"]["warmup_epochs"] = 1
    cfg["training"]["val_interval_epochs"] = 2
    cfg["training"]["num_workers"] = 0
    cfg["preprocessing"]["patch_size_xyz"] = [32, 32, 16]
    cfg["evaluation"]["roi_size_xyz"] = [32, 32, 16]

    tf_hash = compute_transform_hash(cfg["preprocessing"])
    cdir = case_cache_dir(tmp_path / "cache", tf_hash)
    rng = np.random.default_rng(7)
    for cid in (1, 2, 11, 12):
        img = rng.uniform(-1.5, 2.5, size=(1, 32, 32, 16)).astype(np.float16)
        lbl = np.zeros((1, 32, 32, 16), dtype=np.uint8)
        lbl[0, 10:22, 10:22, 4:12] = 1
        fg = np.argwhere(lbl[0] > 0).astype(np.int16)
        bg = np.argwhere(lbl[0] == 0)[:50].astype(np.int16)
        save_cached_case_atomic(cdir, cid, img, lbl, fg, bg, meta={"case_id": cid, "transform_sha256": tf_hash})

    # Run 2 epochs uninterrupted -> check full resume checkpoint keys
    r1 = tmp_path / "run_resume"
    train_model(
        cfg=cfg,
        model_spec=cfg["models"][0],
        cache_root=tmp_path / "cache",
        ckpt_dir=r1 / "ckpt",
        log_dir=r1 / "log",
        train_ids=[1, 2],
        val_ids=[11, 12],
        device_str="cpu",
        stop_after_epoch=2,
    )
    latest = torch.load(r1 / "ckpt" / "latest_state.pt", map_location="cpu", weights_only=False)
    for k in (
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_val_dice",
        "rng_state",
        "transform_sha256",
    ):
        assert k in latest, f"Missing resume key {k}"
    assert latest["epoch"] == 2
    assert (r1 / "ckpt" / "best_model.pth").is_file()

    # Resume epochs 3..4 and compare final weights against uninterrupted 1..4 run
    train_model(
        cfg=cfg,
        model_spec=cfg["models"][0],
        cache_root=tmp_path / "cache",
        ckpt_dir=r1 / "ckpt",
        log_dir=r1 / "log",
        train_ids=[1, 2],
        val_ids=[11, 12],
        device_str="cpu",
    )
    r2 = tmp_path / "run_cont"
    train_model(
        cfg=cfg,
        model_spec=cfg["models"][0],
        cache_root=tmp_path / "cache",
        ckpt_dir=r2 / "ckpt",
        log_dir=r2 / "log",
        train_ids=[1, 2],
        val_ids=[11, 12],
        device_str="cpu",
    )
    sd1 = torch.load(r1 / "ckpt" / "final_epoch200.pth", map_location="cpu", weights_only=True)
    sd2 = torch.load(r2 / "ckpt" / "final_epoch200.pth", map_location="cpu", weights_only=True)
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k], atol=1e-6), f"Resume weight drift in {k}"

    manifest = json.loads((r1 / "log" / "train_manifest.json").read_text(encoding="utf-8"))
    for k in ("git", "config_sha256", "transform_sha256", "seed", "env", "best_epoch", "best_val_dice"):
        assert k in manifest
    assert (r1 / "ckpt" / "DONE.flag").is_file()
    log_df = pd.read_csv(r1 / "log" / "train_log.csv")
    assert list(log_df["epoch"]) == [1, 2, 3, 4]
    assert log_df["val_dice"].notna().sum() == 2  # epochs 2 and 4


# ---------------------------------------------------------------------------
# 6. Eval Pipeline: Widened Grid [0.10..0.70], LF Lock, 10k Bootstrap & Empty Guard
# ---------------------------------------------------------------------------
def test_eval_pipeline_widened_grid_lock_and_10k_bootstrap_stats(tmp_path: Path):
    from retrain_v1_1.src.eval_pipeline import (
        TAU_GRID_V11,
        build_strategy_a_lock,
        compute_strategy_a_stats,
        load_strategy_a_lock,
        write_strategy_a_lock,
    )

    assert TAU_GRID_V11[0] == pytest.approx(0.10)
    assert TAU_GRID_V11[-1] == pytest.approx(0.70)
    assert len(TAU_GRID_V11) == 13

    splits = {"train": list(range(1, 11)), "val": [11, 12, 13], "test": [851, 852, 853, 854]}
    models = ["rasnet_v1_1", "segresnet_v1_1", "vnet_v1_1", "ablation_no_attn_v1_1", "ablation_no_ds_v1_1", "nnunet_v2"]
    val_rows = []
    for m in models:
        peak = 0.15 if m == "nnunet_v2" else 0.40
        for cid in splits["val"]:
            for mode in ("sw050_gauss_1pass", "sw050_gauss_tta8"):
                for tau in TAU_GRID_V11:
                    val_rows.append({
                        "model": m,
                        "case_id": cid,
                        "mode": mode,
                        "tau": tau,
                        "dice": 0.82 - abs(tau - peak),
                    })
    lock = build_strategy_a_lock(pd.DataFrame(val_rows), splits, models)
    assert lock["thresholds"]["nnunet_v2"]["sw050_gauss_tta8"] == pytest.approx(0.15)
    assert lock["thresholds"]["rasnet_v1_1"]["sw050_gauss_tta8"] == pytest.approx(0.40)

    lock_path = tmp_path / "locked_recipes.json"
    sha = write_strategy_a_lock(lock_path, lock)
    assert b"\r\n" not in lock_path.read_bytes()
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == sha
    loaded, sha2 = load_strategy_a_lock(lock_path, splits, models)
    assert sha2 == sha
    assert loaded["thresholds"] == lock["thresholds"]

    # Synthetic test per-case dataframe with 1 empty prediction to verify strict precision & NI
    rng = np.random.default_rng(0)
    t_rows = []
    for m in models:
        for cid in splits["test"]:
            for r in ("R1_raw_1pass", "R2_raw_tta8", "R3_tuned_1pass_top2", "R4_tuned_tta8_top2"):
                row = {
                    "model": m, "case_id": cid, "recipe": r, "tau": 0.40, "status": "ok",
                    "dice": float(rng.uniform(0.78, 0.85)),
                    "iou": float(rng.uniform(0.65, 0.74)),
                    "precision": float(rng.uniform(0.80, 0.88)),
                    "recall": float(rng.uniform(0.75, 0.84)),
                    "hd95": float(rng.uniform(5.0, 9.0)),
                    "asd": float(rng.uniform(1.0, 2.0)),
                    "cldice": float(rng.uniform(0.84, 0.90)),
                    "centerline_recall": float(rng.uniform(0.84, 0.91)),
                    "components_total": 2,
                }
                if m == "vnet_v1_1" and cid == 851 and r == "R4_tuned_tta8_top2":
                    row.update(status="prediction_empty", dice=0.0, iou=0.0, recall=0.0,
                               precision=np.nan, hd95=np.nan, asd=np.nan, cldice=np.nan,
                               centerline_recall=np.nan, components_total=0)
                t_rows.append(row)

    summ, fam, ni = compute_strategy_a_stats(pd.DataFrame(t_rows), n_boot=1000, seed=42)
    v_r4 = summ[(summ["model"] == "vnet_v1_1") & (summ["recipe"] == "R4_tuned_tta8_top2")].iloc[0]
    assert int(v_r4["n_empty_pred"]) == 1
    assert v_r4["precision_strict_mean"] < v_r4["precision_mean"]
    assert set(ni["recipe"]) == {"R1_raw_1pass", "R2_raw_tta8", "R3_tuned_1pass_top2", "R4_tuned_tta8_top2"}
    assert "FA_primary_rasnet_vs_nnunet" in set(fam["family"])
    assert "FB_rasnet_vs_segresnet" in set(fam["family"])
    assert "FE_ablations_vs_rasnet" in set(fam["family"])


def test_restore_rng_state_handles_cuda_mapped_tensors():
    """torch.load(..., map_location='cuda') moves ByteTensors to GPU; _restore_rng_state must restore on CPU."""
    from retrain_v1_1.src.train import _capture_rng_state, _restore_rng_state

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    state = _capture_rng_state()
    # Simulate torch.load(path, map_location=torch.device('cuda')) moving every tensor to GPU
    gpu_state = {
        "python": state["python"],
        "numpy": state["numpy"],
        "torch_cpu": state["torch_cpu"].to("cuda"),
        "torch_cuda": [t.to("cuda") for t in state["torch_cuda"]],
    }
    _restore_rng_state(gpu_state)

