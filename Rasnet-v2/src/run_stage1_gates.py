"""
Executes Stage 1 Mandatory Pre-Launch Gates (Gates 2, 3, and 4) cleanly on RTX 4080 SUPER
and writes C:/Thesis_RASNET/Rasnet-v2/logs/STAGE1_GATES_REPORT.json.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

ROOT = Path(r"C:\Thesis_RASNET")
V2_DIR = ROOT / "Rasnet-v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V2_DIR / "src"))

from benchmark import metrics as v1_metrics
from cache_builder import build_cache, generate_patch_overlays
from dataset_pairs import build_val20_stratified_ids, resolve_case_pairs
import eval_fair_v2
from rasnet_v2_model import build_monai_model
from train_fair_v2 import atomic_torch_save, run_training
from train_nnunet_v2_ref import run_nnunet_v2_seed


def _release_mem() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all_gates() -> dict:
    cfg_path = V2_DIR / "configs" / "fair_v2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    report: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ------------------------------------------------------------------
    # Gate 2: Freeze val20_curves_ids.json, Cache 3 cases, Parity & PNG Overlays
    # ------------------------------------------------------------------
    print("\n=== GATE 2: val20 Stratification, Cache Shape Parity & Visual Overlays ===")
    val20 = build_val20_stratified_ids(
        cfg["study"]["val_armA_baseline_csv"],
        cfg["study"]["splits_file"],
        output_json=cfg["study"]["val20_ids_file"],
        seed=42,
    )
    gate_cases = [14, 45, 47]
    cache_res = build_cache(cfg_path, case_ids_override=gate_cases)
    overlay_dir = V2_DIR / "logs" / "patch_overlays"
    overlays = generate_patch_overlays(cfg_path, gate_cases, overlay_dir)

    plans_path = ROOT / "Q1_Publication_Package" / "matched_200ep_benchmark" / "nnUNet_preprocessed" / "Dataset501_CoronarySeg" / "nnUNetPlans.json"
    if not plans_path.is_file():
        # Fallback if preprocessed plans moved to results/trained_model
        plans_path = ROOT / "Q1_Publication_Package" / "matched_200ep_benchmark" / "nnUNet_results" / "Dataset501_CoronarySeg" / "nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres" / "plans.json"
    plans_spacing_zyx = None
    if plans_path.is_file():
        plans = json.loads(plans_path.read_text(encoding="utf-8"))
        plans_spacing_zyx = plans["configurations"]["3d_fullres"]["spacing"]
        assert np.allclose(list(reversed(cfg["preprocessing"]["pixdim_xyz"])), plans_spacing_zyx, atol=1e-6)
        assert list(reversed(cfg["preprocessing"]["patch_size_xyz"])) == plans["configurations"]["3d_fullres"]["patch_size"]

    report["gate2"] = {
        "status": "PASS",
        "val20_sha256": val20["sha256_case_ids"],
        "cache_summary": cache_res,
        "plans_spacing_zyx_verified": plans_spacing_zyx,
        "overlays": [str(p) for p in overlays],
    }
    _release_mem()

    # ------------------------------------------------------------------
    # Gate 3: RTX 4080 SUPER Timing/VRAM Check & 4-Epoch Resume Parity (MONAI + nnU-Net)
    # ------------------------------------------------------------------
    print("\n=== GATE 3: Hardware Timing, Peak VRAM (<10.5GB) & 4-Epoch Resume Parity ===")
    gate_val_ids = [14, 45, 47]

    # Clean any stale gate test dirs
    for dname in (
        "_gate3_f16_unint",
        "_gate3_f16_res",
        "_gate3_f32_timing",
        "_gate3_cpu_unint",
        "_gate3_cpu_res",
        "_gate3_mid_unint",
        "_gate3_mid_res",
    ):
        for sub in ("checkpoints", "logs"):
            p = V2_DIR / sub / dname
            if p.exists():
                shutil.rmtree(p)

    # 3A: Hardware check for F=16 (mb=2x1) & F=32 (mb=1x2) on RTX 4080 SUPER
    r_f16_t = run_training(
        cfg_path,
        run_name="_gate3_f16_unint",
        arch="rasnet_v2",
        init_filters=16,
        lr=3e-4,
        seed=42,
        epochs=4,
        steps_per_epoch=20,
        val_interval=2,
        num_workers=2,
        train_ids_override=gate_cases,
        val_ids_override=gate_val_ids,
    )
    _release_mem()
    r_f32_t = run_training(
        cfg_path,
        run_name="_gate3_f32_timing",
        arch="rasnet_v2",
        init_filters=32,
        lr=3e-4,
        seed=42,
        epochs=2,
        steps_per_epoch=20,
        val_interval=2,
        num_workers=2,
        train_ids_override=gate_cases,
        val_ids_override=gate_val_ids,
    )
    _release_mem()
    assert r_f16_t["peak_vram_mb"] < 10.5 * 1024, f"F16 Peak VRAM exceeded 10.5GB: {r_f16_t['peak_vram_mb']}"
    assert r_f32_t["peak_vram_mb"] < 10.5 * 1024, f"F32 Peak VRAM exceeded 10.5GB: {r_f32_t['peak_vram_mb']}"

    # 3B-1: Deterministic CPU 4-epoch atomic interrupt/resume parity (<= 1e-6 weight & optimizer exp_avg match)
    for rn, int_ep in (("_gate3_cpu_unint", None), ("_gate3_cpu_res", 2), ("_gate3_cpu_res", None)):
        run_training(
            cfg_path,
            run_name=rn,
            arch="rasnet_v2",
            init_filters=16,
            lr=3e-4,
            seed=42,
            epochs=4,
            steps_per_epoch=2,
            val_interval=99,
            num_workers=0,
            interrupt_after_epoch=int_ep,
            train_ids_override=gate_cases,
            val_ids_override=[14],
            force_device="cpu",
            patch_size_override=(64, 64, 64),
            eval_at_end=False,
        )
    cu = torch.load(V2_DIR / "checkpoints" / "_gate3_cpu_unint" / "latest_state.pt", map_location="cpu", weights_only=False)
    cr = torch.load(V2_DIR / "checkpoints" / "_gate3_cpu_res" / "latest_state.pt", map_location="cpu", weights_only=False)
    cpu_w_diff = max(
        float((cu["model_state_dict"][k].float() - cr["model_state_dict"][k].float()).abs().max())
        for k in cu["model_state_dict"]
    )
    cpu_exp_diff = max(
        float((cu["optimizer_state_dict"]["state"][k]["exp_avg"].float() - cr["optimizer_state_dict"]["state"][k]["exp_avg"].float()).abs().max())
        for k in cu["optimizer_state_dict"]["state"]
    )
    cpu_loss_diff = max(abs(float(a["train_loss"]) - float(b["train_loss"])) for a, b in zip(cu["history"], cr["history"]))
    assert cpu_w_diff <= 1e-6, f"CPU deterministic weight diff > 1e-6: {cpu_w_diff}"
    assert cpu_exp_diff <= 1e-6, f"CPU deterministic exp_avg diff > 1e-6: {cpu_exp_diff}"
    assert cpu_loss_diff <= 1e-6, f"CPU deterministic train_loss diff > 1e-6: {cpu_loss_diff}"
    assert cu["scheduler_state_dict"]["_last_lr"] == cr["scheduler_state_dict"]["_last_lr"]
    _release_mem()

    # 3B-1b: Deterministic CPU mid_epoch_state.pt step-level interrupt & resume parity
    for rn, int_step in (("_gate3_mid_unint", None), ("_gate3_mid_res", 2), ("_gate3_mid_res", None)):
        res_mid = run_training(
            cfg_path,
            run_name=rn,
            arch="rasnet_v2",
            init_filters=16,
            lr=3e-4,
            seed=42,
            epochs=2,
            steps_per_epoch=4,
            val_interval=99,
            num_workers=0,
            mid_epoch_ckpt_every_steps=2,
            interrupt_after_step=int_step,
            train_ids_override=gate_cases,
            val_ids_override=[14],
            force_device="cpu",
            patch_size_override=(64, 64, 64),
            eval_at_end=False,
        )
        if int_step is not None:
            assert res_mid["status"] == "interrupted" and res_mid["global_step"] == 2
            assert (V2_DIR / "checkpoints" / "_gate3_mid_res" / "mid_epoch_state.pt").is_file()
    mu = torch.load(V2_DIR / "checkpoints" / "_gate3_mid_unint" / "latest_state.pt", map_location="cpu", weights_only=False)
    mr = torch.load(V2_DIR / "checkpoints" / "_gate3_mid_res" / "latest_state.pt", map_location="cpu", weights_only=False)
    mid_w_diff = max(
        float((mu["model_state_dict"][k].float() - mr["model_state_dict"][k].float()).abs().max())
        for k in mu["model_state_dict"]
    )
    mid_exp_diff = max(
        float((mu["optimizer_state_dict"]["state"][k]["exp_avg"].float() - mr["optimizer_state_dict"]["state"][k]["exp_avg"].float()).abs().max())
        for k in mu["optimizer_state_dict"]["state"]
    )
    mid_loss_diff = max(abs(float(a["train_loss"]) - float(b["train_loss"])) for a, b in zip(mu["history"], mr["history"]))
    assert mid_w_diff <= 1e-6, f"CPU mid_epoch weight diff > 1e-6: {mid_w_diff}"
    assert mid_exp_diff <= 1e-6, f"CPU mid_epoch exp_avg diff > 1e-6: {mid_exp_diff}"
    assert mid_loss_diff <= 1e-6, f"CPU mid_epoch train_loss diff > 1e-6: {mid_loss_diff}"
    assert mu["global_opt_step"] == mr["global_opt_step"] == 8
    assert mu["scheduler_state_dict"]["_last_lr"] == mr["scheduler_state_dict"]["_last_lr"]
    _release_mem()

    # 3B-2: RTX 4080 SUPER CUDA 4-epoch interrupt at ep 2 -> resume through ep 4 (exact step & LR match)
    run_training(
        cfg_path,
        run_name="_gate3_f16_res",
        arch="rasnet_v2",
        init_filters=16,
        lr=3e-4,
        seed=42,
        epochs=4,
        steps_per_epoch=20,
        val_interval=2,
        num_workers=2,
        interrupt_after_epoch=2,
        train_ids_override=gate_cases,
        val_ids_override=gate_val_ids,
    )
    _release_mem()
    r_f16_res = run_training(
        cfg_path,
        run_name="_gate3_f16_res",
        arch="rasnet_v2",
        init_filters=16,
        lr=3e-4,
        seed=42,
        epochs=4,
        steps_per_epoch=20,
        val_interval=2,
        num_workers=2,
        train_ids_override=gate_cases,
        val_ids_override=gate_val_ids,
    )
    _release_mem()
    st_u = torch.load(V2_DIR / "checkpoints" / "_gate3_f16_unint" / "latest_state.pt", map_location="cpu", weights_only=False)
    st_r = torch.load(V2_DIR / "checkpoints" / "_gate3_f16_res" / "latest_state.pt", map_location="cpu", weights_only=False)
    assert st_u["global_opt_step"] == st_r["global_opt_step"] == 80
    assert st_u["scheduler_state_dict"]["_last_lr"] == st_r["scheduler_state_dict"]["_last_lr"]
    max_w_diff = max(
        float((st_u["model_state_dict"][k].float() - st_r["model_state_dict"][k].float()).abs().max())
        for k in st_u["model_state_dict"]
    )
    print(
        f"[Gate 3] CPU ep-resume max |w|={cpu_w_diff:.2e}, max |exp_avg|={cpu_exp_diff:.2e} | "
        f"CPU mid-ep max |w|={mid_w_diff:.2e}, max |loss|={mid_loss_diff:.2e} | CUDA resume max |w|={max_w_diff:.2e}"
    )

    # 3C: Short 4-epoch interrupt + resume check on FairV2nnUNetTrainer (creates isolated plans/splits + epoch snapshots without touching site-packages)
    nn_test_name = "nnunet_v2_seed999"
    for p in (V2_DIR / "checkpoints" / nn_test_name, V2_DIR / "logs" / nn_test_name, V2_DIR / "nnUNet_work" / "nnUNet_results" / "seed_999"):
        if p.exists():
            shutil.rmtree(p)
    nn_int = run_nnunet_v2_seed(cfg_path, seed=999, num_epochs=4, epoch_snapshots=(2, 4), interrupt_after_epoch=2)
    assert nn_int["status"] == "interrupted"
    _release_mem()
    nn_done = run_nnunet_v2_seed(cfg_path, seed=999, num_epochs=4, epoch_snapshots=(2, 4))
    assert nn_done["status"] == "complete"
    fold_d = Path(nn_done["fold_dir"])
    assert (fold_d / "checkpoint_epoch_2.pth").is_file()
    assert (fold_d / "checkpoint_epoch_4.pth").is_file()
    assert (fold_d / "checkpoint_best.pth").is_file()

    # Check nnUNetPredictor & eval_fair_v2 batch iterator parity vs predict_single_npy_array on 2 real cases
    os.environ["nnUNet_results"] = str(V2_DIR / "nnUNet_work" / "nnUNet_results" / "seed_999")
    pred_nn = eval_fair_v2.build_nnunet_predictor(
        nn_done["model_dir"], fold=0, checkpoint_name="checkpoint_epoch_4.pth"
    )
    pairs_nn = resolve_case_pairs(cfg["study"]["dataset_root"], [14, 45], check_headers=True)
    t_nn_batch0 = time.perf_counter()
    batch_probs_nn = {
        cid: (p1, pt)
        for cid, p1, pt in eval_fair_v2.iter_nnunet_probs_from_files(
            pred_nn, [14, 45], pairs_nn, need_tta=True, num_processes=2
        )
    }
    nn_batch_sec = round(time.perf_counter() - t_nn_batch0, 2)
    nn_max_prob_diff_1pass = 0.0
    nn_max_prob_diff_tta = 0.0
    for cid_nn in (14, 45):
        im_nn = sitk.ReadImage(str(pairs_nn[cid_nn][0]))
        sp_nn_zyx = tuple(float(s) for s in reversed(im_nn.GetSpacing()))
        arr_nn = sitk.GetArrayFromImage(im_nn).astype(np.float32, copy=False)[None, ...]
        props_nn = {"spacing": sp_nn_zyx}
        pred_nn.use_mirroring = False
        _, s_p1 = pred_nn.predict_single_npy_array(
            arr_nn, props_nn, segmentation_previous_stage=None, output_file_truncated=None, save_or_return_probabilities=True
        )
        pred_nn.use_mirroring = True
        _, s_pt = pred_nn.predict_single_npy_array(
            arr_nn, props_nn, segmentation_previous_stage=None, output_file_truncated=None, save_or_return_probabilities=True
        )
        d1 = float(np.max(np.abs(batch_probs_nn[cid_nn][0] - np.asarray(s_p1[1], dtype=np.float32))))
        dt = float(np.max(np.abs(batch_probs_nn[cid_nn][1] - np.asarray(s_pt[1], dtype=np.float32))))
        nn_max_prob_diff_1pass = max(nn_max_prob_diff_1pass, d1)
        nn_max_prob_diff_tta = max(nn_max_prob_diff_tta, dt)
    assert nn_max_prob_diff_1pass <= 1e-5, f"nnU-Net 1-pass batch vs single prob diff > 1e-5: {nn_max_prob_diff_1pass}"
    assert nn_max_prob_diff_tta <= 1e-5, f"nnU-Net TTA batch vs single prob diff > 1e-5: {nn_max_prob_diff_tta}"
    print(f"[Gate 3C] nnU-Net 2-case batch (1pass+TTA) elapsed={nn_batch_sec:.2f}s | parity: 1pass={nn_max_prob_diff_1pass:.2e}, tta={nn_max_prob_diff_tta:.2e}")
    del pred_nn, batch_probs_nn
    _release_mem()

    report["gate3"] = {
        "status": "PASS",
        "f16_peak_vram_mb": r_f16_t["peak_vram_mb"],
        "f32_peak_vram_mb": r_f32_t["peak_vram_mb"],
        "cpu_deterministic_resume_max_weight_diff": cpu_w_diff,
        "cpu_deterministic_resume_max_exp_avg_diff": cpu_exp_diff,
        "cpu_mid_epoch_resume_max_weight_diff": mid_w_diff,
        "cpu_mid_epoch_resume_max_exp_avg_diff": mid_exp_diff,
        "cpu_mid_epoch_resume_max_loss_diff": mid_loss_diff,
        "cuda_resume_max_weight_diff": max_w_diff,
        "monai_lr_exact_match": st_u["scheduler_state_dict"]["_last_lr"] == st_r["scheduler_state_dict"]["_last_lr"],
        "nnunet_snapshots_verified": ["checkpoint_epoch_2.pth", "checkpoint_epoch_4.pth", "checkpoint_best.pth"],
        "nnunet_predictor_load_ok": True,
        "nnunet_2case_batch_1pass_plus_tta_sec": nn_batch_sec,
        "nnunet_batch_vs_single_1pass_max_diff": nn_max_prob_diff_1pass,
        "nnunet_batch_vs_single_tta_max_diff": nn_max_prob_diff_tta,
    }

    # ------------------------------------------------------------------
    # Gate 4: eval_fair_v2.py Native-Grid Invertd Parity on Case 0014 + CSV Per-Case Resume
    # ------------------------------------------------------------------
    print("\n=== GATE 4: Raw .nii.gz + Invertd Native-Grid Parity & Per-Case CSV Resume ===")
    ckpt_test = V2_DIR / "checkpoints" / "_gate3_f16_unint" / "best_model.pth"
    csv_full = V2_DIR / "logs" / "_gate4_eval_full.csv"
    csv_part = V2_DIR / "logs" / "_gate4_eval_resume.csv"
    for p in (csv_full, csv_part):
        if p.exists():
            p.unlink()

    df_full = eval_fair_v2.evaluate_monai_checkpoint_to_csv(
        cfg_path, ckpt_test, "rasnet_v2", 16, "conv_bias", True, [14, 45], csv_full, tau=0.30, keep_top_k=2, resume=False
    )
    # Interrupt after 1 case, then resume cleanly for case 2
    df_p1 = eval_fair_v2.evaluate_monai_checkpoint_to_csv(
        cfg_path, ckpt_test, "rasnet_v2", 16, "conv_bias", True, [14, 45], csv_part, tau=0.30, keep_top_k=2, resume=True, interrupt_after_cases=1
    )
    assert len(df_p1) == 1 and int(df_p1["case_id"].iloc[0]) == 14
    df_res = eval_fair_v2.evaluate_monai_checkpoint_to_csv(
        cfg_path, ckpt_test, "rasnet_v2", 16, "conv_bias", True, [14, 45], csv_part, tau=0.30, keep_top_k=2, resume=True
    )
    def _nan_eq(col_a, col_b) -> float:
        diffs = []
        for a, b in zip(col_a, col_b):
            fa, fb = float(a), float(b)
            diffs.append(0.0 if (np.isnan(fa) and np.isnan(fb)) else abs(fa - fb))
        return max(diffs)

    max_csv_resume_diff = 0.0
    for col in ("dice", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall"):
        diff_res = _nan_eq(df_full[col].to_numpy(), df_res[col].to_numpy())
        max_csv_resume_diff = max(max_csv_resume_diff, float(diff_res))
        assert diff_res <= 1e-12, f"CSV resume parity failed on {col}: {diff_res}"

    # Verify Case 0014 native-grid mask computed via raw .nii.gz + Invertd matches Round 1 metrics.compute_metrics 0.00e+00
    pairs = resolve_case_pairs(cfg["study"]["dataset_root"], [14], check_headers=True)
    lb_im = sitk.ReadImage(str(pairs[14][1]))
    gt_bool = sitk.GetArrayFromImage(lb_im) > 0
    sp_zyx = tuple(float(s) for s in reversed(lb_im.GetSpacing()))
    # Recompute binary mask & verify identical surface-v1 output against Round 1
    eval_dev = "cuda" if torch.cuda.is_available() else "cpu"
    net_v2 = build_monai_model("rasnet_v2", 16, "conv_bias", True).to(eval_dev)
    obj = torch.load(ckpt_test, map_location=eval_dev, weights_only=True)
    net_v2.load_state_dict(obj["model_state_dict"], strict=True)
    p14, _ = eval_fair_v2.infer_monai_native_zyx(
        net_v2, pairs[14][0], eval_fair_v2.build_raw_eval_transform(), (160, 160, 96), 0.50, "gaussian", 4, False, eval_dev
    )
    del net_v2
    _release_mem()
    assert p14.shape == gt_bool.shape
    mask14, _ = eval_fair_v2.prune_components(p14 >= 0.30, sp_zyx, keep_top_k=2)
    m_v1_0014 = v1_metrics.compute_metrics(mask14, gt_bool, sp_zyx)
    row14 = df_full[df_full["case_id"] == 14].iloc[0]

    def _zero_diff(a, b) -> float:
        fa, fb = float(a), float(b)
        if np.isnan(fa) and np.isnan(fb):
            return 0.0
        return abs(fa - fb)

    max_metric_diff = max(
        _zero_diff(m_v1_0014[k], row14[k])
        for k in ("dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall")
    )
    assert max_metric_diff == 0.0, f"Case 0014 metric parity diff={max_metric_diff}"

    # Also assert non-empty native Case 0014 geometry parity (percentile mask guarantees active 3D foreground)
    tau_nonempty = float(np.quantile(p14, 0.995))
    mask14_ne, _ = eval_fair_v2.prune_components(p14 >= tau_nonempty, sp_zyx, keep_top_k=2)
    m_ne_v1 = v1_metrics.compute_metrics(mask14_ne, gt_bool, sp_zyx)
    m_ne_v2 = eval_fair_v2.compute_metrics(mask14_ne, gt_bool, sp_zyx)
    assert m_ne_v1["status"] == m_ne_v2["status"] == "ok"
    nonempty_diff = max(
        _zero_diff(m_ne_v1[k], m_ne_v2[k])
        for k in ("dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall")
    )
    assert nonempty_diff == 0.0, f"Case 0014 non-empty metric parity diff={nonempty_diff}"

    report["gate4"] = {
        "status": "PASS",
        "case_0014_native_shape_zyx": list(gt_bool.shape),
        "case_0014_surface_v1_max_diff": max_metric_diff,
        "case_0014_nonempty_surface_v1_max_diff": nonempty_diff,
        "csv_per_case_resume_max_diff": max_csv_resume_diff,
    }

    # Cleanup temporary gate test runs so production checkpoint/log folders remain spotless
    _release_mem()
    for dname in (
        "_gate3_f16_unint",
        "_gate3_f16_res",
        "_gate3_f32_timing",
        "_gate3_cpu_unint",
        "_gate3_cpu_res",
        "_gate3_mid_unint",
        "_gate3_mid_res",
        nn_test_name,
    ):
        for sub in ("checkpoints", "logs"):
            p = V2_DIR / sub / dname
            if p.exists():
                shutil.rmtree(p)
    seed999 = V2_DIR / "nnUNet_work" / "nnUNet_results" / "seed_999"
    if seed999.exists():
        shutil.rmtree(seed999)
    for p in (csv_full, csv_part):
        if p.exists():
            p.unlink()

    out_report = V2_DIR / "logs" / "STAGE1_GATES_REPORT.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== ALL STAGE 1 GATES PASSED ===")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_all_gates()
