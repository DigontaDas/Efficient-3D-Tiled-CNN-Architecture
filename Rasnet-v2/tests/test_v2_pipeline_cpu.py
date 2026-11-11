"""Comprehensive CPU unit and integration tests for RASNet-v2 fixes and Stages 2-6 pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import torch
import yaml

ROOT = Path(r"C:\Thesis_RASNET")
V2_DIR = ROOT / "Rasnet-v2"
V2_SRC = V2_DIR / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V2_SRC))

from benchmark import postprocess as v1_postprocess
from cache_builder import build_cache, case_cache_paths
import eval_fair_v2
from eval_fair_v2 import ArmEvalSpec
from rasnet_v2_model import build_monai_model
import run_queue
import stage2_pilot_eval
import stage4a_val20_curves
import stage4b_val94_lock
import stage4c_test150_eval
import stage5_stats_and_efficiency
import stage6_phaseI_clinical
from study_registry import build_stage3_eval_manifest, build_stage3_jobs
from train_nnunet_v2_ref import check_preprocessed_complete


def _make_synthetic_nifti_dataset(base_dir: Path, cids: list[int], shape=(24, 24, 24), spacing=(0.5, 0.5, 0.5)):
    base_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    pairs = {}
    for idx, cid in enumerate(cids):
        vol = np.full(shape, -100.0, dtype=np.float32)
        vol[6:18, 6:18, 6:18] = 280.0 + idx * 5.0
        lbl = np.zeros(shape, dtype=np.uint8)
        lbl[8:16, 8:16, 8:16] = 1
        im_p = base_dir / f"{cid}.img.nii.gz"
        lb_p = base_dir / f"{cid}.label.nii.gz"
        nib.save(nib.Nifti1Image(vol, affine), im_p)
        nib.save(nib.Nifti1Image(lbl, affine), lb_p)
        pairs[int(cid)] = (im_p, lb_p)
    return pairs


def test_hysteresis_empty_m_high_and_seeded_validation():
    sp = (0.5, 0.35, 0.35)
    prob = np.full((24, 24, 24), 0.35, dtype=np.float32)
    prob[5:12, 5:12, 5:12] = 0.48  # passes tau_low=0.30, below tau_high=0.60

    m1, s1 = v1_postprocess.hysteresis_threshold_3d(prob, sp, tau_low=0.30, tau_high=0.60, min_seed_voxels=5, keep_top_k=2)
    m2, s2 = eval_fair_v2.hysteresis_threshold_3d(prob, sp, tau_low=0.30, tau_high=0.60, min_seed_voxels=5, keep_top_k=2)
    assert np.array_equal(m1, m2)
    assert s1 == s2
    assert s2["components_total"] == 0

    # Seeded filter input validations
    with pytest.raises(ValueError):
        eval_fair_v2.seeded_component_filter_3d(prob, sp, tau=0.0)
    with pytest.raises(ValueError):
        eval_fair_v2.seeded_component_filter_3d(prob, sp, tau=1.5)
    with pytest.raises(ValueError):
        eval_fair_v2.seeded_component_filter_3d(prob, sp, tau=0.4, tau_seed=0.0)
    with pytest.raises(ValueError):
        eval_fair_v2.seeded_component_filter_3d(prob[0], sp, tau=0.4)
    with pytest.raises(ValueError):
        eval_fair_v2.seeded_component_filter_3d(prob, (0.5, -0.35, 0.35), tau=0.4)
    with pytest.raises(ValueError):
        eval_fair_v2.seeded_component_filter_3d(prob, sp, tau=0.4, min_seed_voxels=0)
    with pytest.raises(ValueError):
        eval_fair_v2.seeded_component_filter_3d(prob, sp, tau=0.4, min_comp_vol_mm3=-1.0)

    # Empty mask returns zero components cleanly
    m_emp, s_emp = eval_fair_v2.seeded_component_filter_3d(np.zeros_like(prob), sp, tau=0.4, tau_seed=0.5)
    assert int(m_emp.sum()) == 0 and s_emp["components_total"] == 0


def test_sweep_case_arm_c_grid_exact_parity():
    rng = np.random.default_rng(99)
    sp = (0.5, 0.35, 0.35)
    prob = rng.uniform(0.0, 0.25, size=(28, 32, 32)).astype(np.float32)
    prob[4:12, 4:10, 4:10] = 0.72  # large comp 1
    prob[16:22, 16:20, 16:20] = 0.62  # medium comp 2
    prob[2:5, 24:28, 24:28] = 0.56  # 3rd branch (48 vox)
    prob[22:24, 2:4, 2:4] = 0.42  # small noise island (8 vox)

    gt = np.zeros_like(prob, dtype=bool)
    gt[4:12, 4:10, 4:10] = True
    gt[16:22, 16:20, 16:20] = True
    gt[2:5, 24:28, 24:28] = True

    taus = [0.20, 0.35, 0.50, 0.65, 0.85]
    ts_vals = [0.50, 0.55]
    msv_vals = [40, 100]
    mvol_vals = [10.0, 50.0]

    t2_fast, sd_fast = eval_fair_v2.sweep_case_arm_c_grid(prob, gt, sp, taus, ts_vals, msv_vals, mvol_vals)

    def _dice(mask_u8):
        p = mask_u8.astype(bool)
        denom = int(p.sum()) + int(gt.sum())
        return 1.0 if denom == 0 else float(2.0 * int((p & gt).sum()) / denom)

    for tau in taus:
        m_t2, _ = eval_fair_v2.prune_components(prob >= tau, sp, keep_top_k=2)
        assert abs(t2_fast[tau] - _dice(m_t2)) == 0.0
        for ts in ts_vals:
            for msv in msv_vals:
                for mvol in mvol_vals:
                    m_sd, _ = eval_fair_v2.seeded_component_filter_3d(
                        prob, sp, tau=tau, tau_seed=ts, min_seed_voxels=msv, min_comp_vol_mm3=mvol
                    )
                    assert abs(sd_fast[(tau, ts, msv, mvol)] - _dice(m_sd)) == 0.0


def test_stats_validation_nan_filtering_rank_biserial_and_ties():
    with pytest.raises(ValueError):
        eval_fair_v2.holm_adjust([0.01, float("nan")])
    with pytest.raises(ValueError):
        eval_fair_v2.holm_adjust([-0.1, 0.2])
    with pytest.raises(ValueError):
        eval_fair_v2.holm_adjust([])

    df_a = pd.DataFrame({"case_id": [1, 2, 3, 4], "dice": [0.80, 0.70, 0.90, 0.60], "hd95": [5.0, float("nan"), 3.0, 10.0]})
    df_b = pd.DataFrame({"case_id": [1, 2, 3, 4], "dice": [0.75, 0.70, 0.85, 0.65], "hd95": [6.0, 7.0, 3.0, 8.0]})

    # Mismatched case_id set raises
    with pytest.raises(ValueError):
        eval_fair_v2.paired_with_noninferiority(df_a.iloc[:3], df_b, "dice")

    # Ties & rank_biserial & non-inferiority
    res = eval_fair_v2.paired_with_noninferiority(df_a, df_b, "dice", ni_margin=-0.05)
    assert res["wins"] == 2 and res["ties"] == 1 and res["losses"] == 1
    assert "rank_biserial" in res and np.isfinite(res["rank_biserial"])
    assert res["non_inferior"] is True  # mean diff > 0, ci_lo > -0.05

    # Strict NI margin fails when CI crosses
    res_fail = eval_fair_v2.paired_with_noninferiority(df_a, df_b, "dice", ni_margin=0.05)
    assert res_fail["non_inferior"] is False

    # NaN filtering
    with pytest.raises(ValueError):
        eval_fair_v2.paired_with_noninferiority(df_a, df_b, "hd95", allow_nan_filtering=False)
    res_nan = eval_fair_v2.paired_with_noninferiority(df_a, df_b, "hd95", allow_nan_filtering=True, direction="lower")
    assert res_nan["n"] == 3 and res_nan["n_excluded_nan"] == 1
    assert res_nan["ties"] == 1 and res_nan["wins"] == 1 and res_nan["losses"] == 1


def test_empty_csv_resume_and_invertd_prob_reuse(tmp_path: Path):
    pairs = _make_synthetic_nifti_dataset(tmp_path / "dset", [14, 45])
    cfg = yaml.safe_load((V2_DIR / "configs" / "fair_v2.yaml").read_text(encoding="utf-8"))
    cfg["study"]["root_dir"] = str(tmp_path / "v2")
    cfg["study"]["dataset_root"] = str(tmp_path / "dset")
    cfg["preprocessing"]["pixdim_xyz"] = [0.5, 0.5, 0.5]
    cfg["inference_eval"]["roi_size_xyz"] = [16, 16, 16]
    cfg_p = tmp_path / "cfg.yaml"
    cfg_p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Save dummy checkpoint
    net = build_monai_model("rasnet_v2", init_filters=16, gate_mode="conv_bias", enable_ds=True)
    ckpt_p = tmp_path / "m.pth"
    torch.save({"model_state_dict": net.state_dict()}, ckpt_p)

    # Verify Invertd reuse parity on Case 14
    tf = eval_fair_v2.build_raw_eval_transform((0.5, 0.5, 0.5), -950.0)
    inv_op = eval_fair_v2.build_monai_invert_op(tf)
    p1_fresh, pt_fresh = eval_fair_v2.infer_monai_native_zyx(net, pairs[14][0], tf, (16, 16, 16), 0.50, "gaussian", 4, True, "cpu")
    p1_reuse, pt_reuse = eval_fair_v2.infer_monai_native_zyx(net, pairs[14][0], tf, (16, 16, 16), 0.50, "gaussian", 4, True, "cpu", inv_op=inv_op)
    assert np.array_equal(p1_fresh, p1_reuse) and np.array_equal(pt_fresh, pt_reuse)

    # Empty 0-byte CSV and header-only CSV resume guard for both MONAI & nnU-Net wrappers
    csv_zero = tmp_path / "zero.csv"
    csv_zero.write_text("", encoding="utf-8")
    csv_hdr = tmp_path / "hdr.csv"
    csv_hdr.write_text("case_id,dice\n", encoding="utf-8")

    pre = {cid: (np.full((24, 24, 24), 0.7, dtype=np.float32), np.full((24, 24, 24), 0.75, dtype=np.float32)) for cid in (14, 45)}
    df_m = eval_fair_v2.evaluate_monai_checkpoint_to_csv(
        cfg_p, ckpt_p, "rasnet_v2", 16, "conv_bias", True, [14, 45], csv_zero, precomputed_probs=pre, pairs_override=pairs
    )
    df_n = eval_fair_v2.evaluate_nnunet_checkpoint_to_csv(
        cfg_p, tmp_path, "checkpoint_best.pth", [14, 45], csv_hdr, precomputed_probs=pre, pairs_override=pairs
    )
    assert len(df_m) == 2 and len(df_n) == 2

    # Multi-arm single pass + disk prob_cache_dir reuse
    prob_cache = tmp_path / "pcache"
    specs = [
        ArmEvalSpec("A", tmp_path / "a.csv", tau=0.50, need_tta=False, postprocess_mode="top_k", keep_top_k=None),
        ArmEvalSpec("B", tmp_path / "b.csv", tau=0.50, need_tta=True, postprocess_mode="top_k", keep_top_k=None),
        ArmEvalSpec("C_top2", tmp_path / "ct2.csv", tau=0.45, need_tta=False, postprocess_mode="top_k", keep_top_k=2),
    ]
    res1 = eval_fair_v2.evaluate_monai_multi_arm_to_csvs(
        cfg_p, ckpt_p, "rasnet_v2", 16, "conv_bias", True, [14, 45], specs, pairs_override=pairs, prob_cache_dir=prob_cache
    )
    assert (prob_cache / "case_0014_1p.npy").is_file() and (prob_cache / "case_0014_tta.npy").is_file()
    assert len(res1["A"]) == 2 and len(res1["B"]) == 2 and len(res1["C_top2"]) == 2

    # Second call with nonexistent checkpoint MUST succeed from prob_cache without GPU/model load
    spec_reuse = [ArmEvalSpec("C_seeded", tmp_path / "cs.csv", tau=0.40, need_tta=False, postprocess_mode="seeded")]
    res2 = eval_fair_v2.evaluate_monai_multi_arm_to_csvs(
        cfg_p, tmp_path / "missing_should_not_load.pth", "rasnet_v2", 16, "conv_bias", True, [14, 45], spec_reuse, pairs_override=pairs, prob_cache_dir=prob_cache
    )
    assert len(res2["C_seeded"]) == 2


def test_nnunet_ensure_preprocessed_completeness_and_cache_parallel_rng(tmp_path: Path):
    # 1. nnU-Net preprocessed completeness check
    prep = tmp_path / "nnPrep"
    dname = "Dataset501_CoronarySeg"
    dset_dir = prep / dname
    cfg_dir = dset_dir / "nnUNetPlans_3d_fullres"
    cfg_dir.mkdir(parents=True)
    for fn in ("nnUNetPlans.json", "dataset_fingerprint.json", "splits_final.json", "dataset.json"):
        (dset_dir / fn).write_text("{}", encoding="utf-8")
    train_ids = list(range(1, 691))
    val_ids = list(range(691, 785))
    for cid in train_ids + val_ids:
        for ext in (".b2nd", "_seg.b2nd", ".pkl"):
            (cfg_dir / f"case_{cid:04d}{ext}").write_bytes(b"ok")

    ok, missing = check_preprocessed_complete(dname, prep, train_ids, val_ids)
    assert ok and not missing

    # Delete 1 file -> fails closed
    (cfg_dir / "case_0500_seg.b2nd").unlink()
    ok2, missing2 = check_preprocessed_complete(dname, prep, train_ids, val_ids)
    assert not ok2 and "case_0500_seg.b2nd" in missing2

    # Restore as 0-byte file -> still fails closed
    (cfg_dir / "case_0500_seg.b2nd").write_bytes(b"")
    ok3, _ = check_preprocessed_complete(dname, prep, train_ids, val_ids)
    assert not ok3

    # 2. cache_builder parallel deterministic per-case RNG parity
    dset = tmp_path / "dset_cache"
    _make_synthetic_nifti_dataset(dset, [14, 45, 47])
    splits_p = tmp_path / "splits.json"
    splits_p.write_text(json.dumps({"train": [14, 45], "val": [47], "test": [851]}), encoding="utf-8")
    val20_p = tmp_path / "val20.json"
    val20_p.write_text(json.dumps({"case_ids": [47]}), encoding="utf-8")

    cfg = yaml.safe_load((V2_DIR / "configs" / "fair_v2.yaml").read_text(encoding="utf-8"))
    cfg["study"]["dataset_root"] = str(dset)
    cfg["study"]["splits_file"] = str(splits_p)
    cfg["study"]["val20_ids_file"] = str(val20_p)
    cfg["preprocessing"]["pixdim_xyz"] = [0.5, 0.5, 0.5]
    cfg["preprocessing"]["bg_coords_max_per_case"] = 500

    v2_a = tmp_path / "v2_a"
    cfg["study"]["root_dir"] = str(v2_a)
    cfg_a = tmp_path / "cfg_a.yaml"
    cfg_a.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    build_cache(cfg_a, case_ids_override=[14, 45, 47], max_workers=2)

    v2_b = tmp_path / "v2_b"
    cfg["study"]["root_dir"] = str(v2_b)
    cfg_b = tmp_path / "cfg_b.yaml"
    cfg_b.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    build_cache(cfg_b, case_ids_override=[45], max_workers=1)
    build_cache(cfg_b, case_ids_override=[14, 45, 47], max_workers=2)

    for cid in (14, 45, 47):
        pa = case_cache_paths(v2_a / "cache_fp16", cid)
        pb = case_cache_paths(v2_b / "cache_fp16", cid)
        assert np.array_equal(np.load(pa["bg"]), np.load(pb["bg"]))
        assert np.array_equal(np.load(pa["fg"]), np.load(pb["fg"]))


def test_run_queue_preflight_dedup_and_stage3_manifest(tmp_path: Path):
    cfg = yaml.safe_load((V2_DIR / "configs" / "fair_v2.yaml").read_text(encoding="utf-8"))
    v2_root = tmp_path / "v2_q"
    cfg["study"]["root_dir"] = str(v2_root)
    (v2_root / "cache_fp16").mkdir(parents=True)

    # Fails closed when 784 cache files are absent
    with pytest.raises(RuntimeError, match="cache_fp16 preflight failed"):
        run_queue.preflight_check_cache_784(cfg)

    # Stage 2A covers both SegResNet F=32 & RASNet-v2 F=32 (16 jobs total)
    s2a = run_queue.build_stage2a_jobs(cfg)
    assert len(s2a) == 16
    assert sum(1 for j in s2a if j["lr_key"] == "segresnet_f32") == 2
    assert sum(1 for j in s2a if j["lr_key"] == "rasnet_v2_f32") == 2

    # Stage 2B deduplicates identical (F, lr, conv_bias, w_cldice=0.2) runs from Stage 2A
    locked = {
        "segresnet_f16": 3e-4,
        "segresnet_f32": 3e-4,
        "rasnet_v2_f16": 3e-4,
        "rasnet_v2_f32": 3e-4,
        "vnet": 3e-4,
        "unet3d": 1e-3,
    }
    s2b = run_queue.build_stage2b_jobs(cfg, locked)
    assert len(s2b) == 3  # 2 conv_bias cl=0.2 runs deduplicated cleanly!
    alias_log = json.loads((v2_root / "logs" / "stage2b_dedup_aliases.json").read_text(encoding="utf-8"))
    assert len(alias_log["skipped_or_aliased"]) == 2

    # Stage 3 fails if segresnet_f32 missing (no silent fallback)
    bad_lrs = {k: v for k, v in locked.items() if k != "segresnet_f32"}
    with pytest.raises(KeyError, match="segresnet_f32"):
        build_stage3_jobs(cfg, bad_lrs)

    s3 = build_stage3_jobs(cfg, locked)
    assert len(s3) == 20


def test_stages_2_4_5_6_end_to_end_synthetic_cpu(tmp_path: Path):
    val_ids = [10, 11, 12]
    test_ids = [851, 852, 853]
    train_ids = [1, 2]
    pairs_dset = _make_synthetic_nifti_dataset(tmp_path / "dset", train_ids + val_ids + test_ids)

    # Separate prospective Phase I hospital cohort (imagesTr/labelsTr layout)
    clin_dir = tmp_path / "clinical_cohort"
    (clin_dir / "imagesTr").mkdir(parents=True)
    (clin_dir / "labelsTr").mkdir(parents=True)
    aff = np.diag([0.5, 0.5, 0.5, 1.0])
    clin_ids = [101, 102]
    for cid in clin_ids:
        v = np.full((24, 24, 24), -80.0, dtype=np.float32)
        v[6:18, 6:18, 6:18] = 300.0
        l = np.zeros((24, 24, 24), dtype=np.uint8)
        l[8:16, 8:16, 8:16] = 1
        nib.save(nib.Nifti1Image(v, aff), clin_dir / "imagesTr" / f"hosp_{cid:03d}_0000.nii.gz")
        nib.save(nib.Nifti1Image(l, aff), clin_dir / "labelsTr" / f"hosp_{cid:03d}.nii.gz")

    splits_p = tmp_path / "splits.json"
    splits_p.write_text(json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}), encoding="utf-8")
    val20_p = tmp_path / "val20.json"
    val20_p.write_text(json.dumps({"case_ids": val_ids[:2]}), encoding="utf-8")

    v2_root = tmp_path / "v2_e2e"
    cfg = yaml.safe_load((V2_DIR / "configs" / "fair_v2.yaml").read_text(encoding="utf-8"))
    cfg["study"]["root_dir"] = str(v2_root)
    cfg["study"]["dataset_root"] = str(tmp_path / "dset")
    cfg["study"]["splits_file"] = str(splits_p)
    cfg["study"]["val20_ids_file"] = str(val20_p)
    cfg["training_budget"]["epoch_snapshots"] = [50, 200]
    cfg["inference_eval"]["tau_grid"] = [0.35, 0.50]
    cfg["inference_eval"]["seeded_filter_grid"] = {
        "tau_seed": [0.55],
        "min_seed_voxels": [20],
        "min_comp_vol_mm3": [10.0],
    }
    cfg_p = tmp_path / "cfg_e2e.yaml"
    cfg_p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def _prob_map(val_inside: float, val_outside: float = 0.05):
        p1 = np.full((24, 24, 24), val_outside, dtype=np.float32)
        p1[8:16, 8:16, 8:16] = val_inside
        pt = np.clip(p1 + 0.03, 0.0, 1.0)
        return p1, pt

    # Stage 2A + 2B
    s2a_probs = {}
    for j in run_queue.build_stage2a_jobs(cfg):
        score = 0.85 if (j["arch"] == "rasnet_v2" and j["lr"] == 3e-4) else 0.72
        s2a_probs[j["run_name"]] = {cid: _prob_map(score) for cid in val_ids}
    lrs_out = stage2_pilot_eval.evaluate_stage2a_pilots(
        cfg_p, case_ids_override=val_ids, precomputed_probs_by_run=s2a_probs, pairs_override=pairs_dset
    )
    assert lrs_out["locked_lrs"]["rasnet_v2_f16"] == 3e-4

    s2b_probs = {
        "pilot2b_rasnet_v2_f16_gnorig_cl020": {cid: _prob_map(0.75) for cid in val_ids},
        "pilot2b_rasnet_v2_f16_convbias_cl000": {cid: _prob_map(0.76) for cid in val_ids},
        "pilot2b_rasnet_v2_f32_gnorig_cl020": {cid: _prob_map(0.77) for cid in val_ids},
    }
    dec = stage2_pilot_eval.evaluate_stage2b_pilots(
        cfg_p, case_ids_override=val_ids, precomputed_probs_by_run=s2b_probs, pairs_override=pairs_dset
    )
    assert dec["all_gates_passed"] is True
    assert (v2_root / "results" / "pilots_val94" / "pilot_lock_decision.md").is_file()

    # Stage 4A + 4B + 4C across all 20 Stage 3 runs using synthetic probs
    s3_jobs = build_stage3_eval_manifest(cfg)
    s4a_probs = {
        (j["run_name"], ep): {cid: _prob_map(0.70 + 0.0005 * ep) for cid in val_ids[:2]}
        for j in s3_jobs for ep in (50, 200)
    }
    m4a = stage4a_val20_curves.run_stage4a_val20_curves(
        cfg_p, epochs_override=[50, 200], case_ids_override=val_ids[:2], precomputed_probs_by_key=s4a_probs, pairs_override=pairs_dset
    )
    assert m4a["n_runs"] == 20 and (v2_root / "results" / "val20_curves" / "fig_val20_convergence_curves.png").is_file()

    s4b_probs = {}
    s4c_probs = {}
    for j in s3_jobs:
        v_in = 0.85 if "rasnet_v2" in j["family"] else (0.82 if "nnunet" in j["family"] else 0.78)
        s4b_probs[j["run_name"]] = {cid: _prob_map(v_in) for cid in val_ids}
        s4c_probs[j["run_name"]] = {cid: _prob_map(v_in) for cid in test_ids}

    t4b = stage4b_val94_lock.run_stage4b_val94_lock(
        cfg_p, case_ids_override=val_ids, precomputed_probs_by_run=s4b_probs, pairs_override=pairs_dset
    )
    assert len(t4b["per_run"]) == 20 and (v2_root / "results" / "val94_final" / "thresholds_v2.json").is_file()

    m4c = stage4c_test150_eval.run_stage4c_test150(
        cfg_p, case_ids_override=test_ids, precomputed_probs_by_run=s4c_probs, pairs_override=pairs_dset
    )
    assert m4c["n_test_cases"] == 3 and (v2_root / "results" / "test_retrospective_150" / "summary_by_family_multiseed.csv").is_file()

    # Stage 5 (CPU stats + pareto + MANIFEST_V2_LOCKED.json)
    m5 = stage5_stats_and_efficiency.run_stage5_analysis(cfg_p, skip_gpu_profiler=True)
    assert (v2_root / "results" / "stats_and_efficiency" / "MANIFEST_V2_LOCKED.json").is_file()
    assert (v2_root / "results" / "stats_and_efficiency" / "family1_causal_stats.csv").is_file()
    assert (v2_root / "results" / "stats_and_efficiency" / "family2_vs_nnunet_stats.csv").is_file()
    assert m5["lead_init_filters"] in (16, 32)

    # Stage 6 Phase I single-shot evaluation + lock enforcement
    clin_probs = {
        role: {cid: _prob_map(0.84 if role == "rasnet_v2" else 0.78) for cid in clin_ids}
        for role in ("rasnet_v2", "segresnet_v2", "nnunet_v2")
    }
    p6 = stage6_phaseI_clinical.run_stage6_phaseI(
        cfg_p, cohort_dir=clin_dir, verify_hashes=False, precomputed_probs_by_role=clin_probs
    )
    assert p6["status"] == "complete_single_shot"
    assert (v2_root / "results" / "phaseI_primary_dataset" / "PHASE_I_PROSPECTIVE_REPORT.md").is_file()
    with pytest.raises(RuntimeError, match="already complete"):
        stage6_phaseI_clinical.run_stage6_phaseI(
            cfg_p, cohort_dir=clin_dir, verify_hashes=False, precomputed_probs_by_role=clin_probs
        )


def test_sw_batch_size_and_nnunet_batch_iterators(monkeypatch, tmp_path: Path):
    # Verify eval_fair_v2 does NOT throttle unet3d with init_filters=32 while throttling rasnet_v2/segresnet F=32
    observed_sw_bs = []
    orig_infer = eval_fair_v2.infer_monai_native_zyx

    def _spy_infer(*args, **kwargs):
        observed_sw_bs.append(kwargs.get("sw_batch_size"))
        return np.full((24, 24, 24), 0.7, dtype=np.float32), None

    pairs = _make_synthetic_nifti_dataset(tmp_path / "dset_sw", [14])
    cfg = yaml.safe_load((V2_DIR / "configs" / "fair_v2.yaml").read_text(encoding="utf-8"))
    cfg["study"]["root_dir"] = str(tmp_path / "v2_sw")
    cfg["study"]["dataset_root"] = str(tmp_path / "dset_sw")
    cfg_p = tmp_path / "cfg_sw.yaml"
    cfg_p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.setattr(eval_fair_v2, "infer_monai_native_zyx", _spy_infer)
    for arch, f_val in (("unet3d", 32), ("rasnet_v2", 32), ("segresnet", 32)):
        net = build_monai_model(arch, init_filters=f_val, enable_ds=False)
        cp = tmp_path / f"{arch}_{f_val}.pth"
        torch.save({"model_state_dict": net.state_dict()}, cp)
        eval_fair_v2.evaluate_monai_checkpoint_to_csv(
            cfg_p, cp, arch, f_val, "conv_bias", False, [14], tmp_path / f"out_{arch}.csv", resume=False, pairs_override=pairs
        )
    monkeypatch.setattr(eval_fair_v2, "infer_monai_native_zyx", orig_infer)
    assert observed_sw_bs == [4, 2, 2]

    # Verify iter_nnunet_probs_from_npy invokes get_data_iterator_from_raw_npy_data cleanly
    import nnunetv2.inference.export_prediction as exp_mod

    class _FakePredictor:
        plans_manager = object()
        configuration_manager = object()
        label_manager = object()
        device = torch.device("cpu")
        use_mirroring = False

        def get_data_iterator_from_raw_npy_data(self, image_or_list_of_images, segs_from_prev_stage_or_list_of_segs_from_prev_stage, properties_or_list_of_properties, truncated_ofname, num_processes=2):
            for tok, img, pr in zip(truncated_ofname, image_or_list_of_images, properties_or_list_of_properties):
                yield {"ofile": tok, "data": torch.from_numpy(img), "data_properties": pr}

        def predict_logits_from_preprocessed_data(self, data_tensor):
            # shape (2, Z, Y, X)
            z, y, x = data_tensor.shape[1:]
            logits = torch.zeros((2, z, y, x), dtype=torch.float32)
            logits[1] = 2.0 if self.use_mirroring else 1.0
            return logits

    monkeypatch.setattr(
        exp_mod,
        "convert_predicted_logits_to_segmentation_with_correct_shape",
        lambda logits, pm, cm, lm, props, return_probabilities=True, num_threads_torch=8: (
            np.zeros(logits.shape[1:], dtype=np.uint8),
            torch.softmax(logits, dim=0).numpy(),
        ),
    )
    arr = np.zeros((1, 8, 8, 8), dtype=np.float32)
    got = list(eval_fair_v2.iter_nnunet_probs_from_npy(_FakePredictor(), [(14, arr, {"spacing": (0.5, 0.5, 0.5)})], need_tta=True, num_processes=1))
    assert len(got) == 1 and got[0][0] == 14
    assert got[0][1].shape == (8, 8, 8) and got[0][2].shape == (8, 8, 8)
    assert float(got[0][2].mean()) > float(got[0][1].mean())


def test_regression_stage4b_partial_csv_precision_stage5_failclosed_stage6_manifest(tmp_path: Path):
    # 1. Stage 4B non-2-decimal precision grid + partial csv_a ahead of sweep_cache + empty Stage 2B CSV guard
    val_ids = [10, 11, 12]
    test_ids = [851, 852, 853]
    train_ids = [1, 2]
    pairs = _make_synthetic_nifti_dataset(tmp_path / "dset", train_ids + val_ids + test_ids)
    splits_p = tmp_path / "splits.json"
    splits_p.write_text(json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}), encoding="utf-8")
    val20_p = tmp_path / "val20.json"
    val20_p.write_text(json.dumps({"case_ids": val_ids[:2]}), encoding="utf-8")

    v2_root = tmp_path / "v2_reg"
    cfg = yaml.safe_load((V2_DIR / "configs" / "fair_v2.yaml").read_text(encoding="utf-8"))
    cfg["study"]["root_dir"] = str(v2_root)
    cfg["study"]["dataset_root"] = str(tmp_path / "dset")
    cfg["study"]["splits_file"] = str(splits_p)
    cfg["study"]["val20_ids_file"] = str(val20_p)
    cfg["inference_eval"]["tau_grid"] = [0.225, 0.50]
    cfg["inference_eval"]["seeded_filter_grid"] = {
        "tau_seed": [0.525],
        "min_seed_voxels": [20],
        "min_comp_vol_mm3": [12.55],
    }
    cfg_p = tmp_path / "cfg_reg.yaml"
    cfg_p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Pre-seed csv_a with case 10 & 11 (plus a stale case 999) while sweep_cache only has case 10
    val94_dir = v2_root / "results" / "val94_final"
    val94_dir.mkdir(parents=True)
    rname = "s3_rasnet_v2_f16_full_seed42"
    pd.DataFrame([
        {"case_id": 10, "arm": "A", "tau": 0.5, "tta": False, "postprocess_mode": "top_k", "keep_top_k": 0, "tau_seed": np.nan, "min_seed_voxels": 0, "min_comp_vol_mm3": 0.0, "dice": 0.9, "iou": 0.8, "precision": 0.9, "recall": 0.9, "hd95": 1.0, "asd": 0.5, "cldice": 0.9, "centerline_recall": 0.9, "status": "ok", "components_total": 1, "components_retained": 1},
        {"case_id": 11, "arm": "A", "tau": 0.5, "tta": False, "postprocess_mode": "top_k", "keep_top_k": 0, "tau_seed": np.nan, "min_seed_voxels": 0, "min_comp_vol_mm3": 0.0, "dice": 0.9, "iou": 0.8, "precision": 0.9, "recall": 0.9, "hd95": 1.0, "asd": 0.5, "cldice": 0.9, "centerline_recall": 0.9, "status": "ok", "components_total": 1, "components_retained": 1},
        {"case_id": 999, "arm": "A", "tau": 0.5, "tta": False, "postprocess_mode": "top_k", "keep_top_k": 0, "tau_seed": np.nan, "min_seed_voxels": 0, "min_comp_vol_mm3": 0.0, "dice": 0.1, "iou": 0.1, "precision": 0.1, "recall": 0.1, "hd95": 9.0, "asd": 5.0, "cldice": 0.1, "centerline_recall": 0.1, "status": "ok", "components_total": 1, "components_retained": 1},
    ]).to_csv(val94_dir / f"per_case_{rname}_arm_A.csv", index=False)
    (val94_dir / f"_sweep_cache_{rname}.json").write_text(
        json.dumps({"10": {"top2": {"0.5000": 0.9}, "seeded": {}}}),  # intentionally missing 0.2250 & seeded!
        encoding="utf-8",
    )

    p1 = np.full((24, 24, 24), 0.05, dtype=np.float32)
    p1[8:16, 8:16, 8:16] = 0.85
    pre_s4b = {rname: {cid: (p1, None) for cid in val_ids}}
    t4b = stage4b_val94_lock.run_stage4b_val94_lock(
        cfg_p, run_names_filter=[rname], case_ids_override=val_ids, precomputed_probs_by_run=pre_s4b, pairs_override=pairs
    )
    assert rname in t4b["per_run"]
    assert t4b["per_run"][rname]["top2"]["tau"] in (0.225, 0.5)
    df_a_after = pd.read_csv(val94_dir / f"per_case_{rname}_arm_A.csv")
    assert sorted(df_a_after["case_id"].tolist()) == val_ids  # stale 999 filtered out cleanly!

    # 2. Stage 2B empty/header-only CSV guard + Stage 3 pilot_lock inheritance
    pilot_dir = v2_root / "results" / "pilots_val94"
    pilot_dir.mkdir(parents=True)
    # Create valid Stage 2A summary/ref CSVs and an empty 0-byte Stage 2B CSV that must NOT crash
    s2a_probs = {
        j["run_name"]: {cid: (p1, None) for cid in val_ids}
        for j in run_queue.build_stage2a_jobs(cfg)
    }
    stage2_pilot_eval.evaluate_stage2a_pilots(
        cfg_p, case_ids_override=val_ids, precomputed_probs_by_run=s2a_probs, pairs_override=pairs
    )
    (pilot_dir / "per_case_pilot2b_rasnet_v2_f16_gnorig_cl020.csv").write_text("", encoding="utf-8")
    (pilot_dir / "per_case_pilot2b_rasnet_v2_f16_convbias_cl000.csv").write_text("case_id,dice\n", encoding="utf-8")
    s2b_probs = {
        "pilot2b_rasnet_v2_f16_gnorig_cl020": {cid: (p1, None) for cid in val_ids},
        "pilot2b_rasnet_v2_f16_convbias_cl000": {cid: (p1, None) for cid in val_ids},
        "pilot2b_rasnet_v2_f32_gnorig_cl020": {cid: (p1, None) for cid in val_ids},
    }
    dec = stage2_pilot_eval.evaluate_stage2b_pilots(
        cfg_p, case_ids_override=val_ids, precomputed_probs_by_run=s2b_probs, pairs_override=pairs
    )
    assert dec["all_gates_passed"] is True

    lrs = json.loads((pilot_dir / "locked_lrs.json").read_text(encoding="utf-8"))["locked_lrs"]
    custom_lock = {"locked_design": {"gate_mode": "groupnorm_orig", "w_cldice": 0.0}}
    jobs_custom = build_stage3_jobs(cfg, lrs, pilot_lock=custom_lock)
    f16_lead = [j for j in jobs_custom if j["family"] == "rasnet_v2_f16"][0]
    f16_abl = [j for j in jobs_custom if j["family"] == "rasnet_fixed_ag_nods_f16"][0]
    assert f16_lead["gate_mode"] == "groupnorm_orig" and f16_lead["w_cldice"] == 0.0
    assert f16_abl["gate_mode"] == "conv_bias" and f16_abl["enable_ds"] is False

    # 3. Stage 5 fail-closed case/seed mismatch check
    test_dir = v2_root / "results" / "test_retrospective_150"
    test_dir.mkdir(parents=True)
    for rname_a, rname_b in (
        ("s3_rasnet_v2_f16_full_seed42", "s3_segresnet_f16_v2loss_seed42"),
    ):
        pd.DataFrame([{"case_id": 851, "dice": 0.8, "cldice": 0.8, "precision": 0.8, "hd95": 2.0}]).to_csv(
            test_dir / f"per_case_{rname_a}_arm_A.csv", index=False
        )
        pd.DataFrame([
            {"case_id": 851, "dice": 0.7, "cldice": 0.7, "precision": 0.7, "hd95": 3.0},
            {"case_id": 852, "dice": 0.7, "cldice": 0.7, "precision": 0.7, "hd95": 3.0},
        ]).to_csv(test_dir / f"per_case_{rname_b}_arm_A.csv", index=False)
    fam_tables = stage5_stats_and_efficiency._load_family_tables(test_dir, build_stage3_eval_manifest(cfg), arms=("A",))
    with pytest.raises(RuntimeError, match="case_id/seed mismatch"):
        stage5_stats_and_efficiency._run_family1(fam_tables, ("A",))

    # 4. Stage 6 manifest_csv collision prevention ("P001" vs "1") and duplicate rejection
    clin_m = tmp_path / "clin_manifest"
    _make_synthetic_nifti_dataset(clin_m, [1, 2])
    m_csv = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"case_id": "P001", "image_path": str(clin_m / "1.img.nii.gz"), "label_path": str(clin_m / "1.label.nii.gz")},
        {"case_id": "1", "image_path": str(clin_m / "2.img.nii.gz"), "label_path": str(clin_m / "2.label.nii.gz")},
    ]).to_csv(m_csv, index=False)
    c_pairs, c_labels = stage6_phaseI_clinical.discover_clinical_pairs(manifest_csv=m_csv, check_headers=True)
    assert len(c_pairs) == 2 and set(c_labels.values()) == {"P001", "1"}

    pd.DataFrame([
        {"case_id": "P001", "image_path": str(clin_m / "1.img.nii.gz"), "label_path": str(clin_m / "1.label.nii.gz")},
        {"case_id": "P001", "image_path": str(clin_m / "2.img.nii.gz"), "label_path": str(clin_m / "2.label.nii.gz")},
    ]).to_csv(m_csv, index=False)
    with pytest.raises(ValueError, match="Duplicate case_id"):
        stage6_phaseI_clinical.discover_clinical_pairs(manifest_csv=m_csv, check_headers=True)


def test_disk_inventory_and_safe_cache_lifecycle_cleanup(tmp_path: Path):
    cfg = yaml.safe_load((V2_DIR / "configs" / "fair_v2.yaml").read_text(encoding="utf-8"))
    v2_root = tmp_path / "v2_clean"
    cfg["study"]["root_dir"] = str(v2_root)

    # Create mock cache_fp16, nnUNet_preprocessed (.b2nd + JSON), nnUNet_raw, and completed/incomplete checkpoints
    cache_dir = v2_root / "cache_fp16"
    cache_dir.mkdir(parents=True)
    (cache_dir / "dummy.npy").write_bytes(b"0" * 1024)

    nn_prep = v2_root / "nnUNet_work" / "nnUNet_preprocessed" / "Dataset501_CoronarySeg"
    (nn_prep / "nnUNetPlans_3d_fullres").mkdir(parents=True)
    (nn_prep / "nnUNetPlans.json").write_text("{}", encoding="utf-8")
    (nn_prep / "nnUNetPlans_3d_fullres" / "case.b2nd").write_bytes(b"1" * 2048)
    nn_raw = v2_root / "nnUNet_work" / "nnUNet_raw" / "Dataset501_CoronarySeg"
    nn_raw.mkdir(parents=True)
    (nn_raw / "raw.nii.gz").write_bytes(b"2" * 512)

    s2a_jobs = run_queue.build_stage2a_jobs(cfg)
    s3_jobs = build_stage3_eval_manifest(cfg)

    # Mark all Stage 2A complete WITH stale mid_epoch_state.pt; leave 1 Stage 3 MONAI job missing DONE.flag
    for j in s2a_jobs:
        d = v2_root / "checkpoints" / j["run_name"]
        d.mkdir(parents=True)
        (d / "DONE.flag").write_text('{"status":"complete"}', encoding="utf-8")
        (d / "mid_epoch_state.pt").write_bytes(b"m" * 256)
        (d / "latest_state.pt").write_bytes(b"l" * 256)

    pilot_dir = v2_root / "results" / "pilots_val94"
    pilot_dir.mkdir(parents=True)
    (pilot_dir / "pilot_lock_decision.json").write_text('{"all_gates_passed": true}', encoding="utf-8")

    s3_monai = [j for j in s3_jobs if j["kind"] == "monai"]
    s3_nn = [j for j in s3_jobs if j["kind"] == "nnunet"]
    for j in s3_monai[:-1]:
        d = v2_root / "checkpoints" / j["run_name"]
        d.mkdir(parents=True)
        (d / "DONE.flag").write_text('{"status":"complete"}', encoding="utf-8")

    # Pass 1: stale mid_epoch_state.pt cleaned, BUT cache_fp16 & nnUNet_preprocessed MUST remain!
    res1 = run_queue.cleanup_unused_caches(cfg, aggressive_low_disk_gb=0.0)
    assert res1["removed_count"] == len(s2a_jobs)
    assert (cache_dir / "dummy.npy").is_file()
    assert (nn_prep / "nnUNetPlans_3d_fullres" / "case.b2nd").is_file()

    # Complete last Stage 3 MONAI job -> cache_fp16 MUST now be cleaned, nnUNet_preprocessed MUST remain
    d_last = v2_root / "checkpoints" / s3_monai[-1]["run_name"]
    d_last.mkdir(parents=True)
    (d_last / "DONE.flag").write_text('{"status":"complete"}', encoding="utf-8")
    res2 = run_queue.cleanup_unused_caches(cfg, aggressive_low_disk_gb=0.0)
    assert not cache_dir.exists()
    assert (nn_prep / "nnUNetPlans_3d_fullres" / "case.b2nd").is_file()
    assert res2["inventory"]["cache_fp16_gb"] == 0.0

    # Complete all 3 nnU-Net seeds -> bulk blobs + nnUNet_raw cleaned while preserving nnUNetPlans.json
    for j in s3_nn:
        d = v2_root / "checkpoints" / j["run_name"]
        d.mkdir(parents=True)
        (d / "DONE.flag").write_text('{"status":"complete"}', encoding="utf-8")
    run_queue.cleanup_unused_caches(cfg, aggressive_low_disk_gb=0.0)
    assert not nn_raw.exists()
    assert not (nn_prep / "nnUNetPlans_3d_fullres").exists()
    assert (nn_prep / "nnUNetPlans.json").is_file()
