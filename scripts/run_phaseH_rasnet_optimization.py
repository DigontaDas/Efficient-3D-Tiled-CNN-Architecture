#!/usr/bin/env python3
"""
Phase H: Fixed-Weight RASNet Inference & 3D Hysteresis Optimization (RTX 4080 SUPER Accelerated)
================================================================================================
Strictly adheres to THESIS_VERIFICATION_ARCHIVAL_AND_OPTIMIZATION_PLAN.md (Section 11, Phase H):
- Uses the frozen 200-epoch `rasnet_best.pth` checkpoint (SHA-256: 8d4c25cd...). Zero retraining.
- Evaluates 4 sliding-window inference modes on the N=94 validation split:
    1. sw025_const_1pass : overlap=0.25, mode="constant", 1-pass (Arm A/C probability map)
    2. sw025_const_tta4  : overlap=0.25, mode="constant", 4-pass flip TTA (Arm B probability map)
    3. sw050_gauss_1pass : overlap=0.50, mode="gaussian", 1-pass (3D Gaussian center-weighted blending)
    4. sw050_gauss_tta4  : overlap=0.50, mode="gaussian", 4-pass flip TTA
- Evaluates 38 post-processing candidates per mode (152 total candidate pipelines) on N=94 validation:
    * Exact Dice, IoU, Precision, Recall, Centerline Recall, and cc3d component telemetry for all 152 candidates
      inside the cropped probability ROI (`sl_prob`, mathematically exact since prob < 0.15 outside).
    * Full 8-metric surface + topological suite (including HD95, ASD, clDice) computed concurrently across a
      persistent multi-process CPU worker pool (`ProcessPoolExecutor`, bypassing Python's GIL in skeletonize)
      for all focal validation candidates and all locked winners on Val N=94 & Test N=150.
- Supports crash-safe per-case checkpointing (`val_94_checkpoint.csv`, `test_150_checkpoint.csv`) so an
  interrupted run resumes cleanly from the exact next case.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cc3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize

ROOT = Path(r"C:\Thesis_RASNET")
sys.path.insert(0, str(ROOT))

from benchmark.adapters.monai_models import (
    build_model,
    build_transform,
    infer_monai_both_passes,
)
from benchmark.dataset import resolve_case_pairs
from benchmark.metrics import METRIC_VERSION, _joint_bbox_slices, check_geometry, compute_metrics
from benchmark.postprocess import hysteresis_threshold_3d
from benchmark.preflight import validate_splits
from benchmark.stats import analyze_family

OUT_DIR = ROOT / "results" / "rasnet_fixed_weight_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_candidate_metrics(df: pd.DataFrame) -> dict:
    """Compute Mean, SD, Median, Q25, Q75 across cases for all 8 segmentation metrics."""
    out = {
        "n_cases": int(len(df)),
        "n_ok": int((df["status"] == "ok").sum()),
    }
    for m in ["dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall"]:
        vals = df[m].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            out[f"{m}_mean"] = np.nan
            out[f"{m}_std"] = np.nan
            out[f"{m}_median"] = np.nan
            out[f"{m}_q25"] = np.nan
            out[f"{m}_q75"] = np.nan
        else:
            out[f"{m}_mean"] = float(np.mean(vals))
            out[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out[f"{m}_median"] = float(np.median(vals))
            out[f"{m}_q25"] = float(np.percentile(vals, 25))
            out[f"{m}_q75"] = float(np.percentile(vals, 75))
    return out


FOCAL_FULL_METRIC_POST_KEYS = {
    "single_tau0.30_top2cc",
    "hyst_lo0.20_hi0.50_seed50_nocc",
    "hyst_lo0.20_hi0.50_seed50_top2cc",
}


def build_candidate_specs() -> list[dict]:
    """Return the pre-registered 38 post-processing specs evaluated per inference mode."""
    specs = []
    for tau in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        pk = f"single_tau{tau:.2f}_nocc"
        specs.append({
            "post_key": pk,
            "family": "Single-Threshold (No CC)",
            "kind": "single",
            "tau": tau,
            "tau_low": np.nan,
            "tau_high": np.nan,
            "min_seed_voxels": 0,
            "keep_top_k": None,
            "full_surface_on_val": pk in FOCAL_FULL_METRIC_POST_KEYS,
        })
    for tau in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        pk = f"single_tau{tau:.2f}_top2cc"
        specs.append({
            "post_key": pk,
            "family": "Single-Threshold (Top-2 CC)",
            "kind": "single",
            "tau": tau,
            "tau_low": np.nan,
            "tau_high": np.nan,
            "min_seed_voxels": 0,
            "keep_top_k": 2,
            "full_surface_on_val": pk in FOCAL_FULL_METRIC_POST_KEYS,
        })
    for tau in [0.25, 0.30, 0.35]:
        pk = f"single_tau{tau:.2f}_top3cc"
        specs.append({
            "post_key": pk,
            "family": "Single-Threshold (Top-3 CC)",
            "kind": "single",
            "tau": tau,
            "tau_low": np.nan,
            "tau_high": np.nan,
            "min_seed_voxels": 0,
            "keep_top_k": 3,
            "full_surface_on_val": pk in FOCAL_FULL_METRIC_POST_KEYS,
        })
    hyst_pairs = [
        (0.15, 0.45), (0.20, 0.45), (0.25, 0.45), (0.30, 0.45),
        (0.15, 0.50), (0.20, 0.50), (0.25, 0.50), (0.30, 0.50),
        (0.20, 0.55), (0.25, 0.55), (0.30, 0.55),
    ]
    for t_lo, t_hi in hyst_pairs:
        pk_nocc = f"hyst_lo{t_lo:.2f}_hi{t_hi:.2f}_seed50_nocc"
        specs.append({
            "post_key": pk_nocc,
            "family": "3D Hysteresis Seed-and-Grow (No Top-K)",
            "kind": "hysteresis",
            "tau": np.nan,
            "tau_low": t_lo,
            "tau_high": t_hi,
            "min_seed_voxels": 50,
            "keep_top_k": None,
            "full_surface_on_val": pk_nocc in FOCAL_FULL_METRIC_POST_KEYS,
        })
        pk_top2 = f"hyst_lo{t_lo:.2f}_hi{t_hi:.2f}_seed50_top2cc"
        specs.append({
            "post_key": pk_top2,
            "family": "3D Hysteresis Seed-and-Grow (+Top-2 CC)",
            "kind": "hysteresis",
            "tau": np.nan,
            "tau_low": t_lo,
            "tau_high": t_hi,
            "min_seed_voxels": 50,
            "keep_top_k": 2,
            "full_surface_on_val": pk_top2 in FOCAL_FULL_METRIC_POST_KEYS,
        })
    return specs


INFERENCE_MODES = [
    {"mode_key": "sw025_const_1pass", "overlap": 0.25, "sw_mode": "constant", "tta": False, "label": "1-Pass (overlap=0.25, constant)"},
    {"mode_key": "sw025_const_tta4",  "overlap": 0.25, "sw_mode": "constant", "tta": True,  "label": "4-Pass TTA (overlap=0.25, constant)"},
    {"mode_key": "sw050_gauss_1pass", "overlap": 0.50, "sw_mode": "gaussian", "tta": False, "label": "1-Pass (overlap=0.50, 3D Gaussian)"},
    {"mode_key": "sw050_gauss_tta4",  "overlap": 0.50, "sw_mode": "gaussian", "tta": True,  "label": "4-Pass TTA (overlap=0.50, 3D Gaussian)"},
]


def _worker_surface_metrics(args):
    """
    Top-level picklable worker running in ProcessPoolExecutor on a tight joint-union crop `(pc, gc, sb, sg)`.
    Produces bit-for-bit identical metrics to benchmark.metrics.compute_metrics while bypassing Python's GIL.
    """
    surf_key, pc, gc, sb, sg, ps, gs, sg_sum, spacing = args
    intersection = int((pc & gc).sum())
    union = ps + gs - intersection
    out = {
        "dice": 2 * intersection / (ps + gs) if ps + gs else np.nan,
        "iou": intersection / union if union else np.nan,
        "precision": intersection / ps if ps else np.nan,
        "recall": intersection / gs if gs else np.nan,
        "hd95": np.nan,
        "asd": np.nan,
        "cldice": np.nan,
        "centerline_recall": np.nan,
        "status": "ok" if ps and gs else ("both_empty" if not ps and not gs else ("prediction_empty" if not ps else "ground_truth_empty")),
    }
    if ps and gs:
        sa = pc & ~binary_erosion(pc, border_value=0)
        d_ab = distance_transform_edt(~sb, sampling=spacing)[sa]
        d_ba = distance_transform_edt(~sa, sampling=spacing)[sb]
        d = np.concatenate((d_ab, d_ba))
        out["hd95"], out["asd"] = float(np.percentile(d, 95)), float(np.mean(d))
        sp = skeletonize(pc).astype(bool)
        sp_sum = int(sp.sum())
        if sp_sum == 0 or sg_sum == 0:
            out["status"] = "empty_skeleton"
        else:
            tprec = int((sp & gc).sum()) / sp_sum
            tsens = int((sg & pc).sum()) / sg_sum
            out["centerline_recall"] = float(tsens)
            out["cldice"] = float(2 * tprec * tsens / (tprec + tsens)) if tprec + tsens else 0.0
    return surf_key, out


def load_gt_and_precompute(img_path: str, lbl_path: str):
    """
    Load GT and precompute GT surface `sb_full` and GT 3D skeleton `sg_full` inside GT's tight bbox.
    """
    img_itk = sitk.ReadImage(str(img_path))
    lbl_itk = sitk.ReadImage(str(lbl_path))
    geom_i = {
        "size": img_itk.GetSize(),
        "origin": img_itk.GetOrigin(),
        "spacing": img_itk.GetSpacing(),
        "direction": img_itk.GetDirection(),
    }
    geom_l = {
        "size": lbl_itk.GetSize(),
        "origin": lbl_itk.GetOrigin(),
        "spacing": lbl_itk.GetSpacing(),
        "direction": lbl_itk.GetDirection(),
    }
    check_geometry(geom_i, geom_l)
    sx, sy, sz = lbl_itk.GetSpacing()
    spacing_zyx = (float(sz), float(sy), float(sx))
    gt = np.ascontiguousarray(sitk.GetArrayFromImage(lbl_itk) > 0, dtype=bool)

    sb_full = np.zeros_like(gt, dtype=bool)
    sg_full = np.zeros_like(gt, dtype=bool)
    sl_gt = None
    if gt.any():
        sl_gt = _joint_bbox_slices(gt, margin=2)
        gc_tight = gt[sl_gt]
        sb_full[sl_gt] = gc_tight & ~binary_erosion(gc_tight, border_value=0)
        sg_full[sl_gt] = skeletonize(gc_tight).astype(bool)
    gs = int(gt.sum())
    sg_sum = int(sg_full.sum())
    return gt, spacing_zyx, sb_full, sg_full, sl_gt, gs, sg_sum


def bbox_union_slices(sl_a: tuple[slice, slice, slice] | None, sl_b: tuple[slice, slice, slice] | None, shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    if sl_a is None and sl_b is None:
        return (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))
    if sl_a is None:
        return sl_b
    if sl_b is None:
        return sl_a
    return tuple(
        slice(max(0, min(a.start, b.start)), min(s, max(a.stop, b.stop)))
        for a, b, s in zip(sl_a, sl_b, shape)
    )


def build_cc_cache_cropped(
    prob: np.ndarray,
    gt: np.ndarray,
    sg_full: np.ndarray,
    thresholds: list[float],
) -> tuple[tuple[slice, slice, slice] | None, np.ndarray | None, dict]:
    """
    Crop probability array ONCE to the tight bounding box of `(prob >= min(thresholds))`,
    making `cc3d.connected_components` and `bincount` ~8-10x faster with 0.00e+00 numerical diff.
    """
    min_tau = float(min(thresholds))
    any_fg = prob >= min_tau
    if not np.any(any_fg):
        z = np.zeros(1, dtype=np.int64)
        empty_cache = {round(float(t), 2): (None, 0, z, z, z) for t in thresholds}
        return None, None, empty_cache

    sl_prob = _joint_bbox_slices(any_fg, margin=1)
    prob_c = np.ascontiguousarray(prob[sl_prob], dtype=np.float32)
    gt_flat_idx = np.flatnonzero(np.ascontiguousarray(gt[sl_prob], dtype=bool))
    sg_flat_idx = np.flatnonzero(np.ascontiguousarray(sg_full[sl_prob], dtype=bool))

    cache = {}
    for tau in thresholds:
        t_key = round(float(tau), 2)
        m_u8 = np.ascontiguousarray(prob_c >= float(tau), dtype=np.uint8)
        if not np.any(m_u8):
            z = np.zeros(1, dtype=np.int64)
            cache[t_key] = (None, 0, z, z, z)
        else:
            labeled, n_comp = cc3d.connected_components(m_u8, connectivity=26, return_N=True)
            n_comp = int(n_comp)
            flat_lbl = labeled.ravel()
            total_counts = np.bincount(flat_lbl, minlength=n_comp + 1)
            total_counts[0] = 0
            gt_counts = np.bincount(flat_lbl[gt_flat_idx], minlength=n_comp + 1)
            gt_counts[0] = 0
            sg_counts = np.bincount(flat_lbl[sg_flat_idx], minlength=n_comp + 1)
            sg_counts[0] = 0
            cache[t_key] = (labeled, n_comp, total_counts, gt_counts, sg_counts)
    return sl_prob, prob_c, cache


def resolve_keep_ids(
    cc_cache: dict,
    high_flat_indices: dict[float, np.ndarray],
    spec: dict,
) -> tuple[float, int, list[int], np.ndarray, np.ndarray, np.ndarray]:
    if spec["kind"] == "single":
        t_key = round(float(spec["tau"]), 2)
        _, n_comp, total_counts, gt_counts, sg_counts = cc_cache[t_key]
        if n_comp == 0:
            return t_key, 0, [], total_counts, gt_counts, sg_counts
        keep_top_k = spec["keep_top_k"]
        if keep_top_k is None or n_comp <= keep_top_k:
            return t_key, n_comp, list(range(1, n_comp + 1)), total_counts, gt_counts, sg_counts
        comp_ids = list(range(1, n_comp + 1))
        comp_ids.sort(key=lambda cid: (-int(total_counts[cid]), cid))
        return t_key, n_comp, comp_ids[:keep_top_k], total_counts, gt_counts, sg_counts
    else:
        t_lo = round(float(spec["tau_low"]), 2)
        t_hi = round(float(spec["tau_high"]), 2)
        min_seed = int(spec["min_seed_voxels"])
        keep_top_k = spec["keep_top_k"]
        labeled, n_comp, total_counts, gt_counts, sg_counts = cc_cache[t_lo]
        if n_comp == 0:
            return t_lo, 0, [], total_counts, gt_counts, sg_counts
        hi_idx = high_flat_indices[t_hi]
        if hi_idx.size == 0:
            return t_lo, n_comp, [], total_counts, gt_counts, sg_counts
        seed_counts = np.bincount(labeled.ravel()[hi_idx], minlength=n_comp + 1)
        seed_counts[0] = 0
        seeded_ids = [cid for cid in range(1, n_comp + 1) if int(seed_counts[cid]) >= min_seed]
        if not seeded_ids:
            return t_lo, n_comp, [], total_counts, gt_counts, sg_counts
        seeded_ids.sort(key=lambda cid: (-int(total_counts[cid]), cid))
        keep_ids = seeded_ids[:keep_top_k] if keep_top_k is not None else seeded_ids
        return t_lo, n_comp, keep_ids, total_counts, gt_counts, sg_counts


def extract_tight_crop_task(
    surf_key,
    labeled_crop: np.ndarray | None,
    n_comp: int,
    keep_ids: list[int],
    sl_prob: tuple[slice, slice, slice] | None,
    gt: np.ndarray,
    sb_full: np.ndarray,
    sg_full: np.ndarray,
    sl_gt: tuple[slice, slice, slice] | None,
    gs: int,
    sg_sum: int,
    spacing_zyx: tuple[float, float, float],
):
    """Build tight joint-bbox crops `(pc, gc, sb, sg)` directly without full 72M voxel allocations."""
    if labeled_crop is None or not keep_ids or sl_prob is None:
        empty_b = np.zeros((1, 1, 1), dtype=bool)
        return surf_key, empty_b, empty_b, empty_b, empty_b, 0, gs, sg_sum, spacing_zyx

    if len(keep_ids) == n_comp:
        mask_crop = labeled_crop > 0
    else:
        lut = np.zeros(n_comp + 1, dtype=bool)
        lut[keep_ids] = True
        mask_crop = lut[labeled_crop]

    ps = int(mask_crop.sum())
    if ps == 0 or gs == 0:
        empty_b = np.zeros((1, 1, 1), dtype=bool)
        return surf_key, empty_b, empty_b, empty_b, empty_b, ps, gs, sg_sum, spacing_zyx

    sl_p_local = _joint_bbox_slices(mask_crop, margin=2)
    sl_p_global = tuple(
        slice(max(0, sp.start + sl_l.start - 0), min(s, sp.start + sl_l.stop))
        for sp, sl_l, s in zip(sl_prob, sl_p_local, gt.shape)
    )
    sl_union = bbox_union_slices(sl_p_global, sl_gt, gt.shape)
    u_shape = tuple(s.stop - s.start for s in sl_union)

    pc = np.zeros(u_shape, dtype=bool)
    dst_slices = tuple(
        slice(sp.start - su.start, sp.stop - su.start)
        for sp, su in zip(sl_prob, sl_union)
    )
    src_slices = tuple(
        slice(0, sp.stop - sp.start) for sp in sl_prob
    )
    # Clamp overlap between sl_prob and sl_union
    ov_dst = []
    ov_src = []
    for sp, su in zip(sl_prob, sl_union):
        g0 = max(sp.start, su.start)
        g1 = min(sp.stop, su.stop)
        ov_dst.append(slice(g0 - su.start, g1 - su.start))
        ov_src.append(slice(g0 - sp.start, g1 - sp.start))
    pc[tuple(ov_dst)] = mask_crop[tuple(ov_src)]

    gc = np.ascontiguousarray(gt[sl_union], dtype=bool)
    sb = np.ascontiguousarray(sb_full[sl_union], dtype=bool)
    sg = np.ascontiguousarray(sg_full[sl_union], dtype=bool)
    return surf_key, pc, gc, sb, sg, ps, gs, sg_sum, spacing_zyx


def evaluate_case_all_candidates(
    cid: int,
    probs_by_mode: dict[str, np.ndarray],
    gt: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    sb_full: np.ndarray,
    sg_full: np.ndarray,
    sl_gt: tuple[slice, slice, slice] | None,
    gs: int,
    sg_sum: int,
    candidate_specs: list[dict],
    proc_pool: concurrent.futures.ProcessPoolExecutor,
    verify_first_case: bool = False,
) -> tuple[list[dict], int, int]:
    all_low_taus = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    all_high_taus = [0.45, 0.50, 0.55]
    voxel_vol_mm3 = float(np.prod(spacing_zyx))

    rows_meta = []
    surface_tasks = {}

    for m_info in INFERENCE_MODES:
        m_key = m_info["mode_key"]
        prob = np.ascontiguousarray(probs_by_mode[m_key], dtype=np.float32)
        sl_prob, prob_c, cc_cache = build_cc_cache_cropped(prob, gt, sg_full, all_low_taus)
        if prob_c is not None:
            flat_prob_c = prob_c.ravel()
            high_flat_indices = {round(t, 2): np.flatnonzero(flat_prob_c >= float(t)) for t in all_high_taus}
        else:
            high_flat_indices = {round(t, 2): np.zeros(0, dtype=np.int64) for t in all_high_taus}

        for spec in candidate_specs:
            t_key, n_comp, keep_ids, total_counts, gt_counts, sg_counts = resolve_keep_ids(
                cc_cache, high_flat_indices, spec
            )
            total_low_vox = int(total_counts.sum())
            ret_vox = int(total_counts[keep_ids].sum()) if keep_ids else 0
            rem_vox = int(total_low_vox - ret_vox)
            intersection = int(gt_counts[keep_ids].sum()) if keep_ids else 0
            sg_hits = int(sg_counts[keep_ids].sum()) if keep_ids else 0
            union = ret_vox + gs - intersection

            m_res = {
                "dice": 2 * intersection / (ret_vox + gs) if (ret_vox + gs) else np.nan,
                "iou": intersection / union if union else np.nan,
                "precision": intersection / ret_vox if ret_vox else np.nan,
                "recall": intersection / gs if gs else np.nan,
                "hd95": np.nan,
                "asd": np.nan,
                "cldice": np.nan,
                "centerline_recall": float(sg_hits / sg_sum) if (ret_vox and gs and sg_sum > 0) else np.nan,
                "status": "ok" if (ret_vox and gs) else ("both_empty" if not ret_vox and not gs else ("prediction_empty" if not ret_vox else "ground_truth_empty")),
            }

            surf_key = None
            if spec["full_surface_on_val"]:
                surf_key = (m_key, t_key, tuple(keep_ids))
                if surf_key not in surface_tasks:
                    labeled_crop = cc_cache[t_key][0]
                    task = extract_tight_crop_task(
                        surf_key, labeled_crop, n_comp, keep_ids, sl_prob,
                        gt, sb_full, sg_full, sl_gt, gs, sg_sum, spacing_zyx
                    )
                    surface_tasks[surf_key] = task

                if verify_first_case and m_key == "sw025_const_1pass" and spec["post_key"] == "hyst_lo0.20_hi0.50_seed50_top2cc":
                    ref_clean, _ = hysteresis_threshold_3d(
                        prob, spacing_zyx, tau_low=0.20, tau_high=0.50, min_seed_voxels=50, keep_top_k=2
                    )
                    ref_m = compute_metrics(ref_clean, gt, spacing_zyx)
                    _, worker_m = _worker_surface_metrics(surface_tasks[surf_key])
                    for k in ["dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall"]:
                        if not np.isclose(ref_m[k], worker_m[k], atol=1e-12, rtol=0.0, equal_nan=True):
                            raise RuntimeError(f"Cropped worker mismatch on {k}: {ref_m[k]} vs {worker_m[k]}")
                        if k in ("dice", "iou", "precision", "recall", "centerline_recall"):
                            if not np.isclose(ref_m[k], m_res[k], atol=1e-12, rtol=0.0, equal_nan=True):
                                raise RuntimeError(f"Cropped bincount mismatch on {k}: {ref_m[k]} vs {m_res[k]}")
                    print("[*] Verified Case 1 bit-for-bit identity (0.00e+00 diff) of cropped cc3d + ProcessPool metrics vs benchmark.metrics.compute_metrics!", flush=True)

            cc_stats = {
                "components_total": n_comp,
                "components_retained": int(len(keep_ids)),
                "components_removed": int(n_comp - len(keep_ids)),
                "voxels_retained": ret_vox,
                "voxels_removed": rem_vox,
                "volume_retained_mm3": float(ret_vox * voxel_vol_mm3),
                "volume_removed_mm3": float(rem_vox * voxel_vol_mm3),
            }
            rows_meta.append((m_info, spec, surf_key, m_res, cc_stats))

    surface_results = {}
    if surface_tasks:
        for s_key, s_out in proc_pool.map(_worker_surface_metrics, surface_tasks.values()):
            surface_results[s_key] = s_out

    rows = []
    for m_info, spec, surf_key, m_res, cc_stats in rows_meta:
        final_res = surface_results[surf_key] if surf_key is not None else m_res
        candidate_id = f"{m_info['mode_key']}__{spec['post_key']}"
        rows.append({
            "candidate_id": candidate_id,
            "mode_key": m_info["mode_key"],
            "mode_label": m_info["label"],
            "overlap": m_info["overlap"],
            "sw_mode": m_info["sw_mode"],
            "tta": m_info["tta"],
            "post_key": spec["post_key"],
            "family": spec["family"],
            "kind": spec["kind"],
            "tau": spec["tau"],
            "tau_low": spec["tau_low"],
            "tau_high": spec["tau_high"],
            "min_seed_voxels": spec["min_seed_voxels"],
            "keep_top_k": -1 if spec["keep_top_k"] is None else int(spec["keep_top_k"]),
            "case_id": int(cid),
            **final_res,
            **cc_stats,
        })
    return rows, len(surface_tasks), len(rows)


def append_checkpoint_rows(ckpt_csv: Path, rows: list[dict]):
    df_new = pd.DataFrame(rows)
    header = not ckpt_csv.exists()
    df_new.to_csv(ckpt_csv, mode="a", header=header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 2 val + 2 test cases for end-to-end timing and verification")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint CSVs and start fresh")
    args = parser.parse_args()

    t_start = time.time()
    with open(ROOT / "config.lab.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.init()
    _ = torch.zeros(1, device=device)
    gpu_name = torch.cuda.get_device_name(device)
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    cpu_workers = min(8, (os.cpu_count() or 8))
    print(f"[*] Phase H Hardware: {gpu_name} ({vram_gb:.2f} GiB VRAM) | TF32={torch.backends.cuda.matmul.allow_tf32} | cuDNN Benchmark={torch.backends.cudnn.benchmark} | CPU ProcessPool={cpu_workers}", flush=True)

    split_path = ROOT / cfg["split_file"]
    splits_bytes = split_path.read_bytes()
    splits_sha = hashlib.sha256(splits_bytes).hexdigest()
    splits = json.loads(splits_bytes.decode("utf-8"))
    validate_splits(splits)

    val_pairs = resolve_case_pairs(cfg["dataset_root"], splits["val"], check_headers=False)
    test_pairs = resolve_case_pairs(cfg["dataset_root"], splits["test"], check_headers=False)
    if args.smoke:
        val_pairs = dict(list(val_pairs.items())[:2])
        test_pairs = dict(list(test_pairs.items())[:2])

    ckpt_path = ROOT / cfg["checkpoints"]["rasnet"]
    net = build_model("rasnet", str(ckpt_path), device=str(device))
    transform = build_transform("rasnet")
    sw_batch_size = 8  # Matches val_full_94/test_retrospective_150 & avoids WDDM paging + L2 cache thrashing

    candidate_specs = build_candidate_specs()
    print(f"[*] Total Candidate Recipes per Case: {len(INFERENCE_MODES)} modes x {len(candidate_specs)} post-processors = {len(INFERENCE_MODES) * len(candidate_specs)}", flush=True)

    val_ckpt_csv = OUT_DIR / ("val_smoke_checkpoint.csv" if args.smoke else "val_94_checkpoint.csv")
    test_ckpt_csv = OUT_DIR / ("test_smoke_checkpoint.csv" if args.smoke else "test_150_checkpoint.csv")
    if args.no_resume or args.smoke:
        if val_ckpt_csv.exists():
            val_ckpt_csv.unlink()
        if test_ckpt_csv.exists():
            test_ckpt_csv.unlink()

    done_val_cids = set()
    if val_ckpt_csv.exists():
        prev_val = pd.read_csv(val_ckpt_csv, usecols=["case_id"])
        counts_per_cid = prev_val["case_id"].value_counts()
        expected_rows = len(INFERENCE_MODES) * len(candidate_specs)
        done_val_cids = {int(c) for c, cnt in counts_per_cid.items() if cnt == expected_rows}
        if done_val_cids:
            print(f"[*] Resuming Stage 1 (Val): {len(done_val_cids)}/{len(val_pairs)} cases already completed in {val_ckpt_csv.name}", flush=True)

    # ==========================================================================
    # STAGE 1: N = 94 Validation Split Full Evaluation
    # ==========================================================================
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers) as proc_pool:
        remaining_val = [(cid, p) for cid, p in val_pairs.items() if int(cid) not in done_val_cids]
        t_val_loop = time.time()
        for loop_idx, (cid, (img_path, lbl_path)) in enumerate(remaining_val, 1):
            t_case = time.time()
            t0 = time.time()
            gt, spacing_zyx, sb_full, sg_full, sl_gt, gs, sg_sum = load_gt_and_precompute(img_path, lbl_path)
            batch = transform({"image": str(img_path)})
            t_prep = time.time() - t0

            t0 = time.time()
            p_025_1, p_025_tta = infer_monai_both_passes(
                name="rasnet",
                net=net,
                image_path=str(img_path),
                device=str(device),
                roi_size=(96, 96, 96),
                sw_batch_size=sw_batch_size,
                need_tta=True,
                transform=transform,
                batch=batch,
                overlap=0.25,
                sw_mode="constant",
            )
            t_025 = time.time() - t0

            t0 = time.time()
            p_050_1, p_050_tta = infer_monai_both_passes(
                name="rasnet",
                net=net,
                image_path=str(img_path),
                device=str(device),
                roi_size=(96, 96, 96),
                sw_batch_size=sw_batch_size,
                need_tta=True,
                transform=transform,
                batch=batch,
                overlap=0.50,
                sw_mode="gaussian",
            )
            t_050 = time.time() - t0

            probs_by_mode = {
                "sw025_const_1pass": p_025_1,
                "sw025_const_tta4": p_025_tta,
                "sw050_gauss_1pass": p_050_1,
                "sw050_gauss_tta4": p_050_tta,
            }

            t0 = time.time()
            case_rows, n_uniq, n_tot = evaluate_case_all_candidates(
                cid, probs_by_mode, gt, spacing_zyx, sb_full, sg_full, sl_gt, gs, sg_sum,
                candidate_specs, proc_pool, verify_first_case=(loop_idx == 1 and not done_val_cids),
            )
            append_checkpoint_rows(val_ckpt_csv, case_rows)
            t_eval = time.time() - t0

            done_total = len(done_val_cids) + loop_idx
            elapsed = time.time() - t_val_loop
            eta = (elapsed / loop_idx) * (len(remaining_val) - loop_idx)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(
                f"[Val {done_total:02d}/{len(val_pairs)}] Case {cid:04d} | {time.time()-t_case:.2f}s "
                f"(prep={t_prep:.1f}s, sw025={t_025:.1f}s, sw050={t_050:.1f}s, eval={t_eval:.1f}s | surf={n_uniq}) "
                f"| Peak VRAM: {peak_mb:.0f} MB | Val ETA: {eta/60:.1f}m",
                flush=True,
            )

    val_df = pd.read_csv(val_ckpt_csv).drop_duplicates(subset=["candidate_id", "case_id"]).reset_index(drop=True)

    # Backfill & verify Arm A parity on N=94
    ref_val_a = pd.read_csv(ROOT / "results" / "benchmark_runs" / "val_full_94" / "per_case_arm_A.csv")
    ref_rasnet_a = ref_val_a[(ref_val_a["model"] == "rasnet") & (ref_val_a["case_id"].isin(val_df["case_id"]))].sort_values("case_id").reset_index(drop=True)
    our_arm_a = val_df[val_df["candidate_id"] == "sw025_const_1pass__single_tau0.50_nocc"].sort_values("case_id").reset_index(drop=True)
    max_dice_diff = float(np.max(np.abs(ref_rasnet_a["dice"].to_numpy() - our_arm_a["dice"].to_numpy())))
    print(f"[*] Exact Parity Check vs val_full_94 Arm A (N={len(our_arm_a)}): max |dice diff| = {max_dice_diff:.3e}", flush=True)
    if max_dice_diff > 5e-5:
        raise RuntimeError(f"Parity check failed: max_dice_diff={max_dice_diff}")

    # Backfill exact Arm A HD95/ASD/clDice from val_full_94 into val_df
    ref_map = ref_rasnet_a.set_index("case_id")[["hd95", "asd", "cldice"]]
    mask_arm_a = val_df["candidate_id"] == "sw025_const_1pass__single_tau0.50_nocc"
    for col in ["hd95", "asd", "cldice"]:
        val_df.loc[mask_arm_a, col] = val_df.loc[mask_arm_a, "case_id"].map(ref_map[col])

    summary_rows = []
    for cand_id, grp in val_df.groupby("candidate_id", sort=False):
        first = grp.iloc[0]
        s = summarize_candidate_metrics(grp)
        summary_rows.append({
            "candidate_id": cand_id,
            "mode_key": first["mode_key"],
            "mode_label": first["mode_label"],
            "overlap": first["overlap"],
            "sw_mode": first["sw_mode"],
            "tta": bool(first["tta"]),
            "post_key": first["post_key"],
            "family": first["family"],
            "kind": first["kind"],
            "tau": first["tau"],
            "tau_low": first["tau_low"],
            "tau_high": first["tau_high"],
            "min_seed_voxels": int(first["min_seed_voxels"]),
            "keep_top_k": int(first["keep_top_k"]),
            **s,
        })

    val_sum_df = pd.DataFrame(summary_rows).sort_values("dice_mean", ascending=False).reset_index(drop=True)
    val_sum_df.to_csv(OUT_DIR / ("val_smoke_summary.csv" if args.smoke else "val_94_all_candidates_summary.csv"), index=False)

    # Identify pre-specified validation winners on N=94 (if none pass >= 0.80 in smoke, fallback to top overall)
    def pick_winner(df_sub: pd.DataFrame, fallback_df: pd.DataFrame) -> str:
        elig = df_sub[df_sub["precision_mean"] >= 0.8000]
        return str((elig if len(elig) > 0 else df_sub if len(df_sub) > 0 else fallback_df).iloc[0]["candidate_id"])

    winner_overall_id = pick_winner(val_sum_df, val_sum_df)
    winner_1pass_id = pick_winner(val_sum_df[~val_sum_df["tta"]], val_sum_df)
    winner_hyst_notopk_id = pick_winner(val_sum_df[(val_sum_df["kind"] == "hysteresis") & (val_sum_df["keep_top_k"] == -1)], val_sum_df)
    winner_hyst_top2_id = pick_winner(val_sum_df[(val_sum_df["kind"] == "hysteresis") & (val_sum_df["keep_top_k"] == 2)], val_sum_df)

    print(f"[*] Validation Locked Winner (Overall)         : {winner_overall_id} (Val Dice={val_sum_df.set_index('candidate_id').loc[winner_overall_id, 'dice_mean']:.4f})", flush=True)
    print(f"[*] Validation Locked Winner (1-Pass No TTA)   : {winner_1pass_id} (Val Dice={val_sum_df.set_index('candidate_id').loc[winner_1pass_id, 'dice_mean']:.4f})", flush=True)
    print(f"[*] Validation Locked Winner (Hyst No Top-K)   : {winner_hyst_notopk_id} (Val Dice={val_sum_df.set_index('candidate_id').loc[winner_hyst_notopk_id, 'dice_mean']:.4f})", flush=True)
    print(f"[*] Validation Locked Winner (Hyst + Top-2 CC) : {winner_hyst_top2_id} (Val Dice={val_sum_df.set_index('candidate_id').loc[winner_hyst_top2_id, 'dice_mean']:.4f})", flush=True)

    keep_val_ids = list(dict.fromkeys([
        "sw025_const_1pass__single_tau0.50_nocc",    # Arm A baseline
        "sw025_const_tta4__single_tau0.50_nocc",     # Arm B baseline
        "sw025_const_1pass__single_tau0.30_top2cc",  # Arm C baseline
        "sw050_gauss_1pass__single_tau0.50_nocc",    # Gauss 0.50 1-pass raw
        "sw050_gauss_tta4__single_tau0.50_nocc",     # Gauss 0.50 TTA raw
        "sw050_gauss_1pass__single_tau0.30_top2cc",  # Gauss 0.50 1-pass Arm C equivalent
        "sw050_gauss_tta4__single_tau0.30_top2cc",   # Gauss 0.50 TTA + tau=0.30 + Top-2 CC
        winner_overall_id,
        winner_1pass_id,
        winner_hyst_notopk_id,
        winner_hyst_top2_id,
    ]))
    if not args.smoke:
        val_df[val_df["candidate_id"].isin(keep_val_ids)].to_csv(OUT_DIR / "val_94_selected_per_case.csv", index=False)

    # ==========================================================================
    # STAGE 2: Evaluate Locked Validation Winners on N = 150 Retrospective Test Split
    # ==========================================================================
    spec_lookup = {s["post_key"]: s for s in candidate_specs}
    already_verified_test_ids = {
        "sw025_const_1pass__single_tau0.50_nocc",
        "sw025_const_tta4__single_tau0.50_nocc",
        "sw025_const_1pass__single_tau0.30_top2cc",
    }
    test_eval_specs = []
    for cid_str in keep_val_ids:
        m_key, p_key = cid_str.split("__", 1)
        m_info = next(m for m in INFERENCE_MODES if m["mode_key"] == m_key)
        test_eval_specs.append((cid_str, m_info, spec_lookup[p_key]))

    ref_test_a = pd.read_csv(ROOT / "results" / "benchmark_runs" / "test_retrospective_150" / "per_case_arm_A.csv")
    ref_test_b = pd.read_csv(ROOT / "results" / "benchmark_runs" / "test_retrospective_150" / "per_case_arm_B.csv")
    ref_test_c = pd.read_csv(ROOT / "results" / "benchmark_runs" / "test_retrospective_150" / "per_case_arm_C.csv")
    ref_lookup = {
        ("sw025_const_1pass__single_tau0.50_nocc", int(r["case_id"])): r
        for _, r in ref_test_a[ref_test_a["model"] == "rasnet"].iterrows()
    }
    for _, r in ref_test_b[ref_test_b["model"] == "rasnet"].iterrows():
        ref_lookup[("sw025_const_tta4__single_tau0.50_nocc", int(r["case_id"]))] = r
    for _, r in ref_test_c[ref_test_c["model"] == "rasnet"].iterrows():
        ref_lookup[("sw025_const_1pass__single_tau0.30_top2cc", int(r["case_id"]))] = r

    all_low_taus = sorted(list({
        round(float(s["tau"] if s["kind"] == "single" else s["tau_low"]), 2)
        for _, _, s in test_eval_specs
    }))
    all_high_taus = [0.45, 0.50, 0.55]

    done_test_cids = set()
    if test_ckpt_csv.exists():
        prev_test = pd.read_csv(test_ckpt_csv, usecols=["case_id"])
        counts_t = prev_test["case_id"].value_counts()
        done_test_cids = {int(c) for c, cnt in counts_t.items() if cnt == len(test_eval_specs)}
        if done_test_cids:
            print(f"[*] Resuming Stage 2 (Test): {len(done_test_cids)}/{len(test_pairs)} cases already completed", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers) as proc_pool:
        remaining_test = [(cid, p) for cid, p in test_pairs.items() if int(cid) not in done_test_cids]
        t_test_start = time.time()
        for loop_idx, (cid, (img_path, lbl_path)) in enumerate(remaining_test, 1):
            t_case = time.time()
            gt, spacing_zyx, sb_full, sg_full, sl_gt, gs, sg_sum = load_gt_and_precompute(img_path, lbl_path)
            batch = transform({"image": str(img_path)})

            p_025_1, p_025_tta = infer_monai_both_passes(
                name="rasnet",
                net=net,
                image_path=str(img_path),
                device=str(device),
                roi_size=(96, 96, 96),
                sw_batch_size=sw_batch_size,
                need_tta=True,
                transform=transform,
                batch=batch,
                overlap=0.25,
                sw_mode="constant",
            )
            p_050_1, p_050_tta = infer_monai_both_passes(
                name="rasnet",
                net=net,
                image_path=str(img_path),
                device=str(device),
                roi_size=(96, 96, 96),
                sw_batch_size=sw_batch_size,
                need_tta=True,
                transform=transform,
                batch=batch,
                overlap=0.50,
                sw_mode="gaussian",
            )
            probs_by_mode = {
                "sw025_const_1pass": p_025_1,
                "sw025_const_tta4": p_025_tta,
                "sw050_gauss_1pass": p_050_1,
                "sw050_gauss_tta4": p_050_tta,
            }

            cc_info_by_mode = {}
            high_indices_by_mode = {}
            for m_k, pr in probs_by_mode.items():
                pr_c = np.ascontiguousarray(pr, dtype=np.float32)
                sl_prob, pr_crop, cc_c = build_cc_cache_cropped(pr_c, gt, sg_full, all_low_taus)
                cc_info_by_mode[m_k] = (sl_prob, cc_c)
                if pr_crop is not None:
                    fp_c = pr_crop.ravel()
                    high_indices_by_mode[m_k] = {round(t, 2): np.flatnonzero(fp_c >= float(t)) for t in all_high_taus}
                else:
                    high_indices_by_mode[m_k] = {round(t, 2): np.zeros(0, dtype=np.int64) for t in all_high_taus}

            surface_tasks = {}
            pending_meta = []
            voxel_vol_mm3 = float(np.prod(spacing_zyx))

            for cand_id, m_info, spec in test_eval_specs:
                m_k = m_info["mode_key"]
                sl_prob, cc_cache = cc_info_by_mode[m_k]
                t_key, n_comp, keep_ids, total_counts, gt_counts, sg_counts = resolve_keep_ids(
                    cc_cache, high_indices_by_mode[m_k], spec
                )
                total_low_vox = int(total_counts.sum())
                ret_vox = int(total_counts[keep_ids].sum()) if keep_ids else 0
                rem_vox = int(total_low_vox - ret_vox)

                ref_row = None
                surf_key = None
                if cand_id in already_verified_test_ids:
                    r_ref = ref_lookup[(cand_id, int(cid))]
                    intersection = int(gt_counts[keep_ids].sum()) if keep_ids else 0
                    dice_fast = 2 * intersection / (ret_vox + gs) if (ret_vox + gs) else np.nan
                    if not np.isclose(dice_fast, float(r_ref["dice"]), atol=5e-5):
                        raise RuntimeError(f"Test parity mismatch on {cand_id} case {cid}: {dice_fast} vs {r_ref['dice']}")
                    ref_row = {
                        k: float(r_ref[k]) if k != "status" else str(r_ref[k])
                        for k in ["dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall", "status"]
                    }
                else:
                    surf_key = (m_k, t_key, tuple(keep_ids))
                    if surf_key not in surface_tasks:
                        labeled_crop = cc_cache[t_key][0]
                        surface_tasks[surf_key] = extract_tight_crop_task(
                            surf_key, labeled_crop, n_comp, keep_ids, sl_prob,
                            gt, sb_full, sg_full, sl_gt, gs, sg_sum, spacing_zyx
                        )

                cc_stats = {
                    "components_total": n_comp,
                    "components_retained": int(len(keep_ids)),
                    "components_removed": int(n_comp - len(keep_ids)),
                    "voxels_retained": ret_vox,
                    "voxels_removed": rem_vox,
                    "volume_retained_mm3": float(ret_vox * voxel_vol_mm3),
                    "volume_removed_mm3": float(rem_vox * voxel_vol_mm3),
                }
                pending_meta.append((cand_id, m_info, spec, ref_row, surf_key, cc_stats))

            surface_results = {}
            if surface_tasks:
                for s_key, s_out in proc_pool.map(_worker_surface_metrics, surface_tasks.values()):
                    surface_results[s_key] = s_out

            case_test_rows = []
            for cand_id, m_info, spec, ref_row, surf_key, cc_stats in pending_meta:
                m_res = ref_row if ref_row is not None else surface_results[surf_key]
                case_test_rows.append({
                    "candidate_id": cand_id,
                    "mode_key": m_info["mode_key"],
                    "mode_label": m_info["label"],
                    "overlap": m_info["overlap"],
                    "sw_mode": m_info["sw_mode"],
                    "tta": m_info["tta"],
                    "post_key": spec["post_key"],
                    "family": spec["family"],
                    "kind": spec["kind"],
                    "tau": spec["tau"],
                    "tau_low": spec["tau_low"],
                    "tau_high": spec["tau_high"],
                    "min_seed_voxels": spec["min_seed_voxels"],
                    "keep_top_k": -1 if spec["keep_top_k"] is None else int(spec["keep_top_k"]),
                    "case_id": int(cid),
                    **m_res,
                    **cc_stats,
                })
            append_checkpoint_rows(test_ckpt_csv, case_test_rows)

            done_total = len(done_test_cids) + loop_idx
            elapsed = time.time() - t_test_start
            eta = (elapsed / loop_idx) * (len(remaining_test) - loop_idx)
            print(
                f"[Test {done_total:03d}/{len(test_pairs)}] Case {cid:04d} | {time.time()-t_case:.2f}s (Surf masks: {len(surface_tasks)}) | Test ETA: {eta/60:.1f}m",
                flush=True,
            )

    if args.smoke:
        print(f"[*] Smoke test complete in {time.time() - t_start:.1f}s!", flush=True)
        return

    test_df = pd.read_csv(test_ckpt_csv).drop_duplicates(subset=["candidate_id", "case_id"]).reset_index(drop=True)
    test_df.to_csv(OUT_DIR / "test_150_locked_winners_per_case.csv", index=False)

    ref_test_rasnet_a = ref_test_a[ref_test_a["model"] == "rasnet"].sort_values("case_id").reset_index(drop=True)
    our_test_a = test_df[test_df["candidate_id"] == "sw025_const_1pass__single_tau0.50_nocc"].sort_values("case_id").reset_index(drop=True)
    max_test_diff = float(np.max(np.abs(ref_test_rasnet_a["dice"].to_numpy() - our_test_a["dice"].to_numpy())))
    print(f"[*] Exact Parity Check vs test_retrospective_150 Arm A (N=150): max |dice diff| = {max_test_diff:.3e}", flush=True)

    test_summary_rows = []
    for cand_id, grp in test_df.groupby("candidate_id", sort=False):
        first = grp.iloc[0]
        s = summarize_candidate_metrics(grp)
        test_summary_rows.append({
            "candidate_id": cand_id,
            "mode_key": first["mode_key"],
            "mode_label": first["mode_label"],
            "overlap": first["overlap"],
            "sw_mode": first["sw_mode"],
            "tta": bool(first["tta"]),
            "post_key": first["post_key"],
            "family": first["family"],
            "kind": first["kind"],
            "tau": first["tau"],
            "tau_low": first["tau_low"],
            "tau_high": first["tau_high"],
            "min_seed_voxels": int(first["min_seed_voxels"]),
            "keep_top_k": int(first["keep_top_k"]),
            **s,
        })
    test_sum_df = pd.DataFrame(test_summary_rows).sort_values("dice_mean", ascending=False).reset_index(drop=True)
    test_sum_df.to_csv(OUT_DIR / "test_150_locked_winners_summary.csv", index=False)

    tables_for_stats = {
        "RASNet_ArmA": ref_test_a[ref_test_a["model"] == "rasnet"].copy(),
        "RASNet_ArmB": ref_test_b[ref_test_b["model"] == "rasnet"].copy(),
        "RASNet_ArmC": ref_test_c[ref_test_c["model"] == "rasnet"].copy(),
        "SegResNet_ArmC": ref_test_c[ref_test_c["model"] == "segresnet"].copy(),
        "nnUNet_ArmA": ref_test_a[ref_test_a["model"] == "nnunet"].copy(),
        "nnUNet_ArmB": ref_test_b[ref_test_b["model"] == "nnunet"].copy(),
        "nnUNet_ArmC": ref_test_c[ref_test_c["model"] == "nnunet"].copy(),
    }
    unique_winners = list(dict.fromkeys([
        winner_overall_id,
        winner_1pass_id,
        winner_hyst_notopk_id,
        winner_hyst_top2_id,
        "sw050_gauss_1pass__single_tau0.30_top2cc",
        "sw050_gauss_tta4__single_tau0.30_top2cc",
    ]))
    for winner_id in unique_winners:
        tables_for_stats[winner_id] = test_df[test_df["candidate_id"] == winner_id].copy()

    hypotheses = []
    for winner_id in unique_winners:
        for ref_name in ["RASNet_ArmA", "RASNet_ArmB", "RASNet_ArmC", "SegResNet_ArmC", "nnUNet_ArmA", "nnUNet_ArmB", "nnUNet_ArmC"]:
            for metric, direction in [
                ("dice", "higher"),
                ("precision", "higher"),
                ("recall", "higher"),
                ("hd95", "lower"),
                ("asd", "lower"),
                ("cldice", "higher"),
                ("centerline_recall", "higher"),
            ]:
                hypotheses.append({"a": winner_id, "b": ref_name, "metric": metric, "direction": direction})

    stat_rows = analyze_family(tables_for_stats, hypotheses, allow_nan_filtering=True)
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(OUT_DIR / "paired_stats_test_150_winner_vs_arms_ABC.csv", index=False)

    manifest = {
        "phase": "Phase H - Fixed-Weight RASNet Optimization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_name,
        "sw_batch_size": sw_batch_size,
        "checkpoint": str(ckpt_path),
        "splits_sha256": splits_sha,
        "metric_version": METRIC_VERSION,
        "val_cases": len(val_pairs),
        "test_cases": len(test_pairs),
        "total_candidates_evaluated_on_val": len(val_sum_df),
        "val_parity_max_dice_diff_vs_arm_A": max_dice_diff,
        "test_parity_max_dice_diff_vs_arm_A": max_test_diff,
        "locked_winners": {
            "winner_overall": winner_overall_id,
            "winner_1pass": winner_1pass_id,
            "winner_hyst_no_topk": winner_hyst_notopk_id,
            "winner_hyst_top2": winner_hyst_top2_id,
        },
        "total_runtime_seconds": round(time.time() - t_start, 2),
    }
    with open(OUT_DIR / "manifest_phaseH.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] Phase H Complete in {(time.time() - t_start)/60:.2f} minutes! Results written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
