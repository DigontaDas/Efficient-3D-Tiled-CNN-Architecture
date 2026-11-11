"""
Isolated Evaluation, Post-Processing, Statistical Testing & Native-Grid Inference (`eval_fair_v2.py`).
Strictly runs from raw .nii.gz inputs via MONAI Invertd onto native SimpleITK (Z, Y, X)
physical grids using surface-v1 metrics and atomic per-case CSV checkpoint resume.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import cc3d
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Invertd
from scipy import stats
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augmentations import build_raw_eval_transform
from dataset_pairs import check_geometry, image_geometry, load_validated_splits, resolve_case_pairs
from rasnet_v2_model import build_monai_model


METRIC_VERSION = "surface-v1"


# ---------------------------------------------------------------------------
# Exact surface-v1 metric engine (copied/locked from benchmark/metrics.py)
# ---------------------------------------------------------------------------
def _joint_bbox_slices(mask_union: np.ndarray, margin: int = 2) -> tuple[slice, slice, slice]:
    z = np.any(mask_union, axis=(1, 2))
    y = np.any(mask_union, axis=(0, 2))
    x = np.any(mask_union, axis=(0, 1))
    z_idx = np.where(z)[0]
    y_idx = np.where(y)[0]
    x_idx = np.where(x)[0]
    return (
        slice(max(0, int(z_idx[0]) - margin), min(mask_union.shape[0], int(z_idx[-1]) + margin + 1)),
        slice(max(0, int(y_idx[0]) - margin), min(mask_union.shape[1], int(y_idx[-1]) + margin + 1)),
        slice(max(0, int(x_idx[0]) - margin), min(mask_union.shape[2], int(x_idx[-1]) + margin + 1)),
    )


def surface_distances(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    sp = np.asarray(spacing, dtype=float)
    if a.shape != b.shape or a.ndim != 3 or sp.shape != (3,) or not np.all(np.isfinite(sp)) or np.any(sp <= 0):
        raise ValueError("Expected same 3D shape and positive finite axis spacing")
    if not a.any() or not b.any():
        raise ValueError("Surface distance is undefined for an empty mask")
    sl = _joint_bbox_slices(a | b, margin=2)
    ac, bc = a[sl], b[sl]
    sa = ac & ~binary_erosion(ac, border_value=0)
    sb = bc & ~binary_erosion(bc, border_value=0)
    return distance_transform_edt(~sb, sampling=sp)[sa]


def compute_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, Any]:
    p, g = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if p.shape != g.shape or p.ndim != 3:
        raise ValueError("Masks must have equal 3D shape")
    sp = np.asarray(spacing, dtype=float)
    if sp.shape != (3,) or not np.all(np.isfinite(sp)) or np.any(sp <= 0):
        raise ValueError("Invalid spacing")
    ps, gs = int(p.sum()), int(g.sum())
    intersection = int((p & g).sum())
    union = ps + gs - intersection
    out: dict[str, Any] = {
        "dice": 2 * intersection / (ps + gs) if ps + gs else np.nan,
        "iou": intersection / union if union else np.nan,
        "precision": intersection / ps if ps else np.nan,
        "recall": intersection / gs if gs else np.nan,
        "hd95": np.nan,
        "asd": np.nan,
        "cldice": np.nan,
        "centerline_recall": np.nan,
        "status": (
            "ok"
            if ps and gs
            else (
                "both_empty"
                if not ps and not gs
                else ("prediction_empty" if not ps else "ground_truth_empty")
            )
        ),
    }
    if ps and gs:
        sl = _joint_bbox_slices(p | g, margin=2)
        pc, gc = p[sl], g[sl]
        d = np.concatenate((surface_distances(pc, gc, sp), surface_distances(gc, pc, sp)))
        out["hd95"], out["asd"] = float(np.percentile(d, 95)), float(np.mean(d))
        sp_sk, sg_sk = skeletonize(pc).astype(bool), skeletonize(gc).astype(bool)
        if not sp_sk.any() or not sg_sk.any():
            out["status"] = "empty_skeleton"
        else:
            tprec = float((sp_sk & gc).sum() / sp_sk.sum())
            tsens = float((sg_sk & pc).sum() / sg_sk.sum())
            out["centerline_recall"] = tsens
            out["cldice"] = float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) else 0.0
    return out


# ---------------------------------------------------------------------------
# Post-Processing (Top-K CC, Hysteresis, & Pre-Registered Seeded Filter)
# ---------------------------------------------------------------------------
def prune_components(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    keep_top_k: int | None = 2,
    connectivity: int = 26,
    min_comp_vol_mm3: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    sp = np.asarray(spacing_zyx, dtype=float)
    if mask_u8.ndim != 3 or sp.shape != (3,) or np.any(sp <= 0) or not np.all(np.isfinite(sp)):
        raise ValueError("Expected 3D binary mask and positive finite 3D spacing_zyx")
    if keep_top_k is not None and keep_top_k < 1:
        raise ValueError("keep_top_k must be >= 1 or None")
    voxel_vol_mm3 = float(np.prod(sp))
    total_voxels = int(mask_u8.sum())
    if total_voxels == 0:
        return mask_u8, {
            "components_total": 0,
            "components_retained": 0,
            "components_removed": 0,
            "voxels_retained": 0,
            "voxels_removed": 0,
            "volume_retained_mm3": 0.0,
            "volume_removed_mm3": 0.0,
        }
    labeled_sub, n_comp = cc3d.connected_components(mask_u8, connectivity=connectivity, return_N=True)
    n_comp = int(n_comp)
    counts = np.bincount(labeled_sub.ravel())
    counts[0] = 0
    comp_ids = list(range(1, len(counts)))
    if min_comp_vol_mm3 is not None and min_comp_vol_mm3 > 0.0:
        comp_ids = [cid for cid in comp_ids if float(counts[cid] * voxel_vol_mm3) >= float(min_comp_vol_mm3)]
    comp_ids.sort(key=lambda cid: (-int(counts[cid]), cid))
    if keep_top_k is not None:
        comp_ids = comp_ids[:keep_top_k]

    lut = np.zeros(len(counts), dtype=np.uint8)
    for k_id in comp_ids:
        lut[k_id] = 1
    cleaned = lut[labeled_sub]
    retained_voxels = int(sum(int(counts[k]) for k in comp_ids))
    removed_voxels = int(total_voxels - retained_voxels)
    return cleaned, {
        "components_total": n_comp,
        "components_retained": int(len(comp_ids)),
        "components_removed": int(n_comp - len(comp_ids)),
        "voxels_retained": retained_voxels,
        "voxels_removed": removed_voxels,
        "volume_retained_mm3": float(retained_voxels * voxel_vol_mm3),
        "volume_removed_mm3": float(removed_voxels * voxel_vol_mm3),
    }


prune_connected_components_3d = prune_components


def hysteresis_threshold_3d(
    prob: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    tau_low: float,
    tau_high: float,
    min_seed_voxels: int = 1,
    keep_top_k: int | None = None,
    connectivity: int = 26,
    min_comp_vol_mm3: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    prob_arr = np.asarray(prob, dtype=np.float32)
    sp = np.asarray(spacing_zyx, dtype=float)
    if prob_arr.ndim != 3 or sp.shape != (3,) or np.any(sp <= 0) or not np.all(np.isfinite(sp)):
        raise ValueError("Expected 3D probability volume and positive finite 3D spacing_zyx")
    if not (math.isfinite(tau_low) and math.isfinite(tau_high) and 0.0 < float(tau_low) <= float(tau_high) <= 1.0):
        raise ValueError(f"Expected 0 < tau_low <= tau_high <= 1, got tau_low={tau_low}, tau_high={tau_high}")
    if int(min_seed_voxels) < 1:
        raise ValueError("min_seed_voxels must be >= 1")
    if keep_top_k is not None and int(keep_top_k) < 1:
        raise ValueError("keep_top_k must be >= 1 or None")

    voxel_vol_mm3 = float(np.prod(sp))
    m_low = (prob_arr >= float(tau_low)).astype(np.uint8)
    total_low = int(m_low.sum())
    if total_low == 0:
        return m_low, {
            "components_total": 0,
            "components_seeded": 0,
            "components_retained": 0,
            "components_removed": 0,
            "voxels_retained": 0,
            "voxels_removed": 0,
            "volume_retained_mm3": 0.0,
            "volume_removed_mm3": 0.0,
        }
    m_high = prob_arr >= float(tau_high)
    if not np.any(m_high):
        empty = np.zeros_like(m_low, dtype=np.uint8)
        return empty, {
            "components_total": 0,
            "components_seeded": 0,
            "components_retained": 0,
            "components_removed": 0,
            "voxels_retained": 0,
            "voxels_removed": total_low,
            "volume_retained_mm3": 0.0,
            "volume_removed_mm3": float(total_low * voxel_vol_mm3),
        }

    labeled, n_comp = cc3d.connected_components(m_low, connectivity=connectivity, return_N=True)
    n_comp = int(n_comp)
    total_counts = np.bincount(labeled.ravel(), minlength=n_comp + 1)
    total_counts[0] = 0
    seed_counts = np.bincount(labeled[m_high], minlength=n_comp + 1)
    seed_counts[0] = 0

    seeded_ids = [
        cid for cid in range(1, n_comp + 1)
        if int(seed_counts[cid]) >= int(min_seed_voxels)
        and (min_comp_vol_mm3 is None or float(total_counts[cid] * voxel_vol_mm3) >= float(min_comp_vol_mm3))
    ]
    if not seeded_ids:
        return np.zeros_like(m_low, dtype=np.uint8), {
            "components_total": n_comp,
            "components_seeded": 0,
            "components_retained": 0,
            "components_removed": n_comp,
            "voxels_retained": 0,
            "voxels_removed": total_low,
            "volume_retained_mm3": 0.0,
            "volume_removed_mm3": float(total_low * voxel_vol_mm3),
        }
    seeded_ids.sort(key=lambda cid: (-int(total_counts[cid]), cid))
    keep_ids = seeded_ids[:keep_top_k] if keep_top_k is not None else seeded_ids
    lut = np.zeros(n_comp + 1, dtype=np.uint8)
    for k_id in keep_ids:
        lut[k_id] = 1
    cleaned = lut[labeled]
    ret_vox = int(sum(int(total_counts[k]) for k in keep_ids))
    rem_vox = int(total_low - ret_vox)
    return cleaned, {
        "components_total": n_comp,
        "components_seeded": int(len(seeded_ids)),
        "components_retained": int(len(keep_ids)),
        "components_removed": int(n_comp - len(keep_ids)),
        "voxels_retained": ret_vox,
        "voxels_removed": rem_vox,
        "volume_retained_mm3": float(ret_vox * voxel_vol_mm3),
        "volume_removed_mm3": float(rem_vox * voxel_vol_mm3),
    }


def seeded_component_filter_3d(
    prob: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    tau: float,
    tau_seed: float = 0.55,
    min_seed_voxels: int = 50,
    min_comp_vol_mm3: float = 50.0,
    connectivity: int = 26,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Pre-registered topology-preserving dust filter:
    Keep every 26-connected component of (prob >= tau) satisfying EITHER:
      - physical volume >= min_comp_vol_mm3, OR
      - contains >= min_seed_voxels high-confidence voxels (prob >= tau_seed).
    Does NOT enforce keep_top_k <= 2 so disconnected 3rd/ostial branches are retained.
    """
    prob_arr = np.asarray(prob, dtype=np.float32)
    sp = np.asarray(spacing_zyx, dtype=float)
    if prob_arr.ndim != 3 or sp.shape != (3,) or np.any(sp <= 0) or not np.all(np.isfinite(sp)):
        raise ValueError("Expected 3D probability volume and positive finite 3D spacing_zyx")
    if not (math.isfinite(tau) and 0.0 < float(tau) <= 1.0):
        raise ValueError(f"Expected 0 < tau <= 1, got tau={tau}")
    if not (math.isfinite(tau_seed) and 0.0 < float(tau_seed) <= 1.0):
        raise ValueError(f"Expected 0 < tau_seed <= 1, got tau_seed={tau_seed}")
    if int(min_seed_voxels) < 1:
        raise ValueError("min_seed_voxels must be >= 1")
    if not math.isfinite(min_comp_vol_mm3) or float(min_comp_vol_mm3) < 0.0:
        raise ValueError(f"min_comp_vol_mm3 must be finite >= 0, got {min_comp_vol_mm3}")

    voxel_vol_mm3 = float(np.prod(sp))
    mask_u8 = (prob_arr >= float(tau)).astype(np.uint8)
    total_vox = int(mask_u8.sum())
    if total_vox == 0:
        return mask_u8, {
            "components_total": 0,
            "components_retained": 0,
            "components_removed": 0,
            "voxels_retained": 0,
            "voxels_removed": 0,
            "volume_retained_mm3": 0.0,
            "volume_removed_mm3": 0.0,
        }
    labeled, n_comp = cc3d.connected_components(mask_u8, connectivity=connectivity, return_N=True)
    n_comp = int(n_comp)
    total_counts = np.bincount(labeled.ravel(), minlength=n_comp + 1)
    total_counts[0] = 0
    m_seed = prob_arr >= float(tau_seed)
    seed_counts = np.bincount(labeled[m_seed], minlength=n_comp + 1) if np.any(m_seed) else np.zeros(n_comp + 1, dtype=int)
    seed_counts[0] = 0

    keep_ids = [
        cid for cid in range(1, n_comp + 1)
        if (float(total_counts[cid] * voxel_vol_mm3) >= float(min_comp_vol_mm3))
        or (int(seed_counts[cid]) >= int(min_seed_voxels))
    ]
    lut = np.zeros(n_comp + 1, dtype=np.uint8)
    for k_id in keep_ids:
        lut[k_id] = 1
    cleaned = lut[labeled]
    ret_vox = int(sum(int(total_counts[k]) for k in keep_ids))
    rem_vox = int(total_vox - ret_vox)
    return cleaned, {
        "components_total": n_comp,
        "components_retained": int(len(keep_ids)),
        "components_removed": int(n_comp - len(keep_ids)),
        "voxels_retained": ret_vox,
        "voxels_removed": rem_vox,
        "volume_retained_mm3": float(ret_vox * voxel_vol_mm3),
        "volume_removed_mm3": float(rem_vox * voxel_vol_mm3),
    }


def sweep_case_arm_c_grid(
    prob_zyx: np.ndarray,
    gt_bool: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    tau_grid: list[float] | tuple[float, ...],
    tau_seed_vals: list[float] | tuple[float, ...],
    min_seed_voxels_vals: list[int] | tuple[int, ...],
    min_comp_vol_mm3_vals: list[float] | tuple[float, ...],
) -> tuple[dict[float, float], dict[tuple[float, float, int, float], float]]:
    """
    Exact bit-for-bit vectorized Arm C validation Dice sweep over all tau_grid (Top-2 CC)
    and tau_grid x seeded_grid combinations running cc3d once per tau.
    """
    prob_arr = np.asarray(prob_zyx, dtype=np.float32)
    gt_b = np.asarray(gt_bool, dtype=bool)
    sp = np.asarray(spacing_zyx, dtype=float)
    if prob_arr.shape != gt_b.shape or prob_arr.ndim != 3:
        raise ValueError(f"Expected matching 3D shapes, got {prob_arr.shape} vs {gt_b.shape}")
    if sp.shape != (3,) or np.any(sp <= 0) or not np.all(np.isfinite(sp)):
        raise ValueError("Invalid spacing_zyx")

    voxel_vol_mm3 = float(np.prod(sp))
    gt_sum = int(gt_b.sum())
    d_empty = 1.0 if gt_sum == 0 else 0.0

    taus = [round(float(t), 4) for t in tau_grid]
    ts_list = [round(float(ts), 4) for ts in tau_seed_vals]
    msv_list = [int(v) for v in min_seed_voxels_vals]
    mvol_list = [float(v) for v in min_comp_vol_mm3_vals]

    seed_masks = {ts: (prob_arr >= ts) for ts in ts_list}
    top2_scores: dict[float, float] = {}
    seeded_scores: dict[tuple[float, float, int, float], float] = {}

    for tau in taus:
        mask_u8 = (prob_arr >= tau).astype(np.uint8)
        if not np.any(mask_u8):
            top2_scores[tau] = d_empty
            for ts in ts_list:
                for msv in msv_list:
                    for mvol in mvol_list:
                        seeded_scores[(tau, ts, msv, mvol)] = d_empty
            continue

        labeled, n_comp = cc3d.connected_components(mask_u8, connectivity=26, return_N=True)
        n_comp = int(n_comp)
        counts = np.bincount(labeled.ravel(), minlength=n_comp + 1)
        counts[0] = 0
        tp_counts = np.bincount(labeled[gt_b], minlength=n_comp + 1) if gt_sum > 0 else np.zeros(n_comp + 1, dtype=np.int64)
        tp_counts[0] = 0
        comp_vols = counts * voxel_vol_mm3

        # Top-2 CC (exact tie-break matching prune_components)
        cids_sorted = sorted(range(1, n_comp + 1), key=lambda c: (-int(counts[c]), c))[:2]
        pred_top2 = int(sum(int(counts[c]) for c in cids_sorted))
        inter_top2 = int(sum(int(tp_counts[c]) for c in cids_sorted))
        top2_scores[tau] = float(2.0 * inter_top2 / (pred_top2 + gt_sum)) if (pred_top2 + gt_sum) > 0 else 1.0

        # Seeded grid
        seed_cnt_map: dict[float, np.ndarray] = {}
        for ts, smask in seed_masks.items():
            if np.any(smask):
                sc = np.bincount(labeled[smask], minlength=n_comp + 1)
                sc[0] = 0
            else:
                sc = np.zeros(n_comp + 1, dtype=np.int64)
            seed_cnt_map[ts] = sc

        c_arr = counts[1:]
        tp_arr = tp_counts[1:]
        vol_arr = comp_vols[1:]
        for ts in ts_list:
            sc_arr = seed_cnt_map[ts][1:]
            for msv in msv_list:
                seed_ok = sc_arr >= msv
                for mvol in mvol_list:
                    keep = (vol_arr >= mvol) | seed_ok
                    p_sum = int(c_arr[keep].sum())
                    i_sum = int(tp_arr[keep].sum())
                    seeded_scores[(tau, ts, msv, mvol)] = (
                        float(2.0 * i_sum / (p_sum + gt_sum)) if (p_sum + gt_sum) > 0 else 1.0
                    )

    return top2_scores, seeded_scores


# ---------------------------------------------------------------------------
# Statistical Analysis (exact Holm + Bootstrap + Non-Inferiority)
# ---------------------------------------------------------------------------
def holm_adjust(p_values: list[float] | np.ndarray) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_ci(values: np.ndarray, seed: int = 42, n_boot: int = 2000, confidence: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or len(v) == 0 or not np.all(np.isfinite(v)) or int(n_boot) < 1 or not (0.0 < float(confidence) < 1.0):
        raise ValueError("Invalid bootstrap inputs")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(int(n_boot))])
    alpha = (1.0 - float(confidence)) / 2.0
    return tuple(float(x) for x in np.quantile(means, [alpha, 1.0 - alpha]))


def paired_with_noninferiority(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str,
    ni_margin: float | None = None,
    seed: int = 42,
    n_boot: int = 2000,
    allow_nan_filtering: bool = False,
    direction: str = "higher",
    tie_tolerance: float = 0.0,
) -> dict[str, Any]:
    if direction not in ("higher", "lower"):
        raise ValueError(f"Invalid metric direction: {direction!r}")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be >= 0")
    for df in (df_a, df_b):
        if "case_id" not in df or metric not in df or df["case_id"].isna().any() or df["case_id"].duplicated().any():
            raise ValueError("Missing or duplicate case IDs/metric")
    if set(df_a["case_id"].tolist()) != set(df_b["case_id"].tolist()):
        raise ValueError("Case ID sets differ")

    pair = df_a[["case_id", metric]].merge(df_b[["case_id", metric]], on="case_id", validate="one_to_one", suffixes=("_a", "_b"))
    va = pair[metric + "_a"].to_numpy(dtype=float)
    vb = pair[metric + "_b"].to_numpy(dtype=float)
    finite_mask = np.isfinite(va) & np.isfinite(vb)
    n_excluded = int((~finite_mask).sum())

    if allow_nan_filtering:
        va = va[finite_mask]
        vb = vb[finite_mask]
    elif n_excluded > 0 or len(va) == 0:
        raise ValueError("Missing or non-finite paired metric")

    if len(va) == 0:
        out: dict[str, Any] = {
            "n": 0,
            "n_excluded_nan": n_excluded,
            "a_mean": np.nan,
            "b_mean": np.nan,
            "mean_diff": np.nan,
            "ci95_lo": np.nan,
            "ci95_hi": np.nan,
            "median_diff": np.nan,
            "q25_diff": np.nan,
            "q75_diff": np.nan,
            "iqr_diff": np.nan,
            "p_wilcoxon": 1.0,
            "rank_biserial": np.nan,
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
        if ni_margin is not None:
            out["ni_margin"] = float(ni_margin)
            out["non_inferior"] = False
        return out

    d = va - vb
    ci_lo, ci_hi = bootstrap_ci(d, seed=seed, n_boot=n_boot, confidence=0.95)
    nonzero = d[d != 0]
    if len(nonzero):
        ranks = stats.rankdata(np.abs(nonzero))
        rank_biserial = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
        p_val = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
    else:
        rank_biserial = 0.0
        p_val = 1.0

    wins = int((d > tie_tolerance).sum())
    ties = int((np.abs(d) <= tie_tolerance).sum())
    losses = int((d < -tie_tolerance).sum())
    if direction == "lower":
        wins, losses = losses, wins

    q25 = float(np.quantile(d, 0.25))
    q75 = float(np.quantile(d, 0.75))
    out = {
        "n": int(len(d)),
        "n_excluded_nan": n_excluded,
        "a_mean": float(va.mean()),
        "b_mean": float(vb.mean()),
        "mean_diff": float(d.mean()),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "median_diff": float(np.median(d)),
        "q25_diff": q25,
        "q75_diff": q75,
        "iqr_diff": float(q75 - q25),
        "p_wilcoxon": p_val,
        "rank_biserial": rank_biserial,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }
    if ni_margin is not None:
        out["ni_margin"] = float(ni_margin)
        out["non_inferior"] = bool(ci_lo > float(ni_margin))
    return out


# ---------------------------------------------------------------------------
# Raw .nii.gz -> MONAI Invertd Native SimpleITK (Z, Y, X) Grid Inference
# ---------------------------------------------------------------------------
def build_monai_invert_op(transform) -> Invertd:
    return Invertd(
        keys=["pred"],
        transform=transform,
        orig_keys=["image"],
        nearest_interp=False,
        to_tensor=True,
    )


def infer_monai_native_zyx(
    net: torch.nn.Module,
    image_path: str | Path,
    transform,
    roi_size_xyz: tuple[int, int, int] = (160, 160, 96),
    sw_overlap: float = 0.50,
    sw_mode: str = "gaussian",
    sw_batch_size: int = 4,
    need_tta: bool = False,
    device: str = "cuda",
    inv_op: Invertd | None = None,
    batch: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    net.eval()
    if batch is None:
        batch = transform({"image": str(image_path)})
    x_cpu = batch["image"].unsqueeze(0).cpu()
    sw_dev = torch.device(device)
    cpu_dev = torch.device("cpu")
    amp_ok = sw_dev.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_ok and torch.cuda.is_bf16_supported()) else torch.float16

    def _pred(tensor_cpu: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_ok):
            out = sliding_window_inference(
                tensor_cpu,
                roi_size=roi_size_xyz,
                sw_batch_size=int(sw_batch_size),
                predictor=net,
                overlap=float(sw_overlap),
                mode=str(sw_mode),
                sw_device=sw_dev,
                device=cpu_dev,
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
        return torch.softmax(out.float(), dim=1)[:, 1:2]

    with torch.no_grad():
        p_raw = _pred(x_cpu)
        p_tta = None
        if need_tta:
            acc = p_raw.clone()
            for axis in (2, 3, 4):
                acc += torch.flip(_pred(torch.flip(x_cpu, (axis,))), (axis,))
            p_tta = acc / 4.0

    if inv_op is None:
        inv_op = build_monai_invert_op(transform)

    b1 = dict(batch)
    b1["pred"] = p_raw.squeeze(0)
    prob_1pass_zyx = np.transpose(inv_op(b1)["pred"].squeeze(0).detach().cpu().numpy(), (2, 1, 0)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(prob_1pass_zyx)):
        raise RuntimeError(f"Non-finite MONAI 1-pass probability on {image_path}")

    prob_tta_zyx = None
    if p_tta is not None:
        b2 = dict(batch)
        b2["pred"] = p_tta.squeeze(0)
        prob_tta_zyx = np.transpose(inv_op(b2)["pred"].squeeze(0).detach().cpu().numpy(), (2, 1, 0)).astype(np.float32, copy=False)
        if not np.all(np.isfinite(prob_tta_zyx)):
            raise RuntimeError(f"Non-finite MONAI TTA probability on {image_path}")

    return prob_1pass_zyx, prob_tta_zyx


# ---------------------------------------------------------------------------
# nnU-Net V2 Predictor Builders & Batch Iterators
# ---------------------------------------------------------------------------
def ensure_nnunet_trainer_registered() -> None:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    import nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs as xm

    if not hasattr(xm, "nnUNetTrainer_200epochs"):
        class nnUNetTrainer_200epochs(nnUNetTrainer):
            def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                         device: torch.device = torch.device("cuda")):
                super().__init__(plans, configuration, fold, dataset_json, device)
                self.num_epochs = 200
        setattr(xm, "nnUNetTrainer_200epochs", nnUNetTrainer_200epochs)


def build_nnunet_predictor(
    model_folder: str | Path,
    fold: int = 0,
    checkpoint_name: str = "checkpoint_best.pth",
    device: str | None = None,
    use_mirroring: bool = False,
):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    ensure_nnunet_trainer_registered()
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=bool(use_mirroring),
        perform_everything_on_device=(dev.type == "cuda"),
        device=dev,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(int(fold),),
        checkpoint_name=checkpoint_name,
    )
    return predictor


def iter_nnunet_probs_from_files(
    predictor,
    case_ids: list[int],
    pairs: dict[int, tuple[Path, Path]],
    need_tta: bool = False,
    num_processes: int = 4,
):
    """
    Yield (cid, prob_1pass_zyx, prob_tta_zyx_or_None) using nnU-Net's batch
    preprocessing_iterator_fromfiles so preprocessing overlaps with GPU sliding-window
    inference and both 1-pass and 8-pass TTA reuse the same preprocessed tensor.
    """
    if not case_ids:
        return

    if len(case_ids) == 1:
        cid = int(case_ids[0])
        img_p = pairs[cid][0]
        im = sitk.ReadImage(str(img_p))
        spacing_zyx = tuple(float(s) for s in reversed(im.GetSpacing()))
        arr_zyx = sitk.GetArrayFromImage(im).astype(np.float32, copy=False)[None, ...]
        props = {"spacing": spacing_zyx}
        predictor.use_mirroring = False
        _, probs_1 = predictor.predict_single_npy_array(
            arr_zyx, props, segmentation_previous_stage=None, output_file_truncated=None, save_or_return_probabilities=True
        )
        prob_1 = np.asarray(probs_1[1], dtype=np.float32)
        if not np.all(np.isfinite(prob_1)):
            raise RuntimeError(f"Non-finite nnU-Net 1-pass probability on case {cid}")
        prob_tta = None
        if need_tta:
            predictor.use_mirroring = True
            _, probs_t = predictor.predict_single_npy_array(
                arr_zyx, props, segmentation_previous_stage=None, output_file_truncated=None, save_or_return_probabilities=True
            )
            prob_tta = np.asarray(probs_t[1], dtype=np.float32)
            if not np.all(np.isfinite(prob_tta)):
                raise RuntimeError(f"Non-finite nnU-Net TTA probability on case {cid}")
        yield cid, prob_1, prob_tta
        return

    from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

    list_of_lists = [[str(pairs[int(cid)][0])] for cid in case_ids]
    expected_tokens = [str(int(cid)) for cid in case_ids]
    iterator = preprocessing_iterator_fromfiles(
        list_of_lists=list_of_lists,
        list_of_segs_from_prev_stage_files=None,
        output_filenames_truncated=expected_tokens,
        plans_manager=predictor.plans_manager,
        dataset_json=predictor.dataset_json,
        configuration_manager=predictor.configuration_manager,
        num_processes=max(1, min(int(num_processes), len(case_ids))),
        pin_memory=(predictor.device.type == "cuda"),
        verbose=False,
    )

    yielded = 0
    for cid, dct in zip(case_ids, iterator):
        cid = int(cid)
        ofile = dct.get("ofile")
        if ofile is None or int(ofile) != cid:
            raise RuntimeError(f"nnU-Net iterator case mismatch: expected {cid}, got {ofile}")
        data_tensor = dct["data"]
        props = dct["data_properties"]

        predictor.use_mirroring = False
        logits_1 = predictor.predict_logits_from_preprocessed_data(data_tensor).cpu()
        _, probs_1 = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits_1,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.label_manager,
            props,
            return_probabilities=True,
            num_threads_torch=8,
        )
        prob_1 = np.asarray(probs_1[1], dtype=np.float32)
        if not np.all(np.isfinite(prob_1)):
            raise RuntimeError(f"Non-finite nnU-Net 1-pass probability on case {cid}")

        prob_tta = None
        if need_tta:
            predictor.use_mirroring = True
            logits_t = predictor.predict_logits_from_preprocessed_data(data_tensor).cpu()
            _, probs_t = convert_predicted_logits_to_segmentation_with_correct_shape(
                logits_t,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.label_manager,
                props,
                return_probabilities=True,
                num_threads_torch=8,
            )
            prob_tta = np.asarray(probs_t[1], dtype=np.float32)
            if not np.all(np.isfinite(prob_tta)):
                raise RuntimeError(f"Non-finite nnU-Net TTA probability on case {cid}")

        yielded += 1
        yield cid, prob_1, prob_tta

    if yielded != len(case_ids):
        raise RuntimeError(f"nnU-Net iterator yielded {yielded} cases, expected {len(case_ids)}")


def iter_nnunet_probs_from_npy(
    predictor,
    case_items: list[tuple[int, np.ndarray, dict[str, Any]]],
    need_tta: bool = False,
    num_processes: int = 2,
):
    """
    Batch nnUNetPredictor helper using predictor.get_data_iterator_from_raw_npy_data
    on preloaded (case_id, arr_1zyx, props) tuples.
    """
    if not case_items:
        return
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

    cids = [int(c[0]) for c in case_items]
    imgs = [c[1] for c in case_items]
    props_list = [c[2] for c in case_items]
    tokens = [str(c) for c in cids]

    iterator = predictor.get_data_iterator_from_raw_npy_data(
        image_or_list_of_images=imgs,
        segs_from_prev_stage_or_list_of_segs_from_prev_stage=None,
        properties_or_list_of_properties=props_list,
        truncated_ofname=tokens,
        num_processes=max(1, min(int(num_processes), len(case_items))),
    )
    for cid, dct in zip(cids, iterator):
        ofile = dct.get("ofile")
        if ofile is None or int(ofile) != cid:
            raise RuntimeError(f"nnU-Net npy iterator mismatch: expected {cid}, got {ofile}")
        data_tensor = dct["data"]
        props = dct["data_properties"]

        predictor.use_mirroring = False
        logits_1 = predictor.predict_logits_from_preprocessed_data(data_tensor).cpu()
        _, probs_1 = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits_1,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.label_manager,
            props,
            return_probabilities=True,
            num_threads_torch=8,
        )
        prob_1 = np.asarray(probs_1[1], dtype=np.float32)
        if not np.all(np.isfinite(prob_1)):
            raise RuntimeError(f"Non-finite nnU-Net npy 1-pass probability on case {cid}")

        prob_tta = None
        if need_tta:
            predictor.use_mirroring = True
            logits_t = predictor.predict_logits_from_preprocessed_data(data_tensor).cpu()
            _, probs_t = convert_predicted_logits_to_segmentation_with_correct_shape(
                logits_t,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.label_manager,
                props,
                return_probabilities=True,
                num_threads_torch=8,
            )
            prob_tta = np.asarray(probs_t[1], dtype=np.float32)
            if not np.all(np.isfinite(prob_tta)):
                raise RuntimeError(f"Non-finite nnU-Net npy TTA probability on case {cid}")

        yield cid, prob_1, prob_tta


# ---------------------------------------------------------------------------
# CSV Resume, Multi-Arm Evaluation & Shared Post-Processing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmEvalSpec:
    arm_name: str
    output_csv: Path
    tau: float = 0.50
    need_tta: bool = False
    postprocess_mode: str = "top_k"
    keep_top_k: int | None = None
    tau_seed: float = 0.55
    min_seed_voxels: int = 50
    min_comp_vol_mm3: float = 50.0


def _atomic_write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _load_existing_csv_rows(
    out_p: Path,
    resume: bool,
    valid_cids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], set[int]]:
    rows: list[dict[str, Any]] = []
    done_ids: set[int] = set()
    if resume and out_p.is_file() and out_p.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_p)
            if not existing.empty and "case_id" in existing.columns:
                existing = existing[existing["case_id"].notna()].copy()
                existing["case_id"] = existing["case_id"].astype(int)
                if valid_cids is not None:
                    existing = existing[existing["case_id"].isin(valid_cids)]
                existing = existing.drop_duplicates(subset=["case_id"], keep="last")
                if not existing.empty:
                    rows = existing.to_dict("records")
                    done_ids = set(existing["case_id"].tolist())
        except pd.errors.EmptyDataError:
            rows = []
            done_ids = set()
    return rows, done_ids


def _sorted_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def _apply_postprocess(
    prob_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    tau: float,
    postprocess_mode: str = "top_k",
    keep_top_k: int | None = None,
    tau_seed: float = 0.55,
    min_seed_voxels: int = 50,
    min_comp_vol_mm3: float = 50.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if postprocess_mode == "seeded":
        return seeded_component_filter_3d(
            prob_zyx,
            spacing_zyx,
            tau=float(tau),
            tau_seed=float(tau_seed),
            min_seed_voxels=int(min_seed_voxels),
            min_comp_vol_mm3=float(min_comp_vol_mm3),
        )
    if postprocess_mode == "top_k":
        mask = prob_zyx >= float(tau)
        return prune_components(mask, spacing_zyx, keep_top_k=keep_top_k)
    raise ValueError(f"Unsupported postprocess_mode: {postprocess_mode!r}")


def _eval_single_arm_row(
    cid: int,
    prob_zyx: np.ndarray,
    gt_bool: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    spec: ArmEvalSpec,
) -> dict[str, Any]:
    mask_p, pp_stats = _apply_postprocess(
        prob_zyx,
        spacing_zyx,
        tau=spec.tau,
        postprocess_mode=spec.postprocess_mode,
        keep_top_k=spec.keep_top_k,
        tau_seed=spec.tau_seed,
        min_seed_voxels=spec.min_seed_voxels,
        min_comp_vol_mm3=spec.min_comp_vol_mm3,
    )
    m = compute_metrics(mask_p, gt_bool, spacing_zyx)
    return {
        "case_id": int(cid),
        "arm": spec.arm_name,
        "tau": float(spec.tau),
        "tta": bool(spec.need_tta),
        "postprocess_mode": spec.postprocess_mode,
        "keep_top_k": 0 if spec.keep_top_k is None else int(spec.keep_top_k),
        "tau_seed": float(spec.tau_seed) if spec.postprocess_mode == "seeded" else np.nan,
        "min_seed_voxels": int(spec.min_seed_voxels) if spec.postprocess_mode == "seeded" else 0,
        "min_comp_vol_mm3": float(spec.min_comp_vol_mm3) if spec.postprocess_mode == "seeded" else 0.0,
        **m,
        **pp_stats,
    }


def evaluate_case_arms_from_probs(
    cid: int,
    prob_1pass_zyx: np.ndarray,
    prob_tta_zyx: np.ndarray | None,
    gt_bool: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    arm_specs: list[ArmEvalSpec],
    max_workers: int = 4,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate multiple arms/thresholds for one scan concurrently on CPU threads
    reusing already-inferred (prob_1pass_zyx, prob_tta_zyx).
    """
    if len(arm_specs) <= 1 or max_workers <= 1:
        out: dict[str, dict[str, Any]] = {}
        for spec in arm_specs:
            p = prob_tta_zyx if (spec.need_tta and prob_tta_zyx is not None) else prob_1pass_zyx
            out[spec.arm_name] = _eval_single_arm_row(cid, p, gt_bool, spacing_zyx, spec)
        return out

    out = {}
    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(arm_specs))) as pool:
        futs = {}
        for spec in arm_specs:
            p = prob_tta_zyx if (spec.need_tta and prob_tta_zyx is not None) else prob_1pass_zyx
            futs[spec.arm_name] = pool.submit(_eval_single_arm_row, cid, p, gt_bool, spacing_zyx, spec)
        for arm_name, fut in futs.items():
            out[arm_name] = fut.result()
    return out


def _prob_cache_paths(prob_cache_dir: Path, cid: int) -> tuple[Path, Path]:
    return prob_cache_dir / f"case_{int(cid):04d}_1p.npy", prob_cache_dir / f"case_{int(cid):04d}_tta.npy"


def _save_npy_atomic_float32(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.asarray(arr, dtype=np.float32), allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def evaluate_monai_multi_arm_to_csvs(
    config_path: str | Path,
    checkpoint_path: str | Path,
    arch: str,
    init_filters: int,
    gate_mode: str,
    enable_ds: bool,
    case_ids: list[int],
    arm_specs: list[ArmEvalSpec],
    resume: bool = True,
    pairs_override: dict[int, tuple[Path, Path]] | None = None,
    precomputed_probs: dict[int, tuple[np.ndarray, np.ndarray | None]] | None = None,
    prob_cache_dir: str | Path | None = None,
    on_case_probs: Callable[[int, np.ndarray, np.ndarray | None, np.ndarray, tuple[float, float, float]], None] | None = None,
    interrupt_after_cases: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Single-pass MONAI evaluator: computes (prob_1pass, prob_tta) at most once per case,
    reusing Transform & Invertd instances, evaluating all missing arm_specs in parallel on CPU,
    and optionally persisting/reusing probability volumes in precomputed_probs or prob_cache_dir.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    pixdim_xyz = tuple(float(v) for v in cfg["preprocessing"]["pixdim_xyz"])
    roi_xyz = tuple(int(v) for v in cfg["inference_eval"]["roi_size_xyz"])
    sw_overlap = float(cfg["inference_eval"]["sw_overlap"])
    sw_mode = str(cfg["inference_eval"]["sw_mode"])
    sw_bs = min(
        int(cfg["inference_eval"]["sw_batch_size"]),
        2 if (arch.lower() in ("rasnet_v2", "segresnet") and int(init_filters) >= 32) else 4,
    )

    all_cids = [int(c) for c in case_ids]
    valid_cid_set = set(all_cids)
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_done: dict[str, set[int]] = {}
    for spec in arm_specs:
        r, d = _load_existing_csv_rows(Path(spec.output_csv), resume=resume, valid_cids=valid_cid_set)
        arm_rows[spec.arm_name] = r
        arm_done[spec.arm_name] = d

    missing_cids = [
        cid for cid in all_cids
        if any(cid not in arm_done[spec.arm_name] for spec in arm_specs)
    ]
    if not missing_cids and on_case_probs is None:
        return {spec.arm_name: _sorted_df(arm_rows[spec.arm_name]) for spec in arm_specs}

    target_cids = all_cids if on_case_probs is not None else missing_cids
    if not target_cids:
        return {spec.arm_name: _sorted_df(arm_rows[spec.arm_name]) for spec in arm_specs}

    pairs = pairs_override or resolve_case_pairs(cfg["study"]["dataset_root"], all_cids, check_headers=True)
    need_tta = any(spec.need_tta for spec in arm_specs)
    p_cache = Path(prob_cache_dir) if prob_cache_dir is not None else None

    # Determine whether GPU model load is actually needed (all cases might be in precomputed/disk cache)
    need_gpu_cases = []
    for cid in target_cids:
        if precomputed_probs is not None and cid in precomputed_probs:
            p1, pt = precomputed_probs[cid]
            if need_tta and pt is None:
                need_gpu_cases.append(cid)
            continue
        if p_cache is not None:
            p1_path, pt_path = _prob_cache_paths(p_cache, cid)
            if p1_path.is_file() and (not need_tta or pt_path.is_file()):
                continue
        need_gpu_cases.append(cid)

    net = None
    transform = None
    inv_op = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if need_gpu_cases:
        net = build_monai_model(arch=arch, init_filters=init_filters, gate_mode=gate_mode, enable_ds=enable_ds).to(device)
        obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        transform = build_raw_eval_transform(
            pixdim_xyz=pixdim_xyz,
            body_crop_hu=float(cfg["preprocessing"]["body_crop_hu"]),
        )
        inv_op = build_monai_invert_op(transform)

    def _load_case_inputs(cid_item: int, run_tf: bool):
        im_p, lb_p = pairs[cid_item]
        lb_im = sitk.ReadImage(str(lb_p))
        gt_b = sitk.GetArrayFromImage(lb_im) > 0
        sp_zyx = tuple(float(s) for s in reversed(lb_im.GetSpacing()))
        b_dict = transform({"image": str(im_p)}) if (run_tf and transform is not None) else None
        return cid_item, im_p, gt_b, sp_zyx, b_dict

    processed_now = 0
    try:
        need_set = set(need_gpu_cases)
        with ThreadPoolExecutor(max_workers=2) as prep_pool:
            q_prep: deque = deque()
            next_sub = 0
            while next_sub < min(2, len(target_cids)):
                c_next = target_cids[next_sub]
                q_prep.append(prep_pool.submit(_load_case_inputs, c_next, c_next in need_set))
                next_sub += 1

            for _ in range(len(target_cids)):
                cid, img_p, gt_bool, spacing_zyx, batch_dict = q_prep.popleft().result()
                if next_sub < len(target_cids):
                    c_next = target_cids[next_sub]
                    q_prep.append(prep_pool.submit(_load_case_inputs, c_next, c_next in need_set))
                    next_sub += 1

                p_1pass: np.ndarray
                p_tta: np.ndarray | None
                if precomputed_probs is not None and cid in precomputed_probs and (not need_tta or precomputed_probs[cid][1] is not None):
                    p_1pass, p_tta = precomputed_probs[cid]
                elif p_cache is not None and _prob_cache_paths(p_cache, cid)[0].is_file() and (not need_tta or _prob_cache_paths(p_cache, cid)[1].is_file()):
                    p1_path, pt_path = _prob_cache_paths(p_cache, cid)
                    p_1pass = np.load(p1_path)
                    p_tta = np.load(pt_path) if need_tta else None
                else:
                    assert net is not None and transform is not None and inv_op is not None
                    p_1pass, p_tta = infer_monai_native_zyx(
                        net=net,
                        image_path=img_p,
                        transform=transform,
                        roi_size_xyz=roi_xyz,
                        sw_overlap=sw_overlap,
                        sw_mode=sw_mode,
                        sw_batch_size=sw_bs,
                        need_tta=need_tta,
                        device=device,
                        inv_op=inv_op,
                        batch=batch_dict,
                    )
                    if p_cache is not None:
                        p1_path, pt_path = _prob_cache_paths(p_cache, cid)
                        _save_npy_atomic_float32(p_1pass, p1_path)
                        if p_tta is not None:
                            _save_npy_atomic_float32(p_tta, pt_path)
                del batch_dict

                if p_1pass.shape != gt_bool.shape:
                    raise RuntimeError(f"Case {cid}: inverted prob shape {p_1pass.shape} != GT shape {gt_bool.shape}")

                if on_case_probs is not None:
                    on_case_probs(cid, p_1pass, p_tta, gt_bool, spacing_zyx)

                todo_specs = [s for s in arm_specs if cid not in arm_done[s.arm_name]]
                if todo_specs:
                    res_rows = evaluate_case_arms_from_probs(
                        cid=cid,
                        prob_1pass_zyx=p_1pass,
                        prob_tta_zyx=p_tta,
                        gt_bool=gt_bool,
                        spacing_zyx=spacing_zyx,
                        arm_specs=todo_specs,
                    )
                    for spec in todo_specs:
                        arm_rows[spec.arm_name].append(res_rows[spec.arm_name])
                        arm_done[spec.arm_name].add(cid)
                        _atomic_write_df(_sorted_df(arm_rows[spec.arm_name]), Path(spec.output_csv))

                processed_now += 1
                if interrupt_after_cases is not None and processed_now >= interrupt_after_cases:
                    break
    finally:
        del net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {spec.arm_name: _sorted_df(arm_rows[spec.arm_name]) for spec in arm_specs}


def evaluate_nnunet_multi_arm_to_csvs(
    config_path: str | Path,
    model_folder: str | Path,
    checkpoint_name: str,
    case_ids: list[int],
    arm_specs: list[ArmEvalSpec],
    resume: bool = True,
    pairs_override: dict[int, tuple[Path, Path]] | None = None,
    precomputed_probs: dict[int, tuple[np.ndarray, np.ndarray | None]] | None = None,
    prob_cache_dir: str | Path | None = None,
    on_case_probs: Callable[[int, np.ndarray, np.ndarray | None, np.ndarray, tuple[float, float, float]], None] | None = None,
    num_processes: int = 4,
) -> dict[str, pd.DataFrame]:
    """
    Single-pass native nnU-Net V2 batch evaluator using preprocessing_iterator_fromfiles
    and ThreadPoolExecutor metric scoring across arms.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    all_cids = [int(c) for c in case_ids]
    valid_cid_set = set(all_cids)
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_done: dict[str, set[int]] = {}
    for spec in arm_specs:
        r, d = _load_existing_csv_rows(Path(spec.output_csv), resume=resume, valid_cids=valid_cid_set)
        arm_rows[spec.arm_name] = r
        arm_done[spec.arm_name] = d

    missing_cids = [
        cid for cid in all_cids
        if any(cid not in arm_done[spec.arm_name] for spec in arm_specs)
    ]
    target_cids = all_cids if on_case_probs is not None else missing_cids
    if not target_cids:
        return {spec.arm_name: _sorted_df(arm_rows[spec.arm_name]) for spec in arm_specs}

    pairs = pairs_override or resolve_case_pairs(cfg["study"]["dataset_root"], all_cids, check_headers=True)
    need_tta = any(spec.need_tta for spec in arm_specs)
    p_cache = Path(prob_cache_dir) if prob_cache_dir is not None else None

    cached_cids: list[int] = []
    gpu_cids: list[int] = []
    for cid in target_cids:
        if precomputed_probs is not None and cid in precomputed_probs and (not need_tta or precomputed_probs[cid][1] is not None):
            cached_cids.append(cid)
        elif p_cache is not None and _prob_cache_paths(p_cache, cid)[0].is_file() and (not need_tta or _prob_cache_paths(p_cache, cid)[1].is_file()):
            cached_cids.append(cid)
        else:
            gpu_cids.append(cid)

    def _consume_case(cid_item: int, p_1pass: np.ndarray, p_tta: np.ndarray | None):
        _, lbl_p = pairs[cid_item]
        lb_im = sitk.ReadImage(str(lbl_p))
        gt_bool = sitk.GetArrayFromImage(lb_im) > 0
        spacing_zyx = tuple(float(s) for s in reversed(lb_im.GetSpacing()))
        if p_1pass.shape != gt_bool.shape:
            raise RuntimeError(f"Case {cid_item}: nnU-Net prob shape {p_1pass.shape} != GT shape {gt_bool.shape}")
        if on_case_probs is not None:
            on_case_probs(cid_item, p_1pass, p_tta, gt_bool, spacing_zyx)
        todo_specs = [s for s in arm_specs if cid_item not in arm_done[s.arm_name]]
        if todo_specs:
            res_rows = evaluate_case_arms_from_probs(
                cid=cid_item,
                prob_1pass_zyx=p_1pass,
                prob_tta_zyx=p_tta,
                gt_bool=gt_bool,
                spacing_zyx=spacing_zyx,
                arm_specs=todo_specs,
            )
            for spec in todo_specs:
                arm_rows[spec.arm_name].append(res_rows[spec.arm_name])
                arm_done[spec.arm_name].add(cid_item)
                _atomic_write_df(_sorted_df(arm_rows[spec.arm_name]), Path(spec.output_csv))

    for cid in cached_cids:
        if precomputed_probs is not None and cid in precomputed_probs and (not need_tta or precomputed_probs[cid][1] is not None):
            p1, pt = precomputed_probs[cid]
        else:
            assert p_cache is not None
            p1_path, pt_path = _prob_cache_paths(p_cache, cid)
            p1 = np.load(p1_path)
            pt = np.load(pt_path) if need_tta else None
        _consume_case(cid, p1, pt)

    predictor = None
    if gpu_cids:
        try:
            predictor = build_nnunet_predictor(
                model_folder=model_folder,
                fold=int(cfg["nnunet_v2"]["fold"]),
                checkpoint_name=checkpoint_name,
                use_mirroring=False,
            )
            for cid, p1, pt in iter_nnunet_probs_from_files(
                predictor=predictor,
                case_ids=gpu_cids,
                pairs=pairs,
                need_tta=need_tta,
                num_processes=num_processes,
            ):
                if p_cache is not None:
                    p1_path, pt_path = _prob_cache_paths(p_cache, cid)
                    _save_npy_atomic_float32(p1, p1_path)
                    if pt is not None:
                        _save_npy_atomic_float32(pt, pt_path)
                _consume_case(cid, p1, pt)
        finally:
            del predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {spec.arm_name: _sorted_df(arm_rows[spec.arm_name]) for spec in arm_specs}


def evaluate_monai_checkpoint_to_csv(
    config_path: str | Path,
    checkpoint_path: str | Path,
    arch: str,
    init_filters: int,
    gate_mode: str,
    enable_ds: bool,
    case_ids: list[int],
    output_csv: str | Path,
    tau: float = 0.50,
    need_tta: bool = False,
    postprocess_mode: str = "top_k",
    keep_top_k: int | None = None,
    tau_seed: float = 0.55,
    min_seed_voxels: int = 50,
    min_comp_vol_mm3: float = 50.0,
    resume: bool = True,
    interrupt_after_cases: int | None = None,
    pairs_override: dict[int, tuple[Path, Path]] | None = None,
    precomputed_probs: dict[int, tuple[np.ndarray, np.ndarray | None]] | None = None,
) -> pd.DataFrame:
    """Single-arm MONAI checkpoint evaluator wrapping evaluate_monai_multi_arm_to_csvs."""
    spec = ArmEvalSpec(
        arm_name="single",
        output_csv=Path(output_csv),
        tau=float(tau),
        need_tta=bool(need_tta),
        postprocess_mode=postprocess_mode,
        keep_top_k=keep_top_k,
        tau_seed=float(tau_seed),
        min_seed_voxels=int(min_seed_voxels),
        min_comp_vol_mm3=float(min_comp_vol_mm3),
    )
    res = evaluate_monai_multi_arm_to_csvs(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        arch=arch,
        init_filters=init_filters,
        gate_mode=gate_mode,
        enable_ds=enable_ds,
        case_ids=case_ids,
        arm_specs=[spec],
        resume=resume,
        pairs_override=pairs_override,
        precomputed_probs=precomputed_probs,
        interrupt_after_cases=interrupt_after_cases,
    )
    return res["single"]


def evaluate_nnunet_checkpoint_to_csv(
    config_path: str | Path,
    model_folder: str | Path,
    checkpoint_name: str,
    case_ids: list[int],
    output_csv: str | Path,
    tau: float = 0.50,
    need_tta: bool = False,
    postprocess_mode: str = "top_k",
    keep_top_k: int | None = None,
    tau_seed: float = 0.55,
    min_seed_voxels: int = 50,
    min_comp_vol_mm3: float = 50.0,
    resume: bool = True,
    pairs_override: dict[int, tuple[Path, Path]] | None = None,
    precomputed_probs: dict[int, tuple[np.ndarray, np.ndarray | None]] | None = None,
) -> pd.DataFrame:
    """Single-arm nnU-Net V2 evaluator wrapping evaluate_nnunet_multi_arm_to_csvs."""
    spec = ArmEvalSpec(
        arm_name="single",
        output_csv=Path(output_csv),
        tau=float(tau),
        need_tta=bool(need_tta),
        postprocess_mode=postprocess_mode,
        keep_top_k=keep_top_k,
        tau_seed=float(tau_seed),
        min_seed_voxels=int(min_seed_voxels),
        min_comp_vol_mm3=float(min_comp_vol_mm3),
    )
    res = evaluate_nnunet_multi_arm_to_csvs(
        config_path=config_path,
        model_folder=model_folder,
        checkpoint_name=checkpoint_name,
        case_ids=case_ids,
        arm_specs=[spec],
        resume=resume,
        pairs_override=pairs_override,
        precomputed_probs=precomputed_probs,
    )
    return res["single"]


def main() -> None:
    parser = argparse.ArgumentParser(description="RASNet-v2 Native-Grid Evaluation CLI")
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument("--backend", choices=["monai", "nnunet"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to MONAI .pth or nnU-Net model folder")
    parser.add_argument("--nnunet-ckpt-name", default="checkpoint_best.pth")
    parser.add_argument("--arch", default="rasnet_v2")
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--gate-mode", default="conv_bias")
    parser.add_argument("--disable-ds", action="store_true")
    parser.add_argument("--split", choices=["val", "test", "val20"], default="val")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--tau", type=float, default=0.50)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--postprocess-mode", choices=["top_k", "seeded"], default="top_k")
    parser.add_argument("--keep-top-k", type=int, default=0)
    parser.add_argument("--tau-seed", type=float, default=0.55)
    parser.add_argument("--min-seed-voxels", type=int, default=50)
    parser.add_argument("--min-comp-vol-mm3", type=float, default=50.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.split == "val20":
        cids = json.loads(Path(cfg["study"]["val20_ids_file"]).read_text(encoding="utf-8"))["case_ids"]
    else:
        cids = load_validated_splits(cfg["study"]["splits_file"])[args.split]
    k_val = None if int(args.keep_top_k) <= 0 else int(args.keep_top_k)

    if args.backend == "monai":
        df = evaluate_monai_checkpoint_to_csv(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            arch=args.arch,
            init_filters=args.init_filters,
            gate_mode=args.gate_mode,
            enable_ds=not args.disable_ds,
            case_ids=cids,
            output_csv=args.output_csv,
            tau=args.tau,
            need_tta=args.tta,
            postprocess_mode=args.postprocess_mode,
            keep_top_k=k_val,
            tau_seed=args.tau_seed,
            min_seed_voxels=args.min_seed_voxels,
            min_comp_vol_mm3=args.min_comp_vol_mm3,
            resume=not args.no_resume,
        )
    else:
        df = evaluate_nnunet_checkpoint_to_csv(
            config_path=args.config,
            model_folder=args.checkpoint,
            checkpoint_name=args.nnunet_ckpt_name,
            case_ids=cids,
            output_csv=args.output_csv,
            tau=args.tau,
            need_tta=args.tta,
            postprocess_mode=args.postprocess_mode,
            keep_top_k=k_val,
            tau_seed=args.tau_seed,
            min_seed_voxels=args.min_seed_voxels,
            min_comp_vol_mm3=args.min_comp_vol_mm3,
            resume=not args.no_resume,
        )
    print(df[["dice", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall"]].mean().to_dict())


if __name__ == "__main__":
    main()
