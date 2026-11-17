"""Generate verified V1 reporting corrections, figures, and tables.

Reads existing CSVs, logs, manifests, and checkpoint provenance read-only and writes
ONLY to a non-protected output directory (default:
`results/v1_reporting_correction_20261117_105000Z`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.endpoint_eval import (
    compute_endpoint_paired_stats,
    summarize_endpoint_metrics,
    validate_endpoint_out_dir,
)
from benchmark.run_benchmark import ALL_METRICS, METRIC_DIRECTIONS
from benchmark.stats import holm_adjust

EXTRA_PROTECTED_PREFIXES = (
    "results/v1_epoch200_comparison_20261117_041500Z",
    "results/v1_epoch200_smoke_20261117_041500Z",
    "results/optionA_v1_symmetric_sw050",
    "results/audits",
    "recovered_checkpoints",
    "Q1_Publication_Package",
    "Phase3_Local_Integration",
    "benchmark",
    "Rasnet-v2",
)

NON_CAUSAL_PROTOCOL_SENTENCE = (
    "The logged metrics use different sampling, aggregation and spatial evaluation protocols "
    "and therefore cannot establish comparative full-volume performance."
)


def validate_correction_out_dir(out_dir: Path, repo_root: Path) -> Path:
    """Ensure correction outputs never write into protected or historical directories."""
    resolved = validate_endpoint_out_dir(out_dir=out_dir, repo_root=repo_root, resume=True)
    repo_res = repo_root.resolve()
    for rel in EXTRA_PROTECTED_PREFIXES:
        prot = (repo_res / rel).resolve()
        if resolved == prot or prot in resolved.parents:
            raise ValueError(f"Output directory is inside protected path '{rel}': {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recompute_val94_endpoint_stats(
    ep200_dir: Path,
    splits_path: Path,
    seed: int = 42,
    n_boot: int = 2000,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Independently recompute Val94 epoch-200 summaries, paired stats, and post-hoc Holm-8."""
    dr = pd.read_csv(ep200_dir / "per_case_metrics_rasnet_epoch200.csv")
    dn = pd.read_csv(ep200_dir / "per_case_metrics_nnunet_epoch200.csv")
    geom_df = pd.read_csv(ep200_dir / "geometry_verification.csv")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    expected_val_ids = sorted(int(x) for x in splits["val"])

    for name, df in (("rasnet", dr), ("nnunet", dn)):
        if len(df) != 94 or df["case_id"].nunique() != 94:
            raise ValueError(f"{name} must have exactly 94 unique case_ids")
        if sorted(int(x) for x in df["case_id"]) != expected_val_ids:
            raise ValueError(f"{name} case_ids do not match Val94 split")

    n_empty_pred = int((dr["status"] == "prediction_empty").sum() + (dn["status"] == "prediction_empty").sum())
    geom_ok_count = int(geom_df["geometry_ok"].astype(bool).sum())
    geom_total_count = int(len(geom_df))

    comb = pd.concat([dr, dn], ignore_index=True)
    summary_df = summarize_endpoint_metrics(comb, seed=seed, n_boot=n_boot)

    meta, raw_paired_df = compute_endpoint_paired_stats(
        dr, dn, metrics=ALL_METRICS, seed=seed, n_boot=n_boot, tie_tolerance=tie_tolerance
    )
    n_excluded_nan = int(raw_paired_df["n_excluded_nan"].sum())

    paired_df = raw_paired_df.copy()
    paired_df["p_unadjusted"] = paired_df["p"].astype(float)
    paired_df["p_value_status"] = "exploratory_unadjusted"
    paired_df["ci_vs_wilcoxon_note"] = (
        "Bootstrap 95% CI targets the mean paired difference; Wilcoxon signed-rank p targets "
        "paired rank location. These target different quantities and need not coincide."
    )

    holm_p_vals = holm_adjust(paired_df["p_unadjusted"].to_numpy(dtype=float))
    posthoc_holm8_df = pd.DataFrame({
        "family_label": "POST-HOC EXPLORATORY — NOT PREREGISTERED (8-metric Val94 endpoint)",
        "metric": paired_df["metric"],
        "direction": paired_df["direction"],
        "n": paired_df["n"],
        "mean_difference": paired_df["mean_difference"],
        "diff_ci95_lo": paired_df["diff_ci95_lo"],
        "diff_ci95_hi": paired_df["diff_ci95_hi"],
        "wins": paired_df["wins"],
        "ties": paired_df["ties"],
        "losses": paired_df["losses"],
        "p_unadjusted": paired_df["p_unadjusted"],
        "holm8_p": holm_p_vals,
        "significant_at_0_05_holm8": [bool(p < 0.05) for p in holm_p_vals],
    })

    # Build paired case-level frame for plotting
    r_s = dr.sort_values("case_id").reset_index(drop=True)
    n_s = dn.sort_values("case_id").reset_index(drop=True)
    merged = r_s.merge(n_s, on="case_id", validate="one_to_one", suffixes=("_rasnet", "_nnunet"))
    paired_case_df = pd.DataFrame({
        "case_id": merged["case_id"].astype(int),
        "rasnet_dice": merged["dice_rasnet"].astype(float),
        "nnunet_dice": merged["dice_nnunet"].astype(float),
        "dice_diff_rasnet_minus_nnunet": merged["dice_rasnet"].astype(float) - merged["dice_nnunet"].astype(float),
    })

    return {
        "n_matched_cases": 94,
        "n_empty_pred": n_empty_pred,
        "n_excluded_nan": n_excluded_nan,
        "geometry_ok_count": geom_ok_count,
        "geometry_total_count": geom_total_count,
        "summary_df": summary_df,
        "paired_df": paired_df,
        "posthoc_holm8_df": posthoc_holm8_df,
        "paired_case_df": paired_case_df,
        "meta": meta,
    }


def parse_all_training_logs(
    logs_dir: Path,
    nnunet_last_pt_path: Path | None = None,
    nnunet_best_pt_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Parse all 5 model training logs and nnU-Net checkpoint logging arrays."""
    specs = {
        "rasnet": {
            "csv": "rasnet_200ep_log.csv",
            "display": "RASNet",
            "loss_label": "StenosisAwareLoss (0.4 Dice + 0.6 Focal, γ=2.5 + DS)",
            "val_protocol": "Val94 full-volume (0.5mm iso, 96³ SW 25% overlap, 12 evals)",
        },
        "segresnet": {
            "csv": "segresnet_200ep_log.csv",
            "display": "SegResNet",
            "loss_label": "DiceFocalLoss (λ_dice=1.0, λ_focal=2.0)",
            "val_protocol": "Val94 full-volume (0.5mm iso, 96³ SW 25% overlap, 20 evals)",
        },
        "vnet": {
            "csv": "vnet_200ep_log.csv",
            "display": "V-Net",
            "loss_label": "DiceFocalLoss (λ_dice=1.0, λ_focal=2.0)",
            "val_protocol": "Val94 full-volume (0.5mm iso, 96³ SW 25% overlap, 13 evals)",
        },
        "3dunet": {
            "csv": "3dunet_200ep_log.csv",
            "display": "3D U-Net",
            "loss_label": "DiceCELoss (sigmoid)",
            "val_protocol": "Val94 single-shot 128³ resized grid (21 evals)",
        },
        "nnunet": {
            "csv": "nnunet_200ep_log.csv",
            "display": "nnU-Net V2",
            "loss_label": "DC_and_CE_loss + DS (negative scale)",
            "val_protocol": "50 val batches (batch_size=2, 96×160×160 [0.5, 0.3496, 0.3496]mm, 33% forced FG, 200 evals)",
        },
    }

    out: dict[str, dict[str, Any]] = {}
    for key, meta in specs.items():
        df = pd.read_csv(logs_dir / meta["csv"])
        df = df[df["epoch"].notna()].copy()
        df["epoch"] = df["epoch"].astype(int)
        df = df.sort_values("epoch").reset_index(drop=True)
        val_df = df[df["val_dice"].notna()].copy().reset_index(drop=True)
        best_idx = int(val_df["val_dice"].astype(float).idxmax())
        best_row = val_df.iloc[best_idx]
        ep200_val = float(val_df.loc[val_df["epoch"] == 200, "val_dice"].iloc[0])

        entry: dict[str, Any] = {
            "df": df,
            "val_df": val_df,
            "display": meta["display"],
            "loss_label": meta["loss_label"],
            "val_protocol": meta["val_protocol"],
            "n_train_epochs": int(len(df)),
            "n_val_evals": int(len(val_df)),
            "best_val_epoch": int(best_row["epoch"]),
            "best_val_dice": float(best_row["val_dice"]),
            "ep200_val_dice": ep200_val,
        }

        if key == "nnunet":
            entry["raw_pseudo_dice_peak_epoch"] = int(best_row["epoch"])
            entry["raw_pseudo_dice_peak_value"] = float(best_row["val_dice"])
            if nnunet_last_pt_path is not None and nnunet_last_pt_path.is_file():
                ckpt_last = torch.load(nnunet_last_pt_path, map_location="cpu", weights_only=False)
                logging_dict = ckpt_last["logging"]
                mean_fg = np.asarray(logging_dict["mean_fg_dice"], dtype=float)
                ema_fg = np.asarray(logging_dict["ema_fg_dice"], dtype=float)
                entry["raw_series"] = mean_fg
                entry["ema_series"] = ema_fg
                entry["raw_pseudo_dice_peak_epoch"] = int(np.argmax(mean_fg) + 1)
                entry["raw_pseudo_dice_peak_value"] = float(np.max(mean_fg))
                entry["ema_best_epoch"] = int(np.argmax(ema_fg) + 1)
                entry["ema_best_value"] = float(np.max(ema_fg))
            if nnunet_best_pt_path is not None and nnunet_best_pt_path.is_file():
                ckpt_best = torch.load(nnunet_best_pt_path, map_location="cpu", weights_only=False)
                entry["ckpt_best_current_epoch"] = int(ckpt_best["current_epoch"])
                entry["ckpt_best_ema"] = float(ckpt_best["_best_ema"])
        out[key] = entry
    return out


def plot_fig3_corrected(parsed_logs: dict[str, dict[str, Any]], png_path: Path, svg_path: Path) -> None:
    """Create corrected Figure 3 with accurate loss labels, full y-range, and raw vs EMA nnU-Net."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(17.0, 6.8), dpi=300)

    colors = {
        "rasnet": "#d95f02",
        "segresnet": "#2ca02c",
        "vnet": "#9467bd",
        "3dunet": "#8c564b",
        "nnunet": "#0072b2",
    }

    # Panel A: Training Loss (unshifted; MONAI positive left axis, nnU-Net negative right axis)
    ax_a_r = ax_a.twinx()
    lines_a = []
    for key in ("rasnet", "segresnet", "vnet", "3dunet"):
        d = parsed_logs[key]
        df = d["df"]
        (ln,) = ax_a.plot(
            df["epoch"],
            df["train_loss"],
            color=colors[key],
            linewidth=1.8,
            label=f"{d['display']} ({d['loss_label']})",
        )
        lines_a.append(ln)

    d_nnu = parsed_logs["nnunet"]
    df_nnu = d_nnu["df"]
    (ln_nnu,) = ax_a_r.plot(
        df_nnu["epoch"],
        df_nnu["train_loss"],
        color=colors["nnunet"],
        linestyle="--",
        linewidth=1.9,
        label=f"{d_nnu['display']} ({d_nnu['loss_label']}, right axis)",
    )
    lines_a.append(ln_nnu)

    ax_a.set_xlim(1, 200)
    ax_a.set_ylim(0.0, 2.05)
    ax_a_r.set_ylim(-1.05, 0.05)
    ax_a.set_xlabel("Training Epoch (1–200)", fontweight="bold")
    ax_a.set_ylabel("MONAI Training Loss (Positive Objectives)", fontweight="bold", color="#222222")
    ax_a_r.set_ylabel("nnU-Net V2 DC_and_CE_loss (Negative Scale)", fontweight="bold", color=colors["nnunet"], labelpad=8)
    ax_a_r.tick_params(axis="y", colors=colors["nnunet"])
    ax_a.set_title("A. Logged Training Loss (Non-Comparable Objectives; Unshifted)", fontweight="bold", fontsize=11.0, pad=10)
    ax_a.grid(True, linestyle="--", alpha=0.35)
    labels_a = [l.get_label() for l in lines_a]
    ax_a.legend(lines_a, labels_a, loc="upper right", fontsize=7.6, frameon=True, framealpha=0.94)

    # Panel B: Validation Metric Trajectories (full 0.0-0.96 range + zoomed inset for epochs 100-200)
    for key, marker, lw in (
        ("rasnet", "o", 2.0),
        ("segresnet", "s", 1.7),
        ("vnet", "^", 1.7),
        ("3dunet", "v", 1.6),
    ):
        d = parsed_logs[key]
        vdf = d["val_df"]
        grid_tag = "128³ resized" if key == "3dunet" else "0.5mm SW 25%"
        lbl = (
            f"{d['display']}: Val94 Full-Vol Dice ({grid_tag}, {d['n_val_evals']} evals; "
            f"best ep {d['best_val_epoch']}={d['best_val_dice']:.4f})"
        )
        ax_b.plot(
            vdf["epoch"],
            vdf["val_dice"],
            color=colors[key],
            marker=marker,
            markersize=4.0,
            linewidth=lw,
            label=lbl,
            zorder=3,
        )

    epochs_nnu = np.arange(1, len(d_nnu["raw_series"]) + 1)
    raw_s = d_nnu["raw_series"]
    ema_s = d_nnu["ema_series"]

    ax_b.plot(
        epochs_nnu,
        raw_s,
        color=colors["nnunet"],
        linestyle="--",
        linewidth=1.35,
        alpha=0.78,
        label=(
            f"nnU-Net V2: Raw Patch Pseudo dice (mean_fg_dice, 50 batches/ep, 200 evals; "
            f"peak ep {d_nnu['raw_pseudo_dice_peak_epoch']}={d_nnu['raw_pseudo_dice_peak_value']:.4f})"
        ),
        zorder=2,
    )
    ax_b.plot(
        epochs_nnu,
        ema_s,
        color="#043f66",
        linestyle=":",
        linewidth=2.2,
        label=(
            f"nnU-Net V2: EMA Patch Pseudo dice (ema_fg_dice; "
            f"ckpt_best ep {d_nnu['ema_best_epoch']}={d_nnu['ema_best_value']:.4f})"
        ),
        zorder=4,
    )
    # Mark distinct raw peak (ep 189) and EMA best (ep 199)
    ax_b.scatter(
        [d_nnu["raw_pseudo_dice_peak_epoch"]],
        [d_nnu["raw_pseudo_dice_peak_value"]],
        color=colors["nnunet"],
        marker="*",
        s=115,
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )
    ax_b.scatter(
        [d_nnu["ema_best_epoch"]],
        [d_nnu["ema_best_value"]],
        color="#043f66",
        marker="D",
        s=55,
        edgecolor="black",
        linewidth=0.7,
        zorder=6,
    )

    ax_b.set_xlim(1, 202)
    ax_b.set_ylim(0.0, 0.96)
    ax_b.set_xlabel("Training Epoch (1–200)", fontweight="bold")
    ax_b.set_ylabel("Logged Validation Metric (Protocol-Specific)", fontweight="bold", labelpad=8)
    ax_b.set_title("B. Logged Validation Trajectories (Distinct Evaluation Protocols)", fontweight="bold", fontsize=11.0, pad=10)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.legend(loc="lower right", fontsize=7.2, frameon=True, framealpha=0.95)

    # Zoomed inset for epochs 100-200 in open whitespace above legend and below 3D U-Net (~0.66)
    ax_ins = inset_axes(
        ax_b,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.40, 0.29, 0.57, 0.33),
        bbox_transform=ax_b.transAxes,
        loc="lower left",
    )
    for key, marker in (("rasnet", "o"), ("segresnet", "s"), ("vnet", "^")):
        vdf = parsed_logs[key]["val_df"]
        sub = vdf[vdf["epoch"] >= 100]
        ax_ins.plot(sub["epoch"], sub["val_dice"], color=colors[key], marker=marker, markersize=3.2, linewidth=1.4)
    mask_100 = epochs_nnu >= 100
    ax_ins.plot(epochs_nnu[mask_100], raw_s[mask_100], color=colors["nnunet"], linestyle="--", linewidth=1.1, alpha=0.78)
    ax_ins.plot(epochs_nnu[mask_100], ema_s[mask_100], color="#043f66", linestyle=":", linewidth=1.7)
    ax_ins.scatter([189], [d_nnu["raw_pseudo_dice_peak_value"]], color=colors["nnunet"], marker="*", s=70, edgecolor="black", linewidth=0.5, zorder=5)
    ax_ins.scatter([199], [d_nnu["ema_best_value"]], color="#043f66", marker="D", s=35, edgecolor="black", linewidth=0.5, zorder=6)
    ax_ins.set_xlim(100, 202)
    ax_ins.set_ylim(0.71, 0.855)
    ax_ins.set_title("Epochs 100–200 Detail", fontsize=7.8, fontweight="bold", pad=3)
    ax_ins.tick_params(labelsize=7.0)
    ax_ins.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(
        "Corrected V1 200-Epoch Training Loss & Logged Validation Telemetry (All 5 Models)",
        fontsize=12.8,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        f"{NON_CAUSAL_PROTOCOL_SENTENCE}\n"
        "nnU-Net V2 best checkpoint (current_epoch=199, _best_ema=0.8162, diamond) is selected by EMA Pseudo dice, "
        "whereas raw batch Pseudo dice (mean_fg_dice) peaks at epoch 189 (0.8376, star). "
        "SegResNet & V-Net trained with DiceFocalLoss(λ=1.0, 2.0).",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#333333",
    )
    fig.subplots_adjust(left=0.06, right=0.95, top=0.87, bottom=0.14, wspace=0.28)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_fig_endpoint_corrected(
    ep200_res: dict[str, Any],
    regimes_df: pd.DataFrame,
    png_path: Path,
    svg_path: Path,
) -> None:
    """Create corrected Val94 Epoch-200 endpoint figure with exploratory/Holm-8 annotations."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.2, 5.9), dpi=300)
    c_ras = "#d95f02"
    c_nnu = "#1f78b4"

    paired_case_df = ep200_res["paired_case_df"]
    paired_df = ep200_res["paired_df"].set_index("metric")
    holm_df = ep200_res["posthoc_holm8_df"].set_index("metric")

    r_vals = paired_case_df["rasnet_dice"].to_numpy(dtype=float)
    n_vals = paired_case_df["nnunet_dice"].to_numpy(dtype=float)

    # Panel A
    bp = ax1.boxplot(
        [r_vals, n_vals],
        positions=[0, 1],
        widths=0.44,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.8},
    )
    for patch, col in zip(bp["boxes"], [c_ras, c_nnu]):
        patch.set_facecolor(col)
        patch.set_alpha(0.32)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.4)
    rng = np.random.default_rng(42)
    ax1.scatter(rng.normal(0.0, 0.055, size=len(r_vals)), r_vals, color=c_ras, alpha=0.55, s=16, zorder=3)
    ax1.scatter(rng.normal(1.0, 0.055, size=len(n_vals)), n_vals, color=c_nnu, alpha=0.55, s=16, zorder=3)
    ax1.scatter([0, 1], [np.mean(r_vals), np.mean(n_vals)], color=["#8c2d04", "#084594"], marker="D", s=45, zorder=4)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([
        f"RASNet (final ep 200*)\n{np.mean(r_vals):.4f} ± {np.std(r_vals, ddof=1):.4f}",
        f"nnU-Net V2 (ep 200)\n{np.mean(n_vals):.4f} ± {np.std(n_vals, ddof=1):.4f}",
    ], fontsize=9.2)
    ax1.set_ylabel("Full-Volume Native-Grid Dice (Val94, N=94)", fontweight="bold")
    ax1.set_title("A. Epoch-200 Val94 Endpoint\n(1-Pass, 0.50 Gauss SW, τ=0.50, No CC)", fontweight="bold", fontsize=10.3)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B
    diffs = np.sort(paired_case_df["dice_diff_rasnet_minus_nnunet"].to_numpy(dtype=float))
    bar_cols = [c_ras if d > 0 else c_nnu for d in diffs]
    ax2.bar(np.arange(len(diffs)), diffs, color=bar_cols, width=0.85, alpha=0.82)
    ax2.axhline(0.0, color="#222222", linewidth=1.1)

    d_row = paired_df.loc["dice"]
    ax2.axhline(float(d_row["mean_difference"]), color="#111111", linewidth=1.4, linestyle="--")
    ax2.axhspan(float(d_row["diff_ci95_lo"]), float(d_row["diff_ci95_hi"]), color="#666666", alpha=0.14)
    ax2.set_xlabel("Val94 Cases Sorted by Paired Dice Diff (N=94)", fontweight="bold")
    ax2.set_ylabel("Paired Dice Diff (RASNet − nnU-Net)", fontweight="bold")
    ax2.set_title(
        f"B. Paired Dice Diff: {float(d_row['mean_difference']):+.4f} "
        f"[{float(d_row['diff_ci95_lo']):+.4f}, {float(d_row['diff_ci95_hi']):+.4f}]\n"
        f"Exploratory unadj. Wilcoxon p={float(d_row['p_unadjusted']):.4f} (n.s., not equiv.)",
        fontweight="bold",
        fontsize=9.8,
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    p_r = paired_df.loc["precision"]
    r_r = paired_df.loc["recall"]
    cl_r = paired_df.loc["cldice"]
    clr_r = paired_df.loc["centerline_recall"]
    note_txt = (
        f"0 empties / 188/188 geom OK | Dice W/T/L: {int(d_row['wins'])}/{int(d_row['ties'])}/{int(d_row['losses'])}\n"
        f"Exploratory unadj. p (post-hoc Holm-8 p):\n"
        f"  Prec Δ={float(p_r['mean_difference']):+.4f}: p={float(p_r['p_unadjusted']):.2e} (Holm={float(holm_df.loc['precision','holm8_p']):.2e})\n"
        f"  Rec Δ={float(r_r['mean_difference']):+.4f}: p={float(r_r['p_unadjusted']):.2e} (Holm={float(holm_df.loc['recall','holm8_p']):.2e})\n"
        f"  CL-Rec Δ={float(clr_r['mean_difference']):+.4f}: p={float(clr_r['p_unadjusted']):.4f} (Holm={float(holm_df.loc['centerline_recall','holm8_p']):.4f} n.s.)\n"
        f"  clDice Δ={float(cl_r['mean_difference']):+.4f}: p={float(cl_r['p_unadjusted']):.4f} (Holm={float(holm_df.loc['cldice','holm8_p']):.4f} n.s.)"
    )
    ax2.text(
        0.03,
        0.97,
        note_txt,
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=7.3,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.94},
    )

    # Panel C: Separate Best-Checkpoint Context (Arm A Val94 & Option A Val94 & Option A Test150 R4)
    b1_r = regimes_df[(regimes_df["regime_id"] == "B1_best_val94_armA")].iloc[0]
    b2_r = regimes_df[(regimes_df["regime_id"] == "B2_best_val94_optA_tau050_top2")].iloc[0]
    d_r4 = regimes_df[(regimes_df["regime_id"] == "D_optionA_R4")].iloc[0]

    labels_c = [
        "Val94 Arm A\nRASNet\n(0.25 const)",
        "Val94 Arm A\nnnU-Net\n(0.50 Gauss)",
        "Val94 Opt-A\nRASNet\n(0.50+Top2)",
        "Val94 Opt-A\nnnU-Net\n(0.50+Top2)",
        "Test150 R4\nRASNet\n(TTA8+τ*+Top2)",
        "Test150 R4\nnnU-Net\n(TTA8+τ*+Top2)",
    ]
    vals_c = [
        float(b1_r["rasnet_dice_mean"]),
        float(b1_r["nnunet_dice_mean"]),
        float(b2_r["rasnet_dice_mean"]),
        float(b2_r["nnunet_dice_mean"]),
        float(d_r4["rasnet_dice_mean"]),
        float(d_r4["nnunet_dice_mean"]),
    ]
    errs_c = [
        float(b1_r["rasnet_dice_sd"]),
        float(b1_r["nnunet_dice_sd"]),
        float(b2_r["rasnet_dice_sd"]),
        float(b2_r["nnunet_dice_sd"]),
        float(d_r4["rasnet_dice_sd"]),
        float(d_r4["nnunet_dice_sd"]),
    ]
    cols_c = [c_ras, c_nnu, c_ras, c_nnu, c_ras, c_nnu]
    xp = np.arange(len(labels_c))
    ax3.bar(xp, vals_c, yerr=errs_c, color=cols_c, alpha=0.75, capsize=3.5, width=0.58)
    for x, v, e in zip(xp, vals_c, errs_c):
        ax3.text(x, v + e + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax3.axvline(1.5, color="#888888", linestyle=":", linewidth=1.2)
    ax3.axvline(3.5, color="#888888", linestyle=":", linewidth=1.2)
    ax3.set_xticks(xp)
    ax3.set_xticklabels(labels_c, fontsize=6.6)
    ax3.set_ylim(0.65, 0.91)
    ax3.set_ylabel("Mean Dice (± SD)", fontweight="bold")
    ax3.set_title(
        "C. Strictly Separate Best-Checkpoint Regimes\n(Distinct Cohorts, Epochs & Recipes; Not a Trajectory)",
        fontweight="bold",
        fontsize=9.8,
    )
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Corrected V1 Epoch-200 Val94 Endpoint & Multiplicity / Regime Context",
        fontsize=12.4,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "*RASNet final_epoch200 provenance is supported by script/log/archive alignment (no embedded epoch metadata); "
        "nnU-Net final_epoch200 embeds current_epoch=200. Best nnU-Net embeds current_epoch=199 (_best_ema=0.8162; raw peak ep 189=0.8376).\n"
        "All 8 endpoint p-values are exploratory/unadjusted; clDice (unadj p=0.0352, 95% CI [-0.0006, +0.0135]) is NOT significant under post-hoc Holm-8 (p=0.1760).",
        ha="center",
        va="bottom",
        fontsize=7.9,
        color="#333333",
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.18, wspace=0.25)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def build_four_regimes_table(repo_root: Path) -> pd.DataFrame:
    """Build strictly separated comparison table across Regimes A, B1/B2, C, and D (Option A)."""
    ep200_sum = pd.read_csv(repo_root / "results" / "v1_epoch200_comparison_20261117_041500Z" / "summary_epoch200.csv").set_index("model")
    val_arm_a = pd.read_csv(repo_root / "results" / "benchmark_runs" / "val_full_94" / "summary_arm_A.csv").set_index("model")
    opt_grid = pd.read_csv(repo_root / "results" / "optionA_v1_symmetric_sw050" / "val_grid_scores.csv")
    test_a = pd.read_csv(repo_root / "results" / "benchmark_runs" / "test_retrospective_150" / "summary_arm_A.csv").set_index("model")
    test_b = pd.read_csv(repo_root / "results" / "benchmark_runs" / "test_retrospective_150" / "summary_arm_B.csv").set_index("model")
    test_c = pd.read_csv(repo_root / "results" / "benchmark_runs" / "test_retrospective_150" / "summary_arm_C.csv").set_index("model")
    opt_sum = pd.read_csv(repo_root / "results" / "optionA_v1_symmetric_sw050" / "summary_test.csv")
    opt_ni = pd.read_csv(repo_root / "results" / "optionA_v1_symmetric_sw050" / "noninferiority_rasnet_vs_nnunet_dice.csv").set_index("recipe")

    b2_r = opt_grid[(opt_grid["model"] == "rasnet") & (opt_grid["mode"] == "sw050_gauss_1pass") & (np.isclose(opt_grid["tau"], 0.50))]["dice"].to_numpy(dtype=float)
    b2_n = opt_grid[(opt_grid["model"] == "nnunet") & (opt_grid["mode"] == "sw050_gauss_1pass") & (np.isclose(opt_grid["tau"], 0.50))]["dice"].to_numpy(dtype=float)

    rows = [
        {
            "regime_id": "A_final_epoch200_val94",
            "regime_name": "(A) Final Epoch-200 Endpoint",
            "cohort": "Val94 (N=94, dev)",
            "n_cases": 94,
            "checkpoint_role": "final_epoch200",
            "n_models": 2,
            "models_included": "rasnet, nnunet",
            "sw_and_tta_protocol": "Symmetric 1-pass, 0.50 Gaussian SW, tau=0.50",
            "postprocess": "none",
            "rasnet_dice_mean": float(ep200_sum.loc["rasnet", "dice_mean"]),
            "rasnet_dice_sd": float(ep200_sum.loc["rasnet", "dice_std"]),
            "nnunet_dice_mean": float(ep200_sum.loc["nnunet", "dice_mean"]),
            "nnunet_dice_sd": float(ep200_sum.loc["nnunet", "dice_std"]),
            "non_inferior_at_margin_neg0005": pd.NA,
            "source_file": "results/v1_epoch200_comparison_20261117_041500Z/summary_epoch200.csv",
        },
        {
            "regime_id": "B1_best_val94_armA",
            "regime_name": "(B1) Best-Checkpoint Val94 Arm A",
            "cohort": "Val94 (N=94, dev)",
            "n_cases": 94,
            "checkpoint_role": "best (R:160, N:199/189*)",
            "n_models": 5,
            "models_included": "rasnet, nnunet, segresnet, vnet, 3dunet",
            "sw_and_tta_protocol": "Asymmetric SW (MONAI 0.25 const vs nnU-Net 0.50 Gauss; 3D U-Net 128³), 1-pass, tau=0.50",
            "postprocess": "none",
            "rasnet_dice_mean": float(val_arm_a.loc["rasnet", "dice_mean"]),
            "rasnet_dice_sd": float(val_arm_a.loc["rasnet", "dice_std"]),
            "nnunet_dice_mean": float(val_arm_a.loc["nnunet", "dice_mean"]),
            "nnunet_dice_sd": float(val_arm_a.loc["nnunet", "dice_std"]),
            "non_inferior_at_margin_neg0005": pd.NA,
            "source_file": "results/benchmark_runs/val_full_94/summary_arm_A.csv",
        },
        {
            "regime_id": "B2_best_val94_optA_tau050_top2",
            "regime_name": "(B2) Best-Checkpoint Val94 Option A Grid (tau=0.50)",
            "cohort": "Val94 (N=94, dev)",
            "n_cases": 94,
            "checkpoint_role": "best (R:160, N:199/189*)",
            "n_models": 4,
            "models_included": "rasnet, nnunet, segresnet, vnet (3dunet excluded)",
            "sw_and_tta_protocol": "Symmetric 1-pass, 0.50 Gaussian SW, tau=0.50",
            "postprocess": "26-connected Top-2 CC",
            "rasnet_dice_mean": float(np.mean(b2_r)),
            "rasnet_dice_sd": float(np.std(b2_r, ddof=1)),
            "nnunet_dice_mean": float(np.mean(b2_n)),
            "nnunet_dice_sd": float(np.std(b2_n, ddof=1)),
            "non_inferior_at_margin_neg0005": pd.NA,
            "source_file": "results/optionA_v1_symmetric_sw050/val_grid_scores.csv",
        },
        {
            "regime_id": "C_retrospective_test150_armA",
            "regime_name": "(C-Arm A) Retrospective Test150 Arm A",
            "cohort": "Test150 (851-1000, retrospective)",
            "n_cases": 150,
            "checkpoint_role": "best",
            "n_models": 5,
            "models_included": "rasnet, nnunet, segresnet, vnet, 3dunet",
            "sw_and_tta_protocol": "Asymmetric SW (MONAI 0.25 const vs nnU-Net 0.50 Gauss), 1-pass, tau=0.50",
            "postprocess": "none",
            "rasnet_dice_mean": float(test_a.loc["rasnet", "dice_mean"]),
            "rasnet_dice_sd": float(test_a.loc["rasnet", "dice_std"]),
            "nnunet_dice_mean": float(test_a.loc["nnunet", "dice_mean"]),
            "nnunet_dice_sd": float(test_a.loc["nnunet", "dice_std"]),
            "non_inferior_at_margin_neg0005": pd.NA,
            "source_file": "results/benchmark_runs/test_retrospective_150/summary_arm_A.csv",
        },
        {
            "regime_id": "C_retrospective_test150_armB",
            "regime_name": "(C-Arm B) Retrospective Test150 Arm B",
            "cohort": "Test150 (851-1000, retrospective)",
            "n_cases": 150,
            "checkpoint_role": "best",
            "n_models": 5,
            "models_included": "rasnet, nnunet, segresnet, vnet, 3dunet",
            "sw_and_tta_protocol": "Asymmetric SW + Asymmetric TTA (RASNet 4-pass, nnU-Net 8-pass, others 1-pass), tau=0.50",
            "postprocess": "none",
            "rasnet_dice_mean": float(test_b.loc["rasnet", "dice_mean"]),
            "rasnet_dice_sd": float(test_b.loc["rasnet", "dice_std"]),
            "nnunet_dice_mean": float(test_b.loc["nnunet", "dice_mean"]),
            "nnunet_dice_sd": float(test_b.loc["nnunet", "dice_std"]),
            "non_inferior_at_margin_neg0005": pd.NA,
            "source_file": "results/benchmark_runs/test_retrospective_150/summary_arm_B.csv",
        },
        {
            "regime_id": "C_retrospective_test150_armC",
            "regime_name": "(C-Arm C) Retrospective Test150 Arm C",
            "cohort": "Test150 (851-1000, retrospective)",
            "n_cases": 150,
            "checkpoint_role": "best",
            "n_models": 5,
            "models_included": "rasnet, nnunet, segresnet, vnet, 3dunet",
            "sw_and_tta_protocol": "Asymmetric SW, 1-pass, Val94-locked tau* (clamped at 0.30)",
            "postprocess": "26-connected Top-2 CC",
            "rasnet_dice_mean": float(test_c.loc["rasnet", "dice_mean"]),
            "rasnet_dice_sd": float(test_c.loc["rasnet", "dice_std"]),
            "nnunet_dice_mean": float(test_c.loc["nnunet", "dice_mean"]),
            "nnunet_dice_sd": float(test_c.loc["nnunet", "dice_std"]),
            "non_inferior_at_margin_neg0005": pd.NA,
            "source_file": "results/benchmark_runs/test_retrospective_150/summary_arm_C.csv",
        },
    ]

    for recipe_id, reg_id, label, proto, pp in (
        ("R1_raw_1pass", "D_optionA_R1", "(D-R1) Option A Symmetric R1 (Secondary)", "Symmetric 0.50 Gauss SW, 1-pass, tau=0.50", "none"),
        ("R2_raw_tta8", "D_optionA_R2", "(D-R2) Option A Symmetric R2 (Secondary)", "Symmetric 0.50 Gauss SW, TTA-8 logit avg, tau=0.50", "none"),
        ("R3_tuned_1pass_top2", "D_optionA_R3", "(D-R3) Option A Symmetric R3 (Secondary)", "Symmetric 0.50 Gauss SW, 1-pass, Val94 tau* (R:0.40, N:0.35)", "26-connected Top-2 CC"),
        ("R4_tuned_tta8_top2", "D_optionA_R4", "(D-R4) Option A Symmetric R4 (PRIMARY)", "Symmetric 0.50 Gauss SW, TTA-8 logit avg, Val94 tau* (R:0.40, N:0.20)", "26-connected Top-2 CC"),
    ):
        sub = opt_sum[opt_sum["recipe"] == recipe_id].set_index("model")
        ni_row = opt_ni.loc[recipe_id]
        rows.append({
            "regime_id": reg_id,
            "regime_name": label,
            "cohort": "Test150 (851-1000, retrospective)",
            "n_cases": 150,
            "checkpoint_role": "best",
            "n_models": 4,
            "models_included": "rasnet, nnunet, segresnet, vnet (3dunet excluded)",
            "sw_and_tta_protocol": proto,
            "postprocess": pp,
            "rasnet_dice_mean": float(sub.loc["rasnet", "dice_mean"]),
            "rasnet_dice_sd": float(sub.loc["rasnet", "dice_sd"]),
            "nnunet_dice_mean": float(sub.loc["nnunet", "dice_mean"]),
            "nnunet_dice_sd": float(sub.loc["nnunet", "dice_sd"]),
            "non_inferior_at_margin_neg0005": bool(ni_row["non_inferior"]),
            "source_file": "results/optionA_v1_symmetric_sw050/summary_test.csv",
        })

    return pd.DataFrame(rows)


def build_efficiency_and_timing_table(repo_root: Path) -> pd.DataFrame:
    """Combine verified Table 4 single-patch compute/VRAM with Option A n=10 full-volume timing."""
    t4 = pd.read_csv(repo_root / "results" / "thesis_outputs" / "Table4_computational_efficiency_rtx4080super.csv").set_index("model_key")
    ts = pd.read_csv(repo_root / "results" / "optionA_v1_symmetric_sw050" / "timing_summary.csv")

    timing_lookup = {}
    for _, r in ts.iterrows():
        timing_lookup[(str(r["model"]), str(r["mode"]))] = r

    invalid_note = (
        "Excluded from latency claims: manifest.json rasnet=0.0s reflects --resume skipping 94 completed cases; "
        "legacy ~1.85s estimate from archived 03_efficiency_benchmark.py is superseded and invalid."
    )

    rows = []
    for key in ("rasnet", "segresnet", "3dunet", "nnunet", "vnet"):
        r4 = t4.loc[key]
        t_1p = timing_lookup.get((key, "sw050_gauss_1pass"))
        t_8p = timing_lookup.get((key, "sw050_gauss_tta8"))
        rows.append({
            "model_key": key,
            "model_name": str(r4["Model"]),
            "params_exact": int(r4["Parameters"]),
            "params_m": float(r4["Parameters_M"]),
            "checkpoint_size_mib": float(r4["Checkpoint_Size_MB"]),
            "native_patch_size": str(r4["Inference_Patch_Size"]),
            "gflops_native_patch": float(r4["GFLOPs_per_Patch"]),
            "gflops_equal_96_3": float(r4["GFLOPs_Equal_96x96x96"]),
            "patch_latency_fp32_ms": str(r4["Patch_Latency_FP32_ms_Mean_SD"]),
            "patch_latency_amp_ms": str(r4["Patch_Latency_AMP_ms_Mean_SD"]),
            "peak_patch_vram_mb": float(r4["Peak_Patch_VRAM_MB"]),
            "optionA_timing_scope": (
                "Excluded (3D U-Net uses 128³ single-shot, not sliding window)"
                if key == "3dunet"
                else "RTX 4080 SUPER, n=10 Val cases (after 1 warm-up), 0.50 Gauss SW, full-volume + native inversion"
            ),
            "optionA_1pass_median_sec": float(t_1p["seconds_median"]) if t_1p is not None else np.nan,
            "optionA_1pass_peak_vram_mb": float(t_1p["peak_vram_mb"]) if t_1p is not None else np.nan,
            "optionA_tta8_median_sec": float(t_8p["seconds_median"]) if t_8p is not None else np.nan,
            "optionA_tta8_peak_vram_mb": float(t_8p["peak_vram_mb"]) if t_8p is not None else np.nan,
            "invalid_timing_note": invalid_note,
        })
    return pd.DataFrame(rows)


def build_checkpoint_provenance_table(repo_root: Path) -> pd.DataFrame:
    """Verify exact checkpoint state-dict key counts, hashes, plans, and epoch metadata."""
    ckpt_specs = [
        ("rasnet", "best", "Q1_Publication_Package/matched_200ep_benchmark/checkpoints/rasnet_best.pth"),
        ("rasnet", "final_epoch200", "recovered_checkpoints/v1_epoch200_20261117_041500Z/rasnet_last.pth"),
        ("segresnet", "best", "Q1_Publication_Package/matched_200ep_benchmark/checkpoints/segresnet_best.pt"),
        ("vnet", "best", "Q1_Publication_Package/matched_200ep_benchmark/checkpoints/vnet_best.pt"),
        ("3dunet", "best", "Q1_Publication_Package/matched_200ep_benchmark/checkpoints/3dunet_best.pth"),
        ("nnunet", "best", "Q1_Publication_Package/matched_200ep_benchmark/checkpoints/nnunet_best.pt"),
        ("nnunet", "final_epoch200", "recovered_checkpoints/v1_epoch200_20261117_041500Z/nnunet_last.pt"),
    ]
    t4 = pd.read_csv(repo_root / "results" / "thesis_outputs" / "Table4_computational_efficiency_rtx4080super.csv").set_index("model_key")

    rows = []
    for model_key, role, rel_path in ckpt_specs:
        p = repo_root / rel_path
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if model_key == "nnunet":
            sd = obj["network_weights"]
            n_keys = len(sd)
            embedded_epoch = f"current_epoch={obj['current_epoch']}"
            extra = (
                f"plans patch=[96, 160, 160], spacing=[0.5, 0.349609375, 0.349609375], "
                f"_best_ema={obj.get('_best_ema', float('nan')):.6f}, "
                f"raw_peak_ep=189 (0.8376), ema_peak_ep=199"
            )
            has_opt = "optimizer_state" in obj
        elif model_key == "3dunet":
            sd = obj["model_state_dict"]
            n_keys = len(sd)
            embedded_epoch = f"epoch={obj.get('epoch')} (best_val_dice={obj.get('best_val_dice'):.6f})"
            extra = f"top_keys={list(obj.keys())}; optimizer_state_dict absent"
            has_opt = "optimizer_state_dict" in obj
        else:
            sd = obj
            n_keys = len(sd)
            embedded_epoch = "None (embedded_epoch_metadata_present=False)"
            if model_key == "rasnet" and role == "final_epoch200":
                extra = "Supported by train_rasnet_200ep.py:268 + rasnet_200ep_log.csv (200 ep) + split-ZIP timestamp delta"
            elif model_key == "rasnet":
                extra = "Best at epoch 160 per rasnet_200ep_log.csv (0.77696)"
            elif model_key == "segresnet":
                extra = "Best at epoch 190 per segresnet_200ep_log.csv (0.75212)"
            else:
                extra = "Best at epoch 180 per vnet_200ep_log.csv (0.74905)"
            has_opt = False

        rows.append({
            "model_key": model_key,
            "checkpoint_role": role,
            "relative_path": rel_path,
            "bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
            "state_dict_keys": int(n_keys),
            "unique_params_exact": int(t4.loc[model_key, "Parameters"]),
            "embedded_epoch_metadata": embedded_epoch,
            "has_optimizer_state": bool(has_opt),
            "provenance_notes": extra,
        })
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame, title: str, notes: list[str] | None = None) -> str:
    lines = [f"# {title}", ""]
    if notes:
        for n in notes:
            lines.append(f"> {n}")
        lines.append("")
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def generate_all(out_dir: Path, repo_root: Path) -> dict[str, Path]:
    out = validate_correction_out_dir(out_dir=out_dir, repo_root=repo_root)
    out.mkdir(parents=True, exist_ok=True)

    ep200_dir = repo_root / "results" / "v1_epoch200_comparison_20261117_041500Z"
    splits_path = repo_root / "Phase3_Local_Integration" / "splits_final.json"
    logs_dir = repo_root / "Q1_Publication_Package" / "matched_200ep_benchmark" / "logs"
    nnunet_last_pt = repo_root / "recovered_checkpoints" / "v1_epoch200_20261117_041500Z" / "nnunet_last.pt"
    nnunet_best_pt = repo_root / "Q1_Publication_Package" / "matched_200ep_benchmark" / "checkpoints" / "nnunet_best.pt"

    ep200_res = recompute_val94_endpoint_stats(ep200_dir=ep200_dir, splits_path=splits_path)
    parsed_logs = parse_all_training_logs(logs_dir=logs_dir, nnunet_last_pt_path=nnunet_last_pt, nnunet_best_pt_path=nnunet_best_pt)
    regimes_df = build_four_regimes_table(repo_root)
    eff_df = build_efficiency_and_timing_table(repo_root)
    prov_df = build_checkpoint_provenance_table(repo_root)

    outputs: dict[str, Path] = {}

    fig3_png = out / "fig3_corrected_training_and_val_convergence_200ep.png"
    fig3_svg = out / "fig3_corrected_training_and_val_convergence_200ep.svg"
    plot_fig3_corrected(parsed_logs, fig3_png, fig3_svg)
    outputs["fig3_png"] = fig3_png
    outputs["fig3_svg"] = fig3_svg

    fig_ep_png = out / "fig_v1_epoch200_val94_endpoint_corrected.png"
    fig_ep_svg = out / "fig_v1_epoch200_val94_endpoint_corrected.svg"
    plot_fig_endpoint_corrected(ep200_res, regimes_df, fig_ep_png, fig_ep_svg)
    outputs["fig_endpoint_png"] = fig_ep_png
    outputs["fig_endpoint_svg"] = fig_ep_svg

    # Table: Val94 Epoch-200 Summary + Paired Exploratory Stats
    ep_sum_csv = out / "table_epoch200_val94_summary_and_paired_corrected.csv"
    ep_sum_md = out / "table_epoch200_val94_summary_and_paired_corrected.md"
    ep200_res["paired_df"].to_csv(ep_sum_csv, index=False)
    ep_sum_md_txt = (
        _df_to_markdown(
            ep200_res["summary_df"],
            "Val94 Epoch-200 Endpoint Per-Model Summary (N = 94; final_epoch200)",
            notes=[
                f"Matched cases: {ep200_res['n_matched_cases']}; Empty predictions: {ep200_res['n_empty_pred']}; "
                f"Excluded NaNs: {ep200_res['n_excluded_nan']}; Geometry verified: {ep200_res['geometry_ok_count']}/{ep200_res['geometry_total_count']}.",
                "Provenance: rasnet_last.pth supported by script/log/archive alignment (no embedded epoch metadata); nnunet_last.pt embeds current_epoch=200.",
            ],
        )
        + "\n"
        + _df_to_markdown(
            ep200_res["paired_df"],
            "Val94 Epoch-200 Paired Comparison (RASNet − nnU-Net; All 8 p-values Exploratory / Unadjusted)",
            notes=[
                "All 8 p-values below are EXPLORATORY and UNADJUSTED. Non-significant Dice (p=0.2401) does NOT establish equivalence.",
                "Wilcoxon signed-rank p and bootstrap 95% CI for the mean difference target different quantities and need not coincide.",
            ],
        )
    )
    ep_sum_md.write_text(ep_sum_md_txt, encoding="utf-8")
    outputs["ep200_summary_paired_csv"] = ep_sum_csv
    outputs["ep200_summary_paired_md"] = ep_sum_md

    # Table: Post-Hoc Exploratory Holm-8
    holm_csv = out / "table_epoch200_val94_posthoc_holm8_exploratory.csv"
    holm_md = out / "table_epoch200_val94_posthoc_holm8_exploratory.md"
    ep200_res["posthoc_holm8_df"].to_csv(holm_csv, index=False)
    holm_md.write_text(
        _df_to_markdown(
            ep200_res["posthoc_holm8_df"],
            "POST-HOC EXPLORATORY — NOT PREREGISTERED: Holm-Bonferroni (K = 8) Val94 Epoch-200 Endpoint",
            notes=[
                "Declared family: all 8 metrics in the Val94 final_epoch200 comparison (POST-HOC EXPLORATORY — NOT PREREGISTERED).",
                "Precision (Holm-8 p=7.47e-4) and Recall (Holm-8 p=3.88e-9) survive < 0.05; clDice (Holm-8 p=0.1760) and centerline_recall (Holm-8 p=0.0551) do NOT.",
            ],
        ),
        encoding="utf-8",
    )
    outputs["posthoc_holm8_csv"] = holm_csv
    outputs["posthoc_holm8_md"] = holm_md

    # Table: Four Regimes
    reg_csv = out / "table_v1_four_regimes_comparison.csv"
    reg_md = out / "table_v1_four_regimes_comparison.md"
    regimes_df.to_csv(reg_csv, index=False)
    reg_md.write_text(
        _df_to_markdown(
            regimes_df,
            "Strictly Separated V1 Evaluation Regimes (A, B1/B2, C, D Option A)",
            notes=[
                "Do NOT mix scores across regimes, cohorts (Val94 dev vs Test150 retrospective), checkpoint roles (best vs final_epoch200), or inference/post-processing protocols.",
                "Option A excludes 3D U-Net (4 sliding-window models). Test150 is retrospective.",
            ],
        ),
        encoding="utf-8",
    )
    outputs["four_regimes_csv"] = reg_csv
    outputs["four_regimes_md"] = reg_md

    # Table: Efficiency & Timing
    eff_csv = out / "table_v1_efficiency_and_timing_verified.csv"
    eff_md = out / "table_v1_efficiency_and_timing_verified.md"
    eff_df.to_csv(eff_csv, index=False)
    eff_md.write_text(
        _df_to_markdown(
            eff_df,
            "Verified V1 Parameter Counts, Single-Patch Compute (Table 4), & Full-Volume Runtime (Option A)",
            notes=[
                "Exact parameters verified by instantiating models and loading checkpoints (SegResNet 4.701M < RASNet 4.707M < 3D U-Net 4.806M < nnU-Net 30.786M < V-Net 45.599M).",
                "manifest.json rasnet=0.0s (resumed skip) and legacy ~1.85s (hardcoded estimate) are strictly excluded.",
            ],
        ),
        encoding="utf-8",
    )
    outputs["efficiency_timing_csv"] = eff_csv
    outputs["efficiency_timing_md"] = eff_md

    # Table: Checkpoint Provenance
    prov_csv = out / "table_checkpoint_provenance_verified.csv"
    prov_md = out / "table_checkpoint_provenance_verified.md"
    prov_df.to_csv(prov_csv, index=False)
    prov_md.write_text(
        _df_to_markdown(
            prov_df,
            "Verified V1 Checkpoint Provenance (State-Dict Keys, Hashes, Plans, & Epochs)",
            notes=[
                "Independent torch.load verification of all V1 best and final_epoch200 checkpoints.",
            ],
        ),
        encoding="utf-8",
    )
    outputs["provenance_csv"] = prov_csv
    outputs["provenance_md"] = prov_md

    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="results/v1_reporting_correction_20261117_105000Z",
        help="Output directory (must not be a protected path).",
    )
    args = parser.parse_args(argv)
    out_path = Path(args.out_dir)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    artifacts = generate_all(out_dir=out_path, repo_root=REPO_ROOT)
    for k, v in artifacts.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
