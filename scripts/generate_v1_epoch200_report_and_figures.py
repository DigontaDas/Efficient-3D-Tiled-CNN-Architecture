"""Generate paired case-level CSV, separated best-vs-final table, endpoint figure, and thesis replacement text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.endpoint_eval import PROTECTED_DIRS, validate_endpoint_out_dir
from benchmark.run_benchmark import ALL_METRICS
from benchmark.stats import paired_analysis


def validate_report_out_dir(out_dir: Path, repo_root: Path) -> Path:
    """Ensure report outputs never write into protected directories."""
    return validate_endpoint_out_dir(out_dir=out_dir, repo_root=repo_root, resume=True)


def build_paired_case_level_table(
    df_rasnet: pd.DataFrame,
    df_nnunet: pd.DataFrame,
    tie_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Build sorted case-level paired RASNet-minus-nnU-Net table across all metrics."""
    if df_rasnet.empty or df_nnunet.empty:
        raise ValueError("Input per-case DataFrames cannot be empty")
    if df_rasnet["case_id"].isna().any() or df_nnunet["case_id"].isna().any():
        raise ValueError("NaN case_id found in per-case inputs")
    if df_rasnet["case_id"].duplicated().any() or df_nnunet["case_id"].duplicated().any():
        raise ValueError("Duplicate case_id found in per-case inputs")
    if set(df_rasnet["case_id"]) != set(df_nnunet["case_id"]):
        raise ValueError("Case ID sets differ between RASNet and nnU-Net inputs")

    r = df_rasnet.sort_values("case_id").reset_index(drop=True)
    n = df_nnunet.sort_values("case_id").reset_index(drop=True)
    merged = r.merge(n, on="case_id", validate="one_to_one", suffixes=("_rasnet", "_nnunet"))

    out = pd.DataFrame({"case_id": merged["case_id"].astype(int)})
    out["rasnet_status"] = merged["status_rasnet"]
    out["nnunet_status"] = merged["status_nnunet"]
    for m in ALL_METRICS:
        out[f"rasnet_{m}"] = merged[f"{m}_rasnet"].astype(float)
        out[f"nnunet_{m}"] = merged[f"{m}_nnunet"].astype(float)
        out[f"{m}_diff_rasnet_minus_nnunet"] = out[f"rasnet_{m}"] - out[f"nnunet_{m}"]

    tol = float(tie_tolerance)
    outcomes = []
    for d in out["dice_diff_rasnet_minus_nnunet"]:
        if np.isnan(d):
            outcomes.append("excluded_nan")
        elif abs(d) <= tol:
            outcomes.append("tie")
        elif d > 0:
            outcomes.append("rasnet_win")
        else:
            outcomes.append("nnunet_win")
    out["dice_outcome"] = outcomes
    out["tie_tolerance"] = tol
    return out


def load_existing_best_references(repo_root: Path) -> list[dict]:
    """Load existing verified Val94 best-checkpoint summaries with explicit recipe tags."""
    rows: list[dict] = []

    val_arm_a_p = repo_root / "results" / "benchmark_runs" / "val_full_94" / "summary_arm_A.csv"
    if val_arm_a_p.is_file():
        df_a = pd.read_csv(val_arm_a_p)
        epoch_map = {
            "rasnet": "epoch 160 (rasnet_200ep_log.csv)",
            "nnunet": "current_epoch=199 (best_ema=0.8162; raw mean_fg_dice peak ep 189=0.8376)",
        }
        sw_map = {
            "rasnet": "1-pass, 0.25 constant SW, tau=0.50",
            "nnunet": "1-pass, 0.50 Gaussian SW, tau=0.50",
        }
        for model in ("rasnet", "nnunet"):
            sub = df_a[df_a["model"] == model]
            if not sub.empty:
                s = sub.iloc[0]
                rows.append({
                    "panel": "B_existing_best_reference",
                    "comparison_group": "Val94 Arm A (best; asymmetric SW 0.25 const vs 0.50 gauss, no CC)",
                    "model": model,
                    "checkpoint_role": "best",
                    "epoch_record": epoch_map[model],
                    "cohort": "Val94 (N=94)",
                    "sw_recipe": sw_map[model],
                    "postprocess": "none",
                    "n_cases": int(s["n_cases"]),
                    "dice_mean": float(s["dice_mean"]),
                    "dice_std": float(s["dice_std"]),
                    "precision_mean": float(s["precision_mean"]),
                    "recall_mean": float(s["recall_mean"]),
                    "hd95_mean": float(s["hd95_mean"]),
                    "asd_mean": float(s["asd_mean"]),
                    "cldice_mean": float(s["cldice_mean"]),
                    "centerline_recall_mean": float(s["centerline_recall_mean"]),
                })

    opt_grid_p = repo_root / "results" / "optionA_v1_symmetric_sw050" / "val_grid_scores.csv"
    if opt_grid_p.is_file():
        df_opt = pd.read_csv(opt_grid_p)
        for model, epoch_str in (
            ("rasnet", "epoch 160 (rasnet_200ep_log.csv)"),
            ("nnunet", "current_epoch=199 (best_ema=0.8162; raw mean_fg_dice peak ep 189=0.8376)"),
        ):
            sub = df_opt[
                (df_opt["model"] == model)
                & (df_opt["mode"] == "sw050_gauss_1pass")
                & (np.isclose(df_opt["tau"], 0.50))
            ]
            if not sub.empty:
                vals = sub["dice"].to_numpy(dtype=float)
                rows.append({
                    "panel": "B_existing_best_reference",
                    "comparison_group": "Option A Val94 (best; 0.50 gauss SW, tau=0.50, WITH Top-2 CC)",
                    "model": model,
                    "checkpoint_role": "best",
                    "epoch_record": epoch_str,
                    "cohort": "Val94 (N=94)",
                    "sw_recipe": "1-pass, 0.50 Gaussian SW, tau=0.50",
                    "postprocess": "26-connected Top-2 CC",
                    "n_cases": int(len(vals)),
                    "dice_mean": float(np.mean(vals)),
                    "dice_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "precision_mean": np.nan,
                    "recall_mean": np.nan,
                    "hd95_mean": np.nan,
                    "asd_mean": np.nan,
                    "cldice_mean": np.nan,
                    "centerline_recall_mean": np.nan,
                })

    return rows


def build_separated_comparison_table(
    summary_ep200: pd.DataFrame,
    best_ref_rows: list[dict],
) -> pd.DataFrame:
    """Combine Panel A (epoch-200) and Panel B (existing best refs) without merging unmatched recipes."""
    panel_a_rows = []
    for _, s in summary_ep200.iterrows():
        model = str(s["model"])
        epoch_str = (
            "epoch 200 (script/log/archive provenance; no embedded ckpt metadata)"
            if model == "rasnet"
            else "current_epoch=200 (embedded ckpt metadata)"
        )
        panel_a_rows.append({
            "panel": "A_epoch200_endpoint",
            "comparison_group": "Epoch-200 Endpoint Val94 (final_epoch200; symmetric 0.50 gauss SW, no CC)",
            "model": model,
            "checkpoint_role": "final_epoch200",
            "epoch_record": epoch_str,
            "cohort": f"Val94 (N={int(s['n_cases'])})",
            "sw_recipe": "1-pass, 0.50 Gaussian SW, tau=0.50",
            "postprocess": "none",
            "n_cases": int(s["n_cases"]),
            "dice_mean": float(s["dice_mean"]),
            "dice_std": float(s["dice_std"]),
            "precision_mean": float(s["precision_mean"]),
            "recall_mean": float(s["recall_mean"]),
            "hd95_mean": float(s["hd95_mean"]),
            "asd_mean": float(s["asd_mean"]),
            "cldice_mean": float(s["cldice_mean"]),
            "centerline_recall_mean": float(s["centerline_recall_mean"]),
        })

    for r in best_ref_rows:
        if r.get("panel") != "B_existing_best_reference":
            raise ValueError(
                f"Panel B reference rows must explicitly set panel='B_existing_best_reference', got {r.get('panel')!r}"
            )

    return pd.DataFrame(panel_a_rows + list(best_ref_rows))


def _fmt(val: float, digits: int = 4) -> str:
    if pd.isna(val):
        return "—"
    return f"{float(val):.{digits}f}"


def render_separated_table_markdown(table_df: pd.DataFrame, paired_json: dict) -> str:
    """Render separated Panel A and Panel B markdown tables."""
    lines = [
        "# V1 Epoch-200 Endpoint Comparison vs. Existing Best-Checkpoint References (Val94)",
        "",
        "> **Evidence Separation Rule**: Panel A (`final_epoch200`, `1-pass, 0.50 Gaussian SW, tau=0.50, no CC`) "
        "and Panel B (existing `best` checkpoint runs at differing epochs and differing inference/post-processing recipes) "
        "are kept strictly distinct. Do **not** merge rows across panels into a single ranking or infer a training trajectory across unmatched recipes.",
        "",
        "## Panel A — Newly Verified Epoch-200 Endpoint Comparison (Val N = 94; Symmetric 1-Pass, 0.50 Gaussian SW, τ = 0.50, No CC)",
        "",
        "| Model | Role | Epoch Provenance | N | Dice (Mean ± SD) | Precision | Recall | HD95 (mm) | ASD (mm) | clDice | Centerline Recall |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    a_df = table_df[table_df["panel"] == "A_epoch200_endpoint"]
    for _, r in a_df.iterrows():
        lines.append(
            f"| **{r['model']}** | `{r['checkpoint_role']}` | {r['epoch_record']} | {int(r['n_cases'])} | "
            f"`{_fmt(r['dice_mean'])} ± {_fmt(r['dice_std'])}` | `{_fmt(r['precision_mean'])}` | `{_fmt(r['recall_mean'])}` | "
            f"`{_fmt(r['hd95_mean'], 2)}` | `{_fmt(r['asd_mean'], 3)}` | `{_fmt(r['cldice_mean'])}` | `{_fmt(r['centerline_recall_mean'])}` |"
        )

    dice_row = next(m for m in paired_json["metrics"] if m["metric"] == "dice")
    lines.extend([
        "",
        f"- **Paired Dice Difference (`RASNet − nnU-Net`, N={dice_row['n']}):** "
        f"`{dice_row['mean_difference']:+.4f}` (95% bootstrap CI `[{dice_row['diff_ci95_lo']:+.4f}, {dice_row['diff_ci95_hi']:+.4f}]`, "
        f"seed `{paired_json['seed']}`, resamples `{paired_json['n_boot']}`; median `{dice_row['median_difference']:+.4f}`, "
        f"exploratory unadjusted Wilcoxon `p = {dice_row['p']:.4f}`, W/T/L = `{dice_row['wins']}/{dice_row['ties']}/{dice_row['losses']}` with tie tolerance `{dice_row['tie_tolerance']}`).",
        "",
        "## Panel B — Existing Best-Checkpoint Val94 References (Strictly Separate Recipes & Selected Epochs)",
        "",
        "| Comparison Group | Model | Role | Epoch Record | SW / Preprocessing Recipe | Post-Processing | N | Dice (Mean ± SD) | Precision | Recall | HD95 (mm) | clDice |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    b_df = table_df[table_df["panel"] == "B_existing_best_reference"]
    for _, r in b_df.iterrows():
        lines.append(
            f"| {r['comparison_group']} | **{r['model']}** | `{r['checkpoint_role']}` | {r['epoch_record']} | "
            f"{r['sw_recipe']} | {r['postprocess']} | {int(r['n_cases'])} | "
            f"`{_fmt(r['dice_mean'])} ± {_fmt(r['dice_std'])}` | `{_fmt(r['precision_mean'])}` | `{_fmt(r['recall_mean'])}` | "
            f"`{_fmt(r['hd95_mean'], 2)}` | `{_fmt(r['cldice_mean'])}` |"
        )
    lines.append("")
    return "\n".join(lines)


def plot_epoch200_comparison_figure(
    paired_case_df: pd.DataFrame,
    summary_ep200: pd.DataFrame,
    paired_json: dict,
    ref_rows: list[dict],
    png_path: Path,
    svg_path: Path,
) -> None:
    """Create 3-panel endpoint figure without fabricating or interpolating trajectories."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=300)
    ax1, ax2, ax3 = axes

    c_ras = "#d95f02"
    c_nnu = "#1f78b4"

    # Panel A: Epoch-200 Per-Case Dice Distributions (Val94)
    r_vals = paired_case_df["rasnet_dice"].to_numpy(dtype=float)
    n_vals = paired_case_df["nnunet_dice"].to_numpy(dtype=float)
    bp = ax1.boxplot(
        [r_vals, n_vals],
        positions=[0, 1],
        widths=0.45,
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
    ax1.scatter([0, 1], [np.mean(r_vals), np.mean(n_vals)], color=["#8c2d04", "#084594"], marker="D", s=45, zorder=4, label="Mean")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([
        f"RASNet (ep 200)\n{np.mean(r_vals):.4f} ± {np.std(r_vals, ddof=1):.4f}",
        f"nnU-Net V2 (ep 200)\n{np.mean(n_vals):.4f} ± {np.std(n_vals, ddof=1):.4f}",
    ], fontsize=9.5)
    ax1.set_ylabel("Full-Volume Native-Grid Dice", fontweight="bold")
    ax1.set_title("A. Epoch-200 Val94 Endpoint\n(1-Pass, 0.50 Gauss SW, τ=0.50, No CC)", fontweight="bold", fontsize=10.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    # Panel B: Paired Case-Level Differences (RASNet - nnU-Net) at Epoch 200
    diffs = paired_case_df["dice_diff_rasnet_minus_nnunet"].to_numpy(dtype=float)
    sorted_idx = np.arange(len(diffs))
    sorted_diffs = np.sort(diffs)
    bar_cols = [c_ras if d > 0 else c_nnu for d in sorted_diffs]
    ax2.bar(sorted_idx, sorted_diffs, color=bar_cols, width=0.85, alpha=0.82)
    ax2.axhline(0.0, color="#222222", linewidth=1.1, linestyle="-")

    dice_stat = next(m for m in paired_json["metrics"] if m["metric"] == "dice")
    prec_stat = next(m for m in paired_json["metrics"] if m["metric"] == "precision")
    rec_stat = next(m for m in paired_json["metrics"] if m["metric"] == "recall")
    clrec_stat = next(m for m in paired_json["metrics"] if m["metric"] == "centerline_recall")
    mean_d = dice_stat["mean_difference"]
    lo_d = dice_stat["diff_ci95_lo"]
    hi_d = dice_stat["diff_ci95_hi"]
    ax2.axhline(mean_d, color="#111111", linewidth=1.4, linestyle="--")
    ax2.axhspan(lo_d, hi_d, color="#666666", alpha=0.14)
    ax2.set_xlabel(f"Val94 Cases Sorted by Paired Diff (N={len(diffs)})", fontweight="bold")
    ax2.set_ylabel("Paired Dice Diff (RASNet − nnU-Net)", fontweight="bold")
    ax2.set_title(
        f"B. Paired Epoch-200 Difference\nMean {mean_d:+.4f} (95% CI [{lo_d:+.4f}, {hi_d:+.4f}]), unadj. p={dice_stat['p']:.3f}",
        fontweight="bold",
        fontsize=10.5,
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    note_txt = (
        f"Exploratory unadjusted p-values:\n"
        f"Dice W/T/L: {dice_stat['wins']}/{dice_stat['ties']}/{dice_stat['losses']}\n"
        f"Precision Δ: {prec_stat['mean_difference']:+.4f} (unadj p={prec_stat['p']:.1e})\n"
        f"Recall Δ: {rec_stat['mean_difference']:+.4f} (unadj p={rec_stat['p']:.1e})\n"
        f"CL-Recall Δ: {clrec_stat['mean_difference']:+.4f} (unadj p={clrec_stat['p']:.3f})"
    )
    ax2.text(
        0.03, 0.97, note_txt,
        transform=ax2.transAxes,
        va="top", ha="left",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    # Panel C: Strictly Separate Existing Best-Checkpoint Val94 Runs
    arm_a_r = next((x for x in ref_rows if "Arm A" in x["comparison_group"] and x["model"] == "rasnet"), None)
    arm_a_n = next((x for x in ref_rows if "Arm A" in x["comparison_group"] and x["model"] == "nnunet"), None)
    opt_r = next((x for x in ref_rows if "Option A" in x["comparison_group"] and x["model"] == "rasnet"), None)
    opt_n = next((x for x in ref_rows if "Option A" in x["comparison_group"] and x["model"] == "nnunet"), None)

    labels_c = []
    vals_c = []
    errs_c = []
    cols_c = []
    for lbl, item, col in (
        ("Arm A Best\nRASNet (ep 160)\n[0.25 const,\nno CC]", arm_a_r, c_ras),
        ("Arm A Best\nnnU-Net (ep 199*)\n[0.50 Gauss,\nno CC]", arm_a_n, c_nnu),
        ("Opt-A Best\nRASNet (ep 160)\n[0.50 Gauss,\n+Top-2 CC]", opt_r, c_ras),
        ("Opt-A Best\nnnU-Net (ep 199*)\n[0.50 Gauss,\n+Top-2 CC]", opt_n, c_nnu),
    ):
        if item is not None:
            labels_c.append(lbl)
            vals_c.append(item["dice_mean"])
            errs_c.append(item["dice_std"])
            cols_c.append(col)

    x_pos = np.arange(len(labels_c))
    if len(labels_c) > 0:
        ax3.bar(x_pos, vals_c, yerr=errs_c, color=cols_c, alpha=0.75, capsize=4, width=0.52)
        for xp, v, e in zip(x_pos, vals_c, errs_c):
            ax3.text(xp, v + e + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    if len(labels_c) >= 4:
        ax3.axvline(1.5, color="#888888", linestyle=":", linewidth=1.2)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels_c, fontsize=7.8)
    ax3.set_ylim(0.65, 0.90)
    ax3.set_ylabel("Mean Val94 Dice (± SD)", fontweight="bold")
    ax3.set_title(
        "C. Separate Reference: Existing Best Runs\n(Distinct Epochs & Unmatched Recipes; Not a Trajectory)",
        fontweight="bold",
        fontsize=10.2,
    )
    ax3.grid(axis="y", linestyle="--", alpha=0.45)

    fig.suptitle(
        "V1 Epoch-200 Full-Volume Val94 Endpoint Comparison (RASNet vs. nnU-Net V2) & Separate Best-Checkpoint Context",
        fontsize=12.0,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.02,
        "*nnU-Net best checkpoint carries current_epoch=199 (_best_ema=0.8162); unsmoothed batch Pseudo-Dice peaked at ep 189 (0.8376). "
        "No intermediate epochs (50/100/150) are plotted or interpolated.",
        ha="center",
        va="top",
        fontsize=8.3,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def render_thesis_curve_and_endpoint_text(
    summary_ep200: pd.DataFrame,
    paired_json: dict,
    ref_rows: list[dict],
    manifest: dict,
) -> str:
    """Draft objective replacement thesis & defense text without post-hoc or causal fallacies."""
    r_row = summary_ep200[summary_ep200["model"] == "rasnet"].iloc[0]
    n_row = summary_ep200[summary_ep200["model"] == "nnunet"].iloc[0]
    by_m = {m["metric"]: m for m in paired_json["metrics"]}
    d = by_m["dice"]
    p_m = by_m["precision"]
    rec_m = by_m["recall"]
    cl_rec_m = by_m["centerline_recall"]
    hd_m = by_m["hd95"]
    asd_m = by_m["asd"]
    cl_m = by_m["cldice"]

    return "\n".join([
        "# Corrected Thesis & Defense Text: Training Curves vs. Epoch-200 Full-Volume Val94 Endpoint",
        "",
        "> **Integration Status (`OUTSTANDING`):**",
        "> - Protected manuscript file [`results/thesis_outputs/THESIS_METHODS_RESULTS_AND_LIMITATIONS.md`](../../results/thesis_outputs/THESIS_METHODS_RESULTS_AND_LIMITATIONS.md) and [`results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.svg`](../../results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.svg) are preserved untouched per repository protection rules.",
        "> - [`Answers-thesis-def.md`](../../Answers-thesis-def.md) was not overwritten, and external thesis manuscript/slide sources (`.tex`/`.docx`/`.pptx`) are not present in this repository. Integrating the replacement paragraphs below into the external thesis manuscript/defense deck remains **OUTSTANDING**.",
        "",
        "---",
        "",
        "## 1. Why the Original Logged Training Curves Are Not Directly Comparable",
        "",
        "In `Q1_Publication_Package/matched_200ep_benchmark/`, the logged validation curves plotted in `results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.svg` measure **different quantities under different protocols**:",
        "- **RASNet (`train_rasnet_200ep.py`):** Full-volume sliding-window validation Dice (`96x96x96`, `overlap = 0.25`) across all `N = 94` validation scans (`Val94`) on the `0.5 mm` isotropic foreground-cropped grid at **12** sparse checkpoints (`every 20 epochs` plus `50, 100, 150, 200`), aggregated as mean full-volume Dice (`DiceMetric(include_background=False)`).",
        "- **SegResNet (`train_segresnet_200ep.py`):** Full-volume sliding-window validation Dice (`96x96x96`, `overlap = 0.25`) across `Val94` (`N = 94`) on the `0.5 mm` isotropic grid at **20** checkpoints (`every 10 epochs`), trained with `DiceFocalLoss(lambda_dice=1.0, lambda_focal=2.0)`.",
        "- **V-Net (`train_vnet_200ep.py`):** Full-volume sliding-window validation Dice (`96x96x96`, `overlap = 0.25`) across `Val94` (`N = 94`) on the `0.5 mm` isotropic grid at **13** checkpoints logged (`[10, 20, 40, 50, 60, 80, 100, 120, 140, 150, 160, 180, 200]` — epoch 10 prior to auto-resume, then `every 20` or `50`), trained with `DiceFocalLoss(lambda_dice=1.0, lambda_focal=2.0)`.",
        "- **3D U-Net (`train_3dunet_200ep.py`):** Single-shot validation Dice on `Val94` (`N = 94`) volumes resized to `128x128x128` (`128³`, no sliding window, non-0.5 mm grid) at **21** checkpoints (`1, 10, 20, ..., 200`), trained with `DiceCELoss`.",
        "- **nnU-Net V2 (`nnUNetTrainer_200epochs`):** Logged at **200** epochs (`every epoch 1..200`) over `50` validation batches (`batch_size=2`) of `96x160x160` patches (`[0.5, 0.3496, 0.3496] mm`, `CTNormalization`) with `33.3%` forced foreground oversampling and batch-pooled `mean_fg_dice` (raw unsmoothed `Pseudo dice`, peaking at epoch `189` = `0.8376`), smoothed by `MetaLogger` into `ema_fg_dice` (which selects `checkpoint_best.pth` at `current_epoch = 199` with `_best_ema = 0.8162`).",
        "- **Loss axis note:** Panel A of `fig3` also plots non-comparable training objectives (`StenosisAwareLoss` positive values vs. nnU-Net negative `DC_and_CE_loss`; and note that Panel A's historical legend mislabels SegResNet and V-Net as `DiceCE` even though `train_segresnet_200ep.py` and `train_vnet_200ep.py` actually trained with `DiceFocalLoss(lambda_dice=1.0, lambda_focal=2.0)`).",
        "- The logged metrics use different sampling, aggregation and spatial evaluation protocols and therefore cannot establish comparative full-volume performance. Because intermediate epoch checkpoints (`50`, `100`, `150`) are absent in the recovered archive, intermediate trajectories on a common full-volume protocol were not measured.",
        "",
        "---",
        "",
        "## 2. What the Completed Epoch-200 Endpoint Evaluation Actually Shows (Val94, N = 94)",
        "",
        f"Evaluating both recovered epoch-200 final checkpoints (`rasnet_last.pth` via script/log/archive provenance, `nnunet_last.pt` with embedded `current_epoch = 200`) on the exact same `N = 94` native-grid validation scans under a frozen symmetric 1-pass protocol (`0.50` Gaussian sliding-window overlap, threshold `tau = 0.50`, no TTA, no connected-component pruning; `run_id = {manifest.get('run_id')}`):",
        "",
        "> **Statistical Qualification (Exploratory / Unadjusted 8-Metric Table):** All eight metric p-values reported in `paired_stats_epoch200.csv` are **exploratory** and **unadjusted** for multiple comparisons. Moreover, the two-sided **Wilcoxon** signed-rank test (rank-based) and the **bootstrap CI** for the mean paired difference target **different quantities** (paired location of ranks vs. mean difference), so their conclusions need not coincide.",
        "",
        f"- **Primary descriptive endpoint (Dice):** RASNet achieved **`{r_row['dice_mean']:.4f} ± {r_row['dice_std']:.4f}`** (median `{r_row['dice_median']:.4f}`) and nnU-Net V2 achieved **`{n_row['dice_mean']:.4f} ± {n_row['dice_std']:.4f}`** (median `{n_row['dice_median']:.4f}`). The paired case-level difference (`RASNet − nnU-Net`) was **`{d['mean_difference']:+.4f}`** (95% bootstrap CI `[{d['diff_ci95_lo']:+.4f}, {d['diff_ci95_hi']:+.4f}]`, `54W / 0T / 40L`, exploratory unadjusted Wilcoxon `p = {d['p']:.4f}`). This difference is not statistically significant and **does not establish superiority or equivalence**.",
        f"- **Precision vs. Recall trade-off (including nnU-Net advantages):** Under this unpruned raw threshold (`tau = 0.50`), RASNet achieved higher **Precision** (`{r_row['precision_mean']:.4f}` vs. `{n_row['precision_mean']:.4f}`, paired diff `{p_m['mean_difference']:+.4f}`, 95% CI `[{p_m['diff_ci95_lo']:+.4f}, {p_m['diff_ci95_hi']:+.4f}]`, `66W / 0T / 28L`, exploratory unadjusted `p = {p_m['p']:.2e}`), whereas **nnU-Net V2 achieved higher voxel Recall (`{n_row['recall_mean']:.4f}` vs. `{r_row['recall_mean']:.4f}`, diff `{rec_m['mean_difference']:+.4f}`, 95% CI `[{rec_m['diff_ci95_lo']:+.4f}, {rec_m['diff_ci95_hi']:+.4f}]`, `16W / 0T / 78L`, exploratory unadjusted `p = {rec_m['p']:.2e}`) and higher Centerline Recall (`{n_row['centerline_recall_mean']:.4f}` vs. `{r_row['centerline_recall_mean']:.4f}`, diff `{cl_rec_m['mean_difference']:+.4f}`, 95% CI `[{cl_rec_m['diff_ci95_lo']:+.4f}, {cl_rec_m['diff_ci95_hi']:+.4f}]`, `37W / 2T / 55L`, exploratory unadjusted `p = {cl_rec_m['p']:.4f}`)**.",
        f"- **Surface and topological descriptors (with uncertainty):** Mean `HD95` (`{r_row['hd95_mean']:.2f}` vs. `{n_row['hd95_mean']:.2f} mm`, diff `{hd_m['mean_difference']:+.2f} mm`, 95% CI `[{hd_m['diff_ci95_lo']:+.2f}, {hd_m['diff_ci95_hi']:+.2f}]`, exploratory unadjusted `p = {hd_m['p']:.4f}`), `ASD` (`{r_row['asd_mean']:.3f}` vs. `{n_row['asd_mean']:.3f} mm`, diff `{asd_m['mean_difference']:+.3f} mm`, 95% CI `[{asd_m['diff_ci95_lo']:+.3f}, {asd_m['diff_ci95_hi']:+.3f}]`, exploratory unadjusted `p = {asd_m['p']:.4f}`), and `clDice` (`{r_row['cldice_mean']:.4f}` vs. `{n_row['cldice_mean']:.4f}`, diff `{cl_m['mean_difference']:+.4f}`, 95% CI `[{cl_m['diff_ci95_lo']:+.4f}, {cl_m['diff_ci95_hi']:+.4f}]`, exploratory unadjusted `p = {cl_m['p']:.4f}`) are mixed; `clDice`'s 95% bootstrap CI includes `0` and its unadjusted `p = 0.0352` does **not** survive post-hoc Holm-8 adjustment (`p = 0.1760`), so topology superiority is not established.",
        "- **Strict scope limits:** Because only the final epoch-200 endpoint was re-evaluated (and Val94 is a development cohort previously used during V1 development), these results do **not** show that RASNet outperformed nnU-Net throughout epochs `1–200`, do **not** imply equal training losses or absence of overfitting, and do **not** establish a causal effect of parameter count or capacity.",
        "",
        "---",
        "",
        "## 3. Ready-to-Paste Replacement Text for Thesis Section 2.1.1 & Defense Q&A",
        "",
        "### Replacement Paragraph for Thesis Results (§2.1.1)",
        "",
        "> **Reconciling Training-Log Validation Telemetry and the Completed Epoch-200 Endpoint (`N = 94`):**",
        f"> The training-log trajectories in Figure 3 reflect fundamentally different validation definitions: `train_rasnet_200ep.py` logged full-volume sliding-window Dice on the `0.5 mm` grid across all `94` validation scans at 12 checkpoints (`every 20 epochs` plus `50, 100, 150, 200`), `SegResNet` logged 20 full-volume evaluations (`DiceFocalLoss`), `V-Net` logged 13 (`DiceFocalLoss`), `3D U-Net` logged 21 single-shot evaluations on `128³` resized volumes (`DiceCELoss`), and `nnUNetTrainer_200epochs` logged patch-level `Pseudo dice` (`mean_fg_dice`, raw peak `0.8376` at epoch 189; `0.8145` at epoch 200) and `ema_fg_dice` (`_best_ema = 0.8162` at `current_epoch = 199`) over 50 foreground-oversampled `96x160x160` patches per epoch across all 200 epochs. The logged metrics use different sampling, aggregation and spatial evaluation protocols and therefore cannot establish comparative full-volume performance. When both recovered epoch-200 final checkpoints (`rasnet_last.pth` supported by script/log/archive provenance; `nnunet_last.pt` with embedded `current_epoch = 200`) are evaluated on the identical `94` full validation volumes under a single-pass, `0.50`-overlap Gaussian sliding-window protocol (`tau = 0.50`, no post-processing), both models reach comparable overall Dice (`0.7792 ± 0.0569` for RASNet vs. `0.7767 ± 0.0562` for nnU-Net V2; paired difference `+0.0025`, 95% bootstrap CI `[-0.0027, +0.0075]`, exploratory unadjusted Wilcoxon `p = 0.240`; not statistically significant, which does not establish superiority or equivalence). At this unpruned `tau = 0.50` operating point (where all 8 metric p-values are exploratory and unadjusted), RASNet exhibits higher Precision (`0.7586` vs. `0.7428`, exploratory unadjusted `p = 1.07e-4`), while nnU-Net V2 exhibits higher voxel Recall (`0.8203` vs. `0.8065`, exploratory unadjusted `p = 4.85e-10`) and Centerline Recall (`0.9142` vs. `0.9068`, exploratory unadjusted `p = 0.0092`). Because intermediate epoch checkpoints (`50`, `100`, `150`) are absent from the archive, this single-endpoint check characterizes epoch 200 only and does not reconstruct a full-volume trajectory across earlier epochs. Best-checkpoint comparisons (`rasnet_best.pth` from epoch 160; `nnunet_best.pt` carrying `current_epoch = 199` under `_best_ema`, vs. raw `Pseudo dice` peak at epoch 189) use separate evaluation configurations (`val_full_94` Arm A and `Option A` Val94) and are reported separately.",
        "",
        "### Defense Talking Points (Response to Examiner Question on Figure 3 Curves)",
        "",
        "- **Why did nnU-Net's logged curve differ from MONAI's validation Dice in Figure 3B?** The logged metrics use different sampling, aggregation and spatial evaluation protocols and therefore cannot establish comparative full-volume performance (nnU-Net logs batch-pooled `Pseudo dice` and `ema_fg_dice` over 50 validation batches of `96x160x160` patches at `[0.5, 0.3496, 0.3496] mm` with `33.3%` forced foreground oversampling, whereas MONAI models log full-volume Dice across `Val94` on `0.5 mm` or `128³` grids).",
        "- **What happens when we compare both epoch-200 checkpoints on the full `Val94` volumes?** Under symmetric single-pass `0.50` Gaussian sliding-window inference at `tau = 0.50` without pruning, full-volume Dice is `0.7792` (RASNet) vs. `0.7767` (nnU-Net), paired difference `+0.0025` (`95% CI [-0.0027, +0.0075]`, exploratory unadjusted `p = 0.240`). RASNet is higher on Precision (`+0.0158`), while nnU-Net is higher on Recall (`+0.0138`) and Centerline Recall (`+0.0074`).",
        "- **What can we NOT claim?** We cannot claim RASNet beat nnU-Net throughout training (no epoch 50/100/150 checkpoints exist), we cannot interpret `p = 0.240` as statistical equivalence, and we must keep best-checkpoint runs (`val_full_94` Arm A and `Option A` Val94 with Top-2 CC) distinct from the epoch-200 unpruned endpoint.",
        "",
    ])


def render_endpoint_report(
    out_dir: Path,
    manifest: dict,
    summary_ep200: pd.DataFrame,
    paired_json: dict,
    geom_df: pd.DataFrame,
    table_md: str,
) -> str:
    """Render concise factual summary of the completed epoch-200 endpoint run."""
    n_cases = int(manifest["split"]["evaluated_case_count"])
    n_geom_ok = int(geom_df["geometry_ok"].sum()) if not geom_df.empty else 0
    n_empty = int(summary_ep200["n_empty_pred"].sum()) if "n_empty_pred" in summary_ep200.columns else 0
    timing = manifest.get("timing_seconds", {})
    r_time = timing.get("rasnet")
    n_time = timing.get("nnunet")
    timing_str = (
        f"`rasnet = {r_time}s`, `nnunet = {n_time}s` (`rasnet = 0.0s` reflects skipped already-completed 94/94 cases on `--resume` and is NOT inference latency; use Option A `timing_summary.csv` and `Table 4` for valid runtime/patch timing)"
        if timing
        else "not recorded in manifest"
    )
    return "\n".join([
        f"# V1 Epoch-200 Val94 Endpoint Evaluation Report (`{manifest.get('run_id')}`)",
        "",
        f"- **Generated (UTC):** `{manifest.get('generated_at_utc')}`",
        f"- **Checkpoint Role:** `{manifest.get('checkpoint_role')}` (`RASNet`: epoch 200 by script/log/archive provenance, `embedded_epoch_metadata_present=False`; `nnU-Net V2`: embedded `current_epoch=200`)",
        f"- **Validation Split:** `Val94` (`N = {n_cases}`, `splits_sha256: {manifest['split']['splits_sha256']}`)",
        f"- **Statuses (`RASNet` / `nnU-Net`):** `{paired_json['status_summary']['rasnet']}` / `{paired_json['status_summary']['nnunet']}` (`0` dropped, `{n_empty}` empty predictions)",
        f"- **Mask Geometry Verification:** `{n_geom_ok}/{len(geom_df)}` exported masks verified against source image and label headers.",
        f"- **Timing (`manifest.json`):** {timing_str}.",
        "",
        table_md,
    ])


def generate_outputs(out_dir: Path, repo_root: Path) -> dict[str, Path]:
    """Generate paired case CSV, separated comparison tables, endpoint figures, and reports."""
    out = validate_report_out_dir(out_dir, repo_root)
    dr = pd.read_csv(out / "per_case_metrics_rasnet_epoch200.csv")
    dn = pd.read_csv(out / "per_case_metrics_nnunet_epoch200.csv")
    summary_ep200 = pd.read_csv(out / "summary_epoch200.csv")
    paired_json = json.loads((out / "paired_stats_epoch200.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    geom_p = out / "geometry_verification.csv"
    geom_df = pd.read_csv(geom_p) if geom_p.is_file() else pd.DataFrame(columns=["geometry_ok"])

    tie_tol = float(paired_json.get("tie_tolerance", 1e-12))
    paired_case_df = build_paired_case_level_table(dr, dn, tie_tolerance=tie_tol)
    paired_case_csv = out / "paired_case_level_differences_epoch200.csv"
    paired_case_df.to_csv(paired_case_csv, index=False)

    ref_rows = load_existing_best_references(repo_root)
    sep_df = build_separated_comparison_table(summary_ep200, ref_rows)
    table_csv = out / "table_epoch200_vs_existing_best_references.csv"
    sep_df.to_csv(table_csv, index=False)

    table_md_text = render_separated_table_markdown(sep_df, paired_json)
    table_md = out / "table_epoch200_vs_existing_best_references.md"
    table_md.write_text(table_md_text, encoding="utf-8")

    fig_png = out / "fig_v1_epoch200_val94_endpoint_comparison.png"
    fig_svg = out / "fig_v1_epoch200_val94_endpoint_comparison.svg"
    plot_epoch200_comparison_figure(paired_case_df, summary_ep200, paired_json, ref_rows, fig_png, fig_svg)

    thesis_md = out / "THESIS_CURVE_AND_ENDPOINT_CORRECTION.md"
    thesis_md.write_text(
        render_thesis_curve_and_endpoint_text(summary_ep200, paired_json, ref_rows, manifest),
        encoding="utf-8",
    )

    report_md = out / "ENDPOINT_EVALUATION_REPORT.md"
    report_md.write_text(
        render_endpoint_report(out, manifest, summary_ep200, paired_json, geom_df, table_md_text),
        encoding="utf-8",
    )

    return {
        "paired_case_csv": paired_case_csv,
        "comparison_table_csv": table_csv,
        "comparison_table_md": table_md,
        "figure_png": fig_png,
        "figure_svg": fig_svg,
        "thesis_text_md": thesis_md,
        "report_md": report_md,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    artifacts = generate_outputs(Path(args.out_dir), REPO_ROOT)
    for k, v in artifacts.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
