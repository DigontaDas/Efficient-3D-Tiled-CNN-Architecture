#!/usr/bin/env python3
r"""
generate_unseen_66_publication_figures.py — Publication-Quality Figures for Unseen 66 Cohort
=============================================================================================
Generates 300 DPI PNG and Vector SVG figures for the IEEE JBHI Paper:
  1. fig1_unseen66_model_comparison_bars.png / .svg (Overlap vs Boundary Distance)
  2. fig2_unseen66_distribution_boxplots.png / .svg (Metric distributions across 66 cases)
  3. fig3_unseen66_stenosis_fidelity.png / .svg (Ground Truth vs RASNet % Area Stenosis)
  4. fig4_unseen66_radar_performance.png / .svg (Multi-dimensional performance radar)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNSEEN_DIR = os.path.join(BASE_DIR, "results", "unseen_66_cohort")
METRICS_DIR = os.path.join(UNSEEN_DIR, "metrics")
FIG_DIR = os.path.join(UNSEEN_DIR, "figures")
TABLES_DIR = os.path.join(UNSEEN_DIR, "final_tables")

os.makedirs(FIG_DIR, exist_ok=True)

# Set global publication styling
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

MODEL_COLORS = {
    "RASNet (Ours)": "#1e40af",    # Deep Blue
    "RASNet (0.50)": "#3b82f6",    # Blue
    "SegResNet": "#059669",        # Emerald Green
    "V-Net": "#7c3aed",            # Purple
    "nnU-Net V2": "#d97706",       # Amber
    "3D U-Net": "#dc2626"          # Crimson Red
}


def load_all_case_metrics():
    """Loads all per-case CSVs into a single concatenated dataframe."""
    model_files = [
        ("RASNet (Ours)", os.path.join(METRICS_DIR, "unseen_66_rasnet_case_metrics.csv")),
        ("RASNet (0.50)", os.path.join(METRICS_DIR, "unseen_66_rasnet_thresh05_case_metrics.csv")),
        ("SegResNet", os.path.join(METRICS_DIR, "unseen_66_segresnet_case_metrics.csv")),
        ("V-Net", os.path.join(METRICS_DIR, "unseen_66_vnet_case_metrics.csv")),
        ("nnU-Net V2", os.path.join(METRICS_DIR, "unseen_66_nnunet_case_metrics.csv")),
        ("3D U-Net", os.path.join(METRICS_DIR, "unseen_66_3dunet_case_metrics.csv")),
    ]
    dfs = []
    for model_name, path in model_files:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["model"] = model_name
            dfs.append(df)
        else:
            print(f"[!] Warning: missing metrics file {path}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def plot_fig1_comparison_bars():
    print("[*] Generating Fig 1: Model Comparison Bar Chart (Overlap & Boundary Error)...")
    models = ["RASNet (Ours)", "SegResNet", "V-Net", "nnU-Net V2", "3D U-Net"]
    colors = [MODEL_COLORS[m] for m in models]
    
    # Values from Table_Unseen66_Model_Comparison
    dice_vals = [0.7497, 0.7369, 0.7272, 0.6580, 0.5428]
    dice_err = [0.0601, 0.0606, 0.0659, 0.0890, 0.0417]
    
    cldice_vals = [0.8183, 0.7775, 0.7890, 0.7360, 0.6878]
    cldice_err = [0.0643, 0.0680, 0.0747, 0.0867, 0.0594]
    
    hd95_vals = [15.7227, 25.7737, 21.1055, 15.8774, 10.8413]
    hd95_err = [14.5796, 18.6838, 18.2661, 11.6041, 7.9886]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    
    x = np.arange(len(models))
    width = 0.35

    # Subplot 1: Dice & clDice
    rects1 = ax1.bar(x - width/2, dice_vals, width, yerr=dice_err, capsize=4,
                     label="Dice (Voxel Overlap)", color="#2563eb", alpha=0.9, edgecolor="black")
    rects2 = ax1.bar(x + width/2, cldice_vals, width, yerr=cldice_err, capsize=4,
                     label="clDice (Vessel Topology)", color="#059669", alpha=0.9, edgecolor="black")

    ax1.set_ylabel("Score (0.0 to 1.0)", fontweight="bold")
    ax1.set_title("(A) Segmentation Overlap & Topology (N=66)", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(0.4, 0.95)
    ax1.legend(loc="lower left", frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, weight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, weight="bold")

    # Subplot 2: Boundary Error (HD95)
    bars_hd = ax2.bar(x, hd95_vals, width=0.5, yerr=hd95_err, capsize=4,
                      color=colors, alpha=0.85, edgecolor="black")
    ax2.set_ylabel("Hausdorff Distance 95% (mm) ↓", fontweight="bold")
    ax2.set_title("(B) Boundary Distance Error HD95 (mm) (Lower is Better)", fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(0, 50)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars_hd, hd95_vals):
        ax2.annotate(f"{val:.2f} mm", xy=(bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    plt.suptitle("Pure Unseen External Cohort Benchmark (N=66 Cases)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_unseen66_model_comparison_bars.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig1_unseen66_model_comparison_bars.svg"), bbox_inches="tight")
    plt.close()


def plot_fig2_distributions(df):
    print("[*] Generating Fig 2: Metric Distribution Boxplots...")
    if df.empty:
        print("[!] Dataframe empty, skipping boxplots")
        return

    # Filter to main 5 models
    target_models = ["RASNet (Ours)", "SegResNet", "V-Net", "nnU-Net V2", "3D U-Net"]
    plot_df = df[df["model"].isin(target_models)].copy()

    metrics = [("dice", "Dice Score", (0.35, 0.95)),
               ("precision", "Precision", (0.35, 0.98)),
               ("cldice", "clDice (Topology)", (0.45, 0.98)),
               ("hd95_mm", "HD95 Error (mm) ↓", (0, 75))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    axes = axes.flatten()

    palette = [MODEL_COLORS[m] for m in target_models]

    for idx, (m_col, m_title, y_lim) in enumerate(metrics):
        ax = axes[idx]
        sns.boxplot(data=plot_df, x="model", y=m_col, order=target_models,
                    palette=palette, ax=ax, width=0.45, showmeans=True,
                    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"})
        sns.stripplot(data=plot_df, x="model", y=m_col, order=target_models,
                      color="black", alpha=0.25, jitter=0.18, size=4, ax=ax)
        ax.set_title(m_title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(m_title, fontweight="bold")
        ax.set_ylim(y_lim)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Case-Level Metric Dispersion on Pure Unseen External Cohort (N=66)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_unseen66_distribution_boxplots.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig2_unseen66_distribution_boxplots.svg"), bbox_inches="tight")
    plt.close()


def plot_fig3_stenosis_fidelity():
    print("[*] Generating Fig 3: Clinical Stenosis Constriction Fidelity Chart...")
    stenosis_csv = os.path.join(UNSEEN_DIR, "stenosis_blocks", "stenosis_detection_unseen_summary.csv")
    if not os.path.exists(stenosis_csv):
        print(f"[!] Stenosis CSV missing: {stenosis_csv}")
        return

    df = pd.read_csv(stenosis_csv)
    df = df.sort_values(by="gt_pct_area_stenosis", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor="white")
    x = np.arange(len(df))
    width = 0.38

    ax.bar(x - width/2, df["gt_pct_area_stenosis"], width, label="Ground Truth Stenosis (%)",
           color="#1e293b", alpha=0.9, edgecolor="black")
    ax.bar(x + width/2, df["rasnet_pct_area_stenosis"], width, label="RASNet Reconstructed Stenosis (%)",
           color="#2563eb", alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Case {cid}" for cid in df["case_id"]], rotation=45, ha="right", fontsize=9.5)
    ax.set_ylabel("% Luminal Area Stenosis", fontweight="bold")
    ax.set_ylim(50, 105)
    ax.set_title("Longitudinal Luminal Area Stenosis Tracking Across 20 Severe Constriction Cases (Mean Abs Error: 6.5%)",
                 fontweight="bold", pad=12)
    ax.axhline(70, color="crimson", linestyle="--", linewidth=1.5, label="Severe Stenosis Threshold (CAD-RADS 4: ≥70%)")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_unseen66_stenosis_fidelity.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig3_unseen66_stenosis_fidelity.svg"), bbox_inches="tight")
    plt.close()


def plot_fig4_radar_performance():
    print("[*] Generating Fig 4: Multi-Dimensional Performance Radar...")
    categories = ["Dice", "IoU", "Precision", "clDice", "Centerline\nRecall", "Boundary\nFidelity (1-ASD/5)"]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Model metric vectors
    # RASNet: Dice 0.7497, IoU 0.6033, Prec 0.7803, clDice 0.8183, CtlRec 0.8502, Bound: 1 - 2.347/5 = 0.531
    rasnet_vals = [0.7497, 0.6033, 0.7803, 0.8183, 0.8502, max(0, 1 - 2.347/5)]
    rasnet_vals += rasnet_vals[:1]

    # SegResNet: Dice 0.7369, IoU 0.5870, Prec 0.7239, clDice 0.7775, CtlRec 0.8721, Bound: 1 - 4.062/5 = 0.188
    segres_vals = [0.7369, 0.5870, 0.7239, 0.7775, 0.8721, max(0, 1 - 4.062/5)]
    segres_vals += segres_vals[:1]

    # V-Net: Dice 0.7272, IoU 0.5755, Prec 0.7286, clDice 0.7890, CtlRec 0.8426, Bound: 1 - 3.275/5 = 0.345
    vnet_vals = [0.7272, 0.5755, 0.7286, 0.7890, 0.8426, max(0, 1 - 3.275/5)]
    vnet_vals += vnet_vals[:1]

    # nnU-Net: Dice 0.6580, IoU 0.4966, Prec 0.7757, clDice 0.7360, CtlRec 0.6490, Bound: 1 - 2.694/5 = 0.461
    nnunet_vals = [0.6580, 0.4966, 0.7757, 0.7360, 0.6490, max(0, 1 - 2.694/5)]
    nnunet_vals += nnunet_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor="white")

    plt.xticks(angles[:-1], categories, color="black", size=11, weight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
    plt.ylim(0, 1.0)

    # Plot each model
    ax.plot(angles, rasnet_vals, linewidth=2.5, linestyle="solid", label="RASNet (Ours)", color=MODEL_COLORS["RASNet (Ours)"])
    ax.fill(angles, rasnet_vals, color=MODEL_COLORS["RASNet (Ours)"], alpha=0.18)

    ax.plot(angles, segres_vals, linewidth=2, linestyle="solid", label="SegResNet", color=MODEL_COLORS["SegResNet"])
    ax.fill(angles, segres_vals, color=MODEL_COLORS["SegResNet"], alpha=0.1)

    ax.plot(angles, vnet_vals, linewidth=1.8, linestyle="dashed", label="V-Net", color=MODEL_COLORS["V-Net"])
    ax.plot(angles, nnunet_vals, linewidth=1.8, linestyle="dotted", label="nnU-Net V2", color=MODEL_COLORS["nnU-Net V2"])

    plt.title("Multi-Dimensional Coronary Segmentation Radar (N=66 Pure Unseen)", size=13, weight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_unseen66_radar_performance.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig4_unseen66_radar_performance.svg"), bbox_inches="tight")
    plt.close()


def main():
    print("=" * 80)
    print("[*] STARTING UNSEEN 66 PUBLICATION FIGURE GENERATION")
    print("=" * 80)
    df = load_all_case_metrics()
    plot_fig1_comparison_bars()
    plot_fig2_distributions(df)
    plot_fig3_stenosis_fidelity()
    plot_fig4_radar_performance()
    print("[+] ALL 4 PUBLICATION FIGURES GENERATED IN:")
    print(f"    {FIG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
