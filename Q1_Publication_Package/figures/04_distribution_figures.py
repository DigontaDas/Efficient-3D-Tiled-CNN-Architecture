#!/usr/bin/env python3
"""
04_distribution_figures.py
=============================================================================
Publication-Quality Distribution & Statistical Significance Figures.
Matched 200-Epoch Benchmark Suite (5 Converged Architectures, N=150 Cases).

Author: Digonta Das / Nafis Mehedi
Project: Efficient 3D Tiled CNN Architecture (RASNet, ImageCAS Dataset)
Target: Q1 Medical Imaging Journal Submission

Features:
  - 1x3 Multi-Panel Layout:
      Panel A: Dice Similarity Coefficient (DSC)
      Panel B: Intersection-over-Union (IoU)
      Panel C: 95% Hausdorff Distance (HD95, mm)
  - Boxplot + Overlaid Jittered Strip Plot (Per-Case Granularity, N=150)
  - Holm-Bonferroni Corrected Statistical Significance Brackets
  - Colorblind-Safe Palette
  - 300 DPI High-Resolution PNG & Vector SVG Export
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set directory paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_DIR = os.path.join(REPO_ROOT, "Q1_Publication_Package", "matched_200ep_benchmark", "evaluation_results")
OUTPUT_DIR = os.path.join(REPO_ROOT, "Q1_Publication_Package", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metric file paths (Authoritative 200-Epoch Evaluation)
MODELS_CONFIG = {
    "RASNet (Ours)": os.path.join(EVAL_DIR, "metrics_rasnet_200ep.csv"),
    "nnU-Net V2": os.path.join(EVAL_DIR, "metrics_nnu_net_v2_200ep.csv"),
    "SegResNet": os.path.join(EVAL_DIR, "metrics_segresnet_200ep.csv"),
    "V-Net": os.path.join(EVAL_DIR, "metrics_v_net_200ep.csv"),
    "3D U-Net": os.path.join(EVAL_DIR, "metrics_3d_u_net_200ep.csv")
}

# Set global publication typography and aesthetics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9.0
plt.rcParams['ytick.labelsize'] = 9.0
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0


def load_long_dataframe() -> pd.DataFrame:
    """Load all 5 CSVs and construct a unified long-format DataFrame."""
    records = []
    for model_name, path in MODELS_CONFIG.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing evaluation metrics file: {path}")
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            records.append({
                "case_id": int(row["case_id"]),
                "Model": model_name,
                "dice": float(row["dice"]),
                "iou": float(row["iou"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "hd95": float(row["hd95"]),
                "asd": float(row["asd"]),
                "cldice": float(row["cldice"]),
                "centerline_recall": float(row["centerline_recall"])
            })
            
    long_df = pd.DataFrame(records)
    return long_df


def draw_significance_bracket(ax, x1: float, x2: float, y: float, h: float, text: str, color: str = "#222222"):
    """Draw clean statistical significance bracket with star annotation."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, c=color)
    ax.text((x1 + x2) * 0.5, y + h + 0.003, text, ha='center', va='bottom', color=color,
            fontsize=8.5, fontweight='bold')


def plot_distributions(long_df: pd.DataFrame):
    """Generate 1x3 publication figure with boxplots, stripplots, and significance stars across all 5 models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    order = ["RASNet (Ours)", "nnU-Net V2", "SegResNet", "V-Net", "3D U-Net"]
    
    # Distinct, high-contrast palette
    palette = {
        "RASNet (Ours)": "#d95f02",   # Orange highlight
        "nnU-Net V2": "#e7298a",      # Magenta
        "SegResNet": "#1b9e77",       # Teal/Green
        "V-Net": "#386cb0",           # Blue
        "3D U-Net": "#7570b3"         # Purple
    }
    
    # ── PANEL A: DICE SIMILARITY (DSC) ──────────────────────────────────────────
    ax = axes[0]
    sns.boxplot(
        data=long_df, x="Model", y="dice", hue="Model", order=order, ax=ax,
        palette=palette, width=0.52, linewidth=1.2, fliersize=0, showcaps=True, legend=False
    )
    sns.stripplot(
        data=long_df, x="Model", y="dice", order=order, ax=ax,
        color="#111111", size=2.8, alpha=0.30, jitter=0.20, zorder=3
    )
    
    ax.set_title("A. Dice Similarity Coefficient (DSC)", fontweight='bold', pad=22)
    ax.set_ylabel("Dice Similarity Coefficient", fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylim(0.35, 1.10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='#dddddd')
    ax.set_axisbelow(True)
    
    # Significance brackets from Holm-Bonferroni tests:
    # RASNet (0) vs nnU-Net V2 (1): p = 0.4523 (n.s.)
    draw_significance_bracket(ax, 0, 1, 0.90, 0.015, "n.s.")
    # RASNet (0) vs SegResNet (2): p = 7.36e-25 (***)
    draw_significance_bracket(ax, 0, 2, 0.94, 0.015, "***")
    # RASNet (0) vs V-Net (3): p = 7.36e-25 (***)
    draw_significance_bracket(ax, 0, 3, 0.98, 0.015, "***")
    # RASNet (0) vs 3D U-Net (4): p = 7.36e-25 (***)
    draw_significance_bracket(ax, 0, 4, 1.02, 0.015, "***")
    
    # ── PANEL B: INTERSECTION OVER UNION (IoU) ─────────────────────────────────
    ax = axes[1]
    sns.boxplot(
        data=long_df, x="Model", y="iou", hue="Model", order=order, ax=ax,
        palette=palette, width=0.52, linewidth=1.2, fliersize=0, showcaps=True, legend=False
    )
    sns.stripplot(
        data=long_df, x="Model", y="iou", order=order, ax=ax,
        color="#111111", size=2.8, alpha=0.30, jitter=0.20, zorder=3
    )
    
    ax.set_title("B. Intersection-over-Union (IoU)", fontweight='bold', pad=22)
    ax.set_ylabel("Intersection-over-Union (Jaccard Index)", fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylim(0.20, 1.02)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='#dddddd')
    ax.set_axisbelow(True)
    
    draw_significance_bracket(ax, 0, 1, 0.81, 0.015, "n.s.")
    draw_significance_bracket(ax, 0, 2, 0.85, 0.015, "***")
    draw_significance_bracket(ax, 0, 3, 0.89, 0.015, "***")
    draw_significance_bracket(ax, 0, 4, 0.93, 0.015, "***")
    
    # ── PANEL C: 95% HAUSDORFF DISTANCE (HD95, mm) ─────────────────────────────
    ax = axes[2]
    sns.boxplot(
        data=long_df, x="Model", y="hd95", hue="Model", order=order, ax=ax,
        palette=palette, width=0.52, linewidth=1.2, fliersize=0, showcaps=True, legend=False
    )
    sns.stripplot(
        data=long_df, x="Model", y="hd95", order=order, ax=ax,
        color="#111111", size=2.8, alpha=0.30, jitter=0.20, zorder=3
    )
    
    ax.set_title("C. 95% Hausdorff Distance (HD95)", fontweight='bold', pad=22)
    ax.set_ylabel("HD95 (mm) — Lower is Better", fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylim(-2, 115)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='#dddddd')
    ax.set_axisbelow(True)
    
    # Significance brackets (HD95):
    # RASNet (0) vs nnU-Net V2 (1): p = 9.20e-10 (***)
    draw_significance_bracket(ax, 0, 1, 68, 2.5, "***")
    # RASNet (0) vs SegResNet (2): p = 4.30e-13 (***)
    draw_significance_bracket(ax, 0, 2, 78, 2.5, "***")
    # RASNet (0) vs V-Net (3): p = 5.35e-10 (***)
    draw_significance_bracket(ax, 0, 3, 88, 2.5, "***")
    # RASNet (0) vs 3D U-Net (4): p = 0.4523 (n.s.)
    draw_significance_bracket(ax, 0, 4, 98, 2.5, "n.s.")
    
    # Rotate tick labels slightly for clean readability
    for ax in axes:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=15, ha='right', fontweight='semibold')
        
    plt.suptitle("Per-Case Segmentation Performance Distributions with Holm-Corrected Wilcoxon Significance ($N=150$, 200 Epochs)",
                 fontsize=13.5, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "dice_iou_hd95_distributions.png")
    svg_path = os.path.join(OUTPUT_DIR, "dice_iou_hd95_distributions.svg")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_path} (300 DPI)")
    print(f"Saved: {svg_path} (Vector)")


def main():
    print("=" * 80)
    print("STEP 4: PUBLICATION-GRADE DISTRIBUTION FIGURES (5 CONVERGED MODELS, 200 EP)")
    print("=" * 80)
    
    long_df = load_long_dataframe()
    print(f"Loaded {len(long_df)} case evaluations across 5 models.")
    
    plot_distributions(long_df)
    print("=" * 80)


if __name__ == "__main__":
    main()
