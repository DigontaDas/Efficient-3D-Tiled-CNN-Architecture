#!/usr/bin/env python3
"""
04_distribution_figures.py
=============================================================================
Publication-Quality Distribution & Statistical Significance Figures.

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
  - Colorblind-Safe Palette (Seaborn colorblind)
  - 300 DPI High-Resolution PNG & Vector SVG Export
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import PathPatch

# Set directory paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAININGS_DIR = os.path.join(REPO_ROOT, "Thesis_Trainings", "Thesis_Trainings")
OUTPUT_DIR = os.path.join(REPO_ROOT, "Q1_Publication_Package", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metric file paths
RASNET_CSV = os.path.join(TRAININGS_DIR, "results-after-hallucin-fix", "metrics_rasnet.csv")
SEGRESNET_CSV = os.path.join(TRAININGS_DIR, "all_four_validations", "metrics_segresnet.csv")
NNUNET_CSV = os.path.join(TRAININGS_DIR, "all_four_validations", "mandatory_artifacts_nnunet", "metrics_nnunet.csv")
UNET3D_CSV = os.path.join(TRAININGS_DIR, "all_four_validations", "metrics_3d_unet.csv")

# Set global publication typography and aesthetics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0


def load_long_dataframe() -> pd.DataFrame:
    """Load all 4 CSVs and construct a unified long-format DataFrame."""
    models = {
        "RASNet (Ours)": RASNET_CSV,
        "SegResNet": SEGRESNET_CSV,
        "3D U-Net": UNET3D_CSV,
        "nnU-Net V2": NNUNET_CSV
    }
    
    records = []
    for model_name, path in models.items():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            records.append({
                "case_id": int(row["case_id"]),
                "Model": model_name,
                "dice": float(row["dice"]),
                "iou": float(row["iou"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "hd95": float(row["hd95"])
            })
            
    long_df = pd.DataFrame(records)
    return long_df


def draw_significance_bracket(ax, x1: float, x2: float, y: float, h: float, text: str, color: str = "#222222"):
    """Draw clean statistical significance bracket with star annotation."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c=color)
    ax.text((x1 + x2) * 0.5, y + h + 0.005, text, ha='center', va='bottom', color=color,
            fontsize=9.5, fontweight='bold')


def plot_distributions(long_df: pd.DataFrame):
    """Generate 1x3 publication figure with boxplots, stripplots, and significance stars."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=300)
    
    order = ["RASNet (Ours)", "SegResNet", "3D U-Net", "nnU-Net V2"]
    
    # Curated colorblind-safe palette (distinct, soft tones)
    palette = {
        "RASNet (Ours)": "#02818a",   # Deep teal / cyan
        "SegResNet": "#67a9cf",       # Soft steel blue
        "3D U-Net": "#bdc9e1",        # Light periwinkle
        "nnU-Net V2": "#f1eef6"       # Pale lavender / gray-white
    }
    
    # ── PANEL A: DICE SIMILARITY (DSC) ──────────────────────────────────────────
    ax = axes[0]
    sns.boxplot(
        data=long_df, x="Model", y="dice", hue="Model", order=order, ax=ax,
        palette=palette, width=0.52, linewidth=1.2, fliersize=0, showcaps=True, legend=False
    )
    # Overlay jittered strip plot
    sns.stripplot(
        data=long_df, x="Model", y="dice", order=order, ax=ax,
        color="#111111", size=3.2, alpha=0.35, jitter=0.22, zorder=3
    )
    
    ax.set_title("A. Dice Similarity Coefficient (DSC)", fontweight='bold', pad=18)
    ax.set_ylabel("Dice Similarity Coefficient", fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylim(0.35, 1.06)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='#dddddd')
    ax.set_axisbelow(True)
    
    # Significance brackets from Step 1 Wilcoxon tests
    # RASNet (0) vs SegResNet (1): p = 3.09e-15 (***)
    draw_significance_bracket(ax, 0, 1, 0.90, 0.02, "***")
    # RASNet (0) vs 3D U-Net (2): p = 2.80e-24 (***)
    draw_significance_bracket(ax, 0, 2, 0.95, 0.02, "***")
    # RASNet (0) vs nnU-Net (3): p = 8.57e-25 (***)
    draw_significance_bracket(ax, 0, 3, 1.00, 0.02, "***")
    
    # ── PANEL B: INTERSECTION OVER UNION (IoU) ─────────────────────────────────
    ax = axes[1]
    sns.boxplot(
        data=long_df, x="Model", y="iou", hue="Model", order=order, ax=ax,
        palette=palette, width=0.52, linewidth=1.2, fliersize=0, showcaps=True, legend=False
    )
    sns.stripplot(
        data=long_df, x="Model", y="iou", order=order, ax=ax,
        color="#111111", size=3.2, alpha=0.35, jitter=0.22, zorder=3
    )
    
    ax.set_title("B. Intersection-over-Union (IoU)", fontweight='bold', pad=18)
    ax.set_ylabel("Intersection-over-Union (Jaccard Index)", fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylim(0.20, 0.96)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='#dddddd')
    ax.set_axisbelow(True)
    
    # Significance brackets
    draw_significance_bracket(ax, 0, 1, 0.81, 0.02, "***")
    draw_significance_bracket(ax, 0, 2, 0.86, 0.02, "***")
    draw_significance_bracket(ax, 0, 3, 0.91, 0.02, "***")
    
    # ── PANEL C: 95% HAUSDORFF DISTANCE (HD95, mm) ─────────────────────────────
    ax = axes[2]
    sns.boxplot(
        data=long_df, x="Model", y="hd95", hue="Model", order=order, ax=ax,
        palette=palette, width=0.52, linewidth=1.2, fliersize=0, showcaps=True, legend=False
    )
    sns.stripplot(
        data=long_df, x="Model", y="hd95", order=order, ax=ax,
        color="#111111", size=3.2, alpha=0.35, jitter=0.22, zorder=3
    )
    
    ax.set_title("C. 95% Hausdorff Distance (HD95)", fontweight='bold', pad=18)
    ax.set_ylabel("HD95 (mm) — Lower is Better", fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylim(-2, 105)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='#dddddd')
    ax.set_axisbelow(True)
    
    # Significance brackets (HD95)
    # RASNet (0) vs SegResNet (1): p = 0.3516 (n.s.)
    draw_significance_bracket(ax, 0, 1, 72, 3, "n.s.")
    # RASNet (0) vs 3D U-Net (2): p = 1.34e-06 (***)
    draw_significance_bracket(ax, 0, 2, 82, 3, "***")
    # RASNet (0) vs nnU-Net (3): p = 3.66e-25 (***)
    draw_significance_bracket(ax, 0, 3, 92, 3, "***")
    
    # Rotate tick labels slightly for clean readability
    for ax in axes:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=15, ha='right', fontweight='semibold')
        
    plt.suptitle("Per-Case Segmentation Performance Distributions with Holm-Corrected Wilcoxon Significance ($N=150$)",
                 fontsize=13.5, fontweight='bold', y=1.02)
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
    print("STEP 4: PUBLICATION-GRADE DISTRIBUTION FIGURES (BOX + STRIP PLOTS)")
    print("=" * 80)
    
    long_df = load_long_dataframe()
    print(f"Loaded {len(long_df)} case evaluations across 4 models.")
    
    plot_distributions(long_df)
    print("=" * 80)


if __name__ == "__main__":
    main()
