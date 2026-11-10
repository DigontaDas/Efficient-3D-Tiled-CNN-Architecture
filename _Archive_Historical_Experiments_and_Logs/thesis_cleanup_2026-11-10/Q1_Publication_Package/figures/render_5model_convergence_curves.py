#!/usr/bin/env python3
"""
render_5model_convergence_curves.py
=============================================================================
Generates Figure 2: Multi-Model 200-Epoch Training Convergence & Validation
Dice Trajectories across all 5 benchmarked architectures:
  - RASNet (Champion)
  - nnU-Net V2
  - SegResNet
  - V-Net (Stabilized)
  - 3D U-Net

Outputs:
  - Q1_Publication_Package/figures/multi_model_convergence_curves.png (300 DPI)
  - Q1_Publication_Package/figures/multi_model_convergence_curves.svg (Vector)
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = r"H:\Thesis_Trainings"
LOG_DIR = os.path.join(REPO_ROOT, "Q1_Publication_Package", "matched_200ep_benchmark", "logs")
OUTPUT_DIR = os.path.join(REPO_ROOT, "Q1_Publication_Package", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PNG_PATH = os.path.join(OUTPUT_DIR, "multi_model_convergence_curves.png")
SVG_PATH = os.path.join(OUTPUT_DIR, "multi_model_convergence_curves.svg")

# Style configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

def smooth_curve(points, factor=0.85):
    """Exponential moving average for smoother loss visualization."""
    smoothed = []
    for p in points:
        if np.isnan(p):
            smoothed.append(smoothed[-1] if smoothed else 0.0)
            continue
        if smoothed:
            previous = smoothed[-1]
            smoothed.append(previous * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return np.array(smoothed)

def main():
    model_configs = [
        {"name": "RASNet (Ours)", "file": "rasnet_200ep_log.csv", "color": "#d95f02", "lw": 2.2, "zorder": 5},
        {"name": "nnU-Net V2", "file": "nnunet_200ep_log.csv", "color": "#e7298a", "lw": 1.8, "zorder": 4},
        {"name": "SegResNet", "file": "segresnet_200ep_log.csv", "color": "#1b9e77", "lw": 1.6, "zorder": 3},
        {"name": "V-Net (Stabilized)", "file": "vnet_200ep_log.csv", "color": "#386cb0", "lw": 1.6, "zorder": 2},
        {"name": "3D U-Net", "file": "3dunet_200ep_log.csv", "color": "#7570b3", "lw": 1.6, "zorder": 1}
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    for cfg in model_configs:
        fp = os.path.join(LOG_DIR, cfg["file"])
        if not os.path.exists(fp):
            print(f"[WARN] Missing log file: {fp}")
            continue

        df = pd.read_csv(fp)
        df = df[df["epoch"] <= 200].copy()
        epochs = df["epoch"].values

        # ── Panel A: Training Loss Convergence (Normalized / Smoothed) ──
        raw_loss = df["train_loss"].values
        # Apply gentle EMA smoothing for clarity
        smooth_loss = smooth_curve(raw_loss, factor=0.8)

        ax1.plot(epochs, smooth_loss, label=cfg["name"], color=cfg["color"],
                 linewidth=cfg["lw"], zorder=cfg["zorder"])

        # ── Panel B: Validation Dice Trajectory ──
        if "val_dice" in df.columns:
            val_df = df.dropna(subset=["val_dice"]).copy()
            if len(val_df) > 0:
                ax2.plot(val_df["epoch"], val_df["val_dice"], label=cfg["name"],
                         color=cfg["color"], linewidth=cfg["lw"], marker="o", markersize=4,
                         zorder=cfg["zorder"])

    # Panel A styling
    ax1.set_title("A. Training Loss Convergence (200 Epochs)", fontweight="bold", pad=12)
    ax1.set_xlabel("Training Epoch", fontweight="bold")
    ax1.set_ylabel("Loss Function Value", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax1.set_xlim(1, 200)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#dddddd", fontsize=9.5, loc="upper right")

    # Panel B styling
    ax2.set_title("B. Validation Dice (DSC) Trajectory (200 Epochs)", fontweight="bold", pad=12)
    ax2.set_xlabel("Training Epoch", fontweight="bold")
    ax2.set_ylabel("Validation Dice Similarity Coefficient", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.set_xlim(1, 200)
    ax2.set_ylim(0.35, 0.85)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#dddddd", fontsize=9.5, loc="lower right")

    plt.suptitle("Multi-Model 200-Epoch Matched Training Convergence on ImageCAS ($N=690$ Train, $N=150$ Val)",
                 fontsize=13.0, fontweight="bold", y=0.98)
    plt.tight_layout()

    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.savefig(SVG_PATH, format="svg", bbox_inches="tight")
    plt.close()

    print(f"[OK] Generated Figure 2:")
    print(f"     -> PNG: {PNG_PATH}")
    print(f"     -> SVG: {SVG_PATH}")

if __name__ == "__main__":
    main()
