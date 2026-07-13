"""
Task 3.3 — Comparative Bar Chart
=================================
Generates a publication-quality grouped bar chart comparing Dice, IoU, Precision, 
and Recall across the models, plus a separate subplot for HD95 (mm) since its scale 
is different.
Reads from c:\\Thesis_RASNET\\Thesis_Trainings\\Thesis_Trainings\\all_four_validations\\unified_comparison_table.csv.
Saves the figure to c:\\Thesis_RASNET\\Thesis_Trainings\\Thesis_Trainings\\Final_Generated_assets\\grouped_metrics_barchart.png.

Run with:
    c:\\Thesis_RASNET\\Thesis_Trainings\\Thesis_Trainings\\.venv_cuda\\Scripts\\python.exe Phase3_Local_Integration\\10_plot_comparison_bar.py
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT_DIR, "all_four_validations", "unified_comparison_table.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "Final_Generated_assets", "grouped_metrics_barchart.png")

def parse_val_std(val_str):
    """
    Parses a string of format 'mean +/- std' or returns (None, None) if invalid/unstable.
    """
    if pd.isna(val_str) or val_str.strip() == "-" or "Instability" in val_str:
        return None, None
    match = re.match(r"([0-9\.]+)\s*\+/-\s*([0-9\.]+)", val_str.strip())
    if match:
        return float(match.group(1)), float(match.group(2))
    # Try parsing as float only
    try:
        return float(val_str.strip()), 0.0
    except ValueError:
        return None, None

def main():
    print(f"Reading validation results from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV file not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Process data
    models = df["Model"].tolist()
    metrics = ["Dice", "IoU", "Precision", "Recall"]
    
    # Data structures to hold parsed values
    mean_data = {m: [] for m in metrics}
    std_data = {m: [] for m in metrics}
    hd95_mean = []
    hd95_std = []
    
    for _, row in df.iterrows():
        # Parse HD95 (mm)
        h_mean, h_std = parse_val_std(str(row["HD95 (mm)"]))
        hd95_mean.append(h_mean)
        hd95_std.append(h_std)
        
        # Parse Dice, IoU, Precision, Recall
        for m in metrics:
            val_mean, val_std = parse_val_std(str(row[m]))
            mean_data[m].append(val_mean)
            std_data[m].append(val_std)

    # Setup plots: 1 row, 2 subplots (Left: Overlap Metrics, Right: HD95 Distance)
    # Using a modern style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300, gridspec_kw={'width_ratios': [2.2, 1]})
    
    # --- Subplot 1: Overlap Metrics ---
    # Number of valid models we want to display on X axis
    # Let's show all 4 models. For V-Net it will show N/A.
    x_indices = np.arange(len(models))
    width = 0.18  # width of each bar
    
    # Color palette (harmonious & vibrant)
    colors = {
        "Dice": "#3b82f6",       # Modern blue
        "IoU": "#6366f1",        # Indigo
        "Precision": "#10b981",  # Emerald green
        "Recall": "#f59e0b"      # Amber/Yellow
    }
    
    for idx, metric in enumerate(metrics):
        means = [m if m is not None else 0.0 for m in mean_data[metric]]
        stds = [s if s is not None else 0.0 for s in std_data[metric]]
        
        # Shift bar position for grouping
        pos = x_indices + (idx - 1.5) * width
        
        # Draw bars
        bars = ax1.bar(
            pos, means, width, 
            yerr=stds, 
            label=metric, 
            color=colors[metric], 
            edgecolor='white', 
            linewidth=0.7, 
            capsize=3,
            error_kw={'elinewidth': 1.0, 'ecolor': '#4b5563'}
        )
        
        # Value labels on top of the bars (only for non-zero/valid values)
        for bar, val in zip(bars, means):
            if val > 0.01: # Don't label near-zero / N/A
                ax1.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() + 0.01, 
                    f"{val:.3f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=8, 
                    fontweight='semibold',
                    color='#1f2937'
                )

    ax1.set_title("Overlap & Classification Metrics (Higher is Better)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight='semibold')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(models, fontsize=11, fontweight='semibold')
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='#e5e7eb', framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Subplot 2: HD95 Metric ---
    # We display HD95 for the models on a separate axis
    hd95_colors = ["#ef4444" if m == "SegResNet" else "#6b7280" for m in models]
    
    # Plot HD95
    hd95_means_clean = [m if m is not None else 0.0 for m in hd95_mean]
    hd95_stds_clean = [s if s is not None else 0.0 for s in hd95_std]
    
    hd_bars = ax2.bar(
        x_indices, hd95_means_clean, 
        yerr=hd95_stds_clean, 
        color=['#10b981', '#f59e0b', '#6b7280', '#d1d5db'], # Green for SegResNet (best), yellow for nnU-Net, dark grey for 3D U-Net, light grey V-Net
        edgecolor='white',
        linewidth=0.7,
        capsize=4,
        error_kw={'elinewidth': 1.2, 'ecolor': '#374151'}
    )
    
    # Set colors of bars specifically to match models
    # SegResNet is the champion here (lowest HD95 is best!)
    # Let's color it green, others red/grey
    for i, bar in enumerate(hd_bars):
        m_name = models[i]
        if m_name == "SegResNet":
            bar.set_color("#10b981") # Green (Best)
        elif m_name == "nnU-Net":
            bar.set_color("#f43f5e") # Soft red/rose (Poor)
        elif m_name == "3D U-Net":
            bar.set_color("#f43f5e") # Soft red/rose (Poor)
        else:
            bar.set_color("#d1d5db") # Light Grey for unstable/NA
            
        # Value label
        m_val = hd95_means_clean[i]
        if m_name == "V-Net" or m_val == 0.0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2, 
                2.0, 
                "N/A\n(Unstable)", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                color='#ef4444',
                fontweight='bold'
            )
        else:
            ax2.text(
                bar.get_x() + bar.get_width() / 2, 
                m_val + (hd95_stds_clean[i] if hd95_stds_clean[i] else 0.0) + 1.0, 
                f"{m_val:.2f} mm", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                fontweight='bold',
                color='#1f2937'
            )

    ax2.set_title("95% Hausdorff Distance (Lower is Better)", fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel("Distance (mm)", fontsize=11, fontweight='semibold')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(models, fontsize=11, fontweight='semibold')
    # Leave room for text labels on top
    ax2.set_ylim(0, max([m + (s if s else 0.0) for m, s in zip(hd95_means_clean, hd95_stds_clean)]) + 12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Global title & layout
    plt.suptitle("Quantitative Model Comparison on ImageCAS Test Set (N = 150)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grouped metrics barchart successfully created and saved -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
