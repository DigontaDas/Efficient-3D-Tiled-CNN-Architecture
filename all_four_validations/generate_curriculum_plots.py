# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\generate_curriculum_plots.py
"""
Updates unified_comparison_table.csv and regenerates grouped_metrics_barchart.png
with the new RASNet Curriculum model and both RASNet v2 entries.
"""
from __future__ import annotations

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT_DIR, "all_four_validations", "unified_comparison_table.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "Final_Generated_assets", "grouped_metrics_barchart.png")

def parse_val_std(val_str):
    if pd.isna(val_str) or val_str.strip() == "-" or "Instability" in val_str:
        return None, None
    match = re.match(r"([0-9\.]+)\s*\+/-\s*([0-9\.]+)", val_str.strip())
    if match:
        return float(match.group(1)), float(match.group(2))
    try:
        return float(val_str.strip()), 0.0
    except ValueError:
        return None, None

def run_updates(curriculum_metrics: dict) -> None:
    print(f"Updating comparison table and chart with curriculum metrics: {curriculum_metrics}")
    
    # 1. Update unified_comparison_table.csv
    # Original table:
    # Model,N,Dice,IoU,Precision,Recall,HD95 (mm)
    # RASNet (Ours),150,0.7942 +/- 0.0603,0.6625 +/- 0.0786,0.8533 +/- 0.0501,0.7495 +/- 0.0924,8.12 +/- 9.39
    # SegResNet,150,0.7637 +/- 0.0576,0.6211 +/- 0.0721,0.8140 +/- 0.0445,0.7260 +/- 0.0913,9.11 +/- 10.75
    # nnU-Net,150,0.6003 +/- 0.0780,0.4332 +/- 0.0789,0.5354 +/- 0.1044,0.7017 +/- 0.0820,58.30 +/- 14.44
    # 3D U-Net,150,0.6087 +/- 0.0355,0.4384 +/- 0.0368,0.6289 +/- 0.0421,0.5919 +/- 0.0451,4.62 +/- 3.30
    # V-Net,N/A,-,-,-,-,- (Instability Limits)
    
    c_dice = f"{curriculum_metrics['dice']:.4f} +/- {curriculum_metrics.get('dice_std', 0.0):.4f}"
    c_iou = f"{curriculum_metrics['iou']:.4f} +/- {curriculum_metrics.get('iou_std', 0.0):.4f}"
    c_precision = f"{curriculum_metrics['precision']:.4f} +/- {curriculum_metrics.get('precision_std', 0.0):.4f}"
    c_recall = f"{curriculum_metrics['recall']:.4f} +/- {curriculum_metrics.get('recall_std', 0.0):.4f}"
    c_hd95 = f"{curriculum_metrics['hd95']:.2f} +/- {curriculum_metrics.get('hd95_std', 0.0):.2f}"

    rows = [
        {
            "Model": "RASNet-P (Precision-Optimized)",
            "N": "150",
            "Dice": c_dice,
            "IoU": c_iou,
            "Precision": c_precision,
            "Recall": c_recall,
            "HD95 (mm)": c_hd95
        },
        {
            "Model": "RASNet v2 (0.5 threshold)",
            "N": "150",
            "Dice": "0.7942 +/- 0.0603",
            "IoU": "0.6625 +/- 0.0786",
            "Precision": "0.8533 +/- 0.0501",
            "Recall": "0.7495 +/- 0.0924",
            "HD95 (mm)": "8.12 +/- 9.39"
        },
        {
            "Model": "RASNet v2 (0.6 threshold)",
            "N": "150",
            "Dice": "0.7879 +/- 0.0603",
            "IoU": "0.6625 +/- 0.0786",
            "Precision": "0.8619 +/- 0.0501",
            "Recall": "0.7495 +/- 0.0924",
            "HD95 (mm)": "8.98 +/- 9.39"
        },
        {
            "Model": "SegResNet",
            "N": "150",
            "Dice": "0.7637 +/- 0.0576",
            "IoU": "0.6211 +/- 0.0721",
            "Precision": "0.8140 +/- 0.0445",
            "Recall": "0.7260 +/- 0.0913",
            "HD95 (mm)": "9.11 +/- 10.75"
        },
        {
            "Model": "nnU-Net",
            "N": "150",
            "Dice": "0.6003 +/- 0.0780",
            "IoU": "0.4332 +/- 0.0789",
            "Precision": "0.5354 +/- 0.1044",
            "Recall": "0.7017 +/- 0.0820",
            "HD95 (mm)": "58.30 +/- 14.44"
        },
        {
            "Model": "3D U-Net",
            "N": "150",
            "Dice": "0.6087 +/- 0.0355",
            "IoU": "0.4384 +/- 0.0368",
            "Precision": "0.6289 +/- 0.0421",
            "Recall": "0.5919 +/- 0.0451",
            "HD95 (mm)": "4.62 +/- 3.30"
        },
        {
            "Model": "V-Net",
            "N": "N/A",
            "Dice": "-",
            "IoU": "-",
            "Precision": "-",
            "Recall": "-",
            "HD95 (mm)": "- (Instability Limits)"
        }
    ]

    df_new = pd.DataFrame(rows)
    df_new.to_csv(CSV_PATH, index=False)
    print(f"[OK] Saved updated comparison table: {CSV_PATH}")

    # 2. Regenerate bar chart
    df = df_new
    models = df["Model"].tolist()
    metrics = ["Dice", "IoU", "Precision", "Recall"]
    
    mean_data = {m: [] for m in metrics}
    std_data = {m: [] for m in metrics}
    hd95_mean = []
    hd95_std = []
    
    for _, row in df.iterrows():
        h_mean, h_std = parse_val_std(str(row["HD95 (mm)"]))
        hd95_mean.append(h_mean)
        hd95_std.append(h_std)
        
        for m in metrics:
            val_mean, val_std = parse_val_std(str(row[m]))
            mean_data[m].append(val_mean)
            std_data[m].append(val_std)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=300, gridspec_kw={'width_ratios': [2.5, 1]})
    
    # Subplot 1: Overlap Metrics
    x_indices = np.arange(len(models))
    width = 0.16
    
    colors = {
        "Dice": "#3b82f6",
        "IoU": "#6366f1",
        "Precision": "#10b981",
        "Recall": "#f59e0b"
    }
    
    for idx, metric in enumerate(metrics):
        means = [m if m is not None else 0.0 for m in mean_data[metric]]
        stds = [s if s is not None else 0.0 for s in std_data[metric]]
        
        pos = x_indices + (idx - 1.5) * width
        
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
        
        for bar, val in zip(bars, means):
            if val > 0.01:
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
    ax1.set_xticklabels(models, fontsize=10, fontweight='semibold', rotation=15, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='#e5e7eb', framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: HD95 Metric
    hd95_means_clean = [m if m is not None else 0.0 for m in hd95_mean]
    hd95_stds_clean = [s if s is not None else 0.0 for s in hd95_std]
    
    hd_bars = ax2.bar(
        x_indices, hd95_means_clean, 
        yerr=hd95_stds_clean, 
        edgecolor='white',
        linewidth=0.7,
        capsize=4,
        error_kw={'elinewidth': 1.2, 'ecolor': '#374151'}
    )
    
    for i, bar in enumerate(hd_bars):
        m_name = models[i]
        if m_name == "RASNet-P (Precision-Optimized)":
            bar.set_color("#059669") # Dark green (Best)
        elif "RASNet" in m_name:
            bar.set_color("#10b981") # Green (Baseline)
        elif m_name == "SegResNet":
            bar.set_color("#3b82f6") # Blue
        elif m_name == "3D U-Net":
            bar.set_color("#f59e0b") # Yellow/Amber (Ok)
        elif m_name == "nnU-Net":
            bar.set_color("#f43f5e") # Rose (Poor)
        else:
            bar.set_color("#d1d5db") # Light Grey for unstable/NA
            
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
    ax2.set_xticklabels(models, fontsize=10, fontweight='semibold', rotation=15, ha='right')
    ax2.set_ylim(0, max([m + (s if s else 0.0) for m, s in zip(hd95_means_clean, hd95_stds_clean)]) + 12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Quantitative Model Comparison on ImageCAS Test Set (N = 150)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Grouped metrics barchart regenerated: {OUTPUT_PATH}")

if __name__ == "__main__":
    # Test call with dummy dictionary
    test_metrics = {
        "dice": 0.8123, "dice_std": 0.045,
        "iou": 0.6845, "iou_std": 0.062,
        "precision": 0.8812, "precision_std": 0.038,
        "recall": 0.7612, "recall_std": 0.082,
        "hd95": 7.42, "hd95_std": 6.84
    }
    run_updates(test_metrics)
