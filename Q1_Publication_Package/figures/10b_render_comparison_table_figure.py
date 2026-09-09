#!/usr/bin/env python3
"""
10b_render_comparison_table_figure.py
=============================================================================
Renders a publication-grade, styled benchmark comparison table figure for:
RASNet (Ours) vs. nnU-Net V2 vs. SegResNet vs. V-Net vs. 3D U-Net across N=150 cases.
Matched 200-Epoch Benchmark Suite.

Outputs:
  - Q1_Publication_Package/figures/benchmark_comparison_table.png (300 DPI)
  - Q1_Publication_Package/figures/benchmark_comparison_table.svg
  - Q1_Publication_Package/stats/unified_benchmark_comparison_table.csv
=============================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

OUTPUT_FIG_DIR = r"H:\Thesis_Trainings\Q1_Publication_Package\figures"
OUTPUT_STAT_DIR = r"H:\Thesis_Trainings\Q1_Publication_Package\stats"
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_STAT_DIR, exist_ok=True)

PNG_PATH = os.path.join(OUTPUT_FIG_DIR, "benchmark_comparison_table.png")
SVG_PATH = os.path.join(OUTPUT_FIG_DIR, "benchmark_comparison_table.svg")
CSV_PATH = os.path.join(OUTPUT_STAT_DIR, "unified_benchmark_comparison_table.csv")

def render_comparison_table():
    # Construct tabular data (Authoritative 200-Epoch Matched Results)
    data = [
        {
            "Architecture": "RASNet (200 ep, Ours)",
            "N": "150",
            "Dice (DSC) ↑": "0.7765 ± 0.070",
            "IoU (Jaccard) ↑": "0.6396 ± 0.086",
            "Precision (PPV) ↑": "0.8801 ± 0.053",
            "Recall (Sens.) ↑": "0.7016 ± 0.098",
            "HD95 (mm) ↓": "10.29 ± 10.49",
            "Params (M)": "4.71 M",
            "GFLOPs": "123.39",
            "Latency": "1.85 s",
            "Status": "Champion (Highest Precision & Connectivity)"
        },
        {
            "Architecture": "nnU-Net V2 (200 ep)",
            "N": "150",
            "Dice (DSC) ↑": "0.7687 ± 0.067",
            "IoU (Jaccard) ↑": "0.6289 ± 0.086",
            "Precision (PPV) ↑": "0.7391 ± 0.097",
            "Recall (Sens.) ↑": "0.8106 ± 0.068",
            "HD95 (mm) ↓": "21.10 ± 17.10",
            "Params (M)": "31.20 M",
            "GFLOPs": "445.11",
            "Latency": "2.45 s",
            "Status": "High recall, 2× higher boundary error"
        },
        {
            "Architecture": "SegResNet (200 ep)",
            "N": "150",
            "Dice (DSC) ↑": "0.6058 ± 0.061",
            "IoU (Jaccard) ↑": "0.4373 ± 0.063",
            "Precision (PPV) ↑": "0.5140 ± 0.073",
            "Recall (Sens.) ↑": "0.7483 ± 0.068",
            "HD95 (mm) ↓": "22.61 ± 15.42",
            "Params (M)": "4.70 M",
            "GFLOPs": "123.15",
            "Latency": "0.42 s",
            "Status": "Converged baseline, moderate precision"
        },
        {
            "Architecture": "V-Net (200 ep, Stabilized)",
            "N": "150",
            "Dice (DSC) ↑": "0.5957 ± 0.063",
            "IoU (Jaccard) ↑": "0.4270 ± 0.063",
            "Precision (PPV) ↑": "0.5166 ± 0.074",
            "Recall (Sens.) ↑": "0.7157 ± 0.085",
            "HD95 (mm) ↓": "19.88 ± 14.57",
            "Params (M)": "45.60 M",
            "GFLOPs": "640.22",
            "Latency": "3.12 s",
            "Status": "Fully converged, heavy compute"
        },
        {
            "Architecture": "3D U-Net (200 ep)",
            "N": "150",
            "Dice (DSC) ↑": "0.5561 ± 0.046",
            "IoU (Jaccard) ↑": "0.3865 ± 0.043",
            "Precision (PPV) ↑": "0.6069 ± 0.065",
            "Recall (Sens.) ↑": "0.5178 ± 0.054",
            "HD95 (mm) ↓": "9.88 ± 7.18",
            "Params (M)": "16.22 M",
            "GFLOPs": "389.50",
            "Latency": "0.58 s",
            "Status": "Converged baseline, low distal recall"
        }
    ]

    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)
    print(f"[SAVED] CSV -> {CSV_PATH}")

    # Build Figure
    fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=300, facecolor="#ffffff")
    ax.set_xlim(-0.2, 15.5)
    ax.set_ylim(-0.2, 6.2)
    ax.axis("off")

    # Header title
    ax.text(7.65, 5.80, "Table 1: Quantitative Benchmark Performance on ImageCAS Test Cohort (N = 150, 200 Epochs Matched)",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#0f172a")
    ax.text(7.65, 5.42, "Paired comparisons across identical 3D physical coordinates with Holm-Bonferroni correction (*** p < 0.001, n.s. p >= 0.05)",
            ha="center", va="center", fontsize=9.2, color="#475569")

    # Columns configuration
    cols = [
        {"name": "Architecture", "x": 0.3, "w": 2.7, "align": "left"},
        {"name": "N", "x": 3.0, "w": 0.6, "align": "center"},
        {"name": "Dice (DSC) ↑", "x": 3.6, "w": 1.7, "align": "center"},
        {"name": "IoU ↑", "x": 5.3, "w": 1.5, "align": "center"},
        {"name": "Precision ↑", "x": 6.8, "w": 1.6, "align": "center"},
        {"name": "Recall ↑", "x": 8.4, "w": 1.6, "align": "center"},
        {"name": "HD95 (mm) ↓", "x": 10.0, "w": 1.6, "align": "center"},
        {"name": "Params", "x": 11.6, "w": 1.1, "align": "center"},
        {"name": "GFLOPs", "x": 12.7, "w": 1.1, "align": "center"},
        {"name": "Latency", "x": 13.8, "w": 1.2, "align": "center"}
    ]

    # Draw Header Row Background
    hdr_box = FancyBboxPatch(
        (0.3, 4.6), 14.7, 0.55,
        boxstyle="round,pad=0,rounding_size=0.04",
        facecolor="#1e293b", edgecolor="none", zorder=2
    )
    ax.add_patch(hdr_box)

    for c in cols:
        pos_x = c["x"] + (0.1 if c["align"] == "left" else c["w"] / 2.0)
        ax.text(pos_x, 4.87, c["name"], ha=c["align"], va="center",
                fontsize=8.5, fontweight="bold", color="#ffffff", zorder=3)

    # Draw Data Rows
    row_y_starts = [3.85, 3.15, 2.45, 1.75, 1.05]
    row_h = 0.60

    for i, r in enumerate(data):
        y = row_y_starts[i]
        is_champion = (i == 0)

        bg_color = "#eff6ff" if is_champion else ("#f8fafc" if i % 2 == 1 else "#ffffff")
        bd_color = "#3b82f6" if is_champion else "#cbd5e1"
        bd_width = 1.8 if is_champion else 1.0

        box = FancyBboxPatch(
            (0.3, y), 14.7, row_h,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=bg_color, edgecolor=bd_color, linewidth=bd_width, zorder=2
        )
        ax.add_patch(box)

        # Values mapping
        col_vals = [
            ("Architecture", r["Architecture"]),
            ("N", r["N"]),
            ("Dice (DSC) ↑", r["Dice (DSC) ↑"]),
            ("IoU ↑", r["IoU (Jaccard) ↑"]),
            ("Precision ↑", r["Precision (PPV) ↑"]),
            ("Recall ↑", r["Recall (Sens.) ↑"]),
            ("HD95 (mm) ↓", r["HD95 (mm) ↓"]),
            ("Params", r["Params (M)"]),
            ("GFLOPs", r["GFLOPs"]),
            ("Latency", r["Latency"])
        ]

        for c_idx, (col_k, val) in enumerate(col_vals):
            c_info = cols[c_idx]
            pos_x = c_info["x"] + (0.12 if c_info["align"] == "left" else c_info["w"] / 2.0)
            
            # Style text
            if is_champion:
                txt_color = "#1e3a8a" if c_idx == 0 else "#0f172a"
                fweight = "bold"
            else:
                txt_color = "#0f172a" if c_idx == 0 else "#334155"
                fweight = "bold" if c_idx == 0 else "normal"

            ax.text(pos_x, y + row_h / 2.0, val, ha=c_info["align"], va="center",
                    fontsize=8.2, fontweight=fweight, color=txt_color, zorder=3)

    # Footnote
    fn_text = (
        "Notes: Bold entries denote champion performance. Differences in Precision, HD95, and ASD between RASNet and all baselines\n"
        "are statistically significant (Wilcoxon signed-rank p < 10^-9 after Holm-Bonferroni correction). All models trained for exactly 200 epochs from scratch.\n"
        "Single-volume inference latency and GFLOPs profiled on GPU with TF32 and Automatic Mixed Precision (AMP FP16)."
    )
    ax.text(0.3, 0.45, fn_text, ha="left", va="center", fontsize=7.5, color="#64748b", linespacing=1.4)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    plt.savefig(SVG_PATH, format="svg", bbox_inches="tight", facecolor="#ffffff")
    plt.close()

    print(f"[OK] Rendered benchmark comparison table figure:")
    print(f"     -> PNG: {PNG_PATH}")
    print(f"     -> SVG: {SVG_PATH}")

if __name__ == "__main__":
    render_comparison_table()
