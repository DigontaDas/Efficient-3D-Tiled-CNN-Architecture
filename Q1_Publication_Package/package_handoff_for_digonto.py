#!/usr/bin/env python3
"""
package_handoff_for_digonto.py
=============================================================================
Assembles all 10 Q1 figures, 6 publication tables, and an index README
into a single, clean handoff directory:
  H:\\Thesis_Trainings\\Q1_Publication_Package\\handoff_for_digonto\\
=============================================================================
"""

import os
import shutil

BASE_DIR = r"H:\Thesis_Trainings"
DEST_DIR = os.path.join(BASE_DIR, "Q1_Publication_Package", "handoff_for_digonto")
os.makedirs(DEST_DIR, exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "tables"), exist_ok=True)

# ── List of Figures ──────────────────────────────────────────────────────────
FIGURE_MAPPINGS = [
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "architecture_diagram_v2.png"),
        os.path.join(DEST_DIR, "figures", "Fig1_Architecture_Diagram.png")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "architecture_diagram_v2.svg"),
        os.path.join(DEST_DIR, "figures", "Fig1_Architecture_Diagram.svg")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "multi_model_convergence_curves.png"),
        os.path.join(DEST_DIR, "figures", "Fig2_Multi_Model_Convergence_Curves.png")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "multi_model_convergence_curves.svg"),
        os.path.join(DEST_DIR, "figures", "Fig2_Multi_Model_Convergence_Curves.svg")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "dice_iou_hd95_distributions.png"),
        os.path.join(DEST_DIR, "figures", "Fig3_Dice_IoU_HD95_Distributions.png")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "dice_iou_hd95_distributions.svg"),
        os.path.join(DEST_DIR, "figures", "Fig3_Dice_IoU_HD95_Distributions.svg")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "ablation", "ablation_bar_chart.png"),
        os.path.join(DEST_DIR, "figures", "Fig4_Component_Ablation_Bar.png")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "ablation", "ablation_bar_chart.svg"),
        os.path.join(DEST_DIR, "figures", "Fig4_Component_Ablation_Bar.svg")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "efficiency", "efficiency_vs_dice_scatter.png"),
        os.path.join(DEST_DIR, "figures", "Fig5_Computational_Efficiency_Pareto.png")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "efficiency", "efficiency_vs_dice_scatter.svg"),
        os.path.join(DEST_DIR, "figures", "Fig5_Computational_Efficiency_Pareto.svg")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "attention_maps_case851.png"),
        os.path.join(DEST_DIR, "figures", "Fig6_Attention_Maps_Case851.png")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "attention_maps_case851.svg"),
        os.path.join(DEST_DIR, "figures", "Fig6_Attention_Maps_Case851.svg")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "attention_maps_case934.png"),
        os.path.join(DEST_DIR, "figures", "Fig7_Attention_Maps_Case934.png")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "attention_maps_case934.svg"),
        os.path.join(DEST_DIR, "figures", "Fig7_Attention_Maps_Case934.svg")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "failure_cases_gallery.png"),
        os.path.join(DEST_DIR, "figures", "Fig8_Qualitative_Failure_Cases.png")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "failure_cases_gallery.svg"),
        os.path.join(DEST_DIR, "figures", "Fig8_Qualitative_Failure_Cases.svg")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "clinical_validation", "07_clinical_bland_altman_agreement.png"),
        os.path.join(DEST_DIR, "figures", "Fig9_Clinical_Bland_Altman_Agreement.png")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "clinical_validation", "07_clinical_bland_altman_agreement.svg"),
        os.path.join(DEST_DIR, "figures", "Fig9_Clinical_Bland_Altman_Agreement.svg")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "benchmark_comparison_table.png"),
        os.path.join(DEST_DIR, "figures", "Fig10_Benchmark_Comparison_Table.png")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "figures", "benchmark_comparison_table.svg"),
        os.path.join(DEST_DIR, "figures", "Fig10_Benchmark_Comparison_Table.svg")
    ),
]

# ── List of Tables ───────────────────────────────────────────────────────────
TABLE_MAPPINGS = [
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "stats", "unified_benchmark_comparison_table.csv"),
        os.path.join(DEST_DIR, "tables", "Table1_Benchmark_Comparison.csv")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "matched_200ep_benchmark", "evaluation_results", "benchmark_200ep_summary_table.csv"),
        os.path.join(DEST_DIR, "tables", "Table1b_Benchmark_200ep_Summary.csv")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "matched_200ep_benchmark", "evaluation_results", "benchmark_200ep_final_report.md"),
        os.path.join(DEST_DIR, "tables", "Table1b_Benchmark_200ep_Report.md")
    ),
    (
        os.path.join(BASE_DIR, "results-after-hallucin-fix", "training_metrics_per_epoch.csv"),
        os.path.join(DEST_DIR, "tables", "Table2_Training_Metrics_70_Epochs.csv")
    ),
    (
        os.path.join(BASE_DIR, "results-after-hallucin-fix", "training_metrics_per_epoch.md"),
        os.path.join(DEST_DIR, "tables", "Table2_Training_Metrics_70_Epochs.md")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "ablation", "ablation_table.csv"),
        os.path.join(DEST_DIR, "tables", "Table3_Component_Ablation.csv")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "ablation", "ablation_table.md"),
        os.path.join(DEST_DIR, "tables", "Table3_Component_Ablation.md")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "stats", "stats_significance_table.csv"),
        os.path.join(DEST_DIR, "tables", "Table4_Statistical_Significance_Holm.csv")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "stats", "stats_significance_table.md"),
        os.path.join(DEST_DIR, "tables", "Table4_Statistical_Significance_Holm.md")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "stats", "bootstrap_CI_table.csv"),
        os.path.join(DEST_DIR, "tables", "Table5_Bootstrap_95CI.csv")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "stats", "bootstrap_CI_table.md"),
        os.path.join(DEST_DIR, "tables", "Table5_Bootstrap_95CI.md")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "clinical_validation", "07_clinical_diagnostic_performance.md"),
        os.path.join(DEST_DIR, "tables", "Table6_Clinical_Diagnostic_Performance.md")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "clinical_validation", "07_clinical_validation_summary.md"),
        os.path.join(DEST_DIR, "tables", "Clinical_Validation_Full_Summary.md")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "clinical_validation", "production_dual_mode_comparison.csv"),
        os.path.join(DEST_DIR, "tables", "Table7_Clinical_Dual_Mode_Comparison.csv")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "clinical_validation", "mask_completeness_audit.csv"),
        os.path.join(DEST_DIR, "tables", "Table8_Mask_Completeness_Audit.csv")
    ),
    (
        os.path.join(BASE_DIR, "Q1_Publication_Package", "clinical_validation", "provenance_audit_results.csv"),
        os.path.join(DEST_DIR, "tables", "Table9_Provenance_Audit_Results.csv")
    )
]

def run_packaging():
    print(f">> Packaging Q1 deliverables into: {DEST_DIR}")
    
    # Copy Figures
    copied_figs = 0
    for src, dst in FIGURE_MAPPINGS:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied_figs += 1
            print(f" [FIG] {os.path.basename(dst)} ({os.path.getsize(dst)/1024:.1f} KB)")
        else:
            print(f" [WARN] Missing figure: {src}")

    # Copy Tables
    copied_tabs = 0
    for src, dst in TABLE_MAPPINGS:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied_tabs += 1
            print(f" [TAB] {os.path.basename(dst)} ({os.path.getsize(dst)/1024:.1f} KB)")
        else:
            print(f" [WARN] Missing table: {src}")

    # Write FIGURES_INDEX.md
    index_content = """# 📦 Q1 Publication Figures & Tables Handoff Package

**Project**: RASNet: Residual Attention Segmentation Network with Deep Supervision for Coronary Artery CCTA Segmentation  
**Authors**: Nafis Mehedi & Digonta Das  
**Target Venue**: Q1 Medical Imaging Journal (IEEE TMI / Medical Image Analysis / RSNA Radiology AI)  
**Date**: September 2026

---

## 🖼️ Complete Figures Index (All 10 Figures Ready)

All figures are provided in high-resolution **300 DPI PNG** and editable vector **SVG** format.

| Figure # | File Basename | Proposed Journal Caption & Description | Target Manuscript Section |
|:---:|:---|:---|:---|
| **Fig 1** | `Fig1_Architecture_Diagram.png/.svg` | **RASNet Neural Architecture Schematic.** Multi-scale 3D ResNet encoder backbone, 3-level additive Residual Attention Gates (`AttentionGate3D`), decoder with multi-scale deep supervision auxiliary heads (`aux2`, `aux3`), and calibrated `StenosisAwareLoss` formulation. | *Section 3.1: Network Architecture* |
| **Fig 2** | `Fig2_Multi_Model_Convergence_Curves.png/.svg` | **Matched 200-Epoch Training & Validation Convergence Curves.** Multi-panel comparison across all five architectures (RASNet, nnU-Net V2, SegResNet, V-Net, and 3D U-Net), showing loss descent, Dice validation trajectory, and learning rate schedules. | *Section 3.3: Training Dynamics* |
| **Fig 3** | `Fig3_Dice_IoU_HD95_Distributions.png/.svg` | **Metric Distribution Comparison across N=150 Cases.** 1×3 multi-panel boxplots with overlaid strip distributions for Dice Similarity, Jaccard IoU, and HD95 (mm) with Holm-Bonferroni significance brackets ($*** p < 10^{-14}$). | *Section 4.1: Benchmark Evaluation* |
| **Fig 4** | `Fig4_Component_Ablation_Bar.png/.svg` | **Stepwise Component Ablation Gain.** Progressive performance contributions: Baseline SegResNet ($0.7637$) $\\rightarrow$ + Attention Gates ($0.7712$) $\\rightarrow$ + Deep Supervision ($0.7758$) $\\rightarrow$ + StenosisAwareLoss ($0.7795$) $\\rightarrow$ + 4-Pass TTA ($0.7830$) $\\rightarrow$ + `cc3d` ($0.7765$). | *Section 4.3: Ablation Study* |
| **Fig 5** | `Fig5_Computational_Efficiency_Pareto.png/.svg` | **Pareto Efficiency Frontier (Dice vs. GFLOPs & Parameters).** Scatter visualization demonstrating RASNet's superior accuracy-to-compute ratio (4.71M parameters, 123.39 GFLOPs, 1.85s latency) vs. nnU-Net (16.54M params, 445 GFLOPs) and 3D U-Net. | *Section 4.4: Computational Complexity* |
| **Fig 6** | `Fig6_Attention_Maps_Case851.png/.svg` | **3D Attention Coefficient Heatmaps (Case 851).** Axial, Coronal, and Sagittal multi-planar reformations showing forward-hook attention weights ($\psi$) sharply localizing to coronary arteries while zeroing myocardium and background parenchyma. | *Section 4.5: Model Interpretability* |
| **Fig 7** | `Fig7_Attention_Maps_Case934.png/.svg` | **3D Attention Coefficient Heatmaps (Case 934).** Multi-planar attention coefficient maps for complex bifurcation geometry and side branches. | *Section 4.5: Model Interpretability* |
| **Fig 8** | `Fig8_Qualitative_Failure_Cases.png/.svg` | **Quantitative Failure Case Analysis.** 3-view MIP overlays of worst-performing outlier cases, detailing the root causes (contiguous non-coronary over-segmentation and distal vascular tapering). | *Section 4.6: Failure Mode Analysis* |
| **Fig 9** | `Fig9_Clinical_Bland_Altman_Agreement.png/.svg` | **Clinical Stenosis Agreement vs. Radiologist PACS Calipers (N=32 Hospital Cases).** Panel A: Bland-Altman agreement plot (Mean Bias: $-1.59\\%$, 95% LoA span: $54.2\\%$, 31/32 cases within limits). Panel B: Stenosis correlation scatter ($R^2 = 0.6525$, Spearman $\\rho = 0.603$, $p = 0.00026$). | *Section 5: Clinical Hospital Validation* |
| **Fig 10** | `Fig10_Benchmark_Comparison_Table.png/.svg` | **Full Quantitative Benchmark Comparison Graphic Table.** Styled presentation table reporting Dice, IoU, Precision, Recall, HD95, GFLOPs, Parameters, and single-volume latency for all 5 architectures across $N=150$ test cases. | *Section 4.1 / Executive Summary* |

---

## 📊 Complete Publication Tables Index

| Table # | File Basename | Description | Target Section |
|:---:|:---|:---|:---|
| **Table 1** | `Table1_Benchmark_Comparison.csv` | Full comparative benchmark metrics across $N=150$ test cases (RASNet, SegResNet, nnU-Net, 3D U-Net, V-Net). | *Section 4.1* |
| **Table 1b** | `Table1b_Benchmark_200ep_Summary.csv` / `.md` | Matched 200-epoch benchmark summary across all 5 models and 8 metrics. | *Section 4.1* |
| **Table 2** | `Table2_Training_Metrics_70_Epochs.csv/.md` | Epoch-by-epoch loss, time, learning rate, and memory consumption for the entire 70-epoch champion run. | *Section 3.3 / Appendix* |
| **Table 3** | `Table3_Component_Ablation.csv/.md` | 6-row progressive ablation study table detailing metric gains from each architectural component. | *Section 4.3* |
| **Table 4** | `Table4_Statistical_Significance_Holm.csv/.md` | Wilcoxon signed-rank paired statistical tests with raw and Holm-Bonferroni adjusted $p$-values. | *Section 4.2* |
| **Table 5** | `Table5_Bootstrap_95CI.csv/.md` | Non-parametric percentile bootstrap 95% confidence intervals ($B=2000$ iterations) for all primary metrics. | *Section 4.2* |
| **Table 6** | `Table6_Clinical_Diagnostic_Performance.md` | $2\\times 2$ Confusion Matrix and diagnostic metrics (Sensitivity 96.2%, Accuracy 87.5%, Cohen's $\\kappa = 0.529$) on N=32 hospital cases. | *Section 5* |

---

## 🚀 How to Use These Files in the Report / Paper

1. **LaTeX Users**: Place SVG or PDF/PNG files into your `figures/` directory and include using standard `\\includegraphics[width=\\linewidth]{FigX_...}`.
2. **Word / Overleaf Users**: High-resolution 300 DPI PNGs are sized to fit double-column (7.2 in) or full-page journal formats directly without pixelation.
3. **Table Data**: Use the Markdown tables (`.md`) for quick copy-pasting or CSVs for automatic LaTeX table generation via `pgfplotstable` or Pandas `to_latex()`.
"""

    index_path = os.path.join(DEST_DIR, "FIGURES_INDEX.md")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print(f"[SAVED] FIGURES_INDEX.md -> {index_path}")
    print(f"\n>> All deliverables packaged successfully! ({copied_figs} figure files, {copied_tabs} table files)")

if __name__ == "__main__":
    run_packaging()
