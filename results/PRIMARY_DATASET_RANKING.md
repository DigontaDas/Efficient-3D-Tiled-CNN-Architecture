# 🏅 PRIMARY DATASET BENCHMARK RANKING & FORENSIC VERDICT

**Target Document**: `results/PRIMARY_DATASET_RANKING.md`  
**Dataset**: Primary ImageCAS Test Cohort ($N=150$ Held-Out Cases, 851–1000)  
**Evaluation Standard**: Matched 200-Epoch Benchmark  

---

## 1. Primary Metric Ranking Table

| Metric | #1 Rank | #2 Rank | #3 Rank | #4 Rank | #5 Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dice (DSC) ↑** | **RASNet** (0.7765) | **nnU-Net V2** (0.7687) | **V-Net** (0.7474) | **SegResNet** (0.7469) | **3D U-Net** (0.5561) |
| **Precision (PPV) ↑** | **RASNet** (0.8801) | **V-Net** (0.7545) | **nnU-Net V2** (0.7391) | **SegResNet** (0.7313) | **3D U-Net** (0.6069) |
| **Recall / Sensitivity ↑**| **nnU-Net V2** (0.8106) | **SegResNet** (0.7713) | **V-Net** (0.7499) | **RASNet** (0.7016) | **3D U-Net** (0.5178) |
| **95% Hausdorff (HD95) ↓**| **3D U-Net** (9.88 mm) | **RASNet** (10.29 mm) | **nnU-Net V2** (21.10 mm) | **V-Net** (21.81 mm) | **SegResNet** (31.48 mm) |
| **Average Surface Dist (ASD) ↓**| **RASNet** (1.60 mm) | **3D U-Net** (1.69 mm) | **nnU-Net V2** (3.10 mm) | **V-Net** (3.18 mm) | **SegResNet** (4.66 mm) |
| **Centerline Dice (clDice) ↑**| **RASNet** (0.8592) | **nnU-Net V2** (0.8201) | **V-Net** (0.8075) | **SegResNet** (0.7769) | **3D U-Net** (0.7027) |
| **Tree Recall ($T_{sens}$) ↑**| **nnU-Net V2** (0.9111) | **SegResNet** (0.8866) | **V-Net** (0.8571) | **RASNet** (0.8018) | **3D U-Net** (0.6534) |
| **Intersection over Union (IoU) ↑**| **RASNet** (0.6396) | **nnU-Net V2** (0.6289) | **V-Net** (0.6006) | **SegResNet** (0.6001) | **3D U-Net** (0.3865) |

*(Note: 3D U-Net ranks #1 on HD95 purely because it severely under-segments, predicting a small smooth core while failing to reconstruct peripheral branches, avoiding boundary penalties).*

---

## 2. Forensic Answers to Core Research Questions

### Q1: Does RASNet win the majority of important metrics?
**YES, but with a critical qualification.**
- RASNet wins **5 out of 8 headline metrics**: Dice (0.7765), Precision (0.8801), clDice (0.8592), ASD (1.598 mm), and IoU (0.6396).
- However, RASNet **clearly loses Recall** (0.7016 vs nnU-Net 0.8106, SegResNet 0.7713) and **Tree Recall** (0.8018 vs nnU-Net 0.9111).
- RASNet represents an intentionally conservative operating regime driven by `StenosisAwareLoss` and $\tau = 0.60$: it almost completely eliminates false positives (Precision 88.01%), but misses thin distal vessels that nnU-Net captures.

### Q2: Is the improvement practically meaningful?
- **In Precision & Boundary Fidelity: YES, SUBSTANTIAL.**
  - +14.1% higher Precision over nnU-Net and +14.9% over SegResNet.
  - HD95 is reduced by more than half compared to nnU-Net (10.29 mm vs 21.10 mm) and by two-thirds compared to SegResNet (31.48 mm).
  - Surface error (ASD) is cut in half (1.60 mm vs 3.10 mm).
- **In Volumetric Overlap (Dice): SMALL TO NEGLIGIBLE.**
  - RASNet Dice = 0.7765 vs nnU-Net 0.7687 ($\Delta = +0.0078$, a +1.0% relative gain).

### Q3: Is it statistically significant?
- **RASNet vs SegResNet**: **STATISTICALLY SIGNIFICANT across all 8 metrics** ($p < 10^{-11}$).
- **RASNet vs V-Net**: **STATISTICALLY SIGNIFICANT across all 8 metrics** ($p < 10^{-10}$).
- **RASNet vs 3D U-Net**: **STATISTICALLY SIGNIFICANT across 7 metrics** ($p < 10^{-22}$); HD95 is not significant ($p = 0.1971$).
- **RASNet vs nnU-Net V2**:
  - **Dice & IoU**: **NOT STATISTICALLY SIGNIFICANT** (Wilcoxon raw $p = 0.1152$, Holm-Bonferroni adjusted $p = 0.2262$).
  - **Precision**: **STATISTICALLY SIGNIFICANT favor RASNet** ($p = 1.84 \times 10^{-25}$).
  - **HD95 & ASD**: **STATISTICALLY SIGNIFICANT favor RASNet** ($p = 4.60 \times 10^{-10}$ and $p = 6.29 \times 10^{-12}$).
  - **clDice**: **STATISTICALLY SIGNIFICANT favor RASNet** ($p = 3.96 \times 10^{-09}$).
  - **Recall & Tree Recall**: **STATISTICALLY SIGNIFICANT favor nnU-Net** ($p = 1.84 \times 10^{-25}$).

### Q4: Is RASNet consistently better across cases or only better on average?
- **In Median Comparison**:
  - RASNet Median Dice is **0.7965** vs nnU-Net **0.7768**, SegResNet **0.7551**, and V-Net **0.7588**.
  - RASNet Median HD95 is **6.33 mm** vs nnU-Net **16.68 mm** and SegResNet **29.90 mm**.
- **In Paired Case Differences**:
  - For **Precision**, RASNet beats nnU-Net in **147 out of 150 cases (98.0%)**.
  - For **HD95**, RASNet has lower error than nnU-Net in **114 out of 150 cases (76.0%)**.
  - For **Dice**, RASNet beats nnU-Net in **83 out of 150 cases (55.3%)**, while nnU-Net beats RASNet in 67 cases (44.7%).
  - For **Recall**, nnU-Net beats RASNet in **143 out of 150 cases (95.3%)**.
- **Conclusion**: RASNet is **consistently superior** in boundary precision and topological quality, but operates at a fundamental sensitivity/specificity trade-off vs nnU-Net.
