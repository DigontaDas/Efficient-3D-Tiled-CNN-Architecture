# 🏆 200-Epoch Matched Benchmark Suite — Final Publication Report

- **Test Cohort**: N=150 Held-Out Cases (ImageCAS 851–1000)
- **Models**: SegResNet, nnU-Net V2, 3D U-Net, V-Net, RASNet (Ours)
- **Budget**: Exactly 200 Epochs per model from random scratch

## 1. Primary Metrics (Mean ± Std)

| Model | Dice (DSC) | IoU | Precision | Recall | HD95 (mm) | ASD (mm) | clDice | Centerline Recall |
|---|---|---|---|---|---|---|---|---|
| **SegResNet** | 0.6058 ± 0.0612 | 0.4373 ± 0.0632 | 0.5140 ± 0.0733 | 0.7483 ± 0.0678 | 22.6145 ± 15.4238 | 3.8455 ± 2.4528 | 0.7457 ± 0.0676 | 0.7274 ± 0.0746 |
| **3D U-Net** | 0.5561 ± 0.0458 | 0.3865 ± 0.0429 | 0.6069 ± 0.0646 | 0.5178 ± 0.0536 | 9.8780 ± 7.1789 | 1.6903 ± 1.0444 | 0.7027 ± 0.0611 | 0.6534 ± 0.0739 |
| **V-Net** | 0.5957 ± 0.0628 | 0.4270 ± 0.0633 | 0.5166 ± 0.0742 | 0.7157 ± 0.0848 | 19.8783 ± 14.5715 | 3.2127 ± 2.1601 | 0.7337 ± 0.0715 | 0.6890 ± 0.0840 |
| **RASNet** | 0.7765 ± 0.0695 | 0.6396 ± 0.0864 | 0.8801 ± 0.0530 | 0.7016 ± 0.0976 | 10.2924 ± 10.4861 | 1.5981 ± 1.8235 | 0.8592 ± 0.0719 | 0.8018 ± 0.1022 |
| **nnU-Net V2** | 0.7687 ± 0.0668 | 0.6289 ± 0.0856 | 0.7391 ± 0.0967 | 0.8106 ± 0.0680 | 21.0983 ± 17.1028 | 3.0966 ± 2.6419 | 0.8201 ± 0.0766 | 0.9111 ± 0.0556 |

## 2. Statistical Significance (Paired Two-Sided Wilcoxon Signed-Rank Test with Holm-Bonferroni Correction)

| Comparison | Metric | RASNet | Baseline | p-value (Holm-Bonferroni) | Sig |
|---|---|---|---|---|---|
| RASNet vs. SegResNet | Dice Similarity Coefficient (DSC) | 0.7765 | 0.6058 | 2.22e-25 | YES (***) |
| RASNet vs. SegResNet | Intersection-over-Union (IoU) | 0.6396 | 0.4373 | 2.22e-25 | YES (***) |
| RASNet vs. SegResNet | Precision (PPV) | 0.8801 | 0.5140 | 1.84e-25 | YES (***) |
| RASNet vs. SegResNet | Recall / Sensitivity | 0.7016 | 0.7483 | 1.53e-08 | YES (***) |
| RASNet vs. SegResNet | 95% Hausdorff Distance (HD95, mm) | 10.2924 | 22.6145 | 1.07e-13 | YES (***) |
| RASNet vs. SegResNet | Average Surface Distance (ASD, mm) | 1.5981 | 3.8455 | 5.92e-18 | YES (***) |
| RASNet vs. SegResNet | Centerline Dice (clDice) | 0.8592 | 0.7457 | 3.03e-22 | YES (***) |
| RASNet vs. SegResNet | Centerline Recall / Tree Completeness (T_sens) | 0.8018 | 0.7274 | 4.82e-13 | YES (***) |
| RASNet vs. 3D U-Net | Dice Similarity Coefficient (DSC) | 0.7765 | 0.5561 | 1.84e-25 | YES (***) |
| RASNet vs. 3D U-Net | Intersection-over-Union (IoU) | 0.6396 | 0.3865 | 1.84e-25 | YES (***) |
| RASNet vs. 3D U-Net | Precision (PPV) | 0.8801 | 0.6069 | 1.84e-25 | YES (***) |
| RASNet vs. 3D U-Net | Recall / Sensitivity | 0.7016 | 0.5178 | 2.46e-25 | YES (***) |
| RASNet vs. 3D U-Net | 95% Hausdorff Distance (HD95, mm) | 10.2924 | 9.8780 | 0.1971 | NO |
| RASNet vs. 3D U-Net | Average Surface Distance (ASD, mm) | 1.5981 | 1.6903 | 0.0011 | YES (*) |
| RASNet vs. 3D U-Net | Centerline Dice (clDice) | 0.8592 | 0.7027 | 2.46e-25 | YES (***) |
| RASNet vs. 3D U-Net | Centerline Recall / Tree Completeness (T_sens) | 0.8018 | 0.6534 | 3.03e-23 | YES (***) |
| RASNet vs. V-Net | Dice Similarity Coefficient (DSC) | 0.7765 | 0.5957 | 2.05e-25 | YES (***) |
| RASNet vs. V-Net | Intersection-over-Union (IoU) | 0.6396 | 0.4270 | 2.05e-25 | YES (***) |
| RASNet vs. V-Net | Precision (PPV) | 0.8801 | 0.5166 | 1.84e-25 | YES (***) |
| RASNet vs. V-Net | Recall / Sensitivity | 0.7016 | 0.7157 | 0.2999 | NO |
| RASNet vs. V-Net | 95% Hausdorff Distance (HD95, mm) | 10.2924 | 19.8783 | 1.19e-10 | YES (***) |
| RASNet vs. V-Net | Average Surface Distance (ASD, mm) | 1.5981 | 3.2127 | 2.35e-14 | YES (***) |
| RASNet vs. V-Net | Centerline Dice (clDice) | 0.8592 | 0.7337 | 1.04e-22 | YES (***) |
| RASNet vs. V-Net | Centerline Recall / Tree Completeness (T_sens) | 0.8018 | 0.6890 | 1.34e-17 | YES (***) |
| RASNet vs. nnU-Net V2 | Dice Similarity Coefficient (DSC) | 0.7765 | 0.7687 | 0.2262 | NO |
| RASNet vs. nnU-Net V2 | Intersection-over-Union (IoU) | 0.6396 | 0.6289 | 0.2262 | NO |
| RASNet vs. nnU-Net V2 | Precision (PPV) | 0.8801 | 0.7391 | 1.84e-25 | YES (***) |
| RASNet vs. nnU-Net V2 | Recall / Sensitivity | 0.7016 | 0.8106 | 1.84e-25 | YES (***) |
| RASNet vs. nnU-Net V2 | 95% Hausdorff Distance (HD95, mm) | 10.2924 | 21.0983 | 4.60e-10 | YES (***) |
| RASNet vs. nnU-Net V2 | Average Surface Distance (ASD, mm) | 1.5981 | 3.0966 | 6.29e-12 | YES (***) |
| RASNet vs. nnU-Net V2 | Centerline Dice (clDice) | 0.8592 | 0.8201 | 3.96e-09 | YES (***) |
| RASNet vs. nnU-Net V2 | Centerline Recall / Tree Completeness (T_sens) | 0.8018 | 0.9111 | 1.84e-25 | YES (***) |
