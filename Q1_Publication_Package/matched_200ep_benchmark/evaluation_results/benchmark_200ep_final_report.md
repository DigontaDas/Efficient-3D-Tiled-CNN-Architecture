# 🏆 200-Epoch Matched Benchmark Suite — Final Publication Report

- **Test Cohort**: N=150 Held-Out Cases (ImageCAS 851–1000)
- **Models**: SegResNet, nnU-Net V2, 3D U-Net, V-Net, RASNet (Ours)
- **Budget**: Exactly 200 Epochs per model from random scratch

## 1. Primary Metrics (Mean ± Std)

| Model | Dice (DSC) | IoU | Precision | Recall | HD95 (mm) | ASD (mm) | clDice | Centerline Recall |
|---|---|---|---|---|---|---|---|---|
| **SegResNet** | 0.7469 ± 0.0640 | 0.6001 ± 0.0788 | 0.7313 ± 0.0911 | 0.7713 ± 0.0654 | 31.4775 ± 18.1586 | 4.6614 ± 2.7181 | 0.7769 ± 0.0819 | 0.8866 ± 0.0579 |
| **3D U-Net** | 0.5561 ± 0.0458 | 0.3865 ± 0.0429 | 0.6069 ± 0.0646 | 0.5178 ± 0.0536 | 9.8780 ± 7.1789 | 1.6903 ± 1.0444 | 0.7027 ± 0.0611 | 0.6534 ± 0.0739 |
| **V-Net** | 0.7474 ± 0.0633 | 0.6006 ± 0.0786 | 0.7545 ± 0.0982 | 0.7499 ± 0.0678 | 21.8050 ± 19.1359 | 3.1758 ± 2.7710 | 0.8075 ± 0.0759 | 0.8571 ± 0.0613 |
| **RASNet** | 0.7765 ± 0.0695 | 0.6396 ± 0.0864 | 0.8801 ± 0.0530 | 0.7016 ± 0.0976 | 10.2924 ± 10.4861 | 1.5981 ± 1.8235 | 0.8592 ± 0.0719 | 0.8018 ± 0.1022 |
| **nnU-Net V2** | 0.7687 ± 0.0668 | 0.6289 ± 0.0856 | 0.7391 ± 0.0967 | 0.8106 ± 0.0680 | 21.0983 ± 17.1028 | 3.0966 ± 2.6419 | 0.8201 ± 0.0766 | 0.9111 ± 0.0556 |

## 2. Statistical Significance (Paired Two-Sided Wilcoxon Signed-Rank Test with Holm-Bonferroni Correction)

| Comparison | Metric | RASNet | Baseline | p-value (Holm-Bonferroni) | Sig |
|---|---|---|---|---|---|
| RASNet vs. SegResNet | Dice Similarity Coefficient (DSC) | 0.7765 | 0.7469 | 2.47e-12 | YES (***) |
| RASNet vs. SegResNet | Intersection-over-Union (IoU) | 0.6396 | 0.6001 | 1.11e-12 | YES (***) |
| RASNet vs. SegResNet | Precision (PPV) | 0.8801 | 0.7313 | 1.84e-25 | YES (***) |
| RASNet vs. SegResNet | Recall / Sensitivity | 0.7016 | 0.7713 | 1.84e-25 | YES (***) |
| RASNet vs. SegResNet | 95% Hausdorff Distance (HD95, mm) | 10.2924 | 31.4775 | 2.05e-20 | YES (***) |
| RASNet vs. SegResNet | Average Surface Distance (ASD, mm) | 1.5981 | 4.6614 | 6.26e-22 | YES (***) |
| RASNet vs. SegResNet | Centerline Dice (clDice) | 0.8592 | 0.7769 | 2.05e-20 | YES (***) |
| RASNet vs. SegResNet | Centerline Recall / Tree Completeness (T_sens) | 0.8018 | 0.8866 | 2.37e-25 | YES (***) |
| RASNet vs. 3D U-Net | Dice Similarity Coefficient (DSC) | 0.7765 | 0.5561 | 1.84e-25 | YES (***) |
| RASNet vs. 3D U-Net | Intersection-over-Union (IoU) | 0.6396 | 0.3865 | 1.84e-25 | YES (***) |
| RASNet vs. 3D U-Net | Precision (PPV) | 0.8801 | 0.6069 | 1.84e-25 | YES (***) |
| RASNet vs. 3D U-Net | Recall / Sensitivity | 0.7016 | 0.5178 | 2.46e-25 | YES (***) |
| RASNet vs. 3D U-Net | 95% Hausdorff Distance (HD95, mm) | 10.2924 | 9.8780 | 0.1971 | NO |
| RASNet vs. 3D U-Net | Average Surface Distance (ASD, mm) | 1.5981 | 1.6903 | 0.0011 | YES (*) |
| RASNet vs. 3D U-Net | Centerline Dice (clDice) | 0.8592 | 0.7027 | 2.46e-25 | YES (***) |
| RASNet vs. 3D U-Net | Centerline Recall / Tree Completeness (T_sens) | 0.8018 | 0.6534 | 3.03e-23 | YES (***) |
| RASNet vs. V-Net | Dice Similarity Coefficient (DSC) | 0.7765 | 0.7474 | 5.70e-11 | YES (***) |
| RASNet vs. V-Net | Intersection-over-Union (IoU) | 0.6396 | 0.6006 | 2.09e-11 | YES (***) |
| RASNet vs. V-Net | Precision (PPV) | 0.8801 | 0.7545 | 1.84e-25 | YES (***) |
| RASNet vs. V-Net | Recall / Sensitivity | 0.7016 | 0.7499 | 7.01e-22 | YES (***) |
| RASNet vs. V-Net | 95% Hausdorff Distance (HD95, mm) | 10.2924 | 21.8050 | 2.46e-10 | YES (***) |
| RASNet vs. V-Net | Average Surface Distance (ASD, mm) | 1.5981 | 3.1758 | 1.04e-11 | YES (***) |
| RASNet vs. V-Net | Centerline Dice (clDice) | 0.8592 | 0.8075 | 1.44e-14 | YES (***) |
| RASNet vs. V-Net | Centerline Recall / Tree Completeness (T_sens) | 0.8018 | 0.8571 | 9.15e-15 | YES (***) |
| RASNet vs. nnU-Net V2 | Dice Similarity Coefficient (DSC) | 0.7765 | 0.7687 | 0.2262 | NO |
| RASNet vs. nnU-Net V2 | Intersection-over-Union (IoU) | 0.6396 | 0.6289 | 0.2262 | NO |
| RASNet vs. nnU-Net V2 | Precision (PPV) | 0.8801 | 0.7391 | 1.84e-25 | YES (***) |
| RASNet vs. nnU-Net V2 | Recall / Sensitivity | 0.7016 | 0.8106 | 1.84e-25 | YES (***) |
| RASNet vs. nnU-Net V2 | 95% Hausdorff Distance (HD95, mm) | 10.2924 | 21.0983 | 4.60e-10 | YES (***) |
| RASNet vs. nnU-Net V2 | Average Surface Distance (ASD, mm) | 1.5981 | 3.0966 | 6.29e-12 | YES (***) |
| RASNet vs. nnU-Net V2 | Centerline Dice (clDice) | 0.8592 | 0.8201 | 3.96e-09 | YES (***) |
| RASNet vs. nnU-Net V2 | Centerline Recall / Tree Completeness (T_sens) | 0.8018 | 0.9111 | 1.84e-25 | YES (***) |
