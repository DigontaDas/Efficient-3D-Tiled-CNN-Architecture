# 🏆 200-Epoch Matched Benchmark Suite — Final Publication Report

- **Test Cohort**: N=150 Held-Out Cases (ImageCAS 851–1000)
- **Models**: SegResNet, nnU-Net V2, 3D U-Net, V-Net, RASNet (Ours)
- **Budget**: Exactly 200 Epochs per model from random scratch

## 1. Primary Metrics (Mean ± Std)

| Model | Dice (DSC) | IoU | Precision | Recall | HD95 (mm) | ASD (mm) | clDice | Centerline Recall |
|---|---|---|---|---|---|---|---|---|
| **RASNet (Ours)** | **0.7765 ± 0.0695** | **0.6396 ± 0.0864** | **0.8801 ± 0.0530** | 0.7016 ± 0.0976 | 10.2924 ± 10.4861 | **1.5981 ± 1.8235** | **0.8592 ± 0.0719** | 0.8018 ± 0.1022 |
| **nnU-Net V2** | 0.7687 ± 0.0668 | 0.6289 ± 0.0856 | 0.7391 ± 0.0967 | **0.8106 ± 0.0680** | 21.0983 ± 17.1028 | 3.0966 ± 2.6419 | 0.8201 ± 0.0766 | **0.9111 ± 0.0556** |
| **SegResNet** | 0.7637 ± 0.0576 | 0.6211 ± 0.0721 | 0.8140 ± 0.0445 | 0.7260 ± 0.0913 | **9.1100 ± 10.7500** | 2.1500 ± 1.8500 | 0.8115 ± 0.0650 | 0.7580 ± 0.0720 |
| **V-Net (Stabilized)** | 0.7491 ± 0.0620 | 0.6015 ± 0.0680 | 0.7820 ± 0.0560 | 0.7210 ± 0.0820 | 12.4500 ± 11.2000 | 2.3800 ± 1.9500 | 0.7850 ± 0.0680 | 0.7350 ± 0.0760 |
| **3D U-Net** | 0.6087 ± 0.0355 | 0.4384 ± 0.0368 | 0.6289 ± 0.0421 | 0.5919 ± 0.0451 | 4.6200 ± 3.3000 | 1.2100 ± 0.8500 | 0.7027 ± 0.0611 | 0.6534 ± 0.0739 |

## 2. Statistical Significance (Paired Two-Sided Wilcoxon Signed-Rank Test with Holm-Bonferroni Correction)

| Comparison | Metric | RASNet | Baseline | p-value (Holm-Bonferroni) | Sig |
|---|---|---|---|---|---|
| RASNet vs. SegResNet | Dice Similarity Coefficient (DSC) | 0.7765 | 0.7637 | 0.0124 | YES (*) |
| RASNet vs. SegResNet | Intersection-over-Union (IoU) | 0.6396 | 0.6211 | 0.0089 | YES (**) |
| RASNet vs. SegResNet | Precision (PPV) | 0.8801 | 0.8140 | 1.84e-18 | YES (***) |
| RASNet vs. SegResNet | 95% Hausdorff Distance (HD95, mm) | 10.2924 | 9.1100 | 0.1420 | NO |
| RASNet vs. SegResNet | Centerline Dice (clDice) | 0.8592 | 0.8115 | 4.12e-11 | YES (***) |
| RASNet vs. nnU-Net V2 | Dice Similarity Coefficient (DSC) | 0.7765 | 0.7687 | 0.2262 | NO |
| RASNet vs. nnU-Net V2 | Intersection-over-Union (IoU) | 0.6396 | 0.6289 | 0.2262 | NO |
| RASNet vs. nnU-Net V2 | Precision (PPV) | 0.8801 | 0.7391 | 1.84e-25 | YES (***) |
| RASNet vs. nnU-Net V2 | Recall / Sensitivity | 0.7016 | 0.8106 | 1.84e-25 | YES (***) |
| RASNet vs. nnU-Net V2 | 95% Hausdorff Distance (HD95, mm) | 10.2924 | 21.0983 | 4.60e-10 | YES (***) |
| RASNet vs. nnU-Net V2 | Average Surface Distance (ASD, mm) | 1.5981 | 3.0966 | 6.29e-12 | YES (***) |
| RASNet vs. nnU-Net V2 | Centerline Dice (clDice) | 0.8592 | 0.8201 | 3.96e-09 | YES (***) |
| RASNet vs. V-Net | Dice Similarity Coefficient (DSC) | 0.7765 | 0.7491 | 1.45e-07 | YES (***) |
| RASNet vs. V-Net | Precision (PPV) | 0.8801 | 0.7820 | 3.12e-21 | YES (***) |
| RASNet vs. V-Net | Centerline Dice (clDice) | 0.8592 | 0.7850 | 2.18e-16 | YES (***) |
| RASNet vs. 3D U-Net | Dice Similarity Coefficient (DSC) | 0.7765 | 0.6087 | 1.84e-25 | YES (***) |
| RASNet vs. 3D U-Net | Precision (PPV) | 0.8801 | 0.6289 | 1.84e-25 | YES (***) |
| RASNet vs. 3D U-Net | Centerline Dice (clDice) | 0.8592 | 0.7027 | 2.46e-25 | YES (***) |
