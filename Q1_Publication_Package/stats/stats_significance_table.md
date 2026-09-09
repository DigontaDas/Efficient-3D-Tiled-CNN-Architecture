# 📊 Statistical Significance Testing: RASNet vs. SOTA Baselines (200 Epochs Matched)

- **Test Cohort**: ImageCAS Reserved Test Set ($N=150$, Cases 851–1000)
- **Statistical Test**: Paired Two-Sided Wilcoxon Signed-Rank Test (`scipy.stats.wilcoxon`)
- **Multiple Hypothesis Adjustment**: Step-Down Holm-Bonferroni Family-Wise Correction
- **Significance Thresholds**: `***` $p < 0.001$, `**` $p < 0.01$, `*` $p < 0.05$, `n.s.` $p \ge 0.05$

| Comparison            | Metric                                         | RASNet (Mean ± SD)   | Baseline (Mean ± SD)   |   Δ Mean |   Raw p-value |   Holm-Adj p-value | Sig.   |
|:----------------------|:-----------------------------------------------|:---------------------|:-----------------------|---------:|--------------:|-------------------:|:-------|
| RASNet vs. SegResNet  | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.7637 ± 0.0576        |  +0.0128 |    0.0028     |         0.0124     | *      |
| RASNet vs. SegResNet  | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.6211 ± 0.0721        |  +0.0185 |    0.0019     |         0.0089     | **     |
| RASNet vs. SegResNet  | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.8140 ± 0.0445        |  +0.0661 |    1.84e-18   |         2.21e-17   | ***    |
| RASNet vs. SegResNet  | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.7260 ± 0.0913        |  -0.0244 |    0.0412     |         0.1236     | n.s.   |
| RASNet vs. SegResNet  | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 9.1100 ± 10.7500       |  +1.1824 |    0.0820     |         0.1640     | n.s.   |
| RASNet vs. SegResNet  | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 2.1500 ± 1.8500        |  -0.5519 |    0.0031     |         0.0124     | *      |
| RASNet vs. SegResNet  | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.8115 ± 0.0650        |  +0.0477 |    4.12e-11   |         3.71e-10   | ***    |
| RASNet vs. SegResNet  | Centerline Recall / Tree Completeness (T_sens) | 0.8018 ± 0.1022      | 0.7580 ± 0.0720        |  +0.0438 |    1.25e-08   |         1.00e-07   | ***    |
| RASNet vs. nnU-Net V2 | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.7687 ± 0.0668        |  +0.0078 |    0.1152     |         0.4523     | n.s.   |
| RASNet vs. nnU-Net V2 | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.6289 ± 0.0856        |  +0.0106 |    0.1131     |         0.4523     | n.s.   |
| RASNet vs. nnU-Net V2 | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.7391 ± 0.0967        |  +0.1410 |    2.30e-26   |         7.36e-25   | ***    |
| RASNet vs. nnU-Net V2 | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.8106 ± 0.0680        |  -0.1091 |    2.30e-26   |         7.36e-25   | ***    |
| RASNet vs. nnU-Net V2 | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 21.0983 ± 17.1028      | -10.8059 |    1.15e-10   |         9.20e-10   | ***    |
| RASNet vs. nnU-Net V2 | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 3.0966 ± 2.6419        |  -1.4985 |    1.26e-12   |         1.26e-11   | ***    |
| RASNet vs. nnU-Net V2 | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.8201 ± 0.0766        |  +0.0391 |    1.32e-09   |         9.24e-09   | ***    |
| RASNet vs. nnU-Net V2 | Centerline Recall / Tree Completeness (T_sens) | 0.8018 ± 0.1022      | 0.9111 ± 0.0556        |  -0.1093 |    2.30e-26   |         7.36e-25   | ***    |
| RASNet vs. V-Net      | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.7491 ± 0.0620        |  +0.0274 |    1.45e-07   |         1.31e-06   | ***    |
| RASNet vs. V-Net      | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.6015 ± 0.0680        |  +0.0381 |    1.12e-07   |         1.01e-06   | ***    |
| RASNet vs. V-Net      | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.7820 ± 0.0560        |  +0.0981 |    3.12e-21   |         3.74e-20   | ***    |
| RASNet vs. V-Net      | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.7210 ± 0.0820        |  -0.0194 |    0.0825     |         0.1650     | n.s.   |
| RASNet vs. V-Net      | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 12.4500 ± 11.2000      |  -2.1576 |    0.0025     |         0.0100     | **     |
| RASNet vs. V-Net      | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 2.3800 ± 1.9500        |  -0.7819 |    1.85e-05   |         1.48e-04   | ***    |
| RASNet vs. V-Net      | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.7850 ± 0.0680        |  +0.0742 |    2.18e-16   |         2.40e-15   | ***    |
| RASNet vs. 3D U-Net   | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.6087 ± 0.0355        |  +0.1678 |    2.30e-26   |         7.36e-25   | ***    |
| RASNet vs. 3D U-Net   | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.4384 ± 0.0368        |  +0.2012 |    2.30e-26   |         7.36e-25   | ***    |
| RASNet vs. 3D U-Net   | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.6289 ± 0.0421        |  +0.2512 |    2.30e-26   |         7.36e-25   | ***    |
| RASNet vs. 3D U-Net   | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.5919 ± 0.0451        |  +0.1097 |    5.44e-26   |         1.03e-24   | ***    |
| RASNet vs. 3D U-Net   | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.7027 ± 0.0611        |  +0.1566 |    4.92e-26   |         9.85e-25   | ***    |

---
*Note: Statistical testing performed on paired per-case outputs across $N=150$ held-out ImageCAS test volumes.*
