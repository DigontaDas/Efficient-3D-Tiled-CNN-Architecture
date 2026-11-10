# 📊 Statistical Significance Testing: RASNet vs. SOTA Baselines (200 Epochs Matched)

- **Test Cohort**: ImageCAS Reserved Test Set ($N=150$, Cases 851–1000)
- **Statistical Test**: Paired Two-Sided Wilcoxon Signed-Rank Test (`scipy.stats.wilcoxon`)
- **Multiple Hypothesis Adjustment**: Step-Down Holm-Bonferroni Family-Wise Correction
- **Significance Thresholds**: `***` $p < 0.001$, `**` $p < 0.01$, `*` $p < 0.05$, `n.s.` $p \ge 0.05$

| Comparison            | Metric                                         | RASNet (Mean ± SD)   | Baseline (Mean ± SD)   |   Δ Mean |   Raw p-value |   Holm-Adj p-value | Sig.   |
|:----------------------|:-----------------------------------------------|:---------------------|:-----------------------|---------:|--------------:|-------------------:|:-------|
| RASNet vs. SegResNet  | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.7469 ± 0.0640        |   0.0296 |    2.4709e-12 |         2.718e-11  | ***    |
| RASNet vs. SegResNet  | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.6001 ± 0.0788        |   0.0395 |    5.5409e-13 |         7.2032e-12 | ***    |
| RASNet vs. SegResNet  | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.7313 ± 0.0911        |   0.1488 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. SegResNet  | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.7713 ± 0.0654        |  -0.0697 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. SegResNet  | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 31.4775 ± 18.1586      | -21.1851 |    5.2203e-21 |         8.7176e-20 | ***    |
| RASNet vs. SegResNet  | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 4.6614 ± 2.7181        |  -3.0633 |    1.2518e-22 |         2.2533e-21 | ***    |
| RASNet vs. SegResNet  | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.7769 ± 0.0819        |   0.0823 |    5.128e-21  |         8.7176e-20 | ***    |
| RASNet vs. SegResNet  | Centerline Recall / Tree Completeness (T_sens) | 0.8018 ± 0.1022      | 0.8866 ± 0.0579        |  -0.0848 |    3.9466e-26 |         9.0771e-25 | ***    |
| RASNet vs. nnU-Net V2 | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.7687 ± 0.0668        |   0.0078 |    0.1152     |         0.3393     | n.s.   |
| RASNet vs. nnU-Net V2 | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.6289 ± 0.0856        |   0.0106 |    0.1131     |         0.3393     | n.s.   |
| RASNet vs. nnU-Net V2 | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.7391 ± 0.0967        |   0.141  |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. nnU-Net V2 | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.8106 ± 0.0680        |  -0.1091 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. nnU-Net V2 | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 21.0983 ± 17.1028      | -10.8059 |    1.1498e-10 |         8.0484e-10 | ***    |
| RASNet vs. nnU-Net V2 | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 3.0966 ± 2.6419        |  -1.4985 |    1.2589e-12 |         1.5107e-11 | ***    |
| RASNet vs. nnU-Net V2 | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.8201 ± 0.0766        |   0.0391 |    1.32e-09   |         6.6e-09    | ***    |
| RASNet vs. nnU-Net V2 | Centerline Recall / Tree Completeness (T_sens) | 0.8018 ± 0.1022      | 0.9111 ± 0.0556        |  -0.1093 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. V-Net      | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.7474 ± 0.0633        |   0.0291 |    2.8507e-11 |         2.2806e-10 | ***    |
| RASNet vs. V-Net      | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.6006 ± 0.0786        |   0.0389 |    6.9542e-12 |         6.2588e-11 | ***    |
| RASNet vs. V-Net      | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.7545 ± 0.0982        |   0.1256 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. V-Net      | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.7499 ± 0.0678        |  -0.0483 |    1.0017e-22 |         1.9033e-21 | ***    |
| RASNet vs. V-Net      | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 21.8050 ± 19.1359      | -11.5127 |    2.4597e-10 |         1.4758e-09 | ***    |
| RASNet vs. V-Net      | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 3.1758 ± 2.7710        |  -1.5777 |    2.6069e-12 |         2.718e-11  | ***    |
| RASNet vs. V-Net      | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.8075 ± 0.0759        |   0.0517 |    2.8764e-15 |         4.027e-14  | ***    |
| RASNet vs. V-Net      | Centerline Recall / Tree Completeness (T_sens) | 0.8018 ± 0.1022      | 0.8571 ± 0.0613        |  -0.0554 |    1.5244e-15 |         2.2865e-14 | ***    |
| RASNet vs. 3D U-Net   | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.5561 ± 0.0458        |   0.2205 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. 3D U-Net   | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.3865 ± 0.0429        |   0.2531 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. 3D U-Net   | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.6069 ± 0.0646        |   0.2732 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. 3D U-Net   | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.5178 ± 0.0536        |   0.1838 |    5.4418e-26 |         1.1428e-24 | ***    |
| RASNet vs. 3D U-Net   | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 9.8780 ± 7.1789        |   0.4144 |    0.1971     |         0.3393     | n.s.   |
| RASNet vs. 3D U-Net   | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 1.6903 ± 1.0444        |  -0.0921 |    0.0006     |         0.0022     | **     |
| RASNet vs. 3D U-Net   | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.7027 ± 0.0611        |   0.1566 |    4.9248e-26 |         1.0835e-24 | ***    |
| RASNet vs. 3D U-Net   | Centerline Recall / Tree Completeness (T_sens) | 0.8018 ± 0.1022      | 0.6534 ± 0.0739        |   0.1484 |    1.0096e-23 |         2.0192e-22 | ***    |

---
*Note: Statistical testing performed on paired per-case outputs across identical physical 3D coordinate volumes.*
