# 📊 Statistical Significance Testing: RASNet vs. SOTA Baselines (200 Epochs Matched)

- **Test Cohort**: ImageCAS Reserved Test Set ($N=150$, Cases 851–1000)
- **Statistical Test**: Paired Two-Sided Wilcoxon Signed-Rank Test (`scipy.stats.wilcoxon`)
- **Multiple Hypothesis Adjustment**: Step-Down Holm-Bonferroni Family-Wise Correction
- **Significance Thresholds**: `***` $p < 0.001$, `**` $p < 0.01$, `*` $p < 0.05$, `n.s.` $p \ge 0.05$

| Comparison            | Metric                                         | RASNet (Mean ± SD)   | Baseline (Mean ± SD)   |   Δ Mean |   Raw p-value |   Holm-Adj p-value | Sig.   |
|:----------------------|:-----------------------------------------------|:---------------------|:-----------------------|---------:|--------------:|-------------------:|:-------|
| RASNet vs. SegResNet  | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.6058 ± 0.0612        |   0.1707 |    3.3006e-26 |         7.3587e-25 | ***    |
| RASNet vs. SegResNet  | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.4373 ± 0.0632        |   0.2023 |    3.1709e-26 |         7.3587e-25 | ***    |
| RASNet vs. SegResNet  | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.5140 ± 0.0733        |   0.3661 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. SegResNet  | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.7483 ± 0.0678        |  -0.0467 |    1.5333e-08 |         9.1995e-08 | ***    |
| RASNet vs. SegResNet  | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 22.6145 ± 15.4238      | -12.3221 |    3.5825e-14 |         4.299e-13  | ***    |
| RASNet vs. SegResNet  | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 3.8455 ± 2.4528        |  -2.2474 |    1.4808e-18 |         2.2211e-17 | ***    |
| RASNet vs. SegResNet  | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.7457 ± 0.0676        |   0.1136 |    6.0552e-23 |         9.6883e-22 | ***    |
| RASNet vs. SegResNet  | Centerline Recall / Tree Completeness (T_sens) | 0.8018 ± 0.1022      | 0.7274 ± 0.0746        |   0.0744 |    2.4086e-13 |         2.6495e-12 | ***    |
| RASNet vs. nnU-Net V2 | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.7687 ± 0.0668        |   0.0078 |    0.1152     |         0.4523     | n.s.   |
| RASNet vs. nnU-Net V2 | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.6289 ± 0.0856        |   0.0106 |    0.1131     |         0.4523     | n.s.   |
| RASNet vs. nnU-Net V2 | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.7391 ± 0.0967        |   0.141  |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. nnU-Net V2 | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.8106 ± 0.0680        |  -0.1091 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. nnU-Net V2 | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 21.0983 ± 17.1028      | -10.8059 |    1.1498e-10 |         9.1981e-10 | ***    |
| RASNet vs. nnU-Net V2 | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 3.0966 ± 2.6419        |  -1.4985 |    1.2589e-12 |         1.2589e-11 | ***    |
| RASNet vs. nnU-Net V2 | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.8201 ± 0.0766        |   0.0391 |    1.32e-09   |         9.24e-09   | ***    |
| RASNet vs. nnU-Net V2 | Centerline Recall / Tree Completeness (T_sens) | 0.8018 ± 0.1022      | 0.9111 ± 0.0556        |  -0.1093 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. V-Net      | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.5957 ± 0.0628        |   0.1808 |    2.9264e-26 |         7.3587e-25 | ***    |
| RASNet vs. V-Net      | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.4270 ± 0.0633        |   0.2126 |    2.9264e-26 |         7.3587e-25 | ***    |
| RASNet vs. V-Net      | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.5166 ± 0.0742        |   0.3635 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. V-Net      | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.7157 ± 0.0848        |  -0.0142 |    0.2999     |         0.4523     | n.s.   |
| RASNet vs. V-Net      | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 19.8783 ± 14.5715      |  -9.586  |    5.9391e-11 |         5.3452e-10 | ***    |
| RASNet vs. V-Net      | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 3.2127 ± 2.1601        |  -1.6146 |    7.821e-15  |         1.0167e-13 | ***    |
| RASNet vs. V-Net      | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.7337 ± 0.0715        |   0.1256 |    2.0748e-23 |         3.5272e-22 | ***    |
| RASNet vs. V-Net      | Centerline Recall / Tree Completeness (T_sens) | 0.8018 ± 0.1022      | 0.6890 ± 0.0840        |   0.1128 |    3.3428e-18 |         4.6799e-17 | ***    |
| RASNet vs. 3D U-Net   | Dice Similarity Coefficient (DSC)              | 0.7765 ± 0.0695      | 0.5561 ± 0.0458        |   0.2205 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. 3D U-Net   | Intersection-over-Union (IoU)                  | 0.6396 ± 0.0864      | 0.3865 ± 0.0429        |   0.2531 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. 3D U-Net   | Precision (PPV)                                | 0.8801 ± 0.0530      | 0.6069 ± 0.0646        |   0.2732 |    2.2996e-26 |         7.3587e-25 | ***    |
| RASNet vs. 3D U-Net   | Recall / Sensitivity                           | 0.7016 ± 0.0976      | 0.5178 ± 0.0536        |   0.1838 |    5.4418e-26 |         1.034e-24  | ***    |
| RASNet vs. 3D U-Net   | 95% Hausdorff Distance (HD95, mm)              | 10.2924 ± 10.4861    | 9.8780 ± 7.1789        |   0.4144 |    0.1971     |         0.4523     | n.s.   |
| RASNet vs. 3D U-Net   | Average Surface Distance (ASD, mm)             | 1.5981 ± 1.8235      | 1.6903 ± 1.0444        |  -0.0921 |    0.0006     |         0.0028     | **     |
| RASNet vs. 3D U-Net   | Centerline Dice (clDice)                       | 0.8592 ± 0.0719      | 0.7027 ± 0.0611        |   0.1566 |    4.9248e-26 |         9.8497e-25 | ***    |
| RASNet vs. 3D U-Net   | Centerline Recall / Tree Completeness (T_sens) | 0.8018 ± 0.1022      | 0.6534 ± 0.0739        |   0.1484 |    1.0096e-23 |         1.8173e-22 | ***    |

---
*Note: Statistical testing performed on paired per-case outputs across identical physical 3D coordinate volumes.*
