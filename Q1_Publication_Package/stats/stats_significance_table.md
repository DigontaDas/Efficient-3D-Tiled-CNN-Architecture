# 📊 Statistical Significance Testing: RASNet vs. SOTA Baselines

- **Test Cohort**: ImageCAS Reserved Test Set ($N=150$, Cases 851–1000)
- **Statistical Test**: Paired Two-Sided Wilcoxon Signed-Rank Test (`scipy.stats.wilcoxon`)
- **Multiple Hypothesis Adjustment**: Step-Down Holm-Bonferroni Family-Wise Correction
- **Significance Thresholds**: `***` $p < 0.001$, `**` $p < 0.01$, `*` $p < 0.05$, `n.s.` $p \ge 0.05$

| Comparison            | Metric                            | RASNet (Mean ± SD)   | Baseline (Mean ± SD)   |   Δ Mean |   Raw p-value |   Holm-Adj p-value | Sig.   |
|:----------------------|:----------------------------------|:---------------------|:-----------------------|---------:|--------------:|-------------------:|:-------|
| RASNet vs. SegResNet  | Dice Similarity Coefficient (DSC) | 0.7862 ± 0.0724      | 0.7637 ± 0.0576        |   0.0225 |    6.1831e-16 |         3.0916e-15 | ***    |
| RASNet vs. SegResNet  | Intersection-over-Union (IoU)     | 0.6530 ± 0.0901      | 0.6211 ± 0.0721        |   0.0319 |    2.9412e-16 |         1.7647e-15 | ***    |
| RASNet vs. SegResNet  | Precision (PPV)                   | 0.8585 ± 0.0703      | 0.8140 ± 0.0445        |   0.0445 |    3.8527e-21 |         2.6969e-20 | ***    |
| RASNet vs. SegResNet  | Recall / Sensitivity              | 0.7319 ± 0.0973      | 0.7260 ± 0.0913        |   0.0059 |    0.047      |         0.0941     | n.s.   |
| RASNet vs. SegResNet  | 95% Hausdorff Distance (HD95, mm) | 9.7434 ± 11.4794     | 9.1067 ± 10.7475       |   0.6368 |    0.3516     |         0.3516     | n.s.   |
| RASNet vs. nnU-Net V2 | Dice Similarity Coefficient (DSC) | 0.7862 ± 0.0724      | 0.6003 ± 0.0780        |   0.1859 |    7.7895e-26 |         8.5685e-25 | ***    |
| RASNet vs. nnU-Net V2 | Intersection-over-Union (IoU)     | 0.6530 ± 0.0901      | 0.4332 ± 0.0789        |   0.2199 |    4.1139e-26 |         5.3481e-25 | ***    |
| RASNet vs. nnU-Net V2 | Precision (PPV)                   | 0.8585 ± 0.0703      | 0.5354 ± 0.1044        |   0.3231 |    2.4426e-26 |         3.6638e-25 | ***    |
| RASNet vs. nnU-Net V2 | Recall / Sensitivity              | 0.7319 ± 0.0973      | 0.7017 ± 0.0820        |   0.0302 |    1.5843e-08 |         6.3373e-08 | ***    |
| RASNet vs. nnU-Net V2 | 95% Hausdorff Distance (HD95, mm) | 9.7434 ± 11.4794     | 58.3003 ± 14.4442      | -48.5569 |    2.5427e-26 |         3.6638e-25 | ***    |
| RASNet vs. 3D U-Net   | Dice Similarity Coefficient (DSC) | 0.7862 ± 0.0724      | 0.6087 ± 0.0355        |   0.1775 |    3.1088e-25 |         2.7979e-24 | ***    |
| RASNet vs. 3D U-Net   | Intersection-over-Union (IoU)     | 0.6530 ± 0.0901      | 0.4384 ± 0.0368        |   0.2146 |    9.3156e-26 |         9.3156e-25 | ***    |
| RASNet vs. 3D U-Net   | Precision (PPV)                   | 0.8585 ± 0.0703      | 0.6289 ± 0.0421        |   0.2296 |    5.4418e-26 |         6.5302e-25 | ***    |
| RASNet vs. 3D U-Net   | Recall / Sensitivity              | 0.7319 ± 0.0973      | 0.5919 ± 0.0451        |   0.14   |    2.459e-23  |         1.9672e-22 | ***    |
| RASNet vs. 3D U-Net   | 95% Hausdorff Distance (HD95, mm) | 9.7434 ± 11.4794     | 4.6241 ± 3.3040        |   5.1193 |    4.4641e-07 |         1.3392e-06 | ***    |

---
*Note: Statistical testing performed on paired per-case outputs across identical physical 3D coordinate volumes.*
