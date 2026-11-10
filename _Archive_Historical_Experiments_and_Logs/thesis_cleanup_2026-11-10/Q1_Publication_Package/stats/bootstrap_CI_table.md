# 📈 RASNet Performance with Non-Parametric 95% Bootstrap Confidence Intervals (200 Epochs)

- **Evaluation Dataset**: ImageCAS Test Split ($N=150$ Cases)
- **Bootstrap Resamples**: $B = 2000$ iterations (Percentile Method, Seed = 42)

| Evaluation Metric                              |   N | Mean (95% Bootstrap CI)   | Median [IQR]     |
|:-----------------------------------------------|----:|:--------------------------|:-----------------|
| Dice Similarity Coefficient (DSC)              | 150 | 0.7765 [0.7648, 0.7867]   | 0.7965 [0.0789]  |
| Intersection-over-Union (IoU)                  | 150 | 0.6396 [0.6250, 0.6523]   | 0.6619 [0.1072]  |
| Precision (PPV)                                | 150 | 0.8801 [0.8715, 0.8881]   | 0.8920 [0.0611]  |
| Recall / Sensitivity                           | 150 | 0.7016 [0.6850, 0.7159]   | 0.7178 [0.1249]  |
| 95% Hausdorff Distance (HD95, mm)              | 150 | 10.2924 [8.7938, 12.0175] | 6.3320 [10.1270] |
| Average Surface Distance (ASD, mm)             | 150 | 1.5981 [1.3410, 1.9184]   | 0.9768 [1.1727]  |
| Centerline Dice (clDice)                       | 150 | 0.8592 [0.8478, 0.8696]   | 0.8746 [0.0814]  |
| Centerline Recall / Tree Completeness (T_sens) | 150 | 0.8018 [0.7856, 0.8167]   | 0.8124 [0.1387]  |
