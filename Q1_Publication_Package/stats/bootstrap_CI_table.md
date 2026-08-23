# 📈 RASNet Performance with Non-Parametric 95% Bootstrap Confidence Intervals

- **Evaluation Dataset**: ImageCAS Test Split ($N=150$ Cases)
- **Bootstrap Resamples**: $B = 2000$ iterations (Percentile Method, Seed = 42)

| Evaluation Metric                 |   N | Mean (95% Bootstrap CI)   | Median [IQR]    |
|:----------------------------------|----:|:--------------------------|:----------------|
| Dice Similarity Coefficient (DSC) | 150 | 0.7862 [0.7743, 0.7963]   | 0.8086 [0.0766] |
| Intersection-over-Union (IoU)     | 150 | 0.6530 [0.6383, 0.6657]   | 0.6786 [0.1058] |
| Precision (PPV)                   | 150 | 0.8585 [0.8469, 0.8690]   | 0.8759 [0.0728] |
| Recall / Sensitivity              | 150 | 0.7319 [0.7154, 0.7462]   | 0.7538 [0.1181] |
| 95% Hausdorff Distance (HD95, mm) | 150 | 9.7434 [8.0021, 11.6001]  | 5.2328 [8.9483] |
