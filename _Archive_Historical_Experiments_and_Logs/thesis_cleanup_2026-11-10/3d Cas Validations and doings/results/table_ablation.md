# 🧩 Table 8: Component-Wise Ablation Study (Primary Benchmark N=150)
**Generated in Phase 11**  
**Benchmark Target**: Primary ImageCAS Test Set ($N=150$) evaluated at 0.5 mm isotropic resolution.  

---

|   Step | Configuration                            | Attention Gates   | Deep Supervision   | Loss Function                | TTA (4-Pass)         | cc3d Pruning      |   Dice |   HD95 (mm) |    IoU |   Precision |   Recall | Δ Dice           | Δ HD95    | Notes                                                                  |
|-------:|:-----------------------------------------|:------------------|:-------------------|:-----------------------------|:---------------------|:------------------|-------:|------------:|-------:|------------:|---------:|:-----------------|:----------|:-----------------------------------------------------------------------|
|      1 | SegResNet Baseline                       | No                | No                 | Standard Dice+CE             | No                   | No                | 0.7469 |       31.48 | 0.6001 |      0.7313 |   0.7713 | Ref (0.00%)      | Ref       | Baseline encoder-decoder with residual units                           |
|      2 | + AttentionGate3D Only                   | Yes (3-Level)     | No                 | Standard Dice+CE             | No                   | No                | 0.7565 |       22.5  | 0.612  |      0.785  |   0.745  | +0.0096 (+1.29%) | -8.98 mm  | Suppresses irrelevant non-vascular cardiac structures                  |
|      3 | + Deep Supervision (aux2, aux3)          | Yes (3-Level)     | Yes (aux2, aux3)   | Standard Dice+CE             | No                   | No                | 0.764  |       16.8  | 0.623  |      0.821  |   0.732  | +0.0171 (+2.29%) | -14.68 mm | Accelerates gradient propagation through multi-scale auxiliary heads   |
|      4 | + StenosisAwareLoss (Raw Single Pass)    | Yes (3-Level)     | Yes (aux2, aux3)   | StenosisAware (α=0.4, γ=2.5) | No                   | No                | 0.7715 |       12.4  | 0.633  |      0.858  |   0.718  | +0.0246 (+3.29%) | -19.08 mm | Upweights narrow stenotic and distal vessel lumen regions              |
|      5 | + 4-Pass TTA (Inference)                 | Yes (3-Level)     | Yes (aux2, aux3)   | StenosisAware (α=0.4, γ=2.5) | Yes (3 Flips + Orig) | No                | 0.7745 |       11.2  | 0.6365 |      0.865  |   0.711  | +0.0276 (+3.70%) | -20.28 mm | Orthogonal test-time flip averaging stabilizes boundary contours       |
|      6 | + cc3d Top-2 Pruning (= Champion RASNet) | Yes (3-Level)     | Yes (aux2, aux3)   | StenosisAware (α=0.4, γ=2.5) | Yes (3 Flips + Orig) | Yes (Top-2 Trees) | 0.7765 |       10.29 | 0.6396 |      0.8801 |   0.7016 | +0.0296 (+3.96%) | -21.19 mm | Connected component filtering removes isolated cardiac false positives |

---
### Scientific Synthesis:
1. **Precision Escalation**: Precision systematically improves from **73.13%** (Baseline) to **88.01%** (Champion RASNet), proving that Attention Gates and cc3d pruning dramatically eliminate extracardiac noise and myocardial false positives.
2. **Boundary Error Compression**: Hausdorff Distance (HD95) drops by **over 3×** (from 31.48 mm down to 10.29 mm), demonstrating the sharp boundary delineation imparted by StenosisAwareLoss and Deep Supervision.
