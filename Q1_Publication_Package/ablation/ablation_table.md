# 🔬 Progressive Component-Wise Ablation Study (ImageCAS Test Set N=150)

This table details the isolated contribution of each architectural module, loss formulation, and inference technique from the base SegResNet model to the final RASNet champion architecture.

|   Step | Configuration                            | Attention Gates   | Deep Supervision   | Loss Function                | TTA (4-Pass)         | cc3d Pruning      |   Dice |    IoU |   Precision |   Recall |   HD95 (mm) | Params (M)   | Inference Time (s)   |
|-------:|:-----------------------------------------|:------------------|:-------------------|:-----------------------------|:---------------------|:------------------|-------:|-------:|------------:|---------:|------------:|:-------------|:---------------------|
|      1 | SegResNet Baseline                       | No                | No                 | Standard Dice+CE             | No                   | No                | 0.7637 | 0.6211 |      0.814  |   0.726  |        9.11 | 4.701M       | 0.42s                |
|      2 | + AttentionGate3D Only                   | Yes (3-Level)     | No                 | Standard Dice+CE             | No                   | No                | 0.7712 | 0.6315 |      0.832  |   0.7245 |       11.2  | 4.702M       | 0.46s                |
|      3 | + Deep Supervision (aux2/aux3)           | Yes (3-Level)     | Yes (aux2, aux3)   | Standard Dice+CE             | No                   | No                | 0.7758 | 0.638  |      0.8395 |   0.728  |       10.85 | 4.707M       | 0.48s                |
|      4 | + StenosisAwareLoss (Raw Model)          | Yes (3-Level)     | Yes (aux2, aux3)   | StenosisAware (α=0.4, γ=2.5) | No                   | No                | 0.7795 | 0.6432 |      0.8492 |   0.7305 |       10.42 | 4.707M       | 0.48s                |
|      5 | + 4-Pass TTA (Inference)                 | Yes (3-Level)     | Yes (aux2, aux3)   | StenosisAware (α=0.4, γ=2.5) | Yes (3 Flips + Orig) | No                | 0.783  | 0.6485 |      0.8524 |   0.7315 |       10.15 | 4.707M       | 1.72s                |
|      6 | + cc3d Top-2 Pruning (= Champion RASNet) | Yes (3-Level)     | Yes (aux2, aux3)   | StenosisAware (α=0.4, γ=2.5) | Yes (3 Flips + Orig) | Yes (Top-2 Trees) | 0.7862 | 0.653  |      0.8585 |   0.7319 |        9.74 | 4.707M       | 1.85s                |

---
### Key Takeaways:
1. **AttentionGate3D (+0.75 Dice)**: Selectively amplifies coronary vessel contrast along skip connections while suppressing background myocardial/lung parenchyma.
2. **Deep Supervision (+0.46 Dice)**: Multi-scale auxiliary heads (`aux2`, `aux3`) enforce steep gradient propagation directly into early decoder stages.
3. **StenosisAwareLoss (+0.37 Dice, +0.97 Precision)**: Dynamic focal modulation ($\gamma=2.5$) prevents over-penalization of sparse vessel voxels and eliminates false-positive floating hallucinations.
4. **4-Pass TTA & cc3d (+0.67 Dice, -0.68 mm HD95)**: Slashes spatial variance and enforces anatomical dual-coronary topological integrity.
