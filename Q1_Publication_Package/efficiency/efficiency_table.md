# ⚡ Computational Efficiency & Resource Consumption Benchmark

- **Hardware Platform**: NVIDIA GeForce RTX 4080 SUPER (16 GB VRAM)
- **Patch Benchmark Dimension**: $1 \times 1 \times 96 \times 96 \times 96$ voxels
- **Evaluation Cohort**: ImageCAS Test Split ($N=150$ Cases)

| Model         | Parameters (M)   |   GFLOPs (96³ Patch) | Patch Latency (ms)   | Full-Volume Inference (s)   | Peak VRAM (MB)   | Dice (DSC)   |
|:--------------|:-----------------|---------------------:|:---------------------|:----------------------------|:-----------------|:-------------|
| RASNet (Ours) | 4.71M            |               123.39 | 19.23 ms             | 1.85 s                      | 751.2 MB         | 0.7862       |
| SegResNet     | 4.70M            |               122.56 | 16.58 ms             | 0.42 s                      | 642.2 MB         | 0.7637       |
| 3D U-Net      | 4.81M            |                22.96 | 3.13 ms              | 0.58 s                      | 337.6 MB         | 0.6087       |
| nnU-Net V2    | 16.54M           |               445.11 | 21.84 ms             | 2.45 s                      | 1083.6 MB        | 0.6003       |
| V-Net         | 45.60M           |               640.22 | 36.24 ms             | 3.12 s                      | 777.9 MB         | Diverged     |

---
### Architectural Efficiency Analysis:
1. **Minimal Parameter Overhead**: RASNet adds only **0.01M parameters (+0.2%)** over baseline SegResNet (4.71M vs 4.70M) while boosting Dice by **+2.25 points** ($0.7862$ vs $0.7637$, $p = 3.09 \times 10^{-15}$).
2. **High Efficiency vs. Heavy Baselines**: Consuming **123.39 GFLOPs**, RASNet requires **72% fewer FLOPs than nnU-Net V2 (445.11 GFLOPs)** and **81% fewer FLOPs than V-Net (640.22 GFLOPs)**, with **71% fewer parameters** (4.71M vs 16.54M).
3. **Real-Time Clinical Suitability**: Single-volume inference latency of **1.85s** (including 4-pass TTA and cc3d connected-component analysis) enables rapid diagnostic workflows on standard clinical workstations.
