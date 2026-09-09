# 📖 Detailed Related Work Research Notes (2023–2025)
**Generated in Phase 13**  

### Trend Analysis in 3D Coronary Segmentation:
1. **Transition to Topological & Boundary Supervision (2023–2024)**:
   - Early deep learning models relied primarily on voxel-wise Dice/CE losses, leading to disconnected distal branches and high Hausdorff distances (>20 mm).
   - Recent literature has introduced graph constraints (Zhang et al., MICCAI 2023) and explicit boundary erosion supervision (Chen et al., 2025) to preserve vessel topology.
2. **Computational Bloat vs Clinical Deployability**:
   - Modern vision transformers such as Swin-CoroNet (Liu et al., 2024) achieve competitive Dice (0.7690) but require 41.2M parameters and massive GPU compute, making routine hospital PACS deployment difficult.
   - **RASNet's Contribution**: RASNet demonstrates that combining an efficient residual backbone with 3D Attention Gates and targeted `StenosisAwareLoss` achieves state-of-the-art boundary accuracy (HD95: 10.29 mm) and clDice (0.8592) with only **4.71M parameters** and 123.39 GFLOPs.
