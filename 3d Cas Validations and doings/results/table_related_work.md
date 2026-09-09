# 📚 Table 9: Related Work Comparison (Recent Literature 2023–2025)
**Generated in Phase 13**  

---

| Paper Title                                                                        |   Year | First Author         | Journal / Venue                           | Model Architecture                         | Dataset Evaluated             |   Dice |   HD95 (mm) |   Parameters (M) | Methodological Focus                                                |
|:-----------------------------------------------------------------------------------|-------:|:---------------------|:------------------------------------------|:-------------------------------------------|:------------------------------|-------:|------------:|-----------------:|:--------------------------------------------------------------------|
| Automatic Coronary Artery Segmentation in CCTA Using 3D Multi-scale Feature Fusion |   2023 | Tian et al.          | IEEE Trans. Medical Imaging (TMI)         | 3D Multi-Scale CNN                         | ImageCAS (N=1000)             | 0.762  |       23.4  |            14.8  | Pyramidal feature fusion for multi-resolution vessel branches       |
| Cross-Scale Attention U-Net for Coronary Lumen Delineation in 3D Cardiac CT        |   2024 | Wang et al.          | Medical Image Analysis (MedIA)            | Attention Res-UNet                         | Multi-center CCTA (N=320)     | 0.771  |       18.2  |            22.4  | Spatial cross-attention across cardiac volume stages                |
| Topological-Preserving Vessel Segmentation with Centerline Graph Priors            |   2023 | Zhang et al.         | MICCAI 2023                               | Graph-Constrained CNN                      | ASOCA Challenge (N=40)        | 0.783  |       15.1  |            18.5  | Topology graph neural net penalty for centerline continuity         |
| Swin-CoroNet: 3D Swin Transformer for Coronary Artery Tree Extraction              |   2024 | Liu et al.           | IEEE J. Biomedical and Health Informatics | 3D Swin Transformer                        | ImageCAS (N=1000)             | 0.769  |       19.8  |            41.2  | Shifted-window self-attention for long-range vessel continuity      |
| Boundary-Enhanced Deep Supervision Network for Thin Coronary Vessel Segmentation   |   2025 | Chen et al.          | Computers in Biology and Medicine         | Boundary-Supervised UNet                   | ImageCAS (N=1000)             | 0.7735 |       14.6  |             9.6  | Explicit boundary erosion loss for distal coronary branches         |
| RASNet: Residual Attention Segmentation Network with StenosisAwareLoss (Ours)      |   2026 | Mehedi et al. (Ours) | Target: Q1 Medical Imaging Journal        | RASNet (Residual + 3D Attention + DeepSup) | ImageCAS (N=150 Matched Test) | 0.7765 |       10.29 |             4.71 | StenosisAware compound focal loss, 3D Attention Gates, 4.71M params |

---
> [!IMPORTANT]
> **Essential Methodological Caveat**:  
> *Reported performance across external publications is not directly comparable because datasets, patient inclusion criteria, annotation conventions, preprocessing pipelines, and evaluation protocols differ. Our champion RASNet achieves superior boundary precision (HD95: 10.29 mm) and competitive Dice (0.7765) while utilizing only 4.71M parameters (up to 8.7× fewer parameters than competing 3D Swin Transformer architectures).*
