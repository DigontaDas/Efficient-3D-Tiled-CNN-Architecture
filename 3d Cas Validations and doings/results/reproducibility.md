# 🔬 Reproducibility Specification Document
**Generated in Phase 12**  
**Compliance Standard**: CLAIM 2024 (Checklist for Artificial Intelligence in Medical Imaging)  

---

## 1. Primary Benchmark Dataset Partitioning
- **Source Database**: ImageCAS (1000 coronary CCTA volumes)
- **Total Partitioned Volumes**: 950 volumes
  - **Training Set ($N=690$)**: Used for 200-epoch model parameter optimization.
  - **Validation Set ($N=94$)**: Used for epoch-level checkpoint selection (`best_val_dice`).
  - **Test Set ($N=150$)**: Completely held out, frozen evaluation set ($N=150$).
- **Partitioning Strategy**: Stratified case-ID partitioning via `splits_final.json`.
- **Random Seed**: Fixed globally to `seed = 42` (`random`, `numpy`, `torch`).

---

## 2. Training Hyperparameters & Preprocessing Standard
- **Network Optimization**: AdamW optimizer ($eta_1=0.9, eta_2=0.999$, weight decay $10^{-5}$)
- **Initial Learning Rate**: $2 \times 10^{-4}$
- **Learning Rate Schedule**: Cosine Annealing decay down to $\eta_{min} = 1 \times 10^{-6}$ over 200 epochs.
- **Batch Size**: 4 sub-volume patches per iteration.
- **Sub-Volume Patch Size**: $(96, 96, 96)$ voxels.
- **Positive-to-Negative Patch Ratio**: $2:1$ (centered on coronary foreground vs background).
- **Voxel Spacing**: Resampled to $0.5 \times 0.5 \times 0.5$ mm isotropic spacing using trilinear interpolation for images and nearest-neighbor for ground truth.
- **Spatial Orientation**: Standardized to Right-Anterior-Superior (RAS) coordinate system.
- **Radiometric Normalization**: Fixed Hounsfield window $[-100, 800]$ HU mapped linearly to $[0.0, 1.0]$.
- **Data Augmentation**: Random spatial axis flips ($p=0.5$ on X, Y, Z).

---

## 3. External Dataset Provenance & Leakage Prevention Statement
- **External Dataset**: 3D CAS Images (200 Samples located at `H:\3D CT Images for Coronary Artery Segmentation (200 Samples)`).
- **Provenance Statement**:
  > *"3D CAS Images was evaluated as an independent 200-case dataset. Provenance tracing confirmed it contains Cases 1 to 200 of the ImageCAS database. Within this cohort, 115 cases were included in the primary training set, 19 cases in the validation set, and 66 cases were QC-excluded cases with zero exposure to model training. To preserve absolute scientific integrity, results are reported for the full 200 cases and disaggregated into the genuine unseen sub-cohort ($N=66$) and training-recall sub-cohort ($N=115$)."*

---

## 4. Exact Software & Library Dependencies (Audited)
- **Operating System**: Windows 11 (AMD64)
- **Python**: `3.14.0`
- **PyTorch**: `2.14.0+cu126` (CUDA `12.6`)
- **MONAI**: `1.5.1`
- **SimpleITK**: `2.5.6`
- **SciPy**: `1.17.0`
- **NumPy**: `2.4.1`
- **Connected Components**: `connected-components-3d 4.1.0`
- **Execution Hardware**: NVIDIA GeForce RTX 3060 Ti (8,191.5 MB VRAM)
