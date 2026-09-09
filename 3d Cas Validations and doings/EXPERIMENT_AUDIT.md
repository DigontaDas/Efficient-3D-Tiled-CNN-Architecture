# 🔬 EXPERIMENT AUDIT REPORT
**Generated Automatically during Phase 1**  
**Execution Location**: `H:\Thesis_Trainings\3d Cas Validations and doings`  
**Target Output**: `EXPERIMENT_AUDIT.md`  
**Date**: September 9, 2026  

---

## 1. Executive Environment & Hardware Summary

| Component | Detected Specification |
| :--- | :--- |
| **Operating System** | Windows 10/11 (AMD64) |
| **Local GPU** | **NVIDIA GeForce RTX 3060 Ti** |
| **VRAM Capacity** | **8191.5 MB** |
| **CUDA Available** | **True** (CUDA Version: 12.6) |
| **Python Version** | **3.14.0** |
| **PyTorch Build** | **2.14.0+cu126** |
| **MONAI Build** | **1.5.1** |
| **SimpleITK Build** | **2.5.6** |

---

## 2. Model Architecture Locations & Implementations

1. **RASNet (Ours, Champion Architecture)**:
   - **Path**: `Phase3_Local_Integration/rasnet_model.py`
   - **Architecture**: Asymmetric Encoder-Decoder based on Residual Blocks with:
     - 3-level Attention Gates (`AttentionGate3D` at intermediate skip connections)
     - Multi-scale Deep Supervision (`aux2`, `aux3` auxiliary prediction heads)
     - Stenosis-aware compound loss modulation (`StenosisAwareLoss` with focal weighting $\alpha=0.4, \gamma=2.5$)
   - **Parameters**: 4,706,438 (4.71M)
   - **GFLOPs**: 123.39 GFLOPs at $(96, 96, 96)$

2. **SegResNet (Baseline 1)**:
   - **Path**: MONAI native `monai.networks.nets.SegResNet`
   - **Architecture**: Encoder-Decoder with residual blocks and variational autoencoder branch (disabled during evaluation)
   - **Parameters**: 4,701,474 (4.70M)
   - **Training Runner**: `Q1_Publication_Package/matched_200ep_benchmark/train_segresnet_200ep.py`

3. **nnU-Net (Baseline 2, V2 Configuration)**:
   - **Path**: MONAI native `monai.networks.nets.DynUNet`
   - **Architecture**: Dynamic U-Net self-adapting kernel sizes and deep supervision heads
   - **Parameters**: 31,189,474 (31.19M)
   - **Training Runner**: `Q1_Publication_Package/matched_200ep_benchmark/train_nnunet_200ep.py`

4. **V-Net (Baseline 3)**:
   - **Path**: MONAI native `monai.networks.nets.VNet`
   - **Architecture**: Volumetric convnet with residual connections and PReLU activations
   - **Parameters**: 45,604,018 (45.60M)
   - **Training Runner**: `Q1_Publication_Package/matched_200ep_benchmark/train_vnet_200ep.py`

5. **3D U-Net (Baseline 4)**:
   - **Path**: MONAI native `monai.networks.nets.UNet`
   - **Architecture**: Standard 3D U-Net with 4 resolution stages
   - **Parameters**: 4,809,794 (4.81M)
   - **Training Runner**: `Q1_Publication_Package/matched_200ep_benchmark/train_3dunet_200ep.py`

---

## 3. Matched 200-Epoch Checkpoint Availability

All 5 core checkpoints are stored in `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/` and verified on disk:

| Model | Checkpoint File | Size | Exists? | Verified Architecture |
| :--- | :--- | :--- | :---: | :--- |
| **RASNet** | `rasnet_best.pth` | 18.00 MB | True | RASNet(1->2, filters=16, attngate, deepsup) |
| **SegResNet** | `segresnet_best.pt` | 17.96 MB | True | SegResNet(1->2, filters=16) |
| **nnU-Net** | `nnunet_best.pt` | 235.05 MB | True | DynUNet(1->2, deep supervision) |
| **V-Net** | `vnet_best.pt` | 174.03 MB | True | VNet(1->2, PReLU, residual) |
| **3D U-Net** | `3dunet_best.pth` | 18.36 MB | True | UNet(1->2, channels=[16, 32, 64, 128]) |

---

## 4. Preprocessing & Augmentation Pipeline Audit

Standardized preprocessing pipeline established in `dataset_paths.py` and benchmark training runners:
1. **LoadImaged**: Loads 3D volume and ground truth segmentation NIfTI.
2. **EnsureChannelFirstd**: Adds channel dimension `(C, H, W, D)`.
3. **Orientationd(axcodes="RAS")**: Standardizes anatomical spatial coordinates.
4. **Spacingd(pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest"))**: Re-samples all CT scans to isotropic 0.5 mm resolution.
5. **ScaleIntensityRanged(a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True)**: Standard cardiac CT windowing.
6. **CropForegroundd**: Crops non-informative zero background margins.
7. **RandCropByPosNegLabeld**: Sub-volume patch extraction of size $(96, 96, 96)$ with positive:negative ratio $2:1$.

---

## 5. Post-Processing & Inference Pipeline Audit

1. **Sliding-Window Inference**:
   - Patch Size: $(96, 96, 96)$
   - Overlap: $0.5$ (50% volumetric overlap)
   - Configured Batch Size on RTX 3060 Ti: `sw_batch_size=4` with PyTorch AMP FP16.
2. **Invertd**:
   - Inverts spatial resampling and cropping transformations back to native scan geometry, voxel spacing, and origin.
3. **Probability Thresholding**:
   - Foreground probability cutoff $> 0.60$ (or $0.50$ baseline).
4. **Connected Component Analysis (cc3d)**:
   - 26-connectivity filtering via `cc3d.connected_components`.
   - Top-2 components preserved (Left Coronary Artery and Right Coronary Artery main stems), effectively eliminating spurious cardiac cavity false positives.
5. **Test-Time Augmentation (4-Pass TTA)**:
   - Evaluates: original volume + 3 orthogonal axis flips:
     $$P_{TTA} = \frac{1}{4} \left( P(X) + \text{flip}_x(P(\text{flip}_x(X))) + \text{flip}_y(P(\text{flip}_y(X))) + \text{flip}_z(P(\text{flip}_z(X))) \right)$$

---

## 6. Evaluation Metrics Implementation Status

Audited against `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py`:

| Metric | Supported? | Function / Methodology |
| :--- | :---: | :--- |
| **Dice Similarity Coefficient** | **YES** | `2 * |P & G| / (|P| + |G|)` |
| **Intersection over Union (IoU)** | **YES** | `|P & G| / |P | G|` |
| **Precision (PPV)** | **YES** | `TP / (TP + FP)` |
| **Recall / Sensitivity** | **YES** | `TP / (TP + FN)` |
| **Specificity** | **YES** | `TN / (TN + FP)` (derived via voxel confusion matrix) |
| **Hausdorff Distance 95% (HD95)** | **YES** | `scipy.spatial.distance.directed_hausdorff` at 95th percentile with physical spacing |
| **Average Surface Distance (ASD/ASSD)** | **YES** | Mean bidirectional surface distance computed via boundary erosion distance maps |
| **clDice (Centerline Dice)** | **YES** | MedPy / skeletonize-3d centerline intersection over union |
| **Centerline Recall** | **YES** | Fraction of ground-truth centerline voxels covered by prediction |
| **Volume Metrics** | **YES** | Voxel count $\times \prod(\text{spacing})$ |
| **Statistical Significance** | **YES** | Two-sided paired Wilcoxon signed-rank test + Holm-Bonferroni correction |
| **Confidence Intervals** | **YES** | Percentile bootstrap ($B=2000$) |

---

## 7. Primary Benchmark Dataset Split Verification

Source file: `Phase3_Local_Integration/splits_final.json`

| Split | Number of Cases | Case Range & Nature |
| :--- | :---: | :--- |
| **Train** | **690** | Cases 1 to 998 (includes 115 cases from 1-200) |
| **Validation** | **94** | Cases 3 to 999 (includes 19 cases from 1-200) |
| **Test** | **150** | Cases 201 to 1000 (0 cases from 1-200) |
| **Total Split** | **940** | 950 curated volumes |

---

## 8. Headline Primary Benchmark Results (N=150 ImageCAS Test Cases)

| Model | Dice | IoU | Precision | Recall | HD95 (mm) | ASD (mm) | clDice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RASNet (Ours)** | **0.7765** | **0.6396** | **0.8801** | 0.7016 | **10.29** | **1.598** | **0.8592** |
| **nnU-Net V2** | 0.7687 | 0.6289 | 0.7391 | **0.8106** | 21.10 | 3.097 | 0.8201 |
| **V-Net** | 0.7474 | 0.6006 | 0.7545 | 0.7499 | 21.81 | 3.176 | 0.8075 |
| **SegResNet** | 0.7469 | 0.6001 | 0.7313 | 0.7713 | 31.48 | 4.661 | 0.7769 |
| **3D U-Net** | 0.5561 | 0.3865 | 0.6069 | 0.5178 | 9.88 | 1.690 | 0.7027 |

---

## 9. Conclusion of Phase 1 Audit

1. **Repository Completeness**: All models, weights, preprocessors, post-processors, and metric calculators exist and are verified.
2. **Local Hardware Ready**: The RTX 3060 Ti with 8 GB VRAM is fully operational with PyTorch CUDA 12.6.
3. **Proceed to Phase 2**: The repository audit is complete. We can now proceed immediately to Phase 2: auditing the 200-case dataset structure and creating `3D_CAS_DATASET_AUDIT.md`.
