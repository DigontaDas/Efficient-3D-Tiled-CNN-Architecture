# Thesis Manuscript Sections: Methods, Symmetric 3-Arm Results, Statistical Analysis & Limitations

> **Source Provenance**: Every numerical result in this manuscript draft is computed directly from the verified benchmark runs (`results/benchmark_runs/val_full_94`, `results/benchmark_runs/test_retrospective_150`, `results/thresholds.json`) and audited statistical tables (`results/audits/2026-11-10_verification_audit/`, `results/thesis_outputs/Table1`–`Table4`).

---

## 1. Methods: Architecture, Loss Formulation & Symmetric Three-Arm Evaluation Protocol

### 1.1 Dataset & Disjoint Split Partitioning
Experiments were conducted on the public ImageCAS 3D coronary artery Computed Tomography Angiography (CCTA) dataset (`1,000` volumetric scans). All volumes are stored natively in `LAS` orientation with in-plane spacing $0.29\text{--}0.45\text{ mm}$ and slice thickness $0.50\text{ mm}$. The canonical split manifest (`Phase3_Local_Integration/splits_final.json`, `SHA-256: efe3f06cfe...`) partitions `934` scans into mutually disjoint subsets:
- **Training Split (`N = 690`)**: Subset of cases `1–850` used for 200-epoch network optimization.
- **Validation Split (`N = 94`)**: Disjoint subset of cases `1–850` used for checkpoint selection during training and for selecting frozen probability thresholds ($\tau^*$) in Arm C.
- **Retrospective Test Split (`N = 150`)**: Contiguous cases `851–1000` reserved for final comparative evaluation. Because earlier developmental threshold ($\tau = 0.60$) and loss calibration ($\alpha=0.4, \gamma=2.5$) experiments inspected cases `851–1000`, all evaluations on this 150-case cohort are explicitly reported as **retrospective reassessments** rather than untouched confirmatory tests.

### 1.2 RASNet Architecture & `StenosisAwareLoss`
`RASNet` (`Phase3_Local_Integration/rasnet_model.py`, `4.707M` parameters) extends MONAI's 3D `SegResNet` (`init_filters = 16`, encoder stage blocks `[1, 2, 2, 4]`, decoder stage blocks `[1, 1, 1]`, `GroupNorm` normalization, `dropout_prob = 0.1`) with two structural modifications:
1. **3D Additive Skip-Connection Attention Gates (`AttentionGate3D`)**: At each of the three decoder levels (`64`, `32`, and `16` channels), the upsampled decoder feature map $g \in \mathbb{R}^{B \times C \times D \times H \times W}$ and the same-resolution encoder skip feature map $x \in \mathbb{R}^{B \times C \times D \times H \times W}$ are projected to $F_{\text{int}} = \max(C/2, 1)$ channels via $1 \times 1 \times 1$ convolutions and `GroupNorm(8, F_int)`, summed and activated via `ReLU`, and mapped to a spatial attention coefficient $\alpha \in (0, 1)^{B \times 1 \times D \times H \times W}$ via a $1 \times 1 \times 1$ convolution, `GroupNorm(1, 1)`, and `Sigmoid`. The gated skip tensor $\tilde{x} = \alpha \odot x$ is combined with the upsampled decoder stream by element-wise addition ($g + \tilde{x}$) before entering the decoder residual block. Across all three levels, `AttentionGate3D` and the auxiliary heads add only `5,858` parameters (`+0.12%` over the `4,700,914`-parameter `SegResNet` baseline).
2. **Multi-Scale Deep Supervision & `StenosisAwareLoss`**: During training, auxiliary $1 \times 1 \times 1$ heads at Decoder Level 2 (`aux2`, $1/4$ spatial resolution) and Decoder Level 1 (`aux3`, $1/2$ spatial resolution) are trilinearly upsampled to the `96 × 96 × 96` training patch grid and supervised alongside the full-resolution head (`main`):
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + 0.4\,\mathcal{L}_{\text{aux2}} + 0.2\,\mathcal{L}_{\text{aux3}}$$
   Each head is evaluated using `StenosisAwareLoss` (`Phase3_Local_Integration/rasnet_loss.py`):
   $$\mathcal{L}(\hat{y}, y) = 0.4\,\mathcal{L}_{\text{Dice}}(\hat{y}, y) + 0.6\,\mathcal{L}_{\text{Focal}}(\hat{y}, y; \gamma = 2.5)$$
   where $\mathcal{L}_{\text{Dice}}$ is MONAI's soft `DiceLoss` (`softmax=True, to_onehot_y=True, smooth_dr=1e-5`) and $\mathcal{L}_{\text{Focal}}$ is MONAI's `FocalLoss` (`gamma=2.5, to_onehot_y=True`).

### 1.3 Matched 200-Epoch Baseline Training Protocols
All five architectures (`RASNet`, `SegResNet`, `V-Net`, `3D U-Net`, and `nnU-Net V2`) were trained for `200` epochs on the identical `N = 690` training split (`CHECKPOINTS.md`):
- **`RASNet`, `SegResNet`, and `V-Net`**: Trained on `0.5 × 0.5 × 0.5 mm` isotropic resampled foreground-cropped volumes (`HU [-100, 800]` scaled to `[0, 1]`) with `96 × 96 × 96` random positive/negative patches (`num_samples = 2` for `RASNet` and `SegResNet`; `num_samples = 1` with gradient clipping `max_norm = 1.0` for `V-Net`), `AdamW` ($\text{lr} = 10^{-3}$, weight decay $10^{-5}$), and `CosineAnnealingLR` ($T_{\max} = 200$).
- **`3D U-Net`**: Trained on full volumes trilinearly resized to `128 × 128 × 128` (`HU [-100, 800]` scaled to `[0, 1]`) with `DiceCELoss(sigmoid=True)`.
- **`nnU-Net V2`**: Trained using the native `nnUNetTrainer_200epochs` (`3d_fullres` configuration, `PlainConvUNet` with `30.786M` parameters, patch size `128 × 192 × 192`, target spacing `[0.50, 0.359375, 0.359375] mm`, `CTNormalization`, and deep supervision).

### 1.4 Symmetric Three-Arm Inference & Surface Metric Specification (`surface-v1`)
To disentangle intrinsic network performance from test-time augmentation (TTA) and connected-component post-processing, all five best checkpoints were evaluated by inverting continuous foreground probabilities back onto each scan's native SimpleITK `(Z, Y, X)` physical grid prior to thresholding across three pre-registered arms (`BENCHMARK_PREREGISTRATION.md`):
- **Arm A (Common Raw Inference)**: Single forward pass (`tta=False`, `use_mirroring=False`), fixed probability threshold $\tau = 0.50$, and no connected-component filtering.
- **Arm B (Documented Native / Standard TTA)**: Documented 4-pass spatial flip TTA (`axes=(2, 3, 4)`) for `RASNet`, native 8-pass 3D mirroring (`axes=(0, 1, 2)`) for `nnU-Net V2`, and single-pass inference for `SegResNet`, `V-Net`, and `3D U-Net` (which specify no TTA), at fixed $\tau = 0.50$ without connected-component filtering.
- **Arm C (Validation-Tuned Thresholds $\tau^*$ + Identical Top-2 CC Pruning)**: Single forward pass, per-model threshold $\tau^* \in \{0.30, 0.35, \dots, 0.70\}$ selected strictly on the `N = 94` validation split (`results/thresholds.json`), and identical 26-connected top-2 largest connected-component retention (`cc3d`).

Boundary distances (`HD95` and `ASD`) were computed using `surface-v1` (`benchmark/metrics.py`), which extracts true 6-connected boundary surfaces (`mask & ~binary_erosion(mask, connectivity=1, border_value=0)`) and evaluates physical Euclidean distance transforms (`scipy.ndimage.distance_transform_edt(~surface, sampling=spacing_zyx)`), avoiding the interior distance-zeroing defect of volume-complement distance transforms.

---

## 2. Results

### 2.1 Validation Split Performance (`N = 94`) & Threshold Selection (`Table 1`, `Figure 2`)
On the `N = 94` validation split under raw single-pass inference (`Arm A`, $\tau = 0.50$), `RASNet` achieved the highest mean validation Dice (`0.7789 ± 0.0521`), highest Precision (`0.8165 ± 0.0692`), and highest centerline topological Dice (`clDice = 0.8480 ± 0.0617`), followed by `nnU-Net V2` (`Dice = 0.7761 ± 0.0567`, `Recall = 0.8144 ± 0.0541`), `SegResNet` (`Dice = 0.7532 ± 0.0591`), `V-Net` (`Dice = 0.7511 ± 0.0619`), and `3D U-Net` (`Dice = 0.5516 ± 0.0480`).

When sweeping $\tau \in [0.30, 0.70]$ with top-2 connected-component pruning on the 94 validation cases (`Figure 2`), `nnU-Net V2` and `SegResNet` achieved interior validation Dice peaks at $\tau^* = 0.35$ (`0.8075` and `0.7901`, respectively), whereas `RASNet` (`0.7897`), `V-Net` (`0.7702`), and `3D U-Net` (`0.6047`) selected the lower boundary of the pre-registered grid ($\tau^* = 0.30$).

### 2.2 Retrospective Test Split (`N = 150`) — Three-Arm Comparison (`Table 2`, `Figure 1`, `Figure 4`)

1. **Arm A — Raw Single Pass ($\tau = 0.50$, No Post-Processing)**:
   - Without heuristic post-processing, `RASNet` ranks **first in Dice (`0.7736 ± 0.0616`, median `0.7836 [0.7438, 0.8177]`), first in IoU (`0.6348 ± 0.0787`), first in Precision (`0.8032 ± 0.0830`), and first in clDice (`0.8389 ± 0.0686`)** across all 150 test cases.
   - Compared to its `SegResNet` backbone (`Dice = 0.7481 ± 0.0654`, `Precision = 0.7253 ± 0.0945`, `HD95 = 30.41 ± 18.11 mm`, `ASD = 4.571 ± 2.685 mm`), adding `AttentionGate3D`, Deep Supervision, and `StenosisAwareLoss` in `RASNet` improves raw Dice by **`+0.0255`** ($p_{\text{Holm}} = 2.09 \times 10^{-23}$), increases Precision by **`+0.0780`** ($p_{\text{Holm}} = 2.13 \times 10^{-24}$), and reduces `HD95` by more than half (**`-15.57 mm`**, from `30.41 mm` to `14.83 mm`, $p_{\text{Holm}} = 2.58 \times 10^{-20}$).
   - Compared to raw `nnU-Net V2` (`Dice = 0.7687 ± 0.0668`, `Precision = 0.7391 ± 0.0967`, `Recall = 0.8106 ± 0.0680`, `HD95 = 21.10 ± 17.10 mm`, `ASD = 3.168 ± 2.636 mm`), `RASNet` achieves significantly higher Precision ($\Delta = +0.0642$, 95% CI $[+0.0565, +0.0728]$, $p_{\text{Holm}} = 1.93 \times 10^{-24}$, `139W / 0T / 11L`), lower `HD95` ($\Delta = -6.27\text{ mm}$, 95% CI $[-8.43, -4.38]$, $p_{\text{Holm}} = 3.96 \times 10^{-9}$, `107W / 0T / 43L`), and lower `ASD` ($\Delta = -0.737\text{ mm}$, 95% CI $[-0.995, -0.503]$, $p_{\text{Holm}} = 2.37 \times 10^{-8}$, `107W / 0T / 43L`), while `nnU-Net V2` achieves higher Recall (`0.8106` vs. `0.7521`).

2. **Arm B — Documented Native TTA ($\tau = 0.50$, No Post-Processing)**:
   - Both `RASNet` (4-pass flip TTA: `Dice = 0.7807 ± 0.0606`, `Precision = 0.8138 ± 0.0813`, `HD95 = 13.69 ± 13.88 mm`, `ASD = 2.270 ± 1.863 mm`, `clDice = 0.8453 ± 0.0677`) and `nnU-Net V2` (8-pass mirroring: `Dice = 0.7760 ± 0.0641`, `Precision = 0.7536 ± 0.0936`, `HD95 = 19.25 ± 16.93 mm`, `ASD = 2.914 ± 2.487 mm`, `clDice = 0.8264 ± 0.0750`) improve by `+0.0071` and `+0.0073` Dice, respectively.
   - All three primary hypothesis tests between `RASNet` and `nnU-Net V2` remain highly significant in favor of `RASNet` (`Precision`: $\Delta = +0.0602$, $p_{\text{Holm}} = 5.39 \times 10^{-24}$; `HD95`: $\Delta = -5.56\text{ mm}$, $p_{\text{Holm}} = 4.34 \times 10^{-8}$; `ASD`: $\Delta = -0.644\text{ mm}$, $p_{\text{Holm}} = 4.17 \times 10^{-7}$).

3. **Arm C — Validation-Tuned Thresholds ($\tau^*$) + Identical Top-2 CC Pruning**:
   - Applying 26-connected top-2 component pruning and validation-tuned thresholds symmetrically across all models removes disconnected false-positive background islands (`Table 2C`). Because raw `nnU-Net V2` and `SegResNet` had higher initial false-positive island rates than `RASNet`, top-2 CC pruning increases `nnU-Net V2` Dice by `+0.0367` (`0.7687 -> 0.8055 ± 0.0633`, `HD95 = 8.26 ± 9.81 mm`, `ASD = 1.395 ± 1.862 mm`) and `SegResNet` Dice by `+0.0402` (`0.7481 -> 0.7884 ± 0.0700`, `HD95 = 9.84 ± 11.35 mm`), while `RASNet` increases by `+0.0163` (`0.7736 -> 0.7899 ± 0.0677`, `HD95 = 9.94 ± 11.25 mm`, `ASD = 1.671 ± 2.382 mm`).
   - In Arm C, `nnU-Net V2` achieves the highest overall overlap and lowest surface distance (`HD95` $\Delta = +1.68\text{ mm}$, $p_{\text{Holm}} = 4.20 \times 10^{-4}$; `ASD` $\Delta = +0.277\text{ mm}$, $p_{\text{Holm}} = 1.54 \times 10^{-4}$ in favor of `nnU-Net V2`), whereas `RASNet` retains the **highest Precision (`0.8252 ± 0.0650`)** among all 5 models (statistically indistinguishable from `nnU-Net V2`'s `0.8207`, $p_{\text{Holm}} = 0.160$, and significantly higher than `SegResNet`'s `0.8095`, $\Delta = +0.0158$, $p_{\text{Holm}} = 4.50 \times 10^{-8}$) and ranks second overall in Dice (`0.7899`), ahead of `SegResNet` (`0.7884`), `V-Net` (`0.7710`), and `3D U-Net` (`0.6064`).

### 2.3 Computational Efficiency Profile (`Table 4`, `Figure 4B`)
On an `NVIDIA GeForce RTX 4080 SUPER` (`16,376 MiB` VRAM), `RASNet` requires **`4.707M` parameters (`18.00 MB` checkpoint), `123.39 GFLOPs` per `96 × 96 × 96` patch, `13.55 ± 0.06 ms` AMP FP16 patch latency, and `390.7 MB` peak patch VRAM**. By comparison, `nnU-Net V2` requires **`30.786M` parameters (`235.05 MB` checkpoint — $6.54\times$ larger), `2,149.13 GFLOPs` per `128 × 192 × 192` patch ($17.4\times$ higher compute per patch), `63.04 ± 0.18 ms` AMP FP16 patch latency ($4.65\times$ slower per patch), and `2,381.1 MB` peak patch VRAM ($6.09\times$ higher)**. `V-Net` requires `45.599M` parameters (`174.03 MB`) and `617.23 GFLOPs` per patch while achieving lower accuracy than `RASNet` in every arm.

---

## 3. Methodological Limitations & Discussion Points

1. **Retrospective Nature of the 150-Case ImageCAS Evaluation**:
   Because historical threshold ($\tau=0.60$) and loss hyperparameter ($\alpha=0.4, \gamma=2.5$) exploration inspected cases `851–1000`, all findings on the 150-case ImageCAS test split represent a retrospective reassessment. Prospective validation on an untouched external multi-center CCTA cohort is required for confirmatory generalization claims.
2. **Validation Threshold Grid Lower Boundary ($\tau^* = 0.30$)**:
   In Arm C, `RASNet`, `V-Net`, and `3D U-Net` achieved their highest validation Dice at the lower boundary ($\tau^* = 0.30$) of the pre-registered grid `[0.30, 0.70]`. For `RASNet`, this reflects the conservative foreground probability distribution induced by `FocalLoss(gamma=2.5)`. To prevent post-hoc threshold snooping, the grid was not expanded below `0.30` after inspecting validation/test results.
3. **Anatomical Trade-Off of Top-2 Connected-Component Pruning (`Arm C`)**:
   Retaining at most two 26-connected components assumes the left and right coronary arterial trees each form a single connected component. While effective at eliminating isolated false-positive islands in `Arm C`, top-2 pruning can discard true distal segments beyond a complete total occlusion (CTO) or severe segmentation gap, or retain an aortic root artifact if one coronary ostium is disconnected. Consequently, `Arm A` (unpruned) and `Arm C` (top-2 pruned) should always be interpreted together: `RASNet`'s superiority in `Arm A` demonstrates that `StenosisAwareLoss` and `AttentionGate3D` suppress background false positives directly within the learned representation without relying on topological heuristics.
4. **3D U-Net Coarse Resolution Bottleneck**:
   Following its training specification (`train_3dunet_200ep.py`), `3D U-Net` operates on whole volumes resized to `128 × 128 × 128`, which downsamples sub-millimeter distal coronary branches (`66.38%` Dice on the coarse `128³` grid vs. `55.17%` Arm A / `60.64%` Arm C Dice when inverted to native resolution).

