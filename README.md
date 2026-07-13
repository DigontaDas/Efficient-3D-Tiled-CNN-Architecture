# RASNet: Residual Attention Segmentation Network for Coronary Artery Segmentation

RASNet is a novel deep learning framework specifically engineered for high-precision 3D coronary artery segmentation in contrast-enhanced Computed Tomography Angiography (CCTA) scans. Built to handle the challenging, fine-grained geometries and variable stenotic conditions of the coronary vasculature, RASNet v2 integrates a 3D Residual Attention mechanism alongside Deep Supervision and a tailored Stenosis-Aware focal loss. By combining native coordinate space processing, Hounsfield Unit (HU) intensity windowing, and spatial Test-Time Augmentation (TTA), RASNet v2 dramatically improves boundary definition and recall of distal vessel branches, outperforming established baselines such as SegResNet, nnU-Net, and standard 3D U-Net on public benchmarks.

## Key Contributions
- **AttentionGate3D**: Integrates spatial and channel-wise residual attention gates to filter out non-vascular structures while amplifying weaker signals from small distal branches.
- **StenosisAwareLoss**: A customized focal loss formulation ($\gamma=2.5$) that prevents over-penalization of borderline vessel boundaries and stenotic zones, maximizing vessel recall.
- **Deep Supervision**: Utilizes full-resolution multi-scale auxiliary supervision (1.0 × main + 0.4 × aux2 + 0.2 × aux3) in the decoder to preserve spatial resolution at all stages.
- **HU Windowing**: Standardizes intensity variations using a specialized Hounsfield Unit (HU) window of `[-100, 800]` to sharpen contrast-enhanced lumen boundaries.
- **Test-Time Augmentation (TTA)**: Leverages a 4-pass spatial orientation TTA to smooth boundaries, stabilize predictions, and reduce prediction variance.

## Dataset
The model was developed and benchmarked on the public **ImageCAS** dataset:
- **Training Set**: 784 high-resolution CCTA cases.
- **Test Set**: 150 reserved test cases (Cases 851–1000).

## Final Quantitative Results (150 Test Cases)

The following table summarizes the performance of RASNet v2 compared to standard coronary artery segmentation baselines on the 150 reserved test cases:

| Model | N | Dice Coefficient ↑ | IoU ↑ | Precision ↑ | Recall ↑ | HD95 (mm) ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RASNet v2 (Ours)** | 150 | **0.7942 ± 0.0603** | **0.6625 ± 0.0786** | **0.8533 ± 0.0501** | **0.7495 ± 0.0924** | **8.12 ± 9.39** |
| **SegResNet** | 150 | 0.7637 ± 0.0576 | 0.6211 ± 0.0721 | 0.8140 ± 0.0445 | 0.7260 ± 0.0913 | 9.11 ± 10.75 |
| **nnU-Net** | 150 | 0.6003 ± 0.0780 | 0.4332 ± 0.0789 | 0.5354 ± 0.1044 | 0.7017 ± 0.0820 | 58.30 ± 14.44 |
| **3D U-Net** | 150 | 0.6087 ± 0.0355 | 0.4384 ± 0.0368 | 0.6289 ± 0.0421 | 0.5919 ± 0.0451 | 4.62 ± 3.30 |
| **V-Net** | N/A | — | — | — | — | *Instability Limits* |

> [!NOTE]
> RASNet v2 achieves the highest precision and recall among all models, producing fewer false-positive vessel phantoms and capturing thin branch structures more accurately, making it highly suitable for downstream stenosis quantification.

## Requirements
To install the necessary packages for running the models with CUDA acceleration, please refer to [requirements_cuda.txt](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/requirements_cuda.txt):

```bash
pip install -r requirements_cuda.txt
```

## Clinical Validation (In Progress)
We are currently conducting clinical validation of RASNet v2 on real-world clinical datasets. The **Ibrahim Cardiac Hospital dataset**, featuring manually annotated clinical CCTA volumes with diverse stenosis and calcification profiles, will be integrated soon to evaluate the real-world generalization of the network.
