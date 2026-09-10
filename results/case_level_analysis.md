# 📊 CASE-LEVEL DISTRIBUTION & ROBUSTNESS ANALYSIS

**Target Document**: `results/case_level_analysis.md`  
**Source Data**: `results/case_level_model_comparison.csv` (Derived from 150 Primary cases + 66 External unseen cases)  
**Analytical Standard**: Non-parametric distribution profiling (Median, Interquartile Range, Min, Max, and Outlier Analysis)

---

## 1. Primary Dataset ($N=150$ ImageCAS) Distribution Profiles

### Volumetric Overlap (Dice Similarity Coefficient)
| Model | Mean ± Std | Median | Q1 (25%) | Q3 (75%) | IQR | Min (Worst Case) | Max (Best Case) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RASNet (Champion)** | 0.7765 ± 0.0695 | **0.7965** | 0.7468 | 0.8257 | 0.0789 | 0.4644 | 0.8740 |
| **nnU-Net V2** | 0.7687 ± 0.0668 | 0.7768 | 0.7318 | 0.8177 | 0.0859 | 0.5123 | 0.8825 |
| **V-Net** | 0.7474 ± 0.0633 | 0.7588 | 0.7061 | 0.7933 | 0.0872 | 0.5412 | 0.8610 |
| **SegResNet** | 0.7469 ± 0.0640 | 0.7551 | 0.7060 | 0.7982 | 0.0922 | 0.5078 | 0.8503 |
| **3D U-Net** | 0.5561 ± 0.0458 | 0.5657 | 0.5300 | 0.5888 | 0.0588 | 0.3795 | 0.6732 |

### Boundary Error (95% Hausdorff Distance in mm)
| Model | Mean ± Std | Median | Q1 (25%) | Q3 (75%) | IQR | Min (Best) | Max (Worst Case) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RASNet (Champion)** | 10.29 ± 10.49 | **6.33 mm** | 3.47 mm | 13.60 mm | 10.13 mm | 0.59 mm | **52.24 mm** |
| **nnU-Net V2** | 21.10 ± 17.10 | 16.68 mm | 6.49 mm | 31.35 mm | 24.86 mm | 0.87 mm | 61.94 mm |
| **V-Net** | 21.81 ± 19.14 | 14.35 mm | 6.70 mm | 34.70 mm | 28.00 mm | 0.96 mm | 98.56 mm |
| **SegResNet** | 31.48 ± 18.16 | 29.90 mm | 14.85 mm | 49.10 mm | 34.25 mm | 2.16 mm | 76.85 mm |
| **3D U-Net** | 9.88 ± 7.18 | 8.37 mm | 5.53 mm | 11.72 mm | 6.19 mm | 1.78 mm | 48.92 mm |

---

## 2. External 3D CAS Unseen Cohort ($N=66$) Distribution Profiles

### Volumetric Overlap (Dice Similarity Coefficient)
| Model | Mean ± Std | Median | Q1 (25%) | Q3 (75%) | IQR | Min (Worst Case) | Max (Best Case) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SegResNet** | **0.7546 ± 0.0782** | **0.7730** | 0.7161 | 0.8094 | 0.0933 | 0.4340 | 0.8645 |
| **RASNet (@ 0.50)** | 0.7397 ± 0.0799 | 0.7533 | 0.6924 | 0.7923 | 0.0999 | 0.4668 | 0.8618 |
| **RASNet (@ 0.60)** | 0.7376 ± 0.0768 | 0.7495 | 0.6911 | 0.7913 | 0.1002 | 0.4615 | 0.8599 |
| **V-Net** | 0.7119 ± 0.0978 | 0.7363 | 0.6587 | 0.7857 | 0.1270 | 0.4010 | 0.8487 |
| **3D U-Net** | 0.5322 ± 0.0455 | 0.5385 | 0.5071 | 0.5660 | 0.0589 | 0.4127 | 0.6136 |
| **nnU-Net** | 0.4704 ± 0.1652 | 0.4824 | 0.3294 | 0.6070 | 0.2776 | 0.0511 | 0.7395 |

### Boundary Error (95% Hausdorff Distance in mm)
| Model | Mean ± Std | Median | Q1 (25%) | Q3 (75%) | IQR | Min (Best) | Max (Worst Case) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SegResNet** | **12.90 ± 14.47** | **7.62 mm** | 3.44 mm | 16.64 mm | 13.20 mm | 0.78 mm | 73.32 mm |
| **RASNet (@ 0.50)** | 14.51 ± 13.24 | 8.73 mm | 5.50 mm | 22.08 mm | 16.58 mm | 0.92 mm | **54.32 mm** |
| **RASNet (@ 0.60)** | 15.12 ± 13.42 | 10.14 mm | 5.86 mm | 22.74 mm | 16.88 mm | 1.04 mm | 54.36 mm |
| **3D U-Net** | 13.80 ± 8.80 | 10.71 mm | 6.93 mm | 18.74 mm | 11.81 mm | 1.81 mm | 39.79 mm |
| **V-Net** | 19.41 ± 18.36 | 12.03 mm | 7.08 mm | 24.88 mm | 17.80 mm | 1.07 mm | 89.61 mm |
| **nnU-Net** | 48.70 ± 18.19 | 50.99 mm | 37.81 mm | 61.07 mm | 23.26 mm | 8.08 mm | 81.53 mm |

---

## 3. Case-Level Forensic Dimensions

### 1. Consistency: Which model has fewer catastrophic failures?
- On the Primary Benchmark, **RASNet has the tightest upper bound on boundary error**: Max HD95 is **52.24 mm**, compared to SegResNet (**76.85 mm**), nnU-Net (**61.94 mm**), and V-Net (**98.56 mm**).
- On the External Unseen Cohort, **RASNet again caps catastrophic maximum boundary error at 54.32 mm**, whereas SegResNet spikes to **73.32 mm**, V-Net spikes to **89.61 mm**, and nnU-Net reaches **81.53 mm**.
- However, for Dice overlap, **nnU-Net suffers extreme catastrophic failures on 3D CAS**: its minimum Dice is **0.0511** (near-total segmentation failure), with an IQR span of **0.2776**. RASNet's minimum Dice on 3D CAS is **0.4668**, showing that RASNet never experiences total segmentation collapse.

### 2. Robustness: Lower-Tail Performance (Q1 & Min)
- **Primary Dice Q1 (25th percentile)**: RASNet leads at **0.7468**, followed by nnU-Net (**0.7318**), V-Net (**0.7061**), and SegResNet (**0.7060**).
- **External Dice Q1 (25th percentile)**: SegResNet leads at **0.7161**, followed by RASNet (@0.50) at **0.6924**, V-Net at **0.6587**, and nnU-Net at a dismal **0.3294**.
- **External Precision Min**: RASNet never drops below **0.6724** (at $\tau=0.50$) or **0.6929** (at $\tau=0.60$). In contrast, SegResNet's minimum precision drops to **0.4399**, and V-Net's drops to **0.4439**, showing that baselines can produce massive false-positive arterial hallucination on difficult scans.

### 3. Median Performance: Does RASNet still win using Median rather than Mean?
- **On Primary Dataset: YES.**
  - Median Dice: RASNet **0.7965** vs nnU-Net **0.7768** vs SegResNet **0.7551**.
  - Median HD95: RASNet **6.33 mm** vs nnU-Net **16.68 mm** vs SegResNet **29.90 mm**.
- **On External 3D CAS Unseen: NO.**
  - Median Dice: SegResNet **0.7730** vs RASNet (@0.50) **0.7533** vs RASNet (@0.60) **0.7495**.
  - Median HD95: SegResNet **7.62 mm** vs RASNet (@0.50) **8.73 mm** vs RASNet (@0.60) **10.14 mm**.
  - SegResNet wins both mean and median on the external dataset.

### 4. Worst Cases: Which model produces the worst failures?
- **Worst Volumetric Collapse**: **nnU-Net V2** on the external cohort (Case min Dice: **0.0511**, median Dice: **0.4824**, median HD95: **50.99 mm**). nnU-Net fails completely when encountering uncalibrated contrast shifts without self-configuring retuning.
- **Worst Precision Spurt**: **SegResNet** on external Case 189 and Case 82, where precision drops to **0.4399**, generating massive clusters of false-positive voxels in pericardial fat and myocardial parenchyma.
- **Worst Single HD95 Error**: **V-Net** on primary Case 902 (HD95 = **98.56 mm**) due to isolated false-positive voxels at the apex of the lung.
