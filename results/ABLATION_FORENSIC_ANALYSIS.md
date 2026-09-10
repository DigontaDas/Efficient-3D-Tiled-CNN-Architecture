# 🧩 COMPONENT-WISE ABLATION FORENSIC ANALYSIS

**Target Document**: `results/ABLATION_FORENSIC_ANALYSIS.md`  
**Dataset**: Primary ImageCAS Test Set ($N=150$ Cases)  
**Execution Platform**: Lab Workstation (Matched 200-Epoch Retraining from Scratch)  
**Authentic Data Source**: `Q1_Publication_Package/ablation/evaluation_results/` & `Table8_component_ablation.csv`  

---

## 1. Step-by-Step Cumulative Ablation Progression

| Step | Architectural / Pipeline Configuration | Attention Gates | Deep Supervision | Loss Function | 4-Pass TTA | cc3d Pruning | Dice (DSC) | HD95 (mm) | Precision | Recall | clDice | Δ Dice | Δ HD95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **SegResNet Baseline** | No | No | Standard Dice+CE | No | No | 0.7469 | 31.48 | 0.7313 | 0.7713 | 0.7769 | Baseline | Baseline |
| **2** | **+ AttentionGate3D** | Yes (3-Level) | No | Standard Dice+CE | No | No | 0.7440 | 23.44 | 0.7450 | 0.7512 | 0.7850 | -0.0029 | **-8.04 mm** |
| **3** | **+ Deep Supervision** | Yes (3-Level) | Yes (aux2, aux3)| Standard Dice+CE | No | No | 0.7535 | 21.61 | 0.7700 | 0.7448 | 0.8110 | **+0.0066** | **-9.87 mm** |
| **4** | **+ StenosisAwareLoss (Raw RASNet)**| Yes (3-Level) | Yes (aux2, aux3)| StenosisAware ($\alpha=0.4, \gamma=2.5$)| No | No | **0.7681** | **13.45** | **0.8127** | 0.7340 | **0.8350** | **+0.0212** | **-18.03 mm** |
| **5** | **+ 4-Pass TTA (Inference)** | Yes (3-Level) | Yes (aux2, aux3)| StenosisAware ($\alpha=0.4, \gamma=2.5$)| Yes (4-Flips)| No | 0.7745 | 12.20 | 0.8260 | 0.7347 | 0.8440 | +0.0276 | -19.28 mm |
| **6** | **+ cc3d Pruning (Champion)** | Yes (3-Level) | Yes (aux2, aux3)| StenosisAware ($\alpha=0.4, \gamma=2.5$)| Yes (4-Flips)| Yes (Top-2) | **0.7765** | **10.29** | **0.8801** | 0.7016 | **0.8592** | **+0.0296** | **-21.19 mm** |

---

## 2. Component Impact Dissection

### Component 1: 3-Level Attention Gates (`AttentionGate3D`)
- **Impact on Dice**: Slightly negative ($-0.0029$, from 0.7469 to 0.7440).
- **Impact on HD95**: **Substantial improvement ($-8.04\text{ mm}$, from 31.48 mm down to 23.44 mm)**.
- **Impact on Precision**: Moderate improvement ($+0.0137$, from 0.7313 to 0.7450).
- **Forensic Mechanism**: Soft spatial attention gates gate feature pass-through based on coarser decoder activations. This suppresses non-vascular cardiac structures (e.g., ventricular edges and pulmonary vessels) that cause boundary penalties, but trims boundary voxels slightly, causing a minor dip in raw voxel overlap.

### Component 2: Multi-Scale Deep Supervision (`aux2`, `aux3`)
- **Impact on Dice**: **Positive ($+0.0095$, from 0.7440 to 0.7535)**.
- **Impact on HD95**: **Further improvement ($-1.83\text{ mm}$, from 23.44 mm down to 21.61 mm)**.
- **Impact on Precision**: Substantial boost ($+0.0250$, from 0.7450 to 0.7700).
- **Forensic Mechanism**: Auxiliary gradient injection into intermediate resolution levels ($48^3$ and $24^3$) prevents vanishing gradients across the residual skip connections, accelerating convergence and forcing intermediate layers to learn discriminative tubular representations.

### Component 3: Stenosis-Aware Loss (`StenosisAwareLoss`, $\alpha=0.4, \gamma=2.5$)
- **Impact on Dice**: **Strongest architectural gain ($+0.0146$, from 0.7535 to 0.7681)**.
- **Impact on HD95**: **Dramatic boundary collapse ($-8.16\text{ mm}$, from 21.61 mm down to 13.45 mm)**.
- **Impact on Precision**: **Major jump ($+0.0427$, from 0.7700 to 0.8127)**.
- **Impact on Recall**: Moderate drop ($-0.0108$, from 0.7448 to 0.7340).
- **Forensic Mechanism**: Focal modulating factor $(1 - p_t)^\gamma$ penalizes easy background voxels and focuses gradient updates on ambiguous lumen boundaries and tight stenosis notches. This heavily penalizes false-positive arterial predictions, driving precision above 81%.

### Component 4: 4-Pass Test-Time Augmentation (Inference)
- **Impact on Dice**: Small gain ($+0.0064$, from 0.7681 to 0.7745).
- **Impact on HD95**: Moderate gain ($-1.25\text{ mm}$, from 13.45 mm down to 12.20 mm).
- **Impact on Precision**: $+0.0133$ (from 0.8127 to 0.8260).
- **Forensic Mechanism**: Ensembling 4 spatial orientations (original + 3 orthogonal axis flips) smooths out stochastic patch-boundary artifacts.

### Component 5: `cc3d` Connected-Component Filtering (Post-Processing)
- **Impact on Dice**: Negligible ($+0.0020$, from 0.7745 to 0.7765).
- **Impact on HD95**: **Substantial gain ($-1.91\text{ mm}$, from 12.20 mm down to 10.29 mm)**.
- **Impact on Precision**: **Massive leap ($+0.0541$, from 0.8260 to 0.8801)**.
- **Impact on Recall**: Noticeable drop ($-0.0331$, from 0.7347 down to 0.7016).
- **Forensic Mechanism**: Restricting output to the Top-2 connected components completely purges small disconnected false-positive islands throughout the thorax. Because disconnected distal true branches that failed connectivity are also pruned, recall drops from 73.5% to 70.2%, but precision reaches 88.01%.

---

## 3. Crucial Distinction: Architecture vs. Inference Pipeline

| Dimension | Raw RASNet Architecture (Step 4) | Full RASNet Inference Pipeline (Step 6) | Baseline SegResNet (Step 1) |
| :--- | :---: | :---: | :---: |
| **Inference Mode** | Single-pass, no TTA, no cc3d | 4-pass TTA + cc3d Top-2 | Single-pass, no TTA, no cc3d |
| **Threshold $\tau$** | 0.50 | 0.60 | 0.50 |
| **Dice (DSC)** | **0.7681** | **0.7765** | 0.7469 |
| **HD95 (mm)** | **13.45 mm** | **10.29 mm** | 31.48 mm |
| **Precision** | **0.8127** | **0.8801** | 0.7313 |
| **Recall** | **0.7340** | **0.7016** | 0.7713 |
| **clDice** | **0.8350** | **0.8592** | 0.7769 |

### Definitive Forensic Findings:
1. **The architectural contributions are fully genuine**:
   - The architectural components alone (Attention Gates + Deep Supervision + StenosisAwareLoss) improve Dice by **+0.0212** (0.7469 $\to$ 0.7681) and reduce HD95 by **18.03 mm** (31.48 mm $\to$ 13.45 mm) without any TTA or connected-component post-processing.
   - Raw RASNet matches nnU-Net V2 in Dice (**0.7681 vs 0.7687**) while outperforming it in HD95 (**13.45 mm vs 21.10 mm**) and Precision (**0.8127 vs 0.7391**).
2. **Inference-time post-processing provides the final polish**:
   - TTA and `cc3d` contribute **+0.0084 Dice (28% of total gain)**, **+0.0674 Precision (45% of total gain)**, and **-3.16 mm HD95 (15% of total error reduction)**.
   - **Mandatory Reporting Rule**: The paper must NOT attribute the full 88.01% Precision and 10.29 mm HD95 exclusively to the network architecture. It must explicitly state that raw architecture achieves 81.27% Precision / 13.45 mm HD95, and post-processing refines it to 88.01% / 10.29 mm.
