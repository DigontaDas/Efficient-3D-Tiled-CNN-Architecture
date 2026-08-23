# 🔍 Quantitative Failure Mode & Limitations Analysis
### Analysis of Lowest-Performing Test Cases (ImageCAS Test Cohort $N=150$)

**Target**: Q1 Medical Imaging Journal Peer Review Transparency  
**Evaluation Set**: 150 Reserved Test Scans (Cases 851–1000)

---

## 📊 Summary of Lowest-Dice Test Cases

To eliminate selective reporting bias and provide comprehensive transparency for reviewers, this report analyzes the three lowest-Dice cases from the 150-case test cohort.

| Case ID | Dice (DSC) | IoU | Precision | Recall | HD95 (mm) | Primary Empirical Failure Mechanism |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **930** | `0.4379` | `0.2803` | `0.4781` | `0.4039` | `57.94` | Large contiguous non-coronary false-positive structure retained by size-based Top-2 cc3d pruning over attenuated distal branches. |
| **941** | `0.4611` | `0.2996` | `0.4622` | `0.4599` | `56.73` | Contiguous adjacent high-contrast vascular/chamber over-segmentation surviving component-size ranking. |
| **978** | `0.5846` | `0.4130` | `0.8154` | `0.4556` | `35.16` | Distal sub-millimeter arterial branch tapering with conservative foreground thresholding (high precision `0.8154`, reduced recall `0.4556`). |

---

## 🔬 Detailed Technical & Anatomical Root-Cause Breakdown

### 1. Cases 930 & 941: The Topological Top-2 Component Ranking Blindspot
- **Empirical Observation in Gallery**: In the Maximum Intensity Projection (MIP) overlays for Cases 930 and 941, the false-positive prediction (red) appears as a **large, solid, contiguous 3D anatomical structure** physically separated from the true coronary tree (green/yellow).
- **Underlying Mechanism**:
  1. The network occasionally misclassifies an adjacent high-contrast anatomical structure (e.g., contrast-enhanced cardiac chamber margin, aortic root border, or adjacent mediastinal vessel).
  2. The inference post-processing pipeline applies `cc3d` component ranking to retain the **Top-2 largest 3D connected components** (intended to preserve the Left Coronary Artery [LCA] and Right Coronary Artery [RCA] trees).
  3. When an adjacent false-positive structure is predicted with large volumetric continuity, its voxel count can exceed that of a fragmented or thin true distal arterial branch. Consequently, the size-based filter **preserves the large non-coronary blob and discards the true distal branch**, resulting in lower precision ($0.46–0.48$) and high 95% Hausdorff distance ($56–58\text{ mm}$).

### 2. Case 978: Sub-Millimeter Distal Vessel Tapering
- **Empirical Observation in Gallery**: High spatial alignment along main proximal and mid arterial segments, with boundary termination at peripheral capillary bifurcations.
- **Underlying Mechanism**:
  - The model maintains high precision (**`0.8154`**), demonstrating effective suppression of background floating artifacts.
  - However, where coronary branch diameters fall below 2 voxels ($<1.0\text{ mm}$), the conservative $0.6$ foreground probability threshold truncates distal micro-vessel tips, resulting in reduced recall ($0.4556$).

---

## 📝 Recommended Manuscript Phrasing

### A. Results Section Framing (Accurate Aggregate Precision Gain)
> *"RASNet significantly enhanced false-positive suppression compared to the baseline SegResNet architecture, increasing mean precision from 81.40% to 85.85% ($p = 2.70 \times 10^{-20}$) and reducing boundary distance error across the 150-case test cohort."*
> *(Avoid absolute claims such as '100% hallucination removal', which are contradicted by edge cases with contiguous non-coronary false positives).*

### B. Discussion & Limitations Section Framing (Disclosing the Structural Blindspot)
> *"**Limitations of Heuristic Topological Post-Processing**: While Top-2 connected-component pruning effectively eliminates scattered floating background noise across the vast majority of cases, it introduces a specific vulnerability when the network misclassifies large, contiguous non-target contrast-enhanced structures (e.g., in Cases 930 and 941). Because size-based ranking prioritizes volume rather than anatomical connectivity or vascular centerline topology, large non-coronary false positives can be retained over fragmented true distal branches. Future iterations will replace pure volume ranking with anatomical-prior graph matching, centerline tree continuity tracking, or multi-class whole-heart semantic segmentation."*
