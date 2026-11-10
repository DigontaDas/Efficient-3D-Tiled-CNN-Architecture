# 🔍 Quantitative Failure Mode & Limitations Analysis
### Analysis of Lowest-Performing Test Cases (ImageCAS Test Cohort $N=150$, Matched 200-Epoch Benchmark)

**Target**: Q1 Medical Imaging Journal Peer Review Transparency  
**Evaluation Set**: 150 Reserved Test Scans (Cases 851–1000)

---

## 📊 Summary of Lowest-Dice Test Cases (200-Epoch RASNet)

To eliminate selective reporting bias and provide comprehensive transparency for reviewers, this report analyzes the lowest-Dice cases from the 150-case test cohort.

| Case ID | Dice (DSC) | IoU | Precision | Recall | HD95 (mm) | Primary Empirical Failure Mechanism |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **931** | `0.4644` | `0.3024` | `0.5871` | `0.3841` | `48.09` | Large contiguous non-coronary false-positive structure retained by size-based Top-2 cc3d pruning over attenuated distal branches. |
| **946** | `0.5592` | `0.3881` | `0.7586` | `0.4428` | `22.09` | Contiguous adjacent high-contrast vascular/chamber over-segmentation surviving component-size ranking. |
| **861** | `0.5734` | `0.4019` | `0.9286` | `0.4147` | `36.97` | Very high precision (`0.9286`) with distal sub-millimeter arterial branch tapering truncating low-contrast terminal vessels. |
| **978** | `0.5757` | `0.4042` | `0.8254` | `0.4420` | `35.42` | Distal sub-millimeter arterial branch tapering with conservative foreground thresholding (high precision `0.8254`, reduced recall `0.4420`). |

---

## 🔬 Detailed Technical & Anatomical Root-Cause Breakdown

### 1. Cases 931 & 946: The Topological Top-2 Component Ranking Blindspot
- **Empirical Observation in Gallery**: In the Maximum Intensity Projection (MIP) overlays for Cases 931 and 946, the false-positive prediction appears as a **contiguous 3D anatomical structure** physically adjacent to the true coronary tree.
- **Underlying Mechanism**:
  1. The network occasionally misclassifies an adjacent high-contrast anatomical structure (e.g., contrast-enhanced cardiac chamber margin, aortic root border, or adjacent mediastinal vessel).
  2. The inference post-processing pipeline applies `cc3d` component ranking to retain the **Top-2 largest 3D connected components** (intended to preserve the Left Coronary Artery [LCA] and Right Coronary Artery [RCA] trees).
  3. When an adjacent false-positive structure is predicted with large volumetric continuity, its voxel count can exceed that of a fragmented or thin true distal arterial branch. Consequently, the size-based filter **preserves the large non-coronary blob and discards the true distal branch**, resulting in lower recall ($0.38–0.44$) and high 95% Hausdorff distance ($22–48\text{ mm}$).

### 2. Cases 861 & 978: Sub-Millimeter Distal Vessel Tapering
- **Empirical Observation**: Exceptional spatial alignment along main proximal and mid arterial segments, with boundary termination at peripheral capillary bifurcations.
- **Underlying Mechanism**:
  - The model maintains extremely high precision (**`0.9286`** in Case 861, **`0.8254`** in Case 978), demonstrating near-complete suppression of background floating artifacts.
  - However, where coronary branch diameters fall below 2 voxels ($<1.0\text{ mm}$), the conservative foreground probability threshold truncates distal micro-vessel tips, resulting in reduced recall ($0.41–0.44$).

---

## 📝 Recommended Manuscript Phrasing

### A. Results Section Framing (Accurate Aggregate Precision Gain)
> *"In the matched 200-epoch benchmark, RASNet significantly enhanced false-positive suppression compared to all baseline architectures, increasing mean precision from 51.40% (SegResNet) to 88.01% ($p = 7.36 \times 10^{-25}$) and halving boundary distance error (HD95: 10.29 mm vs. 22.61 mm for SegResNet, 21.10 mm for nnU-Net V2; $p < 10^{-9}$ across all comparators). Single-volume inference latency remained rapid at 1.85s."*

### B. Discussion & Limitations Section Framing (Disclosing the Structural Blindspot)
> *"**Limitations of Heuristic Topological Post-Processing**: While Top-2 connected-component pruning effectively eliminates scattered floating background noise across the vast majority of cases, it introduces a specific vulnerability when the network misclassifies large, contiguous non-target contrast-enhanced structures (e.g., in Cases 931 and 946). Because size-based ranking prioritizes volume rather than anatomical connectivity or vascular centerline topology, large non-coronary false positives can be retained over fragmented true distal branches. Future iterations will replace pure volume ranking with anatomical-prior graph matching, centerline tree continuity tracking, or multi-class whole-heart semantic segmentation."*
