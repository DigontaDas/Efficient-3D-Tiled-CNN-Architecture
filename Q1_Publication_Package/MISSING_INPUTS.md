# 📋 Missing Inputs & Academic Integrity Tracking Log

**Target Repository**: `Efficient-3D-Tiled-CNN-Architecture` (RASNet, ImageCAS Dataset)  
**Date**: August 2026  
**Purpose**: Document all external datasets, annotations, radiologist gradings, or compute resources required by publication guidelines that are not currently present in the repository, explaining why they cannot be simulated or fabricated and outlining the necessary steps to fulfill them.

---

## 🔒 Academic Integrity Policy
In strict compliance with medical imaging AI publication ethics (RSNA *Radiology: Artificial Intelligence*, *Medical Image Analysis*, *IEEE TMI*):
- **Zero Synthetic Clinical Data**: No simulated p-values, synthetic radiologist stenosis grades, or placeholder statistics are permitted.
- **Transparent Gap Reporting**: Any analysis requiring unavailable external data is clearly identified as a limitation rather than omitted silently.

---

## 📌 Missing Inputs Inventory

### 1. Radiologist-Graded % Diameter Stenosis (%DS) for Clinical Agreement
- **Impacted Item**: Step 7 (`07_stenosis_agreement.py`, Bland-Altman agreement plot, ICC(2,1), Cohen's $\kappa$).
- **Current State in Repo**:
  - The repository contains automated skeleton-based diameter extraction and stenosis reports from segmentation masks (`11_clinical_postprocess.py`, `results-after-hallucin-fix/clinical_postprocess/`).
  - However, independent ground truth **radiologist-graded % diameter stenosis values** (e.g. measured via quantitative coronary angiography [QCA] or expert radiologist caliper measurements on CCTA) are not included in the public ImageCAS dataset.
- **Why It Cannot Be Scripted**: Simulating radiologist agreement numbers or synthetic caliper measurements would violate academic integrity and produce fraudulent validation metrics.
- **Action Required**: A certified radiologist/cardiologist must grade a designated subset (e.g., $N=30-50$ test cases) for % diameter stenosis at key anatomical lesions (proximal LAD, mid RCA, LCx). The provided script `07_stenosis_agreement.py` is pre-configured and will immediately compute Spearman $\rho$, Bland-Altman limits of agreement ($\pm 1.96\text{ SD}$), ICC(2,1), and CAD-RADS Cohen's $\kappa$ once the CSV is supplied.

---

### 2. Isolated Anatomical Branch-Level Ground Truth Labels (LAD / LCx / RCA)
- **Impacted Item**: Step 8 (Per-Vessel / Per-Segment Stratified Performance).
- **Current State in Repo**:
  - The public ImageCAS dataset provides binary 3D masks ($1 = \text{coronary arterial tree}$, $0 = \text{background}$) for all 1,000 scans without branch-level anatomical separation (e.g. Left Anterior Descending [LAD], Left Circumflex [LCx], Right Coronary Artery [RCA], or 18-segment AHA model).
- **Why It Cannot Be Scripted**: True anatomical classification requires manual branch labeling by expert annotators or specialized anatomical tree parsing models.
- **Manuscript Disclosure**: This is formally disclosed as an empirical limitation in the thesis and manuscript discussion, as well as the CLAIM 2024 checklist (Item 18 & 34). Whole-tree metrics ($N=150$) are reported.

---

### 3. Local Hospital Clinical CCTA Cohort Ground Truth Annotations
- **Impacted Item**: Step 9 (External Multi-Center / Multi-Scanner Generalization).
- **Current State in Repo**:
  - Phase 3 local integration infrastructure is fully built and verified:
    - DICOM translation pipeline ([`dicom_to_nifti.py`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/dicom_to_nifti.py))
    - Spatial and geometric quality control ([`07_local_data_qc.py`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/07_local_data_qc.py))
    - Fine-tuning adaptation pipeline ([`finetune_segresnet.py`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/finetune_segresnet.py))
  - However, local hospital scans (the ~150 Bangladeshi clinical cohort scans mentioned in thesis plans) require completed radiologist segmentations and institutional review/anonymization before model fine-tuning and quantitative scoring.
- **Why It Cannot Be Scripted**: External validation claims must be based on real annotated patient data from the target clinical scanner.
- **Priority**: Highest clinical priority for post-submission or revision cycle.

---

### 4. Intermediate Ablation Retraining Checkpoints (Rows 2–4)
- **Impacted Item**: Step 2 (Progressive Component-Wise Ablation Study).
- **Current State in Repo**:
  - Checkpoints available:
    - SegResNet Baseline (Row 1): $N=150$ per-case metrics available.
    - Final Post-Fix Champion RASNet (Row 6): Full 70-epoch checkpoint (`rasnet_best.pth`) and $N=150$ per-case metrics available.
    - Inference-only ablations (Row 5: without cc3d, with TTA; Row 4 inference-only: without cc3d, without TTA) can be evaluated directly from `rasnet_best.pth`.
  - Checkpoints missing: Dedicated 70-epoch training runs for Row 2 (+ AttentionGate3D only, plain loss) and Row 3 (+ Deep Supervision only, plain loss).
- **Hardware & Compute Schedule**:
  - Hardware: NVIDIA GeForce RTX 4080 SUPER (16 GB VRAM).
  - Estimated Compute Time: ~1.5 to 2.0 hours per full 70-epoch run ($\approx 3.5-4.0$ hours total).
- **Status**: The ablation script evaluates all available rows and provides clear placeholders with exact compute estimates for full retraining if desired by the user.

---
