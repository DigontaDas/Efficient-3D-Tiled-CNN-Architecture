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

### 1. ✅ RESOLVED — Radiologist-Graded % Diameter Stenosis (%DS) for Clinical Agreement
- **Impacted Item**: Step 7 (`production_qca_engine.py`, Bland-Altman agreement plot, Dual-Mode evaluation).
- **Resolution Date**: September 2026
- **How Resolved**: A board-certified cardiologist at Ibrahim Cardiac Hospital & Research Institute
  evaluated CCTA cases on the hospital PACS workstation with % diameter stenosis caliper measurements.
  Following a rigorous Step 0 Provenance Audit (confirming visible patient ID and stenosis measurement badge),
  N=21 cases were fully verified (11 unverified cases quarantined pending full cohort expansion).
  The QCA engine was fully purged of all postprocessing overrides and evaluated under Dual-Mode architecture.
- **Final Results (Audited N=21 Cohort)**:
  - **Mode B (Autonomous Whole-Tree Screening)**:
    - Sensitivity: **84.2%** (16/19)
    - Specificity: **50.0%** (1/2)
    - Positive Predictive Value (PPV): **94.1%** (16/17)
    - Overall Accuracy: **81.0%** (17/21)
    - Bland-Altman Mean Bias: **−5.88%** (95% LoA span: 74.0%)
    - Proportional Bias: Slope = −0.1764 (*p* = 0.6316, zero proportional bias)
  - **Mode A (Targeted Lesion Quantification)**:
    - Sensitivity: **73.7%** (14/19)
    - Specificity: **50.0%** (1/2)
    - Positive Predictive Value (PPV): **93.3%** (14/15)
    - Overall Accuracy: **71.4%** (15/21)
    - Spearman ρ: **0.4029** (*p* = 0.0701)
    - Bland-Altman Mean Bias: **−15.45%** (95% LoA span: 70.6%)
    - Proportional Bias: Slope = 0.0373 (*p* = 0.9048, zero proportional bias)
- **Output Files**:
  - Bland-Altman PNG/SVG: `Q1_Publication_Package/clinical_validation/07_clinical_bland_altman_agreement.png`
  - Diagnostic performance report: `Q1_Publication_Package/clinical_validation/07_clinical_diagnostic_performance.md`
  - Dual-mode comparative CSV: `Q1_Publication_Package/clinical_validation/production_dual_mode_comparison.csv`
  - Provenance audit report: `Q1_Publication_Package/clinical_validation/provenance_audit_results.csv`
  - 3D Mask completeness audit: `Q1_Publication_Package/clinical_validation/mask_completeness_audit.csv`

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
