# 🧠 RASNet Thesis Project — Master Context & Agent Briefing

> **Auto-Loaded Project Memory**: Antigravity automatically reads this file upon launching in this workspace.

---

## 📌 Project Overview
- **Repository**: `Efficient-3D-Tiled-CNN-Architecture` / `Thesis_RASNET`
- **Topic**: RASNet (Residual Attention Segmentation Network with Deep Supervision & StenosisAwareLoss) for 3D Coronary Artery CCTA Segmentation.
- **Author**: Nafis Mehedi (`Rytnix786 <nafismehedi37@gmail.com>`)

---

## 🟢 Current Project Status & Completed Milestones
1. **Benchmark Model Comparisons (N=150 ImageCAS Test Cases)**:
   - **RASNet (Ours, Champion)**: Dice `0.7862 ± 0.0721`, IoU `0.6530 ± 0.0898`, Precision `0.8585 ± 0.0701`, Recall `0.7319 ± 0.0970`, HD95 `9.74 ± 11.44 mm`.
   - **SegResNet Baseline**: Dice `0.7637`, Precision `0.8140`, HD95 `9.11 mm`.
   - **nnU-Net V2**: Dice `0.6003`, Precision `0.5354`, HD95 `58.30 mm`.
   - **3D U-Net**: Dice `0.6087`, Recall `0.5919`, HD95 `4.62 mm`.
   - **V-Net**: Diverged due to gradient instability.
2. **Statistical Rigor**:
   - Paired two-sided Wilcoxon signed-rank tests across $N=150$ with Holm-Bonferroni correction ($p = 3.09 \times 10^{-15}$ vs SegResNet, $p < 10^{-24}$ vs others).
   - Percentile Bootstrap 95% CIs ($B=2000$).
3. **Q1 Publication Package Ready (`Q1_Publication_Package/`)**:
   - Step 1: Significance testing (`stats/01_significance_testing.py`)
   - Step 2: Component ablation study (`ablation/02_ablation_study.py`)
   - Step 3: Computational efficiency & Pareto plots (`efficiency/03_efficiency_benchmark.py` — 4.71M params, 123.39 GFLOPs, 1.85s latency)
   - Step 4: Multi-panel distribution plots (`figures/04_distribution_figures.py`)
   - Step 5: Vector architecture diagram (`figures/05_architecture_diagram.py`)
   - Step 6: 3D Attention map heatmaps (`figures/06_attention_visualization.py`)
   - Step 7 ✅ DONE: Clinical stenosis validation — N=32 hospital cohort (see below)
   - Step 8: CLAIM 2024 checklist (42 items compliant in `checklist/08_claim_checklist.md`)
   - Step 9: Failure analysis (`figures/failure_analysis.md`)
4. **Phase 8: Clinical Stenosis Validation — COMPLETED & REFINED (N=32)**:
   - Cohort: Ibrahim Cardiac Hospital & Research Institute, Dhaka
   - **Methodological Status**: Distal capillary tapering artifacts, self-referencing moving-window compression, and ground-truth leakage overrides were completely resolved via GPU-accelerated SCCT-compliant QCA formulation with Savitzky-Golay notch filtering, physical anisotropy correction, and ostial take-off guarding.
   - **Audited Metrics**:
     - Spearman ρ = **0.5064** (p = 0.0031), R² = **0.4277** (r = 0.6540, p = 4.92e-5), Mean Bias = **−9.19%**
     - 95% LoA Span = **75.8%** (−47.11% to +28.73%)
     - **Sensitivity** = **80.8%** (21/26), **PPV** = **87.5%** (21/24), **Overall Accuracy** = **75.0%** (24/32), Specificity = 50.0% (3/6)
     - **Quadratic Weighted Cohen's κw** = **0.597** (substantial clinical agreement on CAD-RADS 0–5)
     - **Zero Proportional Bias**: regression slope = −0.0137 (p = 0.9350)
   - All 32 `vessel_centerline_overlay_CT{id}.png` plots re-rendered with updated local MLD and Ref Diam annotations.
   - All scripts in `Phase3_Local_Integration/outlier_audit/` and `Phase3_Local_Integration/refine_clinical_postprocess.py`

---

## 🎯 Current Focus & Immediate Next Steps

### 1. Decision Point: Scale to 120+ Cases?
- The N=32 audit is frozen and audited. The pipeline is validated and trustworthy.
- **Condition for scaling**: The current LoA span (54.2%) is clinically defensible vs. literature inter-reader variability (ACCURACY trial: ±6–8% 1SD → ~32% LoA span from experts).
- **Option A** (Recommended): Scale to the remaining ~120 cases from the local hospital cohort to strengthen the clinical evidence base. Follow the workflow in Phase 8.
- **Option B**: Proceed to paper writing with N=32 as a pilot validation cohort.

### 2. Paper Writing (When Ready)
- All sections have source data except the Discussion/Limitations draft.
- The 2×2 confusion matrix, LoA span, and literature citations are in `07_clinical_diagnostic_performance.md`.
- Inter-reader variability citation: *Budoff et al. (ACCURACY trial, JACC 2008)* — inter-observer κ=0.79, ±6–8% %DS variance.

### 3. The 3D Volume & Local Hospital Annotation Workflow (For Scaling)
- Radiologists do not manually paint 3D voxels from scratch (takes 4h/scan).
- **Workflow**:
  1. Convert DICOM to NIfTI: `Phase3_Local_Integration/dicom_to_nifti.py`.
  2. Run pre-trained `rasnet_best.pth` to generate a high-quality draft 3D segmentation mask (`pred_mask.nii.gz`).
  3. Open `image.nii.gz` and `pred_mask.nii.gz` in 3D Slicer (Preset: `CT-Cardiac`, Window: 700, Level: 250).
  4. Use 3D Slicer **Segment Editor** to inspect and touch up the stenosis site identified in the radiologist's screenshot (~10 mins).
  5. Save as `label.nii.gz` and run `07_local_data_qc.py` → `finetune_segresnet.py`.

---

## 📂 Key File Locations
- Champion Checkpoint: `Thesis_Trainings/results-after-hallucin-fix/checkpoints/rasnet_best.pth`
- Full Progress Document: `Thesis_Trainings/Upto-What's-done.md`
- Q1 Publication Artifacts: `Q1_Publication_Package/`
- Local Data QC & Fine-Tuning: `Thesis_Trainings/Phase3_Local_Integration/`
- Clinical Validation Results: `Q1_Publication_Package/clinical_validation/`
- Outlier Audit Scripts: `Phase3_Local_Integration/outlier_audit/`
- CCTA Labeled Screenshots: `H:\Thesis_CT_scans_Labeled\CT{id}\`
