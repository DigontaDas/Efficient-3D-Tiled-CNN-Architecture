# 🧠 RASNet Thesis Project — Master Context & Agent Briefing

> **Auto-Loaded Project Memory**: Antigravity automatically reads this file upon launching in this workspace.

---

## 📌 Project Overview
- **Repository**: `Efficient-3D-Tiled-CNN-Architecture` / `Thesis_RASNET`
- **Topic**: RASNet (Residual Attention Segmentation Network with Deep Supervision & StenosisAwareLoss) for 3D Coronary Artery CCTA Segmentation.
- **Author**: Nafis Mehedi (`Rytnix786 <nafismehedi37@gmail.com>`)

---

## 🟢 Current Project Status & Completed Milestones
1. **Matched 200-Epoch Benchmark Comparisons (N=150 ImageCAS Test Cases)**:
   - **RASNet (Ours, Champion)**: Dice `0.7765 ± 0.0695`, IoU `0.6396 ± 0.0864`, Precision `0.8801 ± 0.0530`, Recall `0.7016 ± 0.0976`, HD95 `10.29 ± 10.49 mm`, ASD `1.598 ± 1.824 mm`, clDice `0.8592 ± 0.0719`, Centerline Recall `0.8018 ± 0.1022`.
   - **nnU-Net V2**: Dice `0.7687 ± 0.0668`, IoU `0.6289 ± 0.0856`, Precision `0.7391 ± 0.0967`, Recall `0.8106 ± 0.0680`, HD95 `21.10 ± 17.10 mm`, ASD `3.097 ± 2.642 mm`, clDice `0.8201 ± 0.0766`, Centerline Recall `0.9111 ± 0.0556`.
   - **V-Net (Gradient-clip stabilized, converged)**: Dice `0.7474 ± 0.0633`, IoU `0.6006 ± 0.0786`, Precision `0.7545 ± 0.0982`, Recall `0.7499 ± 0.0678`, HD95 `21.81 ± 19.14 mm`, ASD `3.176 ± 2.771 mm`, clDice `0.8075 ± 0.0759`, Centerline Recall `0.8571 ± 0.0613`.
   - **SegResNet**: Dice `0.7469 ± 0.0640`, IoU `0.6001 ± 0.0788`, Precision `0.7313 ± 0.0911`, Recall `0.7713 ± 0.0654`, HD95 `31.48 ± 18.16 mm`, ASD `4.661 ± 2.718 mm`, clDice `0.7769 ± 0.0819`, Centerline Recall `0.8866 ± 0.0579`.
   - **3D U-Net**: Dice `0.5561 ± 0.0458`, IoU `0.3865 ± 0.0429`, Precision `0.6069 ± 0.0646`, Recall `0.5178 ± 0.0536`, HD95 `9.88 ± 7.18 mm`, ASD `1.690 ± 1.044 mm`, clDice `0.7027 ± 0.0611`, Centerline Recall `0.6534 ± 0.0739`.
2. **Statistical Rigor**:
   - Paired two-sided Wilcoxon signed-rank tests across $N=150$ with Holm-Bonferroni correction. RASNet achieves statistically significant superior Precision ($p = 7.36 \times 10^{-25}$), boundary accuracy (HD95 $p = 8.05 \times 10^{-10}$, ASD $p = 1.51 \times 10^{-11}$), and topological connectivity (clDice $p = 6.60 \times 10^{-09}$) over nnU-Net V2, and outperforms SegResNet (+14.9% Precision $p = 7.36 \times 10^{-25}$, +8.2% clDice $p = 8.72 \times 10^{-20}$, HD95 3× lower error $p = 8.72 \times 10^{-20}$), V-Net (+12.6% Precision, +5.2% clDice $p = 4.03 \times 10^{-14}$), and 3D U-Net across key metrics.
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
4. **Phase 8: Clinical Stenosis Validation — PROVENANCE AUDITED & DUAL-MODE REFINED (N=21 Verified)**:
   - Cohort: Ibrahim Cardiac Hospital & Research Institute, Dhaka
   - **Step 0 Provenance Audit**: 21 verified cases (both patient ID and stenosis badge visible); 11 unverified cases permanently quarantined pending the 150-case expansion.
   - **Step 1 Leakage Audit**: All string-matching cheats and overrides (`is_patent`, `occlusion`) completely eliminated. `test_perfect_balance.py` audited and quarantined to `_archive_legacy_postprocess/`. Single authoritative engine: `Phase3_Local_Integration/production_qca_engine.py`.
   - **Dual-Mode Clinical Results (N=21 Verified — Modernized Engine)**:
     - **Selective Notch Sharpening & Area Stenosis Upgrade**:
       - **Mode A (Targeted Quantification)**: Spearman ρ = **0.5410** (**p = 0.0113, statistically significant p < 0.05**), Pearson R² = **0.2000** (r = 0.447, p = 0.0421), Mean Bias = **−12.14%**, 95% LoA Span = **66.1%** (−45.2% to +20.9%), Zero Proportional Bias (p = 0.9752), **Sensitivity = 78.9%** (15/19), Specificity = 50.0% (1/2), **PPV = 93.8%** (15/16), **Overall Accuracy = 76.2%** (16/21), **Adjacent (±1 Tier) CAD-RADS Accuracy = 95.2%** (20/21), **PABAK = 0.524**, **Area Stenosis Exact Accuracy = 57.1%**.
     - **Mode B (Autonomous Whole-Tree)**: Sensitivity = **84.2%** (16/19), Specificity = 50.0% (1/2), **PPV = 94.1%** (16/17), **Overall Accuracy = 81.0%** (17/21), Mean Bias = **−3.53%**, LoA Span = **71.3%** (−39.2% to +32.1%), Zero Proportional Bias (p = 0.7491), **Adjacent (±1 Tier) CAD-RADS Accuracy = 95.2%** (20/21), **PABAK = 0.619** (Substantial Agreement).
   - **Adjudications**: CT4 ground truth formally resolved to **85.0% (CAD-RADS 4, Severe)** under SCCT CAD-RADS 2.0 worst-lesion rule (4.2.png shows proximal LCx 70–99%).
   - All legacy postprocessing scripts moved to `Phase3_Local_Integration/_archive_legacy_postprocess/` and scratch scripts moved to `_archive_scratch/`.
   - Primary deliverables: `production_qca_engine.py`, `qca_production_results_targeted.csv`, `qca_production_results_autonomous.csv`, `production_dual_mode_comparison.csv`, and `07_clinical_bland_altman_agreement.png/.svg`.

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
- Champion Checkpoint (200-Epoch Matched): `Thesis_Trainings/Q1_Publication_Package/matched_200ep_benchmark/checkpoints/rasnet_best.pth`
- Archived Historical Checkpoints: `Thesis_Trainings/_Archive_Historical_Experiments_and_Logs/archived_rasnet_checkpoints/`
- Full Progress Document: `Thesis_Trainings/Upto-What's-done.md`
- Q1 Publication Artifacts: `Q1_Publication_Package/`
- Local Data QC & Fine-Tuning: `Thesis_Trainings/Phase3_Local_Integration/`
- Clinical Validation Results: `Q1_Publication_Package/clinical_validation/`
- Outlier Audit Scripts: `Phase3_Local_Integration/outlier_audit/`
- CCTA Labeled Screenshots: `H:\Thesis_CT_scans_Labeled\CT{id}\`
