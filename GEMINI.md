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
   - Step 8: CLAIM 2024 checklist (42 items compliant in `checklist/08_claim_checklist.md`)
   - Step 9: Failure analysis (`figures/failure_analysis.md`)
   - Gap log: `Q1_Publication_Package/MISSING_INPUTS.md`

---

## 🎯 Current Focus & Immediate Next Steps (Phase 3: Primary Clinical Data)

### 1. The Clinical Stenosis Screenshots from Radiologist:
- The radiologist evaluated CCTA cases on PACS and took 2D screenshots with % stenosis caliper readings.
- **For Step 7 (Clinical Agreement)**: Match the radiologist's % stenosis from screenshots with automated skeleton/EDT % stenosis from `11_clinical_postprocess.py`.
- Fill into `Q1_Publication_Package/clinical_validation/template_radiologist_grades.csv` and run `Q1_Publication_Package/clinical_validation/07_stenosis_agreement.py` for Bland-Altman & Cohen's Kappa.

### 2. The 3D Volume & Local Hospital Annotation Workflow:
- Radiologists do not manually paint 3D voxels from scratch (takes 4h/scan).
- **Workflow**:
  1. Convert DICOM to NIfTI: `Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/dicom_to_nifti.py`.
  2. Run pre-trained `rasnet_best.pth` to generate a high-quality draft 3D segmentation mask (`pred_mask.nii.gz`).
  3. Open `image.nii.gz` and `pred_mask.nii.gz` in 3D Slicer (Preset: `CT-Cardiac`, Window: 700, Level: 250).
  4. Use 3D Slicer **Segment Editor** to inspect and touch up the stenosis site identified in the radiologist's screenshot (takes ~10 mins).
  5. Save as `label.nii.gz` and run `07_local_data_qc.py` $\rightarrow$ `finetune_segresnet.py`.

---

## 📂 Key File Locations
- Champion Checkpoint: `Thesis_Trainings/Thesis_Trainings/results-after-hallucin-fix/checkpoints/rasnet_best.pth`
- Full Progress Document: `Thesis_Trainings/Thesis_Trainings/Upto-What's-done.md`
- Q1 Publication Artifacts: `Q1_Publication_Package/`
- Local Data QC & Fine-Tuning: `Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/`
