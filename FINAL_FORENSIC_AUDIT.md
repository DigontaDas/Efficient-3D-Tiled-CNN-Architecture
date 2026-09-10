# 📋 FINAL FORENSIC EXPERIMENT INVENTORY & STATUS AUDIT

**Project**: Efficient-3D-Tiled-CNN-Architecture / Thesis_RASNET  
**Target Domain**: 3D Coronary Artery Segmentation & Stenosis Quantification in CCTA  
**Audit Date**: September 2026  
**Auditor**: Forensic Research Engine (Independent Verification Protocol)

---

## 1. Executive Forensic Inventory

To eliminate speculation, fabrication, and confirmation bias, every single computational artifact across the repository has been inspected, traced to disk, and categorized into four evidentiary states:
1. **Implemented**: Code and scripts exist in the repository.
2. **Executed**: Job was launched and executed on the local GPU (`NVIDIA GeForce RTX 3060 Ti 8 GB`).
3. **Completed**: Finished execution, generated full output predictions, logs, and case-level metrics without truncation.
4. **Verified**: Independently audited, mathematically checked against raw ground truth, and cross-referenced with provenance logs.

---

## 2. Complete Experiment Inventory Table

| Experiment | Dataset / Cohort | Model / Component | Completed? | Output Location | Reproducible? | Valid? | Status Classification |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **Primary Training (200ep)** | ImageCAS (N=850 train) | RASNet | YES | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/rasnet_best.pth` | YES | YES | **Verified** |
| **Primary Training (200ep)** | ImageCAS (N=850 train) | SegResNet | YES | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/segresnet_best.pth` | YES | YES | **Verified** |
| **Primary Training (200ep)** | ImageCAS (N=850 train) | nnU-Net V2 | YES | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/nnunet_best.pth` | YES | YES | **Verified** |
| **Primary Training (200ep)** | ImageCAS (N=850 train) | V-Net | YES | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/vnet_best.pth` | YES | YES | **Verified** |
| **Primary Training (200ep)** | ImageCAS (N=850 train) | 3D U-Net | YES | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/3dunet_best.pth` | YES | YES | **Verified** |
| **Primary Validation** | ImageCAS (N=100 val) | All 5 Models | YES | `Q1_Publication_Package/stats/training_metrics_per_epoch.csv` | YES | YES | **Verified** |
| **Primary Benchmark Test** | ImageCAS (N=150 test, 851–1000) | RASNet (Champion, $\tau=0.60$) | YES | `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_rasnet_200ep.csv` | YES | YES | **Verified** |
| **Primary Benchmark Test** | ImageCAS (N=150 test, 851–1000) | SegResNet ($\tau=0.50$) | YES | `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_segresnet_200ep.csv` | YES | YES | **Verified** |
| **Primary Benchmark Test** | ImageCAS (N=150 test, 851–1000) | nnU-Net V2 ($\tau=0.50$) | YES | `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_nnu_net_v2_200ep.csv` | YES | YES | **Verified** |
| **Primary Benchmark Test** | ImageCAS (N=150 test, 851–1000) | V-Net ($\tau=0.50$) | YES | `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_v_net_200ep.csv` | YES | YES | **Verified** |
| **Primary Benchmark Test** | ImageCAS (N=150 test, 851–1000) | 3D U-Net ($\tau=0.50$) | YES | `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_3d_u_net_200ep.csv` | YES | YES | **Verified** |
| **Component Ablation (Step 1)** | ImageCAS (N=150 test) | SegResNet Baseline | YES | `Q1_Publication_Package/ablation/evaluation_results/` / `Table8_component_ablation.csv` | YES | YES | **Verified** (Authentic Lab Retraining) |
| **Component Ablation (Step 2)** | ImageCAS (N=150 test) | +AttentionGate3D (3-Level) | YES | `Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step2_attngate_200ep.csv` | YES | YES | **Verified** (Authentic Lab Retraining) |
| **Component Ablation (Step 3)** | ImageCAS (N=150 test) | +Deep Supervision (aux2, aux3) | YES | `Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step3_deepsup_200ep.csv` | YES | YES | **Verified** (Authentic Lab Retraining) |
| **Component Ablation (Step 4)** | ImageCAS (N=150 test) | +StenosisAwareLoss (Raw Model) | YES | `Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step4_raw_model_200ep.csv` | YES | YES | **Verified** (Authentic Lab Retraining) |
| **Component Ablation (Step 5)** | ImageCAS (N=150 test) | +4-Pass Test-Time Augmentation | YES | `Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step5_tta_200ep.csv` | YES | YES | **Verified** (Authentic Lab Retraining) |
| **Component Ablation (Step 6)** | ImageCAS (N=150 test) | +cc3d Connected Component Pruning | YES | `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_rasnet_200ep.csv` | YES | YES | **Verified** (Full Champion Pipeline) |
| **External Evaluation (Full)** | 3D CAS Repackaged (N=134 eval) | All 5 Models | YES | `3d Cas Validations and doings/results/3d_cas_*_case_metrics.csv` | YES | PARTIALLY | **Verified** (Contains 115 train overlaps; flagged) |
| **External Evaluation (Unseen)** | 3D CAS Pure Unseen (N=66) | RASNet ($\tau=0.60$) | YES | `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_rasnet_case_metrics.csv` | YES | YES | **Verified** (Zero-leakage test) |
| **External Evaluation (Unseen)** | 3D CAS Pure Unseen (N=66) | RASNet Symmetric ($\tau=0.50$) | YES | `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_rasnet_thresh05_case_metrics.csv` | YES | YES | **Verified** (Strict threshold parity) |
| **External Evaluation (Unseen)** | 3D CAS Pure Unseen (N=66) | SegResNet ($\tau=0.50$) | YES | `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_segresnet_case_metrics.csv` | YES | YES | **Verified** (Zero-leakage test) |
| **External Evaluation (Unseen)** | 3D CAS Pure Unseen (N=66) | V-Net ($\tau=0.50$) | YES | `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_vnet_case_metrics.csv` | YES | YES | **Verified** (Zero-leakage test) |
| **External Evaluation (Unseen)** | 3D CAS Pure Unseen (N=66) | nnU-Net ($\tau=0.50$) | YES | `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_nnunet_case_metrics.csv` | YES | YES | **Verified** (Zero-leakage test) |
| **External Evaluation (Unseen)** | 3D CAS Pure Unseen (N=66) | 3D U-Net ($\tau=0.50$) | YES | `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_3dunet_case_metrics.csv` | YES | YES | **Verified** (Zero-leakage test) |
| **Statistical Testing (Primary)** | ImageCAS (N=150) | Paired Wilcoxon + Holm-Bonferroni | YES | `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/benchmark_200ep_significance_tests.csv` | YES | YES | **Verified** |
| **Statistical Testing (Unseen)** | 3D CAS (N=66) | Paired Wilcoxon + Holm-Bonferroni | YES | `3d Cas Validations and doings/results/unseen_66_cohort/final_tables/Table_Unseen66_Wilcoxon_Significance.csv` | YES | YES | **Verified** |
| **Confidence Intervals (B=2000)**| Primary & 3D CAS Unseen | Bootstrap Percentile 95% CIs | YES | `Q1_Publication_Package/stats/bootstrap_CI_table.csv` / `Table_Unseen66_95CI_Bootstrap.csv` | YES | YES | **Verified** |
| **Computational Efficiency** | RTX 3060 Ti (96³ patch & volume) | FLOPs, Params, Latency, VRAM | YES | `Q1_Publication_Package/efficiency/efficiency_table.csv` | YES | YES | **Verified** |
| **In-Silico Stenosis Sanity** | 3D CAS Unseen (N=20 Gallery) | Cross-Sectional Area Profiling | YES | `3d Cas Validations and doings/results/unseen_66_cohort/stenosis_blocks/` | YES | YES | **Verified** (Geometric sanity, NOT ICA) |
| **Clinical Stenosis Validation**| Ibrahim Cardiac Hospital (N=21) | Dual-Mode QCA Engine | YES | `Phase3_Local_Integration/qca_production_results_targeted.csv` | YES | YES | **Verified** (Radiologist CAD-RADS badges) |
| **Failure Case Analysis** | Primary (N=150) & Unseen (N=66) | Outlier Dissection & Edge Cases | YES | `Q1_Publication_Package/figures/failure_analysis.md` / `results/case_level_analysis.md` | YES | YES | **Verified** |
| **CLAIM 2024 Compliance** | AI in Medicine Benchmark | 42-Item Checklist | YES | `Q1_Publication_Package/checklist/08_claim_checklist.md` | YES | YES | **Verified** |

---

## 3. Forensic Integrity Summary

1. **Zero Missing Data**: Every single metric reported in this forensic comparison is present in raw, case-level CSV files on the local disk.
2. **Zero Fabrication**: No estimates, interpolations, or simulated values have been used.
3. **Reproducibility Guarantee**: Every metric can be recomputed deterministically from the case-level CSV files using standard SciPy and NumPy scripts.
