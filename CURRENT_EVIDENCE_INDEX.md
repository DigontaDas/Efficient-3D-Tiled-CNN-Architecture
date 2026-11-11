# Canonical Evidence Index (`CURRENT_EVIDENCE_INDEX.md`)

**Last Updated**: `2026-11-10`  
**Workstation**: `NVIDIA GeForce RTX 4080 SUPER (16,376 MiB VRAM)`, Windows 11  
**Python Environment**: `C:\Thesis_RASNET_env\bench\Scripts\python.exe` (`Python 3.12.10`, `torch 2.14.1+cu126`, `monai 1.6.1`, `nnunetv2 2.8.1`, `SimpleITK 2.5.7`)

This file is the **single authoritative index** of every active, verified evidence artifact in `C:\Thesis_RASNET`. Every number reported in active thesis tables, figures, and documentation traces directly to the files indexed below.

---

## 1. Frozen Split Manifest & Validation Thresholds

| Artifact | Path | SHA-256 Digest | Description |
| :--- | :--- | :--- | :--- |
| **Canonical Split Manifest** | [`Phase3_Local_Integration/splits_final.json`](Phase3_Local_Integration/splits_final.json) | `efe3f06cfe9c7dff646224378ad4d0ea5ce50fb4bf920807b17054af4bdf953c` | `690 train`, `94 val`, `150 test` (`IDs 851–1000`); `0` overlaps across splits. |
| **Frozen Validation Thresholds** | [`results/thresholds.json`](results/thresholds.json) | `4f66ac1d415c5ca9bc09ca0e900e9f8f5d106b3620a20d5916f472d1ba0d3acb` | Selected on `val_full_94` (`N=94`) over $\tau \in [0.30..0.70]$ with 26-connected top-2 CC pruning: `rasnet=0.30`, `segresnet=0.35`, `vnet=0.30`, `3dunet=0.30`, `nnunet=0.35`. |

---

## 2. Verified 200-Epoch Model Checkpoints (`CHECKPOINTS.md`)

| Model | Checkpoint Path | SHA-256 Digest | Best Epoch | Native-Grid Val Dice (Arm A) | Native-Grid Val Dice (Arm C) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **RASNet** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/rasnet_best.pth` | `8d4c25cd12d07c40423062b27b1d36836e5fd6cecb30c20c34055eeac976d1b8` | `160` | `0.7789 ± 0.0521` | `0.7897` ($\tau^*=0.30$) |
| **nnU-Net V2** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/nnunet_best.pt` | `9accddf283b1c480e386f30f426b9200ca8d4aa1ecd70d5b5af0b8aad6058711` | `189` | `0.7761 ± 0.0567` | `0.8075` ($\tau^*=0.35$) |
| **SegResNet** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/segresnet_best.pt` | `b8ea61d95106a0bbe15e1ce9f9e5af1a38d9508e59d0639d32bc883d93c6e109` | `190` | `0.7532 ± 0.0591` | `0.7901` ($\tau^*=0.35$) |
| **V-Net** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/vnet_best.pt` | `a89d71bd41c3e773c68e3f4b5e27c5a067f959154aa049eead76f1642c5c1063` | `180` | `0.7511 ± 0.0619` | `0.7702` ($\tau^*=0.30$) |
| **3D U-Net** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/3dunet_best.pth` | `d8a8f5ec95b5f34e2afe0b807ac49bb36d136f9d864238c27a763a29a800a60f` | `180` | `0.5516 ± 0.0480` | `0.6047` ($\tau^*=0.30$) |

---

## 3. Verified Benchmark Runs (`results/benchmark_runs/`)

1. **[`results/benchmark_runs/val_full_94/`](results/benchmark_runs/val_full_94/)** (`N = 94` validation cases, Arm A + validation threshold sweep):
   - Manifest: [`manifest.json`](results/benchmark_runs/val_full_94/manifest.json)
   - Summary table: [`summary_arm_A.csv`](results/benchmark_runs/val_full_94/summary_arm_A.csv)
   - Per-case metrics: [`per_case_arm_A.csv`](results/benchmark_runs/val_full_94/per_case_arm_A.csv) (`470` rows = `94 cases × 5 models`)
2. **[`results/benchmark_runs/test_retrospective_150/`](results/benchmark_runs/test_retrospective_150/)** (`N = 150` retrospective test cases `851–1000`, Arms A, B, C + historical recipe diagnostic):
   - Manifest: [`manifest.json`](results/benchmark_runs/test_retrospective_150/manifest.json)
   - Summary tables: [`summary_arm_A.csv`](results/benchmark_runs/test_retrospective_150/summary_arm_A.csv), [`summary_arm_B.csv`](results/benchmark_runs/test_retrospective_150/summary_arm_B.csv), [`summary_arm_C.csv`](results/benchmark_runs/test_retrospective_150/summary_arm_C.csv)
   - Per-case tables: [`per_case_arm_A.csv`](results/benchmark_runs/test_retrospective_150/per_case_arm_A.csv), [`per_case_arm_B.csv`](results/benchmark_runs/test_retrospective_150/per_case_arm_B.csv), [`per_case_arm_C.csv`](results/benchmark_runs/test_retrospective_150/per_case_arm_C.csv) (`750` rows each = `150 cases × 5 models`)
   - Historical RASNet recipe diagnostic (`4-pass TTA`, $\tau=0.60$, `top-2 CC`): [`diagnostic_rasnet_historical_recipe_tau060_tta_top2.csv`](results/benchmark_runs/test_retrospective_150/diagnostic_rasnet_historical_recipe_tau060_tta_top2.csv)
   - Historical difference audit: [`diff_vs_historical_summary.json`](results/benchmark_runs/test_retrospective_150/diff_vs_historical_summary.json)
3. **[`results/benchmark_runs/smoke_val_all5/`](results/benchmark_runs/smoke_val_all5/)** and **[`results/benchmark_runs/smoke_verified_v2/`](results/benchmark_runs/smoke_verified_v2/)** (`N = 3` validation smoke cases `14, 45, 47` with exported NIfTI predictions):
   - Confirms exact numerical parity of the unified `benchmark/` source tree (`source_tree_sha256: 179d8188047ad596e18b7c32fb66ff4c4997c1fc6407eb59e616e4aa58520af1`) and `nnU-Net V2` batch vs. single-case predictor parity within `7.75e-06` Dice.

---

## 4. Verification Audit Package (`results/audits/2026-11-10_verification_audit/`)

- [`run_state.md`](results/audits/2026-11-10_verification_audit/run_state.md): Complete provenance reconciliation of all benchmark runs and Step 1–2 verification results.
- [`environment.json`](results/audits/2026-11-10_verification_audit/environment.json): Captured workstation hardware and Python package versions.
- [`checkpoint_verification.csv`](results/audits/2026-11-10_verification_audit/checkpoint_verification.csv): Checkpoint SHA-256, epoch, and training-log vs. native-grid validation reconciliation.
- [`source_inventory.csv`](results/audits/2026-11-10_verification_audit/source_inventory.csv) & [`source_inventory_pre_step1.csv`](results/audits/2026-11-10_verification_audit/source_inventory_pre_step1.csv): Pre- and post-Step-1 SHA-256 digests of all code, config, and test files.
- [`artifact_inventory.csv`](results/audits/2026-11-10_verification_audit/artifact_inventory.csv): SHA-256 inventory of all 105 benchmark result files.
- [`issues.csv`](results/audits/2026-11-10_verification_audit/issues.csv): Dated issue ledger mapping every audit finding to its fix and verification test.
- **Augmented Statistical Tables (with Paired Difference Bootstrap 95% CIs & Median/IQR)**:
  - Primary family ($K=3$): [`paired_stats_primary_arm_A_with_ci.csv`](results/audits/2026-11-10_verification_audit/paired_stats_primary_arm_A_with_ci.csv), [`paired_stats_primary_arm_B_with_ci.csv`](results/audits/2026-11-10_verification_audit/paired_stats_primary_arm_B_with_ci.csv), [`paired_stats_primary_arm_C_with_ci.csv`](results/audits/2026-11-10_verification_audit/paired_stats_primary_arm_C_with_ci.csv)
  - Full secondary family ($K=32$): [`paired_stats_full32_arm_A_with_ci.csv`](results/audits/2026-11-10_verification_audit/paired_stats_full32_arm_A_with_ci.csv), [`paired_stats_full32_arm_B_with_ci.csv`](results/audits/2026-11-10_verification_audit/paired_stats_full32_arm_B_with_ci.csv), [`paired_stats_full32_arm_C_with_ci.csv`](results/audits/2026-11-10_verification_audit/paired_stats_full32_arm_C_with_ci.csv)
  - Validation split ($N=94$): [`paired_stats_primary_val_arm_A_with_ci.csv`](results/audits/2026-11-10_verification_audit/paired_stats_primary_val_arm_A_with_ci.csv), [`paired_stats_full32_val_arm_A_with_ci.csv`](results/audits/2026-11-10_verification_audit/paired_stats_full32_val_arm_A_with_ci.csv)
- **Visual Alignment Overlays**:
  - [`smoke_overlays/smoke_case_14_armA_5models_overlay.png`](results/audits/2026-11-10_verification_audit/smoke_overlays/smoke_case_14_armA_5models_overlay.png)
  - [`smoke_overlays/smoke_case_45_armA_5models_overlay.png`](results/audits/2026-11-10_verification_audit/smoke_overlays/smoke_case_45_armA_5models_overlay.png)
  - [`smoke_overlays/smoke_case_47_armA_5models_overlay.png`](results/audits/2026-11-10_verification_audit/smoke_overlays/smoke_case_47_armA_5models_overlay.png)

---

## 5. Publication-Ready Thesis Tables, Figures & Narrative (`results/thesis_outputs/`)

- **Tables (`.csv` and `.md`)**:
  - [`Table1_validation_94_and_thresholds.csv`](results/thesis_outputs/Table1_validation_94_and_thresholds.csv) / [`.md`](results/thesis_outputs/Table1_validation_94_and_thresholds.md)
  - [`Table2_test_150_arms_A_B_C_summary.csv`](results/thesis_outputs/Table2_test_150_arms_A_B_C_summary.csv) / [`.md`](results/thesis_outputs/Table2_test_150_arms_A_B_C_summary.md)
  - [`Table3_paired_wilcoxon_primary_and_secondary.csv`](results/thesis_outputs/Table3_paired_wilcoxon_primary_and_secondary.csv) / [`.md`](results/thesis_outputs/Table3_paired_wilcoxon_primary_and_secondary.md)
  - [`Table4_computational_efficiency_rtx4080super.csv`](results/thesis_outputs/Table4_computational_efficiency_rtx4080super.csv) / [`.md`](results/thesis_outputs/Table4_computational_efficiency_rtx4080super.md)
- **Figures (`.png` and `.svg`)**:
  - [`fig1_multi_arm_metric_distributions.png`](results/thesis_outputs/fig1_multi_arm_metric_distributions.png) / [`.svg`](results/thesis_outputs/fig1_multi_arm_metric_distributions.svg)
  - [`fig2_validation_threshold_sensitivity_n94.png`](results/thesis_outputs/fig2_validation_threshold_sensitivity_n94.png) / [`.svg`](results/thesis_outputs/fig2_validation_threshold_sensitivity_n94.svg)
  - [`fig3_training_and_val_n94_convergence_200ep.png`](results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.png) / [`.svg`](results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.svg)
  - [`fig4_precision_recall_boundary_tradeoff.png`](results/thesis_outputs/fig4_precision_recall_boundary_tradeoff.png) / [`.svg`](results/thesis_outputs/fig4_precision_recall_boundary_tradeoff.svg)
- **Thesis Manuscript Narrative**:
  - [`THESIS_METHODS_RESULTS_AND_LIMITATIONS.md`](results/thesis_outputs/THESIS_METHODS_RESULTS_AND_LIMITATIONS.md)

---

## 6. Archive of Historical & Outside-Scope Materials

- **Archive Root**: [`_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/`](_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/)
- **Manifest & SHA-256 Index (`306` files)**: [`ARCHIVE_INDEX.csv`](_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/ARCHIVE_INDEX.csv)

---

## 7. Phase H: Fixed-Weight RASNet Optimization Results (`results/rasnet_fixed_weight_optimization/`)

- **Optimization Report**: [`PHASE_H_OPTIMIZATION_REPORT.md`](results/rasnet_fixed_weight_optimization/PHASE_H_OPTIMIZATION_REPORT.md)
- **Run Manifest & Parity Digest**: [`manifest_phaseH.json`](results/rasnet_fixed_weight_optimization/manifest_phaseH.json)
- **Validation (`N = 94`) Summaries**:
  - [`val_94_all_candidates_summary.csv`](results/rasnet_fixed_weight_optimization/val_94_all_candidates_summary.csv) (`152` candidate recipes)
  - [`val_94_selected_summary.csv`](results/rasnet_fixed_weight_optimization/val_94_selected_summary.csv) (`11` selected recipes backfilled with complete `HD95`/`ASD`/`clDice`)
- **Retrospective Test (`N = 150`) & Paired Statistics**:
  - [`test_150_locked_winners_summary.csv`](results/rasnet_fixed_weight_optimization/test_150_locked_winners_summary.csv)
  - [`paired_stats_test_150_winner_vs_arms_ABC.csv`](results/rasnet_fixed_weight_optimization/paired_stats_test_150_winner_vs_arms_ABC.csv)
- **Pareto Figure**: [`fig_phaseH_recall_precision_pareto.png`](results/rasnet_fixed_weight_optimization/fig_phaseH_recall_precision_pareto.png) / [`.svg`](results/rasnet_fixed_weight_optimization/fig_phaseH_recall_precision_pareto.svg)

