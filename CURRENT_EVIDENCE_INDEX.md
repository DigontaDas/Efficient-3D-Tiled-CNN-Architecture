# Canonical Evidence Index (`CURRENT_EVIDENCE_INDEX.md`)

**Last Updated**: `2026-11-17`  
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
| **nnU-Net V2** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/nnunet_best.pt` | `9accddf283b1c480e386f30f426b9200ca8d4aa1ecd70d5b5af0b8aad6058711` | `199` (`_best_ema=0.8162`; raw `Pseudo dice` peak ep `189` = `0.8376`) | `0.7761 ± 0.0567` | `0.8075` ($\tau^*=0.35$) |
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
- [`checkpoint_verification.csv`](results/audits/2026-11-10_verification_audit/checkpoint_verification.csv): Checkpoint SHA-256, epoch, and training-log vs. native-grid validation reconciliation (preserved historical CSV; see [`HISTORICAL_ERRATA.md`](results/v1_reporting_correction_20261117_105000Z/HISTORICAL_ERRATA.md) for row-6 nnU-Net epoch/EMA errata).
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
  - [`fig3_training_and_val_n94_convergence_200ep.png`](results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.png) / [`.svg`](results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.svg) (preserved historical figure; see [`HISTORICAL_ERRATA.md`](results/v1_reporting_correction_20261117_105000Z/HISTORICAL_ERRATA.md) and corrected replacement [`fig3_corrected_training_and_val_convergence_200ep.png`](results/v1_reporting_correction_20261117_105000Z/fig3_corrected_training_and_val_convergence_200ep.png) for `DiceFocalLoss` and raw `Pseudo dice` vs EMA fixes)
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

---

## 8. Supplement: Option A (2026-11-15) (`results/optionA_v1_symmetric_sw050/`)

- **Output Folder**: [`results/optionA_v1_symmetric_sw050/`](results/optionA_v1_symmetric_sw050/) (symmetric 50% overlap Gaussian sliding window, `--sw-batch-size 8`, 1-pass and 8-pass mirroring TTA, $\tau \in [0.20, 0.70]$, 26-connected top-2 CC, evaluated across the 4 sliding-window v1 checkpoints).
- **Pre-Test Lock (committed at `558e7eb` before Test-150)**:
  - [`locked_recipes.json`](results/optionA_v1_symmetric_sw050/locked_recipes.json) — SHA-256: `31cb484573b4d941c91f415d7ea4cb3757dc67d205479e96adeeea159ede557c` (`1pass / tta8`: `rasnet=0.40/0.40`, `segresnet=0.40/0.45`, `vnet=0.40/0.45`, `nnunet=0.35/0.20`)
  - [`locked_recipes.sha256`](results/optionA_v1_symmetric_sw050/locked_recipes.sha256) — SHA-256: `8aff03e1789dceaeb89caa9b58a9dd6d72f8872da77d56ac55c248d300f5181c`
- **Stage Manifests (all with `git.dirty_code: false`)**:
  - [`manifest_parity_20261114T195345.json`](results/optionA_v1_symmetric_sw050/manifest_parity_20261114T195345.json) — SHA-256: `1e5780aed77d5ca4c59751bd8c96c8f6c9eac1579134df56c9a5634a73a5e84d`
  - [`manifest_val_20261114T234219.json`](results/optionA_v1_symmetric_sw050/manifest_val_20261114T234219.json) — SHA-256: `da62ee399fceebeaac440a3d4b53126bd10ad54ff1c843ea9756f400d1b1356f`
  - [`manifest_lock_20261114T234245.json`](results/optionA_v1_symmetric_sw050/manifest_lock_20261114T234245.json) — SHA-256: `a8a58709ff8d46a7bfe61f3f2839b70741a6c9d91866b3d6cc572997091bd97c`
  - [`manifest_test_20261115T064137.json`](results/optionA_v1_symmetric_sw050/manifest_test_20261115T064137.json) — SHA-256: `277f8f3a3d6e318d3ff94d4ff3eb581aeaea5751ede66bd2bd4e720d1bbedcb2`
  - [`manifest_timing_20261115T071538.json`](results/optionA_v1_symmetric_sw050/manifest_timing_20261115T071538.json) — SHA-256: `fcf074a18518027bb56c4d9122d094528d0bd17281abfcb336f91d0ee52d289e`
  - [`manifest_stats_20261115T071556.json`](results/optionA_v1_symmetric_sw050/manifest_stats_20261115T071556.json) — SHA-256: `fd51824e92eb2a998ec12d88904296f71c9f343bf79d37ccc9149a38c9657797`
  - [`manifest_figures_20261115T071603.json`](results/optionA_v1_symmetric_sw050/manifest_figures_20261115T071603.json) — SHA-256: `bece4281d8d9b9b9168d8436bfbef9f64b37ace0b08ade7d5be4d742eb920a5c`
- **Report & Tabular Artifacts**:
  - [`OPTION_A_RESULTS.md`](results/optionA_v1_symmetric_sw050/OPTION_A_RESULTS.md) — SHA-256: `38abd4e354a0e1422d6764221775757fafaca9660bd3c1ff866924f8acd98430`
  - [`parity_check.csv`](results/optionA_v1_symmetric_sw050/parity_check.csv) — SHA-256: `dae50f081c4e82cd74355ebe37a7357bec574fd48a36bcda96b62e56ef46f274` (`max |dDice| = 9.08e-06 <= 1e-04`)
  - [`val_grid_scores.csv`](results/optionA_v1_symmetric_sw050/val_grid_scores.csv) — SHA-256: `4f8f6a582eed9c0082bda782784aeb91f0ab9c981516e184f41e306afd2c3084` (`8,272` rows = `94 cases × 4 models × 22 grid entries`)
  - [`test_per_case.csv`](results/optionA_v1_symmetric_sw050/test_per_case.csv) — SHA-256: `2b203f0999b891dd8bfb6691c132d5595c0b220b321ac80e74b83ea9797be958` (`2,400` rows = `150 cases × 4 models × 4 recipes`; `0` non-`ok` rows)
  - [`summary_test.csv`](results/optionA_v1_symmetric_sw050/summary_test.csv) — SHA-256: `e4de75ca2daa1b343df6410b7296c4b21c6c38de1a25b86d9c6ec7fa60c147c2`
  - [`paired_stats_families.csv`](results/optionA_v1_symmetric_sw050/paired_stats_families.csv) — SHA-256: `262ecdd1c146bb18a82175c77f5e48552fb8c3be5a507b02d4206ec3fed1fc9e` (`68` rows; `n_excluded_nan = 0` for all rows)
  - [`noninferiority_rasnet_vs_nnunet_dice.csv`](results/optionA_v1_symmetric_sw050/noninferiority_rasnet_vs_nnunet_dice.csv) — SHA-256: `326fe2261f537777b60fb32d27f306d25515f44282feae40e64143bb4cffc28d`
  - [`timing_val.csv`](results/optionA_v1_symmetric_sw050/timing_val.csv) — SHA-256: `2deb181dc65f874c76a243a0f68d8cab15f59930df82ad4e915f39a89f40dec0` (`80` rows = `10 val cases × 4 models × 2 modes` after `1` warm-up on idle RTX 4080 SUPER)
  - [`timing_summary.csv`](results/optionA_v1_symmetric_sw050/timing_summary.csv) — SHA-256: `db28c133e187537403cf0ebd812f0d39186d18b9b018ab9954e83d4a4cababed`
- **Figures**:
  - [`fig_optionA_dice_vs_params.png`](results/optionA_v1_symmetric_sw050/fig_optionA_dice_vs_params.png) (`6a3f6c91de07605c3af8b192acb47b181c688627626b25328d4b3ae5e9ee7347`) / [`.pdf`](results/optionA_v1_symmetric_sw050/fig_optionA_dice_vs_params.pdf) (`c7e4f6c2dc7012d216e1e975cd9a2561edb887be76955b271bdf974aff995655`)
  - [`fig_optionA_dice_distributions.png`](results/optionA_v1_symmetric_sw050/fig_optionA_dice_distributions.png) (`a8578d5c67509055bc8808c9e2609d2592b8abbf62dd570a4f3c4571ed7f7c64`) / [`.pdf`](results/optionA_v1_symmetric_sw050/fig_optionA_dice_distributions.pdf) (`8905f8af0e0c7fadc9ca9bec9d47b754ed7e44240da281889ad0634562ec650c`)
  - [`fig_optionA_paired_diff_rasnet_vs_nnunet.png`](results/optionA_v1_symmetric_sw050/fig_optionA_paired_diff_rasnet_vs_nnunet.png) (`fe501e45af5710932691d2ac6f00d7ee5d80a82b0d51fc5af831554587af6fd7`) / [`.pdf`](results/optionA_v1_symmetric_sw050/fig_optionA_paired_diff_rasnet_vs_nnunet.pdf) (`b99174e9da7417d4b21ea9d7592e937870cc28c5995ad5fc169bacd75bf26cba`)

---

## 9. V1 Epoch-200 Final Checkpoint Recovery & Val94 Endpoint Comparison (`2026-11-17`)

- **Recovered Final Epoch-200 Checkpoints (`recovered_checkpoints/v1_epoch200_20261117_041500Z/`)**:
  - [`PROVENANCE_REPORT.md`](recovered_checkpoints/v1_epoch200_20261117_041500Z/PROVENANCE_REPORT.md), [`provenance_verification.json`](recovered_checkpoints/v1_epoch200_20261117_041500Z/provenance_verification.json) (`ab792459c200338986c15f656457575c837a0a67372424d3c65f4cffbdfe8224`), [`recover_and_verify.py`](recovered_checkpoints/v1_epoch200_20261117_041500Z/recover_and_verify.py)
  - `rasnet_last.pth` (`71c9e5ef4a02849686c99ccb2d1544ff78c78187e1c0d53e7b22884c708c1b95`, `114` keys, `4,706,772` params; no embedded epoch metadata — corroborated by `train_rasnet_200ep.py`, `rasnet_200ep_log.csv`, `CosineAnnealingLR`, and archive timestamps)
  - `nnunet_last.pt` (`2105bd7238fff7485f1a8d72622f977f635f441e6d44863920c85709b02ffec2`, `292` keys, `30,785,994` unique params, embedded `current_epoch=200`) and isolated native model folder `nnunet_native_model/`
- **2-Case Val94 Smoke Run (`results/v1_epoch200_smoke_20261117_041500Z/`)**:
  - [`manifest.json`](results/v1_epoch200_smoke_20261117_041500Z/manifest.json), [`summary_epoch200.csv`](results/v1_epoch200_smoke_20261117_041500Z/summary_epoch200.csv), [`geometry_verification.csv`](results/v1_epoch200_smoke_20261117_041500Z/geometry_verification.csv) (`N=2`, cases `14, 45`)
- **Full Val94 Endpoint Comparison (`results/v1_epoch200_comparison_20261117_041500Z/`)**:
  - [`manifest.json`](results/v1_epoch200_comparison_20261117_041500Z/manifest.json) (note: `timing_seconds.rasnet = 0.0` reflects `--resume` skipping 94 pre-completed cases, not inference latency), [`geometry_verification.csv`](results/v1_epoch200_comparison_20261117_041500Z/geometry_verification.csv) (`188/188` verified)
  - [`per_case_metrics_rasnet_epoch200.csv`](results/v1_epoch200_comparison_20261117_041500Z/per_case_metrics_rasnet_epoch200.csv), [`per_case_metrics_nnunet_epoch200.csv`](results/v1_epoch200_comparison_20261117_041500Z/per_case_metrics_nnunet_epoch200.csv), [`per_case_metrics_combined.csv`](results/v1_epoch200_comparison_20261117_041500Z/per_case_metrics_combined.csv), [`paired_case_level_differences_epoch200.csv`](results/v1_epoch200_comparison_20261117_041500Z/paired_case_level_differences_epoch200.csv)
  - [`summary_epoch200.csv`](results/v1_epoch200_comparison_20261117_041500Z/summary_epoch200.csv), [`paired_stats_epoch200.csv`](results/v1_epoch200_comparison_20261117_041500Z/paired_stats_epoch200.csv), [`paired_stats_epoch200.json`](results/v1_epoch200_comparison_20261117_041500Z/paired_stats_epoch200.json), [`table_epoch200_vs_existing_best_references.md`](results/v1_epoch200_comparison_20261117_041500Z/table_epoch200_vs_existing_best_references.md)
  - [`ENDPOINT_EVALUATION_REPORT.md`](results/v1_epoch200_comparison_20261117_041500Z/ENDPOINT_EVALUATION_REPORT.md) and [`THESIS_CURVE_AND_ENDPOINT_CORRECTION.md`](results/v1_epoch200_comparison_20261117_041500Z/THESIS_CURVE_AND_ENDPOINT_CORRECTION.md) (preserved; superseded for narrative/multiplicity discipline by §10)

---

## 10. V1 Reporting Correction, Provenance & Defense Package (`results/v1_reporting_correction_20261117_105000Z/`)

- **Preservation & Baseline Digests**: [`pre_edit_backups/`](results/v1_reporting_correction_20261117_105000Z/pre_edit_backups/), [`pre_edit_hashes.json`](results/v1_reporting_correction_20261117_105000Z/pre_edit_hashes.json), [`protected_baseline_hashes.json`](results/v1_reporting_correction_20261117_105000Z/protected_baseline_hashes.json) (`548` protected files)
- **Errata & Audit Reports**: [`HISTORICAL_ERRATA.md`](results/v1_reporting_correction_20261117_105000Z/HISTORICAL_ERRATA.md) (documents protected `fig3` and `checkpoint_verification.csv` errata without modifying them), [`V1_REPORTING_CORRECTION_REPORT.md`](results/v1_reporting_correction_20261117_105000Z/V1_REPORTING_CORRECTION_REPORT.md), [`CLAIM_TO_EVIDENCE_MATRIX.csv`](results/v1_reporting_correction_20261117_105000Z/CLAIM_TO_EVIDENCE_MATRIX.csv)
- **Thesis & Defense Deliverables**: [`THESIS_REPLACEMENT_TEXT.md`](results/v1_reporting_correction_20261117_105000Z/THESIS_REPLACEMENT_TEXT.md), [`DEFENSE_QA_V1.md`](results/v1_reporting_correction_20261117_105000Z/DEFENSE_QA_V1.md), [`DEFENSE_SLIDE_OUTLINE.md`](results/v1_reporting_correction_20261117_105000Z/DEFENSE_SLIDE_OUTLINE.md)
- **Corrected Tables & Figures**:
  - [`table_epoch200_val94_summary_and_paired_corrected.md`](results/v1_reporting_correction_20261117_105000Z/table_epoch200_val94_summary_and_paired_corrected.md) / [`.csv`](results/v1_reporting_correction_20261117_105000Z/table_epoch200_val94_summary_and_paired_corrected.csv)
  - [`table_epoch200_val94_posthoc_holm8_exploratory.md`](results/v1_reporting_correction_20261117_105000Z/table_epoch200_val94_posthoc_holm8_exploratory.md) / [`.csv`](results/v1_reporting_correction_20261117_105000Z/table_epoch200_val94_posthoc_holm8_exploratory.csv)
  - [`table_v1_four_regimes_comparison.md`](results/v1_reporting_correction_20261117_105000Z/table_v1_four_regimes_comparison.md) / [`.csv`](results/v1_reporting_correction_20261117_105000Z/table_v1_four_regimes_comparison.csv)
  - [`table_v1_efficiency_and_timing_verified.md`](results/v1_reporting_correction_20261117_105000Z/table_v1_efficiency_and_timing_verified.md) / [`.csv`](results/v1_reporting_correction_20261117_105000Z/table_v1_efficiency_and_timing_verified.csv)
  - [`table_checkpoint_provenance_verified.md`](results/v1_reporting_correction_20261117_105000Z/table_checkpoint_provenance_verified.md) / [`.csv`](results/v1_reporting_correction_20261117_105000Z/table_checkpoint_provenance_verified.csv)
  - [`fig3_corrected_training_and_val_convergence_200ep.png`](results/v1_reporting_correction_20261117_105000Z/fig3_corrected_training_and_val_convergence_200ep.png) / [`.svg`](results/v1_reporting_correction_20261117_105000Z/fig3_corrected_training_and_val_convergence_200ep.svg)
  - [`fig_v1_epoch200_val94_endpoint_corrected.png`](results/v1_reporting_correction_20261117_105000Z/fig_v1_epoch200_val94_endpoint_corrected.png) / [`.svg`](results/v1_reporting_correction_20261117_105000Z/fig_v1_epoch200_val94_endpoint_corrected.svg)

