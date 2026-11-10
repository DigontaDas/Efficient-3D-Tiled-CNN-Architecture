# Phase A–C Run-State & Provenance Audit (`run_state.md`)

**Audit Date**: `2026-11-10`  
**Workstation**: `NVIDIA GeForce RTX 4080 SUPER (16,376 MiB VRAM)`, Windows 11  
**Python Environment**: `C:\Thesis_RASNET_env\bench\Scripts\python.exe` (`Python 3.12.10`, `torch 2.14.1+cu126`, `monai 1.6.1`, `nnunetv2 2.8.1`, `SimpleITK 2.5.7`)

---

## 1. Existing Benchmark Runs & Source Code Provenance

| Run ID | Split | Cases ($N$) | Arms | Manifest Timestamp (UTC) | Manifest `git.commit` | Actual Source Tree Executed | Provenance Status |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| `smoke_val_all5` | `val` (smoke) | `3` (`14, 45, 47`) | `A, B` | `20261108T054032Z` | `8eb28bb93222fd90...` (`dirty: true`) | Commit `ca692b2` working tree (pre-commit) | **VERIFIED** (`smoke_val_all5/predictions/arm_{A,B}/*` on disk) |
| `val_full_94` | `val` | `94` | `A` + threshold sweep | `20261108T071452Z` | `8eb28bb93222fd90...` (`dirty: true`) | Commit `ca692b2` working tree (pre-commit) | **VERIFIED** (Froze `results/thresholds.json`, `sha256: 4f66ac1d415c5ca9bc09ca0e900e9f8f5d106b3620a20d5916f472d1ba0d3acb`) |
| `test_retrospective_150` | `test` | `150` (`851–1000`) | `A, B, C` + historical recipe | `20261108T110359Z` | `8eb28bb93222fd90...` (`dirty: true`) | Commit `ca692b2` working tree (pre-commit) | **VERIFIED** (`150/150` cases across all 5 models and Arms A, B, C) |
| `smoke_verified_v2` | `val` (smoke) | `3` (`14, 45, 47`) | `A` | `20261110T142935Z` | `d81485d50043c543...` (`source_tree_sha256: 179d8188...`) | Post-Step-1 unified `benchmark/` source tree | **VERIFIED** (Confirms MONAI adapter unification & `nnU-Net` batch iterator parity) |

### Why `val_full_94` and `test_retrospective_150` Recorded `8eb28bb9` (`dirty: true`)
- Both runs executed on `2026-11-08` while `HEAD` was `8eb28bb93222fd906a5cd0765b712252a936b34a` and the working tree contained the uncommitted `benchmark/` implementation that was subsequently committed verbatim as `ca692b20013d0efba8e746baafcaee1ab46a8742` (`source_inventory_pre_step1.csv`).
- Step 1 added `source_tree_sha256` to `git_info()` in `benchmark/run_benchmark.py` so every future manifest records the exact SHA-256 digest of all `benchmark/**/*.py` files independently of git commit state.

---

## 2. Step 1 Robustness Verification (`smoke_verified_v2` vs. `val_full_94` and `smoke_val_all5`)

To verify that Step 1's code unifications did not alter any numerical benchmark behavior:
1. **MONAI Single/TTA Unification (`infer_monai_both_passes` in `benchmark/adapters/monai_models.py`)**:
   - Across smoke cases `14, 45, 47`, `RASNet`, `SegResNet`, and `V-Net` achieved **exact `0.00e+00` maximum absolute difference** between `smoke_verified_v2` and `val_full_94` on all 8 metrics (`dice`, `iou`, `precision`, `recall`, `hd95`, `asd`, `cldice`, `centerline_recall`).
   - `3D U-Net` achieved a maximum absolute Dice difference of `3.67e-05` (`0.000037`, well within the pre-declared `1e-4` tolerance).
2. **`nnU-Net V2` Batch Iterator vs. Single-Case Native Predictor Parity**:
   - Removing the `len(case_ids) <= 3` branch in `_run_nnunet_batch_probabilities` forced `smoke_verified_v2` (`3` cases) to execute the exact same `data_iterator_from_raw_npy_data` batch path used by `val_full_94` (`94` cases) and `test_retrospective_150` (`150` cases).
   - Comparing `smoke_verified_v2` (batch iterator) against `smoke_val_all5` (single-case `infer_nnunet_probability`) on cases `14, 45, 47`:
     - Maximum absolute Dice difference: **`7.75e-06`** (`0.0000078`, `< 1e-4` tolerance)
     - Maximum absolute HD95 difference: **`5.49e-03 mm`** (`0.0055 mm`)
   - Comparing `smoke_verified_v2` against `val_full_94` on cases `14, 45, 47`:
     - Maximum absolute Dice difference: **`5.63e-06`** (`0.0000056`), HD95 difference: **`0.00e+00 mm`**.
3. **Physical Header & Visual Slice Overlay Verification**:
   - All 15 exported NIfTI predictions in `smoke_verified_v2/predictions/arm_A/{rasnet,nnunet,segresnet,vnet,3dunet}/{14,45,47}.nii.gz` have `size_match = True` and `0.0` origin/spacing/direction error (`smoke_header_and_alignment_checks.csv`).
   - Multi-planar (axial, coronal, sagittal) 2D slice overlays with ground-truth (lime) and predicted (red) contours are saved in:
     - `smoke_overlays/smoke_case_14_armA_5models_overlay.png`
     - `smoke_overlays/smoke_case_45_armA_5models_overlay.png`
     - `smoke_overlays/smoke_case_47_armA_5models_overlay.png`

---

## 3. Audit Deliverables Inventory (`results/audits/2026-11-10_verification_audit/`)

- `environment.json`: Exact Python, CUDA, GPU, and library versions.
- `checkpoint_verification.csv`: All 5 best checkpoints with SHA-256 digests, training epochs, training log validation metrics, and native-grid Arm A/C validation Dice.
- `source_inventory_pre_step1.csv` & `source_inventory.csv`: Pre- and post-Step-1 SHA-256 digests of all benchmark source, config, and test files.
- `artifact_inventory.csv`: 105 benchmark result files with SHA-256 digests and provenance status.
- `issues.csv`: 11-row dated issue ledger mapping every audit finding to its resolution and verification test.
- `paired_stats_primary_arm_{A,B,C}_with_ci.csv` & `paired_stats_full32_arm_{A,B,C}_with_ci.csv`: Complete $N=150$ paired statistical comparisons including `diff_ci95_lo`, `diff_ci95_hi`, `median_difference`, `q25_difference`, `q75_difference`, and `iqr_difference`.
- `paired_stats_primary_val_arm_A_with_ci.csv` & `paired_stats_full32_val_arm_A_with_ci.csv`: Complete $N=94$ validation split paired statistical comparisons.
- `smoke_v2_vs_v1_and_val94_comparison.csv` & `smoke_header_and_alignment_checks.csv`: Case-by-case numerical and header parity verification.
- `doc_snapshots_pre_phaseE/`: Pre-edit snapshots of all 12 active documentation files prior to Step 3 updates.

