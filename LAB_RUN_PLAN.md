# Lab Run Plan — Executed & Verified (5-Model ImageCAS Retrospective Benchmark)

> **Status**: **COMPLETED & VERIFIED** on NVIDIA GeForce RTX 4080 SUPER (`16,376 MiB` VRAM).
> **Methodological Scope**: **Retrospective Reassessment** on the ImageCAS secondary dataset (`C:\Thesis_RASNET\secondary-imagcas-dataset`). Because prior historical threshold (`tau = 0.60`) and post-processing sweeps inspected the 150 test cases (`851–1000`) and 66-case internal cohort, neither cohort is presented as an untouched confirmatory test set.

---

## 1. Environment & Preflight Verification Commands

```powershell
# 1. Verify Python, CUDA, GPU, and core package versions
C:\Thesis_RASNET_env\bench\Scripts\python.exe --version
C:\Thesis_RASNET_env\bench\Scripts\python.exe -c "import torch, monai, SimpleITK, scipy, skimage, nibabel, nnunetv2, cc3d; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'monai:', monai.__version__, 'SimpleITK:', SimpleITK.Version_VersionString(), 'scipy:', scipy.__version__, 'skimage:', skimage.__version__, 'nibabel:', nibabel.__version__, 'nnunetv2:', nnunetv2.__version__)"

# 2. Verify SHA-256 digests of all 5 BEST checkpoints and nnU-Net plans
Get-FileHash -Algorithm SHA256 -LiteralPath `
  "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\checkpoints\rasnet_best.pth", `
  "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\checkpoints\segresnet_best.pt", `
  "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\checkpoints\vnet_best.pt", `
  "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\checkpoints\3dunet_best.pth", `
  "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\checkpoints\nnunet_best.pt", `
  "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_results\Dataset501_CoronarySeg\nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres\plans.json", `
  "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_results\Dataset501_CoronarySeg\nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres\dataset.json", `
  "C:\Thesis_RASNET\Phase3_Local_Integration\splits_final.json"
```

### Pass / Fail Criteria (Stage 1)
- **PASS**:
  - `Python 3.12.10`, `torch 2.14.1+cu126`, `CUDA: True (NVIDIA GeForce RTX 4080 SUPER)`, `monai 1.6.1`, `SimpleITK 2.5.7`, `scipy 1.18.1`, `scikit-image 0.26.0`, `nibabel 5.4.2`, `nnunetv2 2.8.1`.
  - Disjoint splits verified in `Phase3_Local_Integration/splits_final.json` (`sha256: efe3f06cfe9c7dff646224378ad4d0ea5ce50fb4bf920807b17054af4bdf953c`): `690 train`, `94 val`, `150 test` (`851–1000`), `0` overlaps.
  - All 1,000 ImageCAS image/label pairs verified on disk with `0` physical geometry (`size`, `origin`, `spacing`, `direction`) mismatches (`atol=1e-4`).
  - All 5 BEST checkpoints verified (`CHECKPOINTS.md`):
    - `rasnet_best.pth`: `8d4c25cd12d07c40423062b27b1d36836e5fd6cecb30c20c34055eeac976d1b8`
    - `segresnet_best.pt`: `b8ea61d95106a0bbe15e1ce9f9e5af1a38d9508e59d0639d32bc883d93c6e109`
    - `vnet_best.pt`: `a89d71bd41c3e773c68e3f4b5e27c5a067f959154aa049eead76f1642c5c1063`
    - `3dunet_best.pth`: `d8a8f5ec95b5f34e2afe0b807ac49bb36d136f9d864238c27a763a29a800a60f`
    - `nnunet_best.pt` / `fold_0/checkpoint_best.pth`: `9accddf283b1c480e386f30f426b9200ca8d4aa1ecd70d5b5af0b8aad6058711`
    - `plans.json`: `c67af4aaeb24f5a72b58c83e351a9c62cf026f8baf1bbb727e3022491d5733d8`
    - `dataset.json`: `d81f40253d7c2d88b9d1cdbd4bd7319a7a0206fce5b2350287317bd34f9c7389`

---

## 2. Stage 4a — Synthetic Test Suite (Including CPU Synthetic End-to-End Dry-Run)

```powershell
C:\Thesis_RASNET_env\bench\Scripts\python.exe -B -m pytest -v -p no:cacheprovider tests
```

### Pass / Fail Criteria (Stage 4a)
- **PASS**: `31 passed` (`0 failed`). Covers brute-force `cdist` and anisotropic surface distances (`surface-v1`), asymmetric non-zero-origin/flipped-direction 3D phantom round-trip alignment, empty-mask/skeleton status handling, paired Wilcoxon + Holm correction + paired difference bootstrap 95% CIs (`diff_ci95_lo`, `diff_ci95_hi`) + median/IQR + NaN-exclusion tracking (`allow_nan_filtering`), connected-component volume logging, validation-only threshold selection & tie-breaking, real MONAI architectures (`RASNet`, `SegResNet`, `VNet`, `3DUNet`) + explicit `_SyntheticNNUNetPredictor` test double, coordinate inversion, `3D U-Net` `128³` resizing, `V-Net` `%16` padding, and fail-closed checks on non-best checkpoints, split leakage, and stale output directories.

---

## 3. Stage 4b — Three-Case Real-Data Smoke Run (`smoke_val_all5`)

```powershell
$env:nnUNet_raw = "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_preprocessed"
$env:nnUNet_results = "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_results"

C:\Thesis_RASNET_env\bench\Scripts\python.exe -m benchmark.run_benchmark `
  --config config.lab.yaml `
  --split val `
  --arm A,B `
  --models all `
  --smoke `
  --run-id smoke_val_all5
```

### Expected Outputs & Pass / Fail Criteria (Stage 4b)
- **Output Directory**: `results/benchmark_runs/smoke_val_all5/`
- **PASS**:
  - All 30 NIfTI masks (`predictions/arm_{A,B}/{rasnet,segresnet,vnet,3dunet,nnunet}/{14,45,47}.nii.gz`) written and verified to match the source image and label headers (`size`, `origin`, `spacing`, `direction` within `1e-4`).
  - `manifest.json` records `"smoke": true`, all 5 checkpoint SHA-256 digests, and `nnunet_metadata`.
  - Smoke run does **not** write or update `results/thresholds.json`.

---

## 4. Stage 4c — Full 94-Case Validation Split & Freezing `thresholds.json` (`val_full_94`)

```powershell
$env:nnUNet_raw = "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_preprocessed"
$env:nnUNet_results = "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_results"

C:\Thesis_RASNET_env\bench\Scripts\python.exe -m benchmark.run_benchmark `
  --config config.lab.yaml `
  --split val `
  --arm A `
  --models all `
  --run-id val_full_94
```

### Expected Outputs & Pass / Fail Criteria (Stage 4c)
- **Output Directory**: `results/benchmark_runs/val_full_94/` and canonical `results/thresholds.json` (`sha256: 4f66ac1d415c5ca9bc09ca0e900e9f8f5d106b3620a20d5916f472d1ba0d3acb`).
- **PASS**:
  - All 94 validation cases evaluated across all 5 models (`n_cases = 94`, `n_ok = 94`).
  - Validation threshold grid `0.30..0.70` (step `0.05`) with 26-connectivity top-2 CC pruning evaluated and frozen in `results/thresholds.json`:
    - `rasnet`: `0.30` (mean val Dice `0.789746`)
    - `segresnet`: `0.35` (mean val Dice `0.790101`)
    - `vnet`: `0.30` (mean val Dice `0.770226`)
    - `3dunet`: `0.30` (mean val Dice `0.604719`)
    - `nnunet`: `0.35` (mean val Dice `0.807510`)

---

## 5. Stage 4d & 4e — 150-Case Retrospective Test Evaluation (`test_retrospective_150`)

```powershell
$env:nnUNet_raw = "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_preprocessed"
$env:nnUNet_results = "C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\nnUNet_results"

C:\Thesis_RASNET_env\bench\Scripts\python.exe -m benchmark.run_benchmark `
  --config config.lab.yaml `
  --split test `
  --arm all `
  --models all `
  --run-id test_retrospective_150 `
  --resume
```

### Expected Outputs & Pass / Fail Criteria (Stage 4d & 4e)
- **Output Directory**: `results/benchmark_runs/test_retrospective_150/`
- **PASS**:
  - `manifest.json` records `"retrospective_evaluation": true`, `"thresholds_sha256": "4f66ac1d415c5ca9bc09ca0e900e9f8f5d106b3620a20d5916f472d1ba0d3acb"`, all 5 checkpoint hashes, and all 150 test case IDs (`851–1000`).
  - Complete per-case and summary tables written for Arms `A`, `B`, and `C` (`per_case_arm_{A,B,C}.csv`, `summary_arm_{A,B,C}.csv`, `metrics_{model}_arm_{A,B,C}.csv`), plus `diagnostic_rasnet_historical_recipe_tau060_tta_top2.csv`.
  - Pre-declared primary (`paired_stats_primary_arm_{A,B,C}.csv`) and 32-test exploratory (`paired_stats_full32_arm_{A,B,C}.csv`) Wilcoxon signed-rank tables with Holm-Bonferroni correction and rank-biserial effect sizes written.
  - Per-case historical comparison CSVs (`diff_vs_historical_{model}_arm_{A,B,C}.csv`, `diff_vs_historical_rasnet_recipe_tau060_tta_top2.csv`) and `diff_vs_historical_summary.json` generated with pre-declared tolerance `1e-4`.

---

## 6. Stage 4f — Post-Unification Verification Smoke Run (`smoke_verified_v2`) & Audit (`2026-11-10`)

```powershell
C:\Thesis_RASNET_env\bench\Scripts\python.exe -m benchmark.run_benchmark `
  --config config.lab.yaml `
  --split val `
  --arm A `
  --models all `
  --smoke `
  --export-masks `
  --run-id smoke_verified_v2
```

### Expected Outputs & Pass / Fail Criteria (Stage 4f)
- **Output Directory**: `results/benchmark_runs/smoke_verified_v2/` and `results/audits/2026-11-10_verification_audit/`
- **PASS**:
  - `manifest.json` records `source_tree_sha256: 179d8188047ad596e18b7c32fb66ff4c4997c1fc6407eb59e616e4aa58520af1`.
  - `RASNet`, `SegResNet`, and `V-Net` match `val_full_94` with `0.00e+00` maximum absolute difference across all 8 metrics; `nnU-Net V2` batch iterator matches single-case `infer_nnunet_probability` (`smoke_val_all5`) within `7.75e-06` on Dice (`< 1e-4` tolerance).
  - Multi-planar 2D slice overlays saved in `results/audits/2026-11-10_verification_audit/smoke_overlays/smoke_case_{14,45,47}_armA_5models_overlay.png`.

---

## 7. Stage 5 — V1 Epoch-200 Final Checkpoint Recovery & Val94 Endpoint Comparison (`2026-11-17`)

> **Preserved Original**: Exact pre-update `LAB_RUN_PLAN.md` (`SHA-256: 8aab7f02dd3b59a66b870bcba328fffa945e2d0a4f03edc38c529129e96c3f3a`) is preserved at `docs/LAB_RUN_PLAN_v1_2026-11-10_original.md`.

### 7.1 Checkpoint Recovery & Provenance Verification (`recovered_checkpoints/v1_epoch200_20261117_041500Z/`)

```powershell
C:\Thesis_RASNET_env\bench\Scripts\python.exe C:\Thesis_RASNET\recovered_checkpoints\v1_epoch200_20261117_041500Z\recover_and_verify.py
```

- **Recovered Artifacts (seekable split-zip extraction; no combined 900 MB zip, no `extractall()`):**
  - `rasnet_last.pth` (`SHA-256: 71c9e5ef4a02849686c99ccb2d1544ff78c78187e1c0d53e7b22884c708c1b95`, `18,871,718 B`, CRC-32 `0x382cc853`): raw `OrderedDict` (`114` tensors, `4,706,772` params). **No embedded epoch metadata**; epoch-200 identity is supported by `train_rasnet_200ep.py` line `268`, contiguous 200-epoch `logs/rasnet_200ep_log.csv`, and archive timestamp alignment (`5120.0 s` delta vs `5119.97 s` logged elapsed time from ep 160 to ep 200).
  - `nnunet_last.pt` (`SHA-256: 2105bd7238fff7485f1a8d72622f977f635f441e6d44863920c85709b02ffec2`, `246,472,688 B`, CRC-32 `0xae7f79e9`): embedded `current_epoch = 200` (`nnUNetTrainer_200epochs`, `3d_fullres`, `fold_0`, `PlainConvUNet`), staged into isolated native model folder `nnunet_native_model/Dataset501_CoronarySeg/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres/` (`plans.json` `c67af4aa...`, `dataset.json` `d81f4025...`, `fold_0/checkpoint_final.pth` `2105bd72...`).

### 7.2 Tests, Preflight Smoke & Full Val94 Endpoint Evaluation

```powershell
# 1. Run all synthetic unit tests (endpoint eval + report/figure + full benchmark suite)
C:\Thesis_RASNET_env\bench\Scripts\python.exe -B -m pytest -v -p no:cacheprovider tests

# 2. 2-case Val94 smoke run (cases 14, 45)
C:\Thesis_RASNET_env\bench\Scripts\python.exe -B C:\Thesis_RASNET\scripts\run_v1_epoch200_endpoint_eval.py `
  --run-id 20261117_041500Z `
  --out C:\Thesis_RASNET\results\v1_epoch200_smoke_20261117_041500Z `
  --smoke --device cuda --sw-batch-size 8

# 3. Full Val94 endpoint evaluation (all 94 validation cases; --resume continues from saved per-model CSV/masks)
C:\Thesis_RASNET_env\bench\Scripts\python.exe -B C:\Thesis_RASNET\scripts\run_v1_epoch200_endpoint_eval.py `
  --run-id 20261117_041500Z `
  --out C:\Thesis_RASNET\results\v1_epoch200_comparison_20261117_041500Z `
  --device cuda --sw-batch-size 8

# 4. Generate paired case-level CSV, separated comparison table, endpoint figure, and thesis curve text
C:\Thesis_RASNET_env\bench\Scripts\python.exe -B C:\Thesis_RASNET\scripts\generate_v1_epoch200_report_and_figures.py `
  --out-dir C:\Thesis_RASNET\results\v1_epoch200_comparison_20261117_041500Z
```

- **Protocol (`v1-epoch200-endpoint-1.0`):** `Val94` (`N = 94`, `splits_final.json` `efe3f06c...`), `final_epoch200`, `1-pass` (no TTA), `0.50` Gaussian SW blending, threshold `tau = 0.50`, no CC pruning, scored on original SimpleITK physical grid (`surface-v1`), `bootstrap_seed = 42`, `n_boot = 2000`, `tie_tolerance = 1e-12`.
- **Verified Smoke (`results/v1_epoch200_smoke_20261117_041500Z/`):** Cases `14, 45`; RASNet mean Dice `0.7679`, nnU-Net mean Dice `0.7995`; physical headers and voxel-axis orientation verified (`flipped_dices <= 0.0666` vs `>= 0.7411` unflipped).
- **Verified Full Val94 Results (`results/v1_epoch200_comparison_20261117_041500Z/`):**
  - `94/94` cases `'ok'` for both models (`0` dropped, `0` empty predictions, `188/188` output mask headers verified).
  - `RASNet` (`final_epoch200`): Dice `0.7792 ± 0.0569`, Precision `0.7586`, Recall `0.8065`, HD95 `17.32 mm`, ASD `2.619 mm`, clDice `0.8413`, Centerline Recall `0.9068`.
  - `nnU-Net V2` (`final_epoch200`): Dice `0.7767 ± 0.0562`, Precision `0.7428`, Recall `0.8203`, HD95 `18.67 mm`, ASD `2.748 mm`, clDice `0.8347`, Centerline Recall `0.9142`.
  - Paired (`RASNet − nnU-Net`, `N = 94`, **8 exploratory / unadjusted** paired Wilcoxon tests): Dice `+0.0025` (`95% CI [-0.0027, +0.0075]`, unadjusted `p = 0.2401`, `54W / 0T / 40L`; non-significant ≠ equivalence); Precision `+0.0158` (`[+0.0072, +0.0245]`, unadjusted `p = 1.07e-4`, `66W / 0T / 28L`, favors RASNet); Recall `-0.0138` (`[-0.0175, -0.0098]`, unadjusted `p = 4.85e-10`, `16W / 0T / 78L`, favors nnU-Net); Centerline Recall `-0.0074` (`[-0.0124, -0.0025]`, unadjusted `p = 0.0092`, `37W / 2T / 55L`, favors nnU-Net); clDice `+0.0067` (`[-0.0006, +0.0135]`, unadjusted `p = 0.0352`, `57W / 0T / 37L`). Under post-hoc exploratory Holm-Bonferroni (`K = 8`), Precision (`Holm-8 p = 7.47e-4`) and Recall (`Holm-8 p = 3.88e-9`) remain `< 0.05`, whereas `clDice` (`Holm-8 p = 0.1760`) and `centerline_recall` (`Holm-8 p = 0.0551`) do not.
  - Distinct best-checkpoint references (`val_full_94` Arm A and `optionA_v1_symmetric_sw050` Val94) are tabulated separately in `table_epoch200_vs_existing_best_references.md` and `fig_v1_epoch200_val94_endpoint_comparison.{png,svg}`; the narrative in `THESIS_CURVE_AND_ENDPOINT_CORRECTION.md` is superseded by `results/v1_reporting_correction_20261117_105000Z/` (`V1_REPORTING_CORRECTION_REPORT.md`, `THESIS_REPLACEMENT_TEXT.md`, `HISTORICAL_ERRATA.md`, and corrected figures/tables).

---

## 8. Stage 6 — V1 Reporting, Provenance & Documentation Correction Run Record (`2026-11-17`)

> **Preservation & Scope**: Pre-edit copies and SHA-256 digests of all 9 modified active files (`pre_edit_hashes.json`) plus 548 protected/historical files (`protected_baseline_hashes.json`, `0` mismatches) are stored in `results/v1_reporting_correction_20261117_105000Z/`. No retraining, new GPU inference, V2 execution, gate overrides, git commits, or git pushes were performed.

### 8.1 Phase 1 Read-Only Audit & Verification Commands

```powershell
# Active process, GPU, and Git state checks
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|conda|uv|nvcc' -or $_.CommandLine -match 'Thesis_RASNET|rasnet|nnunet' }
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
git status
git log -n 10 --oneline

# Independent checkpoint provenance, nnU-Net trainer convention, Val94/Option A recomputation, and model parameter count checks
C:\Thesis_RASNET_env\bench\Scripts\python.exe -c "import torch, json; ..." # verified 114/292/83/164/63 state-dict keys, plans [96,160,160] & [0.5,0.349609375,0.349609375], current_epoch=199 vs ep 189 raw peak vs current_epoch=200
C:\Thesis_RASNET_env\bench\Scripts\python.exe -c "from benchmark import stats, endpoint_eval, symmetric_eval; ..." # verified Val94 endpoint (<=3.55e-15), Holm-8, Option A R1-R4 (<=3.55e-15), and exact model params
```

### 8.2 Phase 2 Tests-First, Corrected Figure/Table Generation, & Full Suite Verification

```powershell
# 1. Tests-first run BEFORE reporting script fixes and new generator (expected FAIL: 4 failed in 5.77s)
C:\Thesis_RASNET_env\bench\Scripts\python.exe -m pytest tests/test_v1_reporting_corrections.py

# 2. Generate corrected training-curve figure, endpoint figure, and 5 verified CSV/MD tables
C:\Thesis_RASNET_env\bench\Scripts\python.exe scripts\generate_v1_reporting_corrections.py

# 3. Targeted tests AFTER fixes (PASS: 4 passed in 2.91s)
C:\Thesis_RASNET_env\bench\Scripts\python.exe -m pytest tests/test_v1_reporting_corrections.py

# 4. Full synthetic & reporting test suite AFTER fixes and documentation updates (PASS: 72 passed)
C:\Thesis_RASNET_env\bench\Scripts\python.exe -m pytest tests/
```

### 8.3 External Thesis & Slide Integration Status

- **Status (`C:\Thesis_RASNET`)**: **`OUTSTANDING`** — recursive search confirmed zero `.tex`, `.docx`, `.pptx`, or `.key` manuscript/slide files exist inside `C:\Thesis_RASNET`. Ready-to-paste replacement text, defense Q&A, and slide outline are supplied in `results/v1_reporting_correction_20261117_105000Z/` (`THESIS_REPLACEMENT_TEXT.md`, `DEFENSE_QA_V1.md`, `DEFENSE_SLIDE_OUTLINE.md`).

