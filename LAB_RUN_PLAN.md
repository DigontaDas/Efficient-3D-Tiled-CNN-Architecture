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
  - `Python 3.12.9`, `torch 2.14.1+cu126`, `CUDA: True (NVIDIA GeForce RTX 4080 SUPER)`, `monai 1.6.1`, `SimpleITK 2.5.7`, `scipy 1.18.1`, `scikit-image 0.26.0`, `nibabel 5.4.2`, `nnunetv2 2.8.1`.
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
- **PASS**: `29 passed` (`0 failed`). Covers brute-force and anisotropic surface distances (`surface-v1`), empty-mask/skeleton status handling, paired Wilcoxon + Holm correction + bootstrap CIs + direction-aware wins/ties/losses, connected-component volume logging, validation-only threshold selection & tie-breaking, real MONAI architectures (`RASNet`, `SegResNet`, `VNet`, `3DUNet`) + explicit `_SyntheticNNUNetPredictor` test double, coordinate inversion, `3D U-Net` `128³` resizing, `V-Net` `%16` padding, and fail-closed checks on non-best checkpoints, split leakage, and stale output directories.

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
