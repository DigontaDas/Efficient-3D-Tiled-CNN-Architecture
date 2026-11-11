# Formal Pre-Registration: `RASNet-v2` Pre-Registered, Multi-Seed, Confounder-Controlled Study (`Rasnet-v2/`)

- **Locked Prior to**: Stage 2 Training Launch
- **Scope Boundary**: All verified `v1` code, checkpoints (`Q1_Publication_Package/`), Phase H outputs (`results/rasnet_fixed_weight_optimization/`), and benchmark Tables 1–4 remain 100% immutable. All `v2` code, caches, checkpoints, logs, and results live strictly inside `C:\Thesis_RASNET\Rasnet-v2\`.

---

## 1. Cohort Roles & Contamination Control
- **Training Split (`N = 690`)**: Canonical `splits_final.json["train"]` (`SHA-256: efe3f06cfe...` prefix verified in preflight).
- **Validation Split (`N = 94`)**: Canonical `splits_final.json["val"]`. Strictly quarantined from `IDs 851–1000`.
- **In-Loop Convergence Subset (`val20_curves`, `N = 20`)**:
  - Frozen in `Rasnet-v2/configs/val20_curves_ids.json` before Stage 2 starts.
  - Stratified jointly into **4 quartiles** over reconstructed ground-truth coronary voxel volume (`Q25 = 86,976`, `Median = 105,396`, `Q75 = 123,199`) balanced with `v1` `RASNet Arm A` Dice difficulty (`Min = 0.6041`, `Q25 = 0.7589`, `Median = 0.7837`, `Q75 = 0.8146`, `Max = 0.8713`), selecting `5` cases per stratum with `seed = 42`.
  - During `train_fair_v2.py` training, validation runs **only** on `val20_curves` every `val_interval = 20` epochs (`0.50 Gauss`, `sw_batch_size = 8`). Full `Val N = 94` evaluation never runs inside training epochs; it executes strictly at Stage 2 pilot lock and Stage 4B final lock.
- **ImageCAS Retrospective Final Cohort (`IDs 851–1000`, `N = 150`)**:
  - Explicitly reported throughout as a **Retrospective Final Run** (never "single-shot"), because `851–1000` were inspected in Round 1 and Phase H. Zero tuning or model/threshold changes are permitted after Stage 4C begins.
- **Primary Prospective Clinical Cohort (`Phase I`, ~20 days out)**:
  - Ground truth is **manually annotated by physicians without any model-assisted draft masks** (non-circularity confirmed).
  - Clinical chapter Option A keeps the ImageCAS `v1` + Hospital chapter frozen on `RASNet-v1`; `Phase I` serves as the single-shot prospective confirmatory benchmark comparing locked `RASNet-v2`, `SegResNet-v2`, and `nnU-Net V2` (zero post-arrival tuning).

---

## 2. Pre-Registered Decision Rules (Locked Strictly on `Val N = 94`)
1. **Stage 2A Learning-Rate Lock**:
   - Evaluate `lr ∈ {1e-4, 3e-4, 1e-3}` equally for 50 epochs (`50 × 250 = 12,500` optimizer steps) on all four MONAI architectures (`SegResNet-v2`, `RASNet-v2`, `V-Net-v2`, `Patch 3D U-Net-v2`) at `init_filters = 16`, plus `F = 32` (`mb=1×2`) LR sanity checks (`3e-4` vs. `1e-3` on both `SegResNet-v2 F=32` and `RASNet-v2 F=32`).
   - Each architecture/width pair (`segresnet_f16`, `segresnet_f32`, `rasnet_v2_f16`, `rasnet_v2_f32`, `vnet`, `unet3d`) explicitly locks its highest `Val N = 94` raw (`0.50 Gauss, 1-Pass, τ = 0.50`) Dice learning rate in `results/pilots_val94/locked_lrs.json` and `results/pilots_val94/pilot_lock_decision.md` (zero silent cross-architecture LR inheritance).
2. **Stage 2B `RASNet-v2` Design Lock**:
   - Select `(F*, gate_mode, w_cldice)` maximizing `Val N = 94` Dice (`0.50 Gauss`) subject to `Val Precision >= 0.8000` and `<=` mean component fragmentation vs. matched-width `SegResNet-v2`.
   - Both `F = 16` (`4.71M`) and `F = 32` (`18.80M`) are carried across `3` seeds in Stage 3 (20-run design) for both `RASNet-v2` and `SegResNet-v2`, while Stage 3 Runs `4` (`Original-AG`) & `5` (`Fixed-AG, no DS`) remain pinned at `F = 16` to cleanly isolate architectural contributions against `SegResNet-v2 F = 16` and `RASNet-v2 Lite F = 16`.
3. **Stage 4B Threshold & Post-Processor Parameter Grids (`Val N = 94`)**:
   - Threshold sweep grid: `τ* ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}`.
   - Pre-registered Arm C variants evaluated side-by-side:
     - **`Arm C-Top2`**: Global `τ*` + 26-connected `keep_top_k = 2` (`prune_components`).
     - **`Arm C-Seeded`**: `seeded_component_filter_3d(prob, tau, tau_seed, min_seed_voxels, min_comp_vol_mm3)` retaining every 26-connected component passing either `comp_vol_mm3 >= min_comp_vol_mm3` OR containing a seed core `(prob >= tau_seed)` with `>= min_seed_voxels`, swept strictly over:
       - `tau_seed ∈ {0.50, 0.55}`
       - `min_seed_voxels ∈ {50, 100}`
       - `min_comp_vol_mm3 ∈ {25.0, 50.0, 100.0}`
       selected strictly to maximize mean `Val N = 94` Dice without requiring `<= 2` components.

---

## 3. Evaluation Arms, Attribution Tiers & Statistical Hypothesis Families (`α = 0.05`)
- **Evaluation Protocol**: Strictly raw `.nii.gz` -> preprocessing -> `sliding_window_inference(0.50 Gauss)` -> MONAI `Invertd(nearest_interp=False)` onto each scan's native SimpleITK `(Z, Y, X)` physical grid -> `surface-v1` metrics (`metrics.py`).
- **Arms**:
  - **Arm A**: Single-pass (`0.50 Gauss`, no TTA), fixed `τ = 0.50`, no CC pruning.
  - **Arm B**: Symmetric TTA (`4-pass` spatial flips for all MONAI models; native `8-pass` mirroring for `nnU-Net V2`), `τ = 0.50`, no CC pruning.
  - **Arm C**: Single-pass (`0.50 Gauss`), frozen validation-tuned thresholds `τ*` (`Val N = 94`), applied with `Top-2 CC` (`Arm C-Top2`) and pre-registered `seeded_component_filter_3d` (`Arm C-Seeded`).
- **Pre-Declared Hypothesis Families (Holm-Bonferroni adjusted within family, `2,000`-bootstrap 95% CIs, seeds `{42, 123, 456}`)**:
  1. **Family 1 — Tier 1 Causal Architecture Comparison (`K = 4`)**:
     `RASNet-v2` vs. matched-width `SegResNet-v2 (+ RASNetV2Loss)` on `Dice`, `clDice`, `Precision`, `HD95` (case-matched seed-averaged & pooled per-seed two-sided Wilcoxon signed-rank tests).
  2. **Family 2 — Tier 2 Full-System Framework Comparison vs. `nnU-Net V2` (`K = 3` + Non-Inferiority)**:
     - Two-sided superiority hypotheses on `Precision`, `HD95`, `clDice` (`Holm K = 3`).
     - Pre-registered **Dice Non-Inferiority Test**: `RASNet-v2` is non-inferior to `nnU-Net V2` if the lower bound of the 95% two-sided bootstrap CI for paired `Dice_RASNetV2 - Dice_nnUNet` exceeds **`Δ_NI = -0.0050`**.
  3. **Tier 3 — Exploratory Efficiency & Parameter Pareto**:
     Full `Dice vs. Parameter Count` curve across `{4.70M, 4.71M, 12.86M, 18.80M, 30.79M, 45.60M}` with measured RTX 4080 SUPER patch latency, GFLOPs, and peak VRAM.

