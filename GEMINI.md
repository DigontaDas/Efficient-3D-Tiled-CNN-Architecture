# RASNet Thesis — Project Context and Rules for Gemini / Antigravity

> **Auto-loaded project memory.** Antigravity reads this file when it opens this workspace. Claude Code reads `CLAUDE.md`. **The rules in §3 are identical in both files; keep them in sync.**
> **Single source of truth for paths and hashes:** [`CURRENT_EVIDENCE_INDEX.md`](CURRENT_EVIDENCE_INDEX.md). Numbers below are copied from result files (source given per row). If a file and this summary ever disagree, **the file wins**; fix this summary.

---

## 1. Project

- **Repository:** `Efficient-3D-Tiled-CNN-Architecture` / `Thesis_RASNET`
- **Author:** Nafis Mehedi (`Rytnix786 <nafismehedi37@gmail.com>`)
- **Topic:** RASNet is a MONAI SegResNet backbone with 3D attention gates, multi-scale deep supervision and `StenosisAwareLoss` (0.4 Dice + 0.6 Focal, γ = 2.5), at 4.71 M parameters. The thesis evaluates it for 3D coronary artery segmentation in CCTA (ImageCAS).
- **Baselines (real public implementations):**
  - nnU-Net V2 (`MIC-DKFZ/nnUNet`, `nnunetv2 2.8.1`, `3d_fullres`, 200-epoch trainer, fold 0)
  - MONAI `SegResNet`
  - MONAI `VNet`
  - MONAI residual `UNet` (Kerfoot 2018 — **not** `wolny/pytorch-3dunet`; trained on 128³ resized volumes)
- **Goal:** show RASNet is an accurate **and** efficient alternative to these baselines. Aim to beat them, but only under protocols that are identical for every model. Efficiency (6.5× fewer parameters than nnU-Net, about 9× fewer GFLOPs per patch) is always a valid claim. Accuracy claims must come from pre-declared tests.
- **Methodological status:** v1 test results on IDs 851–1000 are a **retrospective reassessment**, because earlier sweeps (τ = 0.60, loss calibration) and Phase H inspected that split (`BENCHMARK_PREREGISTRATION.md`). Rasnet-v2 is the controlled, pre-registered, multi-seed study.

## 2. Two machines

| | `H:\Thesis_Trainings` (laptop copy) | `C:\Thesis_RASNET` (lab PC, RTX 4080 SUPER 16 GB) |
| :--- | :--- | :--- |
| Data / checkpoints | **Not present** | Present |
| Allowed work | Code reading, synthetic CPU tests | Training, evaluation, GPU runs |
| Python env | Varies (MONAI may differ, no `nnunetv2`) | `C:\Thesis_RASNET_env\bench` (Python 3.12, torch cu126, MONAI 1.6.1, nnunetv2 2.8.1) |

Anything only verifiable on the lab PC is marked **LAB-CHECK**.

---

## 3. Rules (identical to `CLAUDE.md` §3)

1. **Data paths:** the dataset root lives only in config (`config.lab.yaml`, `Rasnet-v2/configs/fair_v2.yaml`), never in code.
2. **Frozen science:** do not change model architectures, losses, hyperparameters, epoch counts or checkpoints.
   - The only exception is a **dated pre-registration amendment the user has explicitly approved**, committed before any affected result exists.
3. **Locked test set:** test cases 851–1000 are locked. Every selection (threshold, recipe, LR, seed, design, checkpoint) uses Val-94 only.
   - Test inference runs once per frozen recipe, after the validation lock is written and committed.
4. **Symmetry for every comparison:**
   - Every model gets the same inference (overlap, blending), the same TTA definition, the same threshold grid and the same selection rule.
   - Symmetric TTA means 8-pass mirroring: identity + 7 flip-axis combinations, logits averaged before softmax, as in nnU-Net's predictor.
   - Never tune RASNet alone and compare it with untuned baselines.
5. **Evidence:**
   - Every reported number comes from a file produced by a run with a manifest.
   - Stage 2A pilot numbers (`val20_dice`, 50 epochs, one seed) are for LR selection only and must never be cited as results.
   - Do not quote numbers from chat transcripts; quote the file.
6. **Tests first:** write the test, show it failing, then implement and show it passing.
7. **Git:**
   - Never commit data or checkpoints. Record checkpoint SHA-256 in `CHECKPOINTS.md`.
   - Commit only when the user asks. Never push without explicit authorisation.
8. **GPU:** one GPU job at a time. Before launching, check `Rasnet-v2/logs/queue_status.json` and `Rasnet-v2/STATE_HANDOFF.md`. Another agent may own the GPU.
9. **Protected results:** never overwrite `results/benchmark_runs/val_full_94/`, `results/benchmark_runs/test_retrospective_150/`, `results/rasnet_fixed_weight_optimization/` or `results/thesis_outputs/`.
10. **Claims:**
    - Thesis wording is chosen from the pre-written outcome tables (`docs/01_…` §7, `docs/02_…` §5) that match the numbers. Never write a stronger claim.
    - "Not significant" ≠ "equal".
    - Phase H (RASNet-only tuning) is not a fair comparison with nnU-Net.
11. **Status polling:** do not post repeated status-only messages. Report when a milestone changes (epoch with validation, job finished, error).

---

## 4. Verified v1 results (200 epochs, Test N = 150)

Source: `results/thesis_outputs/Table2_test_150_arms_A_B_C_summary.md` (Dice / precision / HD95 / clDice as mean ± SD) and `Table4_computational_efficiency_rtx4080super.csv` (params).

| Model | Params | Arm A Dice | Arm A Precision | Arm A HD95 (mm) | Arm C Dice | Arm C Precision | Arm C HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RASNet | 4.71 M | **0.7736 ± 0.0616** | **0.8032** | 14.83 | 0.7899 ± 0.0677 | **0.8252** | 9.94 |
| nnU-Net V2 | 30.79 M | 0.7687 ± 0.0668 | 0.7391 | 21.10 | **0.8055 ± 0.0633** | 0.8207 | **8.26** |
| SegResNet | 4.70 M | 0.7481 ± 0.0654 | 0.7253 | 30.41 | 0.7884 ± 0.0700 | 0.8095 | 9.84 |
| V-Net | 45.60 M | 0.7492 ± 0.0693 | 0.7524 | 22.10 | 0.7710 ± 0.0736 | 0.8054 | 12.38 |
| 3D U-Net | 4.81 M | 0.5517 ± 0.0480 | 0.7269 | 10.88 | 0.6064 ± 0.0532 | 0.5877 | 11.85 |

Arm definitions:

| Arm | Passes | Threshold | Post-processing |
| :--- | :--- | :--- | :--- |
| A | single pass | τ = 0.50 | none |
| B | native TTA (RASNet 4-pass, nnU-Net 8-pass, others none) | τ = 0.50 | none |
| C | single pass | Val-94 τ\* from `results/thresholds.json` | 26-connected Top-2 CC |

Arm B Dice: RASNet 0.7807, nnU-Net 0.7760.

Historical asymmetric recipe (RASNet, 4-pass TTA, τ = 0.60, Top-2 CC), from `diagnostic_rasnet_historical_recipe_tau060_tta_top2.csv`:

- Dice 0.7755 ± 0.0726
- Precision 0.8968
- Recall 0.6892
- HD95 11.14 mm
- ASD 1.831 mm
- clDice 0.8562

**Known v1 asymmetries.** These are disclosed; Option A addresses the inference ones.

- MONAI models were evaluated with 25 % overlap and constant blending; nnU-Net used 50 % overlap with Gaussian blending.
- TTA differed in Arm B.
- The τ grid floor of 0.30 clamped RASNet, V-Net and 3D U-Net.
- Training differences are baked into the weights: spacing 0.5³ vs 0.35 × 0.35 × 0.5 mm, patch 96³ vs 96 × 160 × 160, normalisation, augmentation, capacity.

### Paired tests, RASNet vs nnU-Net (primary family, Holm K = 3)

Source: `results/audits/2026-11-10_verification_audit/paired_stats_primary_arm_{A,B,C}_with_ci.csv`.

| Arm | Precision Δ [95 % CI], W/T/L | HD95 Δ mm [95 % CI], W/T/L | ASD Δ mm [95 % CI], W/T/L |
| :--- | :--- | :--- | :--- |
| A | +0.0642 [+0.0565, +0.0728], 139/0/11, Holm p < 1e-20 | −6.27 [−8.43, −4.38], 107/0/43 | −0.737 [−0.995, −0.503], 107/0/43 |
| B | +0.0602 [+0.0527, +0.0680], 141/0/9 | −5.56 [−7.67, −3.63], 103/0/47 | −0.644 [−0.885, −0.422], 98/0/52 |
| C | +0.0046 [−0.0046, +0.0124], 84/0/66, Holm p = 0.160 (n.s.) | +1.68 [+0.50, +3.08], 55/1/94, Holm p = 4.2e-4 (favours nnU-Net) | +0.277 [−0.058, +0.713], 51/0/99, Holm p = 1.5e-4 (favours nnU-Net) |

### Arm C, RASNet vs SegResNet (secondary K = 32 family)

Source: `results/audits/2026-11-10_verification_audit/paired_stats_full32_arm_C_with_ci.csv`.

| Metric | Δ [95 % CI] | p | Holm p | W/T/L | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Precision | +0.0158 [+0.0105, +0.0221] | 4.68e-11 | 8.42e-10 | 108/0/42 | Significant |
| Dice | +0.0016 [−0.0026, +0.0065] | 0.520 | 1.0 | 81/0/69 | n.s. |

### Phase H — RASNet-only inference exploration (not a fair nnU-Net comparison)

Source: `results/rasnet_fixed_weight_optimization/PHASE_H_OPTIMIZATION_REPORT.md`.

- **Setup:** frozen `rasnet_best.pth`; 152 recipes searched on Val-94 for RASNet only.
- **Result:** with 0.50 overlap, Gaussian blending and τ = 0.40 plus Top-2 CC, test Dice was 0.8008 single-pass and 0.8043 with 4-pass TTA.
- **Why it is not fair:** the baselines got no equivalent search, and the nnU-Net reference was single-pass. Val-94: RASNet's best was 0.8023, while nnU-Net's single-pass Arm C was 0.8075.
- **Replacement:** Option A is the fair version.

---

## 5. Rasnet-v2 — the controlled study (`Rasnet-v2/`)

- **Protocol:** `PREREGISTRATION_V2.md` and `configs/fair_v2.yaml`.
- **Shared settings for all MONAI models:**
  - spacing 0.3496 × 0.3496 × 0.5 mm, patch 160 × 160 × 96, CT normalisation
  - identical augmentation and loss
  - AdamW, cosine schedule, 250 steps × 200 epochs, effective batch 2
  - 0.50 Gaussian inference, τ grid 0.20–0.70
- **nnU-Net V2** is retrained 3× with its native recipe. Present it as **framework vs framework**.
- **Width tiers:** F = 16 (4.71 M, matched to SegResNet 4.70 M) and F = 32 (18.8 M).
- **Pipeline** (`src/run_queue.py --stage …`):
  - `stage2a`: LR pilots
  - `stage2a_eval`: Val-94 LR lock
  - `stage2b` / `stage2b_eval`: design lock with gates
  - `stage3`: 200-epoch runs with seeds {42, 123, 456}
  - `stage4a` / `stage4b` / `stage4c`: val curves, Val-94 threshold lock, one-shot Test-150
  - `stage5`: families, non-inferiority, efficiency, Pareto
- **Hypothesis families:**
  - Family 1: RASNet-v2 vs matched SegResNet-v2 on Dice, clDice, precision, HD95.
  - Family 2: RASNet-v2 vs nnU-Net on precision, HD95, clDice.
  - Dice non-inferiority vs nnU-Net, margin −0.005.

**Defects found 2026-11-12, fixed by proposed Amendment A1** (`docs/02_…`; needs user sign-off, then commit before Stage 2A resumes):

- the Stage 3 clDice weight can differ between RASNet and its baselines
- the Stage 2B gates are not binding, and empty predictions can help a candidate pass them
- TTA is 4-pass for MONAI models vs 8-pass for nnU-Net
- the F32 LR grid lacks 1e-4
- the deep-supervision ablation has a single seed

---

## 6. Current state and next steps

- **2026-11-10:** all v1 LAB-CHECKs were verified on the lab PC (`results/audits/2026-11-10_verification_audit/`). Superseded material is archived in `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/`.
- **2026-11-12 (lab-reported):** Stage 2A is paused at 14/16. Job 15 (`pilot2a_rasnet_v2_f32_lr3e-4_seed42`) has resume state saved at epoch 30, step 200 (`STATE_HANDOFF.md`).
- **Next, in order** (`docs/03_NEXT_STEPS_RUNBOOK.md`):
  - **P0:** fix the `CHECKPOINTS.md` loss description (SegResNet and V-Net used `DiceFocalLoss(1.0, 2.0)`, not DiceCELoss), and pre-register Option A and Amendment A1.
  - **P1:** write the Option A code (`docs/01_…`).
  - **P2:** apply the Amendment A1 patches.
  - **P3:** run Option A on the GPU.
  - **P4:** resume Rasnet-v2 through Stage 5.
  - **P5:** thesis integration.

## 7. Key files

| What | Where |
| :--- | :--- |
| Evidence index | `CURRENT_EVIDENCE_INDEX.md` |
| Checkpoints | `CHECKPOINTS.md` |
| Inference spec | `INFERENCE_SPEC.md` |
| v1 benchmark | `benchmark/` (`run_benchmark.py`, `metrics.py` surface-v1, `postprocess.py`, `thresholds.py`, `stats.py`, `adapters/`); tests in `tests/` |
| RASNet model + loss | `Phase3_Local_Integration/rasnet_model.py`, `rasnet_loss.py` |
| Splits | `Phase3_Local_Integration/splits_final.json` (690 / 94 / 150) |
| v1 trainers | `Q1_Publication_Package/matched_200ep_benchmark/train_*_200ep.py`, `prepare_and_run_nnunet_200ep.py` |
| v1 results | `results/benchmark_runs/`, `results/thresholds.json`, `results/thesis_outputs/` |
| Phase H | `results/rasnet_fixed_weight_optimization/`, `scripts/run_phaseH_rasnet_optimization.py` |
| Rasnet-v2 | `Rasnet-v2/` |
| Plans | `docs/01_OPTION_A_SYMMETRIC_V1_EVAL.md`, `docs/02_RASNET_V2_FAIRNESS_FIXES_AND_WIN_STRATEGY.md`, `docs/03_NEXT_STEPS_RUNBOOK.md` |

## 8. Technical pitfalls

- **SimpleITK `.img.nii.gz` quirk:** SimpleITK writes `*.img.nii.gz` as Analyze. Write to a temp `.nii.gz` name and rename (see `export_mask_like`).
- **nnU-Net training-log numbers are not comparable:** `Pseudo dice` is patch-level on foreground-oversampled batches, not comparable to full-volume Dice.
- **Empty predictions:** precision and HD95 are NaN for an empty prediction. Count empties; never let NaN-dropping help a candidate.
- **v1 3D U-Net:** excluded from sliding-window comparisons (128³ resize).
