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

| | `H:\Thesis_Trainings` (home PC) | `C:\Thesis_RASNET` (lab PC, RTX 4080 SUPER 16 GB) |
| :--- | :--- | :--- |
| Data / checkpoints | **Not present** | Present |
| Allowed work | Code reading, synthetic CPU tests | Training, evaluation, GPU runs |
| Python env | Varies (MONAI may differ, no `nnunetv2`) | `C:\Thesis_RASNET_env\bench` (Python 3.12, torch cu126, MONAI 1.6.1, nnunetv2 2.8.1) |

Everything that runs for real runs on the lab PC. All commands, settings and performance tuning target it: PowerShell, `$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"`, repo root `C:\Thesis_RASNET`. The synthetic & reporting test suite (`tests/`) collects `94` tests (plus `46` in `Rasnet-v2/tests/`), all passing on the lab PC on `2026-11-22` via `C:\Thesis_RASNET_env\bench\Scripts\python.exe -m pytest tests/`.

Anything only verifiable on the lab PC is marked **LAB-CHECK**.

---

## 3. Rules (identical to `CLAUDE.md` §3)

1. **Data paths:** the dataset root lives only in config (`config.lab.yaml`, `Rasnet-v2/configs/fair_v2.yaml`), never in code.
2. **Frozen science:** do not change model architectures, losses, hyperparameters, epoch counts or checkpoints.
   - The only exception is a **dated pre-registration amendment the user has explicitly approved**, committed before any affected result exists.
   - The amendment goes in `BENCHMARK_PREREGISTRATION.md` (v1) or `Rasnet-v2/PREREGISTRATION_V2.md` (v2).
3. **Locked test set:** test cases 851–1000 are locked. No threshold, recipe, LR, seed, design or checkpoint selection may use them; select on validation (Val-94) only.
   - Test inference happens once per frozen recipe, after the validation lock file is written and committed.
   - Because v1 Arms A/B/C and Phase H already inspected 851–1000, every v1 test result is reported as **retrospective**.
4. **Symmetry for every comparison:**
   - Every compared model gets the same inference (overlap, blending), the same TTA definition, the same threshold grid and the same selection rule.
   - Symmetric TTA means 8-pass mirroring: identity + all 7 flip-axis combinations, logits averaged before softmax. This is what nnU-Net's predictor does.
   - Never tune RASNet alone and then compare it with untuned baselines.
5. **Evidence:**
   - Every reported number must come from a file produced by a run with a manifest (git commit, source hashes, checkpoint SHA-256, env).
   - Pilot numbers (Stage 2A `val20_dice`, 50-epoch, single seed) are for LR selection only and must never be cited as results.
   - Do not quote numbers from chat transcripts; quote the file.
6. **Tests first:** write the test, run it and show it failing, then implement and show it passing.
7. **Git:**
   - Never commit data or checkpoints. Record checkpoint SHA-256 in `CHECKPOINTS.md`.
   - Commit only when the user asks.
   - Never push without explicit authorisation.
8. **GPU:** one GPU job at a time on the lab PC. Before launching anything, check `Rasnet-v2/logs/queue_status.json` and `Rasnet-v2/STATE_HANDOFF.md`. Another agent may own the GPU.
9. **Protected results:** never overwrite `results/benchmark_runs/val_full_94/`, `results/benchmark_runs/test_retrospective_150/`, `results/rasnet_fixed_weight_optimization/` or `results/thesis_outputs/`. New analyses write to new folders.
10. **Claims:**
    - Choose thesis wording from the pre-written outcome tables (`docs-for-next-v2-gen/01_…` §7, `docs-for-next-v2-gen/02_…` §5) that match the numbers. Never write a stronger claim.
    - "Not significant" is not "equal".
    - A Phase H RASNet-only result is not a fair comparison with nnU-Net.
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
- **Replacement:** Option A (`results/optionA_v1_symmetric_sw050/`, committed `22d4581`) is the completed fair 4-model symmetric version (`3dunet` excluded).

### Option A — Symmetric v1 evaluation (4 models, Test N = 150 retrospective)

Source: `results/optionA_v1_symmetric_sw050/OPTION_A_RESULTS.md`, `summary_test.csv`, `noninferiority_rasnet_vs_nnunet_dice.csv`.

- **Primary recipe `R4_tuned_tta8_top2`** (`0.50` Gauss SW, 8-pass logit-averaged TTA, Val94-locked `tau*`, Top-2 CC):
  - `nnU-Net V2` (`tau*=0.20`): Dice `0.8059 ± 0.0630`, Precision `0.8084`, Recall `0.8104`, HD95 `8.17 mm`, clDice `0.8744`.
  - `RASNet` (`tau*=0.40`): Dice `0.8039 ± 0.0691`, Precision `0.8302`, Recall `0.7861`, HD95 `8.93 mm`, clDice `0.8750`.
  - `SegResNet` (`tau*=0.45`): Dice `0.7989 ± 0.0692`; `V-Net` (`tau*=0.45`): Dice `0.7915 ± 0.0677`.
  - **Non-inferiority (RASNet − nnU-Net Dice, margin `−0.005`):** mean diff `−0.0021`, `95% CI [−0.0088, +0.0030]`, `NI = False`, `Superior = False`.
  - **R4 families (Holm within family):** RASNet wins `FA_primary` Precision (`+0.0218`, Holm `p = 7.23e-17`) and `FB`/`FC` Dice over SegResNet/V-Net; HD95/ASD/clDice vs nnU-Net are non-significant; nnU-Net retains Recall (`+0.0244`, Holm `p = 2.40e-17`).
- **Secondary recipes `R1`–`R3`:** `R1` (`0.7788` vs `0.7687`), `R2` (`0.7863` vs `0.7760`), `R3` (`0.8008` vs `0.8055`, `NI = False`).

### V1 Epoch-200 Endpoint Evaluation (`final_epoch200`, Val N = 94)

Source: `results/v1_epoch200_comparison_20261117_041500Z/` & `results/v1_reporting_correction_20261117_105000Z/table_epoch200_val94_summary_and_paired_corrected.md`.

- **Protocol:** `final_epoch200` checkpoints (`rasnet_last.pth` via script/log/archive provenance, `nnunet_last.pt` `current_epoch=200`), symmetric 1-pass, `0.50` Gauss SW, `tau=0.50`, no CC (`N=94`, 0 empty, 188/188 geometry OK).
- **Result (`RASNet` vs `nnU-Net`):** Dice `0.7792 ± 0.0569` vs `0.7767 ± 0.0562` (paired diff `+0.0025`, `95% CI [−0.0027, +0.0075]`, exploratory unadjusted Wilcoxon `p = 0.2401`, `54W/0T/40L`; neither superiority nor equivalence). Across the 8 exploratory unadjusted metrics, post-hoc Holm-8 retains Precision (`+0.0158`, `Holm-8 p = 7.47e-4`) and Recall (`−0.0138`, `Holm-8 p = 3.88e-9`), while `clDice` (`unadj p = 0.0352`, `Holm-8 p = 0.1760`) and `centerline_recall` (`unadj p = 0.0092`, `Holm-8 p = 0.0551`) do not survive `< 0.05`.

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

**Defects found 2026-11-12, fixed by Amendment A1** (`docs-for-next-v2-gen/02_…`; signed off and committed 2026-11-13 at `34ee6ff` / `79e0458`):

- the Stage 3 clDice weight can differ between RASNet and its baselines
- the Stage 2B gates are not binding, and empty predictions can help a candidate pass them
- TTA is 4-pass for MONAI models vs 8-pass for nnU-Net
- the F32 LR grid lacks 1e-4
- the deep-supervision ablation has a single seed

---

## 6. Current state and next steps

- **2026-11-10:** all v1 LAB-CHECKs were verified on the lab PC (`results/audits/2026-11-10_verification_audit/`). Superseded material is archived in `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/`.
- **2026-11-13 (`34ee6ff`, `aecf60e`, `79e0458`):** pre-registrations, Option A code, and Amendment A1 patches committed (`CHECKPOINTS.md` loss description corrected).
- **2026-11-15 (`558e7eb`, `22d4581`):** **P3 (Option A) completed** on the RTX 4080 SUPER (`results/optionA_v1_symmetric_sw050/`): symmetric 4-model evaluation (`R4`: RASNet `0.8039` vs nnU-Net `0.8059`, `NI=False`).
- **2026-11-16 (`b3ca1f1`):** **Rasnet-v2 Stage 2A/2B pilots completed** (`Rasnet-v2/results/pilots_val94/pilot_lock_decision.md`): Stage 2B gate outcome **`FALLBACK (NO CANDIDATE PASSED BOTH GATES)`**. **V2 Stage 3+ remains paused**; do not run V2 or override failed gates.
- **2026-11-17 (`8e9b303` & V1 reporting correction):**
  - V1 Epoch-200 Val94 endpoint evaluation completed (`results/v1_epoch200_comparison_20261117_041500Z/`, `0.7792` vs `0.7767`, `p = 0.2401`, 8 exploratory unadjusted metrics).
  - `CHECKPOINTS.md` updated with verified state-dict key counts (`114`/`292`/`83`/`164`/`63`), nnU-Net plans (`[96, 160, 160]`, `[0.5, 0.349609375, 0.349609375]`), `current_epoch=199` vs ep `189` raw peak, and `final_epoch200` provenance.
  - Corrected figures/tables, `HISTORICAL_ERRATA.md`, and thesis/defense replacement text generated in `results/v1_reporting_correction_20261117_105000Z/`.
- **Next:** **P5 (external thesis & slide integration)** remains **`OUTSTANDING`** (no `.tex`/`.docx`/`.pptx` sources in `C:\Thesis_RASNET`). Keep `git.dirty_code = false` for any future runs (`docs-for-next-v2-gen/00_LAB_PC_OPERATOR_MANUAL.md`).

## 7. Key files

| What | Where |
| :--- | :--- |
| Evidence index | `CURRENT_EVIDENCE_INDEX.md` |
| Checkpoints | `CHECKPOINTS.md` |
| Inference spec | `INFERENCE_SPEC.md` |
| v1 benchmark | `benchmark/` (`run_benchmark.py`, `metrics.py` surface-v1, `postprocess.py`, `thresholds.py`, `stats.py`, `endpoint_eval.py`, `symmetric_eval.py`, `adapters/`); tests in `tests/` (`94` passing) |
| RASNet model + loss | `Phase3_Local_Integration/rasnet_model.py`, `rasnet_loss.py` |
| Splits | `Phase3_Local_Integration/splits_final.json` (690 / 94 / 150) |
| v1 trainers | `Q1_Publication_Package/matched_200ep_benchmark/train_*_200ep.py`, `prepare_and_run_nnunet_200ep.py` |
| v1 results | `results/benchmark_runs/`, `results/thresholds.json`, `results/thesis_outputs/`, `results/optionA_v1_symmetric_sw050/` |
| V1 Epoch-200 endpoint & recovery | `recovered_checkpoints/v1_epoch200_20261117_041500Z/`, `results/v1_epoch200_smoke_20261117_041500Z/`, `results/v1_epoch200_comparison_20261117_041500Z/` |
| V1 reporting correction & defense package | `results/v1_reporting_correction_20261117_105000Z/` |
| Phase H | `results/rasnet_fixed_weight_optimization/`, `scripts/run_phaseH_rasnet_optimization.py` |
| Rasnet-v2 | `Rasnet-v2/` |
| Lab PC manual (start here) | `docs-for-next-v2-gen/00_LAB_PC_OPERATOR_MANUAL.md` |
| Plans | `docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md`, `docs-for-next-v2-gen/02_RASNET_V2_FAIRNESS_FIXES_AND_WIN_STRATEGY.md`, `docs-for-next-v2-gen/03_NEXT_STEPS_RUNBOOK.md` |

## 8. Technical pitfalls

- **SimpleITK `.img.nii.gz` quirk:** SimpleITK writes `*.img.nii.gz` as Analyze. Write to a temp `.nii.gz` name and rename (see `export_mask_like`).
- **nnU-Net training-log numbers are not comparable:** `Pseudo dice` is patch-level on foreground-oversampled batches, not comparable to full-volume Dice.
- **Empty predictions:** precision and HD95 are NaN for an empty prediction. Count empties; never let NaN-dropping help a candidate.
- **v1 3D U-Net:** excluded from sliding-window comparisons (128³ resize).
