# RASNet Critical Audit & nnU-Net Gap Analysis

**Date:** 2026-11-19
**Repository:** `H:\Thesis_Trainings` (home PC; `main` == `origin/main` at `33a70b1`)
**Mode:** read-only. This file is the only file written. No training, inference or repo edits were run.
**Data:** the ImageCAS volumes are not on this machine. Claims that need them are marked **LAB-CHECK**.

The v1 checkpoints *are* on this machine: they are tracked in git as split zips (see P0-2). The trained `rasnet_best.pth` was extracted to a scratch folder outside the repo. Its SHA-256 `8d4c25cd…d1b8` matches `CHECKPOINTS.md`, and it was used for weight-level checks.

**Method:**

- I personally read the core model, loss, training script and evaluators, and ran weight-level and CSV-level checks.
- Two read-only sub-audits covered training-pipeline parity and result integrity. Every claim taken from them that drives a recommendation was re-verified against the source (marked *verified*).
- An earlier, separate audit (`RASNET_CRITICAL_AUDIT.md`, untracked, same date) was not used as evidence. Where it overlaps, its claims were checked independently.
- The prompt's `<folder_path>` placeholder was not filled in. I inspected `results/`, `recovered_checkpoints/`, `Q1_Publication_Package/matched_200ep_benchmark/logs/` and `Rasnet-v2/results/` (for isolation only).

---

## 1. Executive Verdict

**Overall rating:**

| Area | Rating |
| :--- | :--- |
| Evaluation and statistics | **Publication-grade** |
| Core RASNet v1 model and training | **Sound but with documented design weaknesses and fairness confounders** |
| Data governance and defense text | **Critical issues found** |

**"Is everything actually great right now, and can I trust my current RASNet results?"**

- **The headline numbers: yes, you can trust them.** The Option A symmetric evaluation reproduces exactly from the per-case files:
  - 2,400 rows = 150 test cases × 4 models × 4 recipes; 0 duplicates, 0 empty predictions, 0 NaN exclusions.
  - The lock was committed in `558e7eb` before test inference. Every Option A manifest has `dirty_code = false` and `sw_batch_size = 8`.
- **What the numbers actually say:**
  - Under the pre-registered primary recipe **R4**, RASNet (4.71 M parameters) reaches Dice **0.8039 ± 0.0691** vs nnU-Net V2 (30.79 M) **0.8059 ± 0.0630**.
  - The mean difference is **−0.0021**, 95 % CI [−0.0088, +0.0030]. RASNet is *not* non-inferior at the −0.005 margin. The median difference is **+0.0002**, and RASNet wins **76/150** cases.
  - RASNet has clearly **higher precision** (+0.0218, Holm p = 7.2e-17). nnU-Net has clearly **higher recall**.
  - Correct framing: **"comparable Dice at 6.5× fewer parameters, with a precision–recall trade-off."** Not "RASNet beats nnU-Net".
- **RASNet beats all its same-pipeline baselines** (SegResNet, V-Net) on Dice under symmetric inference, but with caveats on effect size and confounding (§4.3).
- **The repository is not "all great".** Four issues need action before the defense (§6 P0):
  1. **Two full patient CT volumes are committed to git and pushed to GitHub.**
  2. Several defense and thesis texts **overclaim** what the statistics support.
  3. The v1 **ablation results** come from a pipeline with a train/validation geometry mismatch.
  4. **nnU-Net's primary threshold is clamped at the grid edge** (0.20), which affects the precision comparison. This must be disclosed.
- **"Bug-free":** no silent bug corrupts the reported Option A or Epoch-200 numbers. The core model does have a **verified design flaw**: its attention gates cannot close (§3.1). The v1 training pipeline also gives RASNet **hyper-parameters the baselines did not get** (§3.3).

---

## 2. Repository Map & Structural Health

| Component | Paths | Status | Health |
| :--- | :--- | :--- | :--- |
| Core model + loss | `Phase3_Local_Integration/rasnet_model.py`, `rasnet_loss.py` | Active, frozen v1 | Loads strictly; 4,706,772 parameters (*verified*). Design issues in §3. |
| Splits | `Phase3_Local_Integration/splits_final.json` | Active | 690 / 94 / 150, disjoint, no IDs 851–1000 in train or val (*verified*). |
| v1 trainers | `Q1_Publication_Package/matched_200ep_benchmark/train_*_200ep.py`, `prepare_and_run_nnunet_200ep.py` | Historical (training done) | Not matched across models (§3.3). No training manifests. |
| Active evaluation | `benchmark/` (`run_benchmark.py`, `symmetric_eval.py`, `endpoint_eval.py`, `metrics.py`, `postprocess.py`, `thresholds.py`, `stats.py`, `adapters/`) | Active | Strong: fail-closed checks, native-grid metrics, paired statistics. |
| Orchestration | `scripts/run_optionA_symmetric_v1_eval.py`, `run_v1_epoch200_endpoint_eval.py`, `generate_v1_*` | Active | Functional. The 10 code-review fixes in `docs-for-next-v2-gen/04_CODE_REVIEW_FIXES.md` were **never applied**: `symmetric_eval.py` and the CLI are unchanged since `09b2cd1`. They did not affect Option A (§4.1). |
| Legacy evaluator | `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py`, `Phase3_Local_Integration/eval_utils.py` | **Stale but tracked** | Test-set evaluator with hard-coded τ = 0.6 (l. 447) and flip TTA (l. 436–439). Easy to confuse with the active benchmark. |
| Rasnet-v2 | `Rasnet-v2/` | **Paused** (Stage 2B gate failed, `b3ca1f1`) | Isolated: its own `src/`, configs and tests; nothing in `benchmark/` imports it. No pipeline clutter, but 12 `--config` defaults hard-code `C:\Thesis_RASNET`. |
| Recovered checkpoints | `recovered_checkpoints/v1_epoch200_20261117_041500Z/` | Active (Epoch-200) | Has its own `recover_and_verify.py` with hard-coded `C:\Thesis_RASNET` (l. 1156, 1166). |
| Tests | `tests/` (72 collected, all passing on the lab PC 2026-11-17 per CLAUDE.md) | Active | The 11 review-fix tests from doc 04 are not in the suite. |

**Reproducibility.**

- **Evaluation is reproducible** from committed code and manifests:
  - Option A manifests record code hashes and checkpoint SHA-256s.
  - The Epoch-200 manifests record `git.dirty_code = true` at `b3ca1f1`. However, the code later committed in `8e9b303` matches the manifest's source hashes byte-for-byte (*sub-audit, verified hashes*).
- **v1 training is not reproducible to the bit:**
  - No training manifest exists.
  - Seed 42 is used, with no cuDNN determinism.
  - Resume restores only the weights (§3.3).
  - `run_all_200ep.py:10` still describes RASNet as "Epoch 71 → 200, continued from champion checkpoint", which contradicts the rest of the documentation. Whether RASNet was trained uninterrupted from scratch is **not verifiable here**.

---

## 3. Core RASNet Code & Pipeline Audit

### 3.1 Architecture (`Phase3_Local_Integration/rasnet_model.py`)

**Strengths:**

- A clean subclass of MONAI `SegResNet`.
- The attention gate sits on every decoder skip (`decode`, l. 79–103).
- Deep-supervision heads are active only in `train()`; inference returns plain logits (l. 101–103).
- Parameter overhead over SegResNet is 5,858 parameters (4,706,772 vs 4,700,914), so the gains are not capacity-driven.

| # | Finding | Evidence | Severity |
| :--- | :--- | :--- | :--- |
| A1 | **Attention gates cannot close.** `psi = Conv → GroupNorm(1, 1) → Sigmoid` (l. 28–32). GroupNorm with one group and one channel z-scores the attention map *per sample over all voxels*. So the share of voxels with ψ > 0.5 is fixed by the two learned affine parameters, whatever the patch contains. | Measured on the real `rasnet_best.pth`: γ/β = (1.061, +0.039), (1.153, −0.102), (1.610, −0.219). That predicts ψ > 0.5 on **51.5 %, 46.5 %, 44.6 %** of voxels per gate. A near-empty patch and a tube-structure patch gave **0.490/0.500, 0.475/0.428, 0.408/0.410**, essentially identical. | **Design flaw** (not a crash bug). The gates act as a soft per-patch re-weighting, not a background suppressor. Claims that the gates "suppress background / irrelevant regions" are not supported. The gate output also depends on each sliding-window tile's own statistics. |
| A2 | `W_g` and `W_x` GroupNorm biases are bit-identical. | Measured: `attention_gates.0.W_g.1.bias[:4]` == `W_x.1.bias[:4]` = [−0.0461, +0.0670, −0.0589, −0.0241]. This is mathematically forced: both biases are added before the same ReLU, so they always get identical gradients. | Harmless redundancy. Do not present it as a bug. |
| A3 | Gate and deep-supervision channel widths are hard-coded `[64, 32, 16]` (l. 66–77). | Valid only for `init_filters = 16` and SegResNet's default `blocks_down` (1, 2, 2, 4). | Latent; v1 always uses F = 16. |
| A4 | Deep-supervision comment is wrong. `rasnet_model.py:73` says "Level 0 (1/2 size): 64 channels". | `ds_outputs[0]` is the **1/4-resolution** head (64 ch) and `ds_outputs[1]` is **1/2** (32 ch). `rasnet_loss.py:16-18` states this correctly. | Documentation only. |
| A5 | Deep-supervision weighting runs **opposite to nnU-Net**. `train_rasnet_200ep.py:215`: `loss_main + 0.4·aux(1/4) + 0.2·aux(1/2)`, so the coarser head gets more weight. Logits are trilinearly *upsampled* to label size (l. 211–212). | nnU-Net weights higher resolutions more and *downsamples labels*. | Design choice, not a bug. Plausibly sub-optimal for thin distal vessels. |

### 3.2 Loss (`Phase3_Local_Integration/rasnet_loss.py`)

`StenosisAwareLoss = 0.4·Dice + 0.6·Focal(γ = 2.5)` (l. 43–84).

| # | Finding | Evidence | Consequence |
| :--- | :--- | :--- | :--- |
| L1 | Dice uses `softmax=True` with `include_background=True` (MONAI default; l. 59–64). | The background-class Dice is ≈ 1 almost everywhere. | The foreground Dice term is roughly halved, so the effective foreground Dice weight is about 0.2, not 0.4. |
| L2 | Focal uses MONAI's default **per-channel sigmoid** (`use_softmax=False`) on the same 2-channel logits that Dice soft-maxes (l. 65–68). | Two different probability models are trained on one output. Inference uses softmax (`benchmark/adapters/monai_models.py:100`), consistent with the Dice term. | Not a bug. Inconsistent and unusual; contributes to the calibration shift that put RASNet's v1 τ\* on the 0.30 floor. |
| L3 | `smooth_nr = 0.0` with per-sample Dice. | On the roughly 1/3 of patches with no foreground (pos 2 : neg 1 sampling), the foreground Dice is a constant 1 with zero gradient. | Wasted signal; not a bug. |
| L4 | **α = 0.4 and γ = 2.5 were calibrated with access to test IDs 851–1000** (l. 20–22). | Disclosed in the docstring. | Every v1 test result must stay labelled **retrospective**, as it currently is. |

### 3.3 Training pipeline: parity with the baselines

Sources: the v1 trainers; table from the sub-audit; the starred rows I verified myself.

| | RASNet | SegResNet | V-Net | 3D U-Net | nnU-Net V2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Spacing / patch | 0.5 mm iso, 96³ | same | same, `num_samples = 1` | **128³ resize, whole volume** | [0.5, 0.35, 0.35], 96 × 160 × 160 |
| Intensity | HU [−100, 800] → [0, 1] | same | same | same | CT normalisation (percentile clip + z-score) |
| Augmentation | 3 random flips | same | same | same | full nnU-Net set (rotation, scaling, elastic, noise, blur, brightness, contrast, gamma, low-res simulation, mirroring) |
| Patches per step | 8 | 8 | **2** | 4 | per plans (LAB-CHECK) |
| Optimiser / LR | AdamW, **2e-4** ★ | AdamW, 1e-4 ★ | AdamW, 1e-4 | AdamW, 1e-4 | SGD + Nesterov, 1e-2, poly |
| Schedule | Cosine, **per epoch**, T_max = 200 ★ | same | same | same | poly, 250 iterations/epoch |
| Loss | StenosisAwareLoss + DS | DiceFocal (incl_bg = False, λ 1/2) | same | DiceCE (sigmoid) | Dice + CE + DS |
| Validation cadence | **every 20 epochs** (+ every 50th) ★ → 12 evaluations | every 10 → 20 evaluations | 13 in log | every 10 | every epoch, patch pseudo-Dice |
| Checkpoint selection | best full-volume Val-94 Dice | same | same | same | EMA patch pseudo-Dice |
| Resume | **weights only**; AdamW moments and GradScaler reset, LR replayed (`train_rasnet_200ep.py:167-180`) ★ | same code; logs show resumes | resumed (logs) | — | native |

**Consequences:**

1. **RASNet vs SegResNet is confounded.** It mixes architecture, loss, LR (2e-4 vs 1e-4) and α/γ calibration. The ~5,858 extra parameters cannot be credited alone with the gain.
2. **Checkpoint selection differs per model** (12 vs 20 vs 13 evaluations, a different metric for nnU-Net). "Best" checkpoints are selected on Val-94, the same set used for the Val-94 tables, so those tables are optimistic by different amounts per model. The Epoch-200 final-checkpoint comparison avoids this and is the cleaner Val comparison.
3. **SegResNet and V-Net trajectories were resumed** with reset optimiser state (logs `*_paused_ep86`, `*_saved_state_ep18/19`). This slightly disadvantages those baselines.
4. **No train/validation/inference preprocessing mismatch** in the five main trainers: none applies `Orientationd`, and `benchmark/adapters/monai_models.py:33-41` matches training.
5. **No leakage:** nnU-Net links only train + val (`prepare_and_run_nnunet_200ep.py:50`).

### 3.4 Inference & evaluation (active)

**Option A (`benchmark/symmetric_eval.py`):**

- Identical 0.50-overlap Gaussian blending (σ = 1/8) for all models.
- 8-pass logit-averaged mirroring, matching nnU-Net's predictor.
- τ grid 0.20–0.70 with Top-2 connected components, selected on Val-94 only.
- Native-grid inversion and surface-v1 metrics in mm.

**Epoch-200 (`benchmark/endpoint_eval.py`):**

- Final-epoch checkpoints, Val-94 only (test IDs refused at l. 145–174).
- 1-pass, τ = 0.50, no post-processing.
- Reuses `infer_monai_symmetric` (l. 1038–1048).

**Legacy note:** `benchmark/adapters/monai_models.py:108-111` still implements the old 4-pass TTA. It is used only by the locked v1 Arm B, not by Option A.

---

## 4. Benchmark & Evaluation Integrity Check

### 4.1 Option A (`results/optionA_v1_symmetric_sw050/`): verified

| Check | Result |
| :--- | :--- |
| Rows / duplicates / status | 2,400 / 0 / all `ok`; `n_excluded_nan = 0` in all 68 family rows |
| Lock before test | `558e7eb` (lock, 2026-11-15 01:30Z) precedes `manifest_test` (06:41Z, records `558e7eb` and lock SHA `31cb4845…`) and the results commit `22d4581` |
| Code state | 7 manifests, all `dirty_code = false`, `sw_batch_size = 8`, identical source and checkpoint hashes |
| Recompute | R4 means/SDs recomputed from `test_per_case.csv` match `summary_test.csv` exactly |
| Locked τ (1-pass / 8-pass) | rasnet 0.40 / 0.40; segresnet 0.40 / 0.45; vnet 0.40 / 0.45; **nnunet 0.35 / 0.20 (grid edge)** |

**Primary recipe R4** (`summary_test.csv`, Test-150, retrospective):

| Model (τ) | Params | Dice | Precision | Recall | HD95 (mm) | ASD (mm) | clDice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nnU-Net V2 (0.20) | 30.79 M | **0.8059 ± 0.0630** | 0.8084 | **0.8104** | **8.17** | **1.39** | 0.8744 |
| **RASNet (0.40)** | **4.71 M** | 0.8039 ± 0.0691 | **0.8302** | 0.7861 | 8.93 | 1.57 | **0.8750** |
| SegResNet (0.45) | 4.70 M | 0.7989 ± 0.0692 | 0.8200 | 0.7855 | 9.12 | 1.58 | 0.8728 |
| V-Net (0.45) | 45.60 M | 0.7915 ± 0.0677 | 0.8216 | 0.7711 | 9.50 | 1.62 | 0.8678 |

**R4 paired families** (`paired_stats_families.csv`; mean diff [95 % bootstrap CI], Holm p, wins/ties/losses):

| Comparison | Result |
| :--- | :--- |
| FA, RASNet vs nnU-Net | precision **+0.0218 [+0.0143, +0.0278]**, 7.2e-17, 132/0/18. HD95 +0.76 [−0.18, +1.93], 0.49 (n.s.). ASD +0.18 [−0.07, +0.56], 0.49 (n.s.) |
| NI (Dice) | −0.0021 [−0.0088, +0.0030] → **not non-inferior** at −0.005. Median +0.0002, 76/150 wins |
| FB, RASNet vs SegResNet | Dice **+0.0050 [−0.0024, +0.0121]**, Holm 4.3e-8, 112/0/38. Precision +0.0102 [+0.0024, +0.0179], 3.2e-6 |
| FC, RASNet vs V-Net | Dice +0.0124 [+0.0056, +0.0183], 3.6e-12. Precision +0.0085 [−0.0000, +0.0162] |

### 4.2 Epoch-200 Val-94 endpoint (`results/v1_epoch200_comparison_20261117_041500Z/`): verified

- **Result:** RASNet 0.7792 ± 0.0569 vs nnU-Net 0.7767 ± 0.0562; difference +0.0025, CI [−0.0027, +0.0075], unadjusted Wilcoxon p = 0.2401, 54 W / 40 L; 94/94 `ok` for each model. All CLAUDE.md §5 figures match the CSVs.
- **Correctly labelled** "POST-HOC EXPLORATORY — NOT PREREGISTERED".
- **Caveats:** it was run *after* the Option A test results, on the **untuned** recipe (τ = 0.50, no Top-2 connected components, 1-pass), which is where RASNet looks best. Under Val-tuned 8-pass recipes nnU-Net also leads on Val (Option A Val grid: 0.8098 vs 0.8010).

### 4.3 Fairness caveats that must accompany the numbers

1. **nnU-Net's R4 τ is clamped at the 0.20 grid edge.** Its Val curve is flat (0.8067–0.8098 across the whole grid; 0.20 beat 0.45 by 0.0006). A lower τ gives nnU-Net more recall and less precision, so **RASNet's R4 precision advantage partly reflects nnU-Net's operating point**. Under R3 (both 1-pass; nnU-Net at 0.35) the precision difference is only **+0.0063, CI [−0.0027, +0.0136]**, although Holm p = 5.9e-4. The pre-registration widened the grid to 0.20 to remove a boundary clamp, so this must be **disclosed as a limitation, not re-tuned**.
2. **The FB Dice result is significant by rank test but not by mean CI.** The median is +0.0050 and RASNet wins 112/150, but the bootstrap CI of the mean includes 0, so a few cases go strongly against RASNet. Report it as "RASNet is better in most cases (112/150, Wilcoxon Holm p = 4e-8); the mean gain +0.005 is small and its CI includes 0". Never write "significantly higher mean Dice".
3. **The architecture vs SegResNet comparison is confounded** with loss, LR and α/γ (§3.3). Do not attribute the gain to attention gates or deep supervision alone.
4. **Every test result is retrospective.** IDs 851–1000 were inspected by α/γ calibration, Arms A/B/C and Phase H.
5. **Timing and VRAM:** `timing_summary.csv` used `sw_batch_size` 8 for the MONAI models and 1 for nnU-Net. Full-volume peak VRAM favours nnU-Net (2,243 MB vs RASNet 4,782 MB). Use parameters and GFLOPs for the efficiency claim and keep this caveat on VRAM.
6. **No empty predictions anywhere:** all **5,308** evaluated case-rows have status `ok`: Val Arm A 470, Test Arms A–C 2,250, Option A 2,400, Epoch-200 188 (verified from each per-case CSV). The CLAUDE.md §6 NaN rule was not triggered in any reported result.

---

## 5. The nnU-Net Gap Analysis (Architecture vs. Engineering)

**How big is the gap?** On the primary recipe it is **−0.0021 Dice** (median +0.0002, 76/150 wins), i.e. small and not reliably different from zero. RASNet trades recall for precision.

**Where the gap comes from.** nnU-Net has 6.5× the parameters but also every pipeline advantage below. Given RASNet's near-parity, **almost all of nnU-Net's remaining edge is plausibly engineering** (native in-plane spacing, larger field of view, augmentation, CT normalisation) rather than capacity. This cannot be quantified without retraining; that is exactly what Rasnet-v2's Family 1/2 design was built to measure.

| Factor | RASNet v1 (current) | nnU-Net V2 | Type | Est. impact on thin vessels | Effort to adopt in v1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| In-plane spacing | resampled to **0.5 mm** | native **0.35 mm** | Engineering | **High**: 2× fewer in-plane voxels per vessel cross-section | Retraining |
| Patch / field of view | 96³ (0.88 M voxels) | 96 × 160 × 160 (2.46 M) | Engineering | Medium | Retraining |
| Intensity normalisation | fixed [−100, 800] window | dataset percentiles + z-score | Engineering | Low–medium | Retraining |
| Augmentation | flips only | rotation, scale, elastic, noise, blur, gamma, contrast, low-res | Engineering | **High** (generalisation) | Retraining |
| Loss | 0.4 Dice (incl. background) + 0.6 sigmoid focal γ 2.5 | Dice + CE | Engineering | Medium (calibration) | Retraining |
| Deep supervision | 2 heads, coarse-weighted, logits upsampled | all scales, fine-weighted, labels downsampled | Engineering | Low–medium | Retraining |
| Optimiser / schedule | AdamW 2e-4, cosine per epoch, ≈173 iterations/epoch | SGD 1e-2 poly, 250 iterations/epoch | Engineering | Low–medium | Retraining |
| Checkpoint | best of 12 Val evaluations | EMA pseudo-Dice, every epoch | Engineering | Low | — |
| Sliding window | **matched** (0.5 Gaussian) in Option A | 0.5 Gaussian | Inference | Done | — |
| TTA | **matched** (8-pass) in Option A | 8-pass mirroring | Inference | Done | — |
| Threshold + Top-2 CC | **matched** (Val-locked) | same | Inference | Done | — |
| Ensembling | single checkpoint | single fold (fold 0) here | Inference | Medium if done symmetrically | Low (no retraining) |
| Skip gating | GroupNorm(1, 1) gate that cannot close (A1) | plain skip | **Architecture** | Unknown | Retraining |
| Capacity | 4.71 M, 123 GFLOPs/patch | 30.79 M, 1,119 GFLOPs/patch | **Architecture** | Unknown | — |

**Bottom line:** every inference-level lever has already been matched by Option A. **There is no remaining high-ROI, no-retraining upgrade that would legitimately overturn nnU-Net's lead** except symmetric checkpoint ensembling (P1-1). Every engineering gap above needs retraining. That is **Rasnet-v2's job**: v2 already uses native spacing, 160 × 160 × 96 patches, CT normalisation, the nnU-Net-style augmentation set and a conv-bias gate. Retraining v1 would also break the frozen-science rule (CLAUDE.md §3.2).

---

## 6. Prioritized Action Plan for Core RASNet

### P0 — Must fix before the defense

1. **Patient CT volumes in a pushed repository.** Verified from the zip directory listings:
   - `Phase3_Local_Integration/local_data/CT4/CT4_Stenosis_Annotation_Package.zip` (`image.nii.gz` 79,925,122 B)
   - `…/CT61/CT61_Stenosis_Annotation_Package.zip` (`image.nii.gz` 73,995,099 B)

   They were committed in `97cf8e7` (2026-09-17) and are on `origin` (github.com/DigontaDas/Efficient-3D-Tiled-CNN-Architecture). The folder name suggests your local clinical cohort, not ImageCAS.

   **Actions:**
   - Check the repository's visibility now and make it private if it isn't.
   - Purge the files from history (`git filter-repo`) and force-push.
   - Tell your supervisor or ethics contact.

   These are your decisions; nothing was changed by this audit.
2. **Correct the overclaims in the defense package** (`results/v1_reporting_correction_20261117_105000Z/`):
   - **`DEFENSE_QA_V1.md:38`, "consistent precision advantage":** it omits Arm C (+0.0046, CI [−0.0046, +0.0124], Holm p = 0.16) and R3 (CI crosses 0). Rewrite it as "precision advantage in Arms A/B and Option A R4, recipe- and threshold-dependent (see 4.3.1)".
   - **`DEFENSE_QA_V1.md:39`, gains attributed to attention gates, DS and loss:** not supported (confounded, §3.3; gates cannot close, A1).
   - **`DEFENSE_QA_V1.md:73`:** remove the untested causal claim "because StenosisAwareLoss suppresses…".
   - **`DEFENSE_SLIDE_OUTLINE.md:81` "RASNet beats SegResNet", and README / correction text "Dice was comparable":** replace with the wording in 4.3.2, and "not significantly different (p = 0.24), not shown to be equal" (rule 10).
3. **Ablation results are invalid as trained.**
   - `_Archive_…/ablation/train_ablation_models_200ep.py:89` reuses `cache/segresnet` in a `PersistentDataset` whose cache key ignores transforms (`transform_hash = ""`; MONAI source, sub-audit).
   - Training declares `Orientationd("RAS")` (verified, transform list l. 145–175), but cached items are served without it.
   - Validation applies RAS and drops `CropForegroundd`.
   - Both ablation checkpoints (`checkpoints/ablation_{attngate,deepsup}_best.pth`, 18.9 MB each) are tracked in git.
   - **Do not cite the v1 ablation table** as evidence of component effects. Mark it invalid in the thesis. **LAB-CHECK:** confirm the cache was hit for every case.
4. **Disclose nnU-Net's τ = 0.20 edge clamp** and the R3 sensitivity of the precision result (4.3.1) wherever the R4 precision win is stated.

### P1 — High-ROI upgrades for core RASNet (no retraining, protocol-safe)

1. **Symmetric checkpoint ensembling (exploratory, pre-register first).** Average probabilities of `best` and `last` checkpoints for *every* model (RASNet `rasnet_best` + `rasnet_last`; nnU-Net `checkpoint_best` + `checkpoint_final`; the same for SegResNet and V-Net). Use the same R4 recipe, with τ re-locked on Val-94 for the ensemble. Low effort (`infer_monai_symmetric` + nnU-Net predictor), no retraining. Only valid if all models get it.
2. **Report the operating curve, not one τ.** Precision–recall (or Dice vs τ) curves for RASNet and nnU-Net on Val-94 and Test-150 from the same 8-pass probabilities. This turns 4.3.1 from a weakness into a transparent result: compare precision at matched recall. It needs probabilities, so it means an inference re-run (inference only).
3. **Apply `docs-for-next-v2-gen/04_CODE_REVIEW_FIXES.md` before any further evaluation runs.** It adds the run fingerprint, lock-commit guard, empty-prediction reporting, primary-recipe labelling, crash-safe CSVs, the sidecar requirement and parity numerics. It was validated (62/62 + 14/14) but never applied.
4. **Fix the provenance gaps:**
   - `env_info()` records `nnunetv2: missing` and `cc3d: missing` in Option A manifests (`benchmark/symmetric_eval.py:88-92` reads `__version__`; use `importlib.metadata.version`).
   - Add a `.gitattributes` (`* text=auto eol=lf` for json/csv/py) so recorded SHA-256s reproduce across CRLF and LF checkouts (`locked_recipes.json` is `31cb48…` on disk vs `508b38…` as the committed blob).

### P2 — Polish & cleanup

1. **Move checkpoints out of git.** The ~900 MB `checkpoints_200ep.zip.001–.010` and the ablation `.pth` break rule 7. Use a release asset or LFS, keep the SHA-256 in `CHECKPOINTS.md`, and record that the home PC *does* hold the weights (CLAUDE.md §2 says otherwise).
2. **Retire the legacy evaluator.** Archive or rename `evaluate_matched_200ep_models.py` and `Phase3_Local_Integration/eval_utils.py`, or add a top-of-file "LEGACY — do not use" banner.
3. **Move the dataset root into config.** `Phase3_Local_Integration/dataset_paths.py:24-30` hard-codes data roots (rule 1). `scripts/run_phaseH_rasnet_optimization.py:43` and `scripts/backfill_phaseH_val94_and_report.py:31` hard-code `ROOT = C:\Thesis_RASNET`.
4. **Fix doc errors:**
   - `rasnet_model.py:73` deep-supervision level comment;
   - the `run_all_200ep.py:10` RASNet description;
   - `CHECKPOINTS.md:40` (`nnUNetTrainer_200epochs` is a stock nnunetv2 variant, not registered dynamically);
   - `nnunet_200ep_log.csv` duplicates epochs 51/101/151 and its `val_dice` is patch pseudo-Dice.
5. **Re-align GEMINI.md with CLAUDE.md:** GEMINI.md §3 claims to be "identical" but drops the "retrospective" bullet and the manifest detail, and adds rule 11.
6. **Document training non-determinism and the weights-only resume** (AdamW and GradScaler reset) as a limitation of v1 training.

### Explicitly deferred to Rasnet-v2 (do not touch v1)

- Replacing the GroupNorm(1, 1) gate (v2 `gate_mode = "conv_bias"`).
- Re-weighting or re-designing deep supervision.
- The loss redesign (foreground-only Dice, consistent softmax).
- Native 0.35 mm spacing, larger patches, CT normalisation and nnU-Net-style augmentation.
- Matched optimiser and LR per model, and multi-seed training.
- The ablation study, re-run correctly with 3 seeds (already in Amendment A1).
- **Note:** v2's Stage 2B gates failed (`Rasnet-v2/results/pilots_val94/pilot_lock_decision.md`). Do not override them; any continuation needs a written amendment first.

---

## 7. Final Go / No-Go Checklist

- [ ] Patient CT zips purged from history; repository visibility checked; supervisor or ethics contact informed (P0-1)
- [ ] `DEFENSE_QA_V1.md` l. 38, 39 and 73, `DEFENSE_SLIDE_OUTLINE.md:81`, README "comparable" wording corrected (P0-2)
- [ ] v1 ablation table marked invalid in thesis and defense (P0-3); LAB-CHECK of the ablation cache recorded
- [ ] nnU-Net τ = 0.20 clamp and R3 precision sensitivity disclosed next to every precision claim (P0-4)
- [ ] Headline wording fixed as: "Dice comparable to nnU-Net V2 (−0.0021, 95 % CI [−0.0088, +0.0030]; not non-inferior at −0.005; median +0.0002, 76/150 wins) with 6.5× fewer parameters and 9.1× fewer GFLOPs per patch; higher precision, lower recall"
- [ ] RASNet vs SegResNet stated with the rank-test and mean-CI distinction and the loss/LR confounder (4.3.2–4.3.3)
- [ ] All v1 test results labelled retrospective (L4)
- [ ] Attention-gate wording limited to "soft spatial re-weighting" (A1)
- [ ] Doc 04 code-review fixes applied before any new evaluation run (P1-3)
- [ ] Optional: symmetric checkpoint ensemble and operating-curve analysis pre-registered before running (P1-1, P1-2)
- [ ] Checkpoints moved out of git, with SHA-256s preserved (P2-1)
- [ ] Rasnet-v2 left paused until a written amendment addresses the failed Stage 2B gates

**Verdict:** **GO** on the evaluated numbers and the efficiency story, **once P0-1 to P0-4 are done**. **NO-GO** on any claim that RASNet beats nnU-Net in Dice, or that attention gates or the loss *cause* the gains.
