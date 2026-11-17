# Thesis Defense Q&A (`DEFENSE_QA_V1.md`)

**Date (UTC):** `2026-11-17`  
**Central Defensible Claim (R7):**  
> *"RASNet is a compact coronary segmentation model with observed precision advantages at specified operating points and competitive full-volume overlap. The epoch-200 comparison did not establish Dice superiority or equivalence; nnU-Net retained recall advantages."*

---

## Part 1: Core Committee Questions (R7 Questions 1–8)

### 1. Why was nnU-Net's logged curve higher?
- **Answer:** The logged validation curves in `fig3` reflect **fundamentally different metrics, spatial grids, sampling rules, and aggregation protocols**:
  - `nnU-Net V2` (`nnUNetTrainer_200epochs`) logged `Pseudo dice` (`mean_fg_dice`) every epoch (`200` evaluations) over `50` validation batches (`batch_size = 2`) of **`96 × 160 × 160` patches** with **`33.3%` forced foreground oversampling**, aggregated via **batch-pooled** `2*TP / (2*TP + FP + FN)`.
  - `RASNet` (`12` evals), `SegResNet` (`20` evals), and `V-Net` (`13` evals) logged **case-mean full-volume** sliding-window Dice across all `N = 94` `Val94` scans on the `0.5 mm` isotropic grid (`96³`, `overlap = 0.25`); `3D U-Net` (`21` evals) evaluated full volumes downsampled to `128³`.
  - Additionally, historical `fig3` mislabeled nnU-Net's curve as `EMA Pseudo-Dice`: SVG verification (`r = 0.99999978`) proves it plotted **raw unsmoothed `mean_fg_dice`** (peaking at epoch `189` = `0.8376`), whereas `checkpoint_best.pth` was selected on `ema_fg_dice` (`_best_ema = 0.816239` at `current_epoch = 199`).
- **Required Non-Causal Statement:** *"The logged metrics use different sampling, aggregation and spatial evaluation protocols and therefore cannot establish comparative full-volume performance."*
- **Sources:** `results/v1_reporting_correction_20261117_105000Z/fig3_corrected_training_and_val_convergence_200ep.{png,svg}`, `Q1_Publication_Package/matched_200ep_benchmark/logs/*_200ep_log.csv`, `nnUNetTrainer.py`.

### 2. What did the same-epoch full-volume evaluation establish?
- **Answer:** Evaluating both recovered `final_epoch200` checkpoints (`rasnet_last.pth` via script/log/archive provenance; `nnunet_last.pt` with embedded `current_epoch = 200`) on the identical `N = 94` `Val94` native-grid scans under a symmetric single-pass protocol (`0.50` Gaussian SW, `τ = 0.50`, no CC; `0` empty predictions, `188/188` verified mask headers) established:
  - **No significant Dice difference (and neither superiority nor equivalence):** RASNet **`0.7792 ± 0.0569`** (median `0.7921`) vs. nnU-Net **`0.7767 ± 0.0562`** (median `0.7824`); paired difference `+0.002473` (`95%` bootstrap CI `[-0.002747, +0.007545]`, `54W / 0T / 40L`, exploratory unadjusted Wilcoxon `p = 0.2401`, post-hoc Holm-8 `p = 0.9605`).
  - **Precision vs. Recall divergence at `τ = 0.50`:** All 8 endpoint p-values are **exploratory / unadjusted**; under post-hoc exploratory Holm-8 (`K = 8`), only two metrics remain `< 0.05`: RASNet achieved higher **Precision** (`0.7586` vs. `0.7428`, diff `+0.0158`, `95%` CI `[+0.0072, +0.0245]`, `66W/0T/28L`, unadj `p = 1.07e-4`, Holm-8 `p = 7.47e-4`), while nnU-Net achieved higher **Recall** (`0.8203` vs. `0.8065`, diff `-0.0138`, `95%` CI `[-0.0175, -0.0098]`, `16W/0T/78L`, unadj `p = 4.85e-10`, Holm-8 `p = 3.88e-9`).
  - **Boundary/topology metrics did not establish superiority:** `HD95` (`p = 0.6742`), `ASD` (`p = 0.3372`), `clDice` (`unadj p = 0.0352`, `95%` CI `[-0.0006, +0.0135]`, Holm-8 `p = 0.1760`), and `centerline_recall` (`unadj p = 0.0092`, `95%` CI `[-0.0124, -0.0025]`, Holm-8 `p = 0.0551`).
- **Sources:** `results/v1_reporting_correction_20261117_105000Z/table_epoch200_val94_summary_and_paired_corrected.{csv,md}`, `table_epoch200_val94_posthoc_holm8_exploratory.{csv,md}`, `fig_v1_epoch200_val94_endpoint_corrected.{png,svg}`.

### 3. Why does this not prove RASNet won throughout training?
- **Answer:** Four strict reasons:
  1. **Single endpoint only:** The epoch-200 comparison evaluates only `final_epoch200` checkpoints (`rasnet_last.pth` and `nnunet_last.pt`). No intermediate checkpoints (`epochs 50, 100, 150`) exist in the recovered V1 archive to evaluate full-volume trajectories across earlier training epochs.
  2. **Non-significance is neither superiority nor equivalence:** `p = 0.2401` (`95%` CI `[-0.0027, +0.0075]`) fails to reject the null and does not establish statistical equivalence.
  3. **RASNet checkpoint metadata qualification:** `rasnet_last.pth` lacks embedded epoch metadata (`embedded_epoch_metadata_present = False`), relying on `train_rasnet_200ep.py:268`, `rasnet_200ep_log.csv`, LR alignment, and archive timestamps.
  4. **Development cohort scope:** `Val94` was used during V1 development, and when best checkpoints are compared under Option A `R4` (`Test150`), nnU-Net has slightly higher Dice (`0.8059` vs. `0.8039`).
- **Sources:** `table_checkpoint_provenance_verified.md`, `recovered_checkpoints/v1_epoch200_20261117_041500Z/PROVENANCE_REPORT.md`.

### 4. What are RASNet's supported contributions?
- **Answer:** Four verified, scope-bounded contributions:
  1. **Compact Parameter & Patch-Compute Footprint:** `4,706,772` parameters (`4.707M`, `18.00 MiB` checkpoint)—**`6.54×` (`84.7%`) fewer parameters** than `3d_fullres` `nnU-Net V2` (`30,785,994`, `30.786M`) and `9.69×` fewer than `V-Net` (`45.599M`), with `123.39` vs. `1,119.34` native (`9.07×`) or `402.96` equal-`96³` (`3.27×`) GFLOPs per patch (`Table 4`).
  2. **Faster End-to-End Inference Under Option A Protocol:** On `n = 10` Val cases (RTX 4080 SUPER, `0.50` Gauss SW, `sw_batch_size = 8` MONAI vs `1` nnU-Net), median **`5.40s` vs. `29.84s`** (`1-pass`) and **`22.95s` vs. `74.37s`** (`TTA-8`), while noting higher MONAI full-volume peak VRAM (`4,782–5,134 MB` vs. `2,243–2,261 MB` due to `sw_batch_size = 8`).
  3. **Consistent Voxel Precision Advantage at Specified Operating Points:** Significant precision gains over `nnU-Net V2` in Epoch-200 `Val94` (`+0.0158`, Holm-8 `p = 7.47e-4`), Option A `R4` (`0.8302` vs. `0.8084`, `+0.0218`, Holm `p = 7.23e-17`, `132W/0T/18L`), and Retrospective Arm A/B (`+0.0642` / `+0.0602`, Holm `p < 1e-20`).
  4. **Outperforming Matched MONAI Baselines (`SegResNet` & `V-Net`):** Under symmetric Option A `R4`, adding `AttentionGate3D` (`+5,858` params), deep supervision, and `StenosisAwareLoss` improved Dice over `SegResNet` (`0.8039` vs. `0.7989`, `+0.0050`, Holm `p = 4.27e-8`) and `V-Net` (`0.7915`, `+0.0124`, Holm `p = 3.58e-12`), plus Precision (`+0.0102` / `+0.0085`).
- **Sources:** `results/optionA_v1_symmetric_sw050/OPTION_A_RESULTS.md`, `results/thesis_outputs/Table4_computational_efficiency_rtx4080super.csv`, `table_v1_efficiency_and_timing_verified.md`.

### 5. Where does nnU-Net perform better?
- **Answer:** `nnU-Net V2` holds clear, verified advantages in:
  1. **Voxel Recall (all primary/symmetric comparisons):** Epoch-200 `Val94` (`0.8203` vs. `0.8065`, `+0.0138`, Holm-8 `p = 3.88e-9`, `78W/0T/16L`); Option A `R4` (`0.8104` vs. `0.7861`, `+0.0244`, Holm `p = 2.40e-17`, `133W/0T/17L`); Retrospective Test150 Arm A (`0.8106` vs. `0.7521`) and Arm C (`0.7977` vs. `0.7649`).
  2. **Centerline Recall (unadjusted / development):** Epoch-200 `Val94` (`0.9142` vs. `0.9068`, `+0.0074`, `95%` CI `[+0.0025, +0.0124]`, unadj `p = 0.0092`, Holm-8 `p = 0.0551`).
  3. **Post-Processed Overlap & Boundary Distances (`Arm C` / `Option A R4`):** Under Retrospective `Arm C`, nnU-Net led on Dice (`0.8055` vs. `0.7899`), `HD95` (`8.26` vs. `9.94 mm`, Holm `p = 4.2e-4`), and `ASD` (`1.395` vs. `1.671 mm`, Holm `p = 1.5e-4`). Under Option A `R4`, nnU-Net achieved `0.8059` vs. `0.8039` Dice (causing RASNet to fail the `-0.005` non-inferiority margin) with lower full-volume peak VRAM (`2,243–2,261 MB` vs. `4,782–5,134 MB`).
- **Sources:** `OPTION_A_RESULTS.md`, `Table2_test_150_arms_A_B_C_summary.md`, `table_epoch200_val94_posthoc_holm8_exploratory.md`, `timing_summary.csv`.

### 6. Why are best-checkpoint and final-checkpoint results separate?
- **Answer:** Combining them violates protocol and checkpoint integrity because they differ in **checkpoint selection epoch**, **inference configuration**, and **post-processing**:
  - **Different epochs:** `final_epoch200` uses epoch `200` for both models (`rasnet_last.pth`, `nnunet_last.pt`), whereas `best` uses `rasnet_best.pth` (epoch `160`) vs. `nnunet_best.pt` (`current_epoch = 199` selected by `_best_ema = 0.816239`, distinct from raw `Pseudo dice` peak at epoch `189`).
  - **Different SW/TTA/CC settings:** Regime A (`final_epoch200`) used symmetric `0.50` Gaussian 1-pass at `τ = 0.50` without CC; Regime B1 (`val_full_94` Arm A) used **asymmetric** SW (`0.25` constant vs. `0.50` Gaussian); Regime B2 (Option A Val94) added Top-2 CC; Regimes C/D evaluated `Test150` with tuned `τ*` and Top-2 CC (plus TTA-8 in `R4`).
  - Mixing them creates phantom trajectories across unmatched protocols.
- **Sources:** `table_v1_four_regimes_comparison.{csv,md}`, `table_checkpoint_provenance_verified.{csv,md}`.

### 7. Why are the Test150 results retrospective?
- **Answer:** Although `IDs 851–1000` (`N = 150`) are disjoint from the `690` training and `94` validation IDs in `splits_final.json`, earlier V1 developmental sweeps—specifically choosing threshold `τ = 0.60`, calibrating `StenosisAwareLoss` (`α = 0.4, γ = 2.5`), and running Phase H fixed-weight comparisons—had already inspected `IDs 851–1000`. As documented in `BENCHMARK_PREREGISTRATION.md`, `Test150` cannot be treated as a blinded, untouched confirmatory test set; it is a **retrospective reassessment** under frozen protocols.
- **Sources:** `BENCHMARK_PREREGISTRATION.md`, `THESIS_METHODS_RESULTS_AND_LIMITATIONS.md:13,75-76`, `GEMINI.md` §1.

### 8. What remains uncertain and what independent evaluation is still needed?
- **Answer:** Five explicit gaps remain:
  1. **No untouched multi-center external CTA evaluation:** `unseen_66` (leaked into legacy 850-case nnU-Net `dataset.json`, `D4`/`I2` audit issues) and the local hospital QCA cohort (`N = 21/32`, unfixed `S1–S6`/`E5`/`D2`/`I5` defects) are excluded. Independent external datasets across scanners/sites are needed.
  2. **V1 training protocol confounders:** V1 MONAI models (`0.5 mm`, `96³`, `AdamW`) and `nnU-Net V2` (`[0.5, 0.3496, 0.3496]`, `96×160×160`, `SGD`) differed in training setup; thus V1 cannot isolate architectural causality (attention gates/capacity).
  3. **Controlled multi-seed study (`Rasnet-v2`) still needed:** Single-seed V1 failed primary Option A `R4` Dice non-inferiority (`-0.0021`, CI `[-0.0088, +0.0030]`); `Rasnet-v2` Stage 2A/2B pilots (`2026-11-16`) ended in `FALLBACK` (no candidate passed both gates), and multi-seed Stage 3–5 confirmatory runs have not been executed.
  4. **Missing intermediate V1 checkpoints (`ep 50/100/150`) and embedded RASNet epoch metadata:** Full-volume dynamics across V1 training remain unobserved.
  5. **Clinical downstream utility unvalidated:** Lumen segmentation overlap (`Dice`/`Precision`) does not prove diagnostic CAD-RADS or invasive `FFR` / `FFR_CT` accuracy.

---

## Part 2: Key Committee Follow-Up Questions

### Follow-up A: Why did Option A primary `R4` fail Dice non-inferiority (`NI = False`) when `R1` and `R2` passed (`NI = True`)?
- In raw unpruned recipes (`R1` 1-pass `τ = 0.50`, `R2` TTA-8 `τ = 0.50`), `RASNet` outperformed `nnU-Net` on Dice (`+0.0101` and `+0.0103`) because `StenosisAwareLoss` suppresses background false positives without post-processing.
- Once symmetric validation-tuned thresholding (`τ*`) and 26-connected **Top-2 CC pruning** were applied in `R3` and primary `R4`, Top-2 CC eliminated nnU-Net's disconnected false-positive islands (`+0.0299` Dice boost from `R2` `0.7760` to `R4` `0.8059`, vs. `+0.0176` for RASNet from `0.7863` to `0.8039`), reversing the point estimate to `-0.0021` (`95%` CI `[-0.0088, +0.0030]`), crossing the pre-specified `-0.005` margin (`non_inferior = False`). `R1–R3` remain strictly secondary. *(Source: `noninferiority_rasnet_vs_nnunet_dice.csv`, `OPTION_A_RESULTS.md`.)*

### Follow-up B: Is RASNet the smallest model in the benchmark? Why is SegResNet smaller?
- **No.** Exact verified parameter counts are: **`SegResNet` (`4,700,914`) < `RASNet` (`4,706,772`) < `3D U-Net` (`4,805,534`) < `nnU-Net V2` (`30,785,994`) < `V-Net` (`45,599,452`)**. RASNet wraps `SegResNet(init_filters=16)` and adds `AttentionGate3D` + deep supervision heads (`+5,858` parameters, `+0.12%`). Thus, RASNet is `6.54×` (`84.7%`) smaller than this `30.786M` `nnU-Net V2`, **not** `75%`/`4×` smaller and **not** smaller than every baseline. *(Source: `table_v1_efficiency_and_timing_verified.md`, `rasnet_model.py`.)*

### Follow-up C: Why are `manifest.json` `rasnet = 0.0s` and legacy `~1.85s` excluded from runtime claims?
- **`0.0s`:** In `results/v1_epoch200_comparison_20261117_041500Z/manifest.json:318`, `benchmark/endpoint_eval.py:1020-1066` skipped all `94` RASNet cases because `--resume` found `per_case_metrics_rasnet_epoch200.csv` and `predictions/rasnet/` already completed (`rasnet_todo == []`). It is a resume-skip counter, not inference latency (the 2-case smoke run took `42.092s` for RASNet vs `58.821s` for nnU-Net).
- **`~1.85s`:** Traced to hardcoded synthetic estimates (`full_vol_inf_time = 1.85`) in archived `03_efficiency_benchmark.py:250-259`, which also benchmarked a synthetic `16.544M` `DynUNet` instead of `30.786M` `PlainConvUNet`. Valid full-volume timing comes strictly from Option A `timing_summary.csv` (`5.40s` 1-pass / `22.95s` TTA-8 on `n = 10` Val cases, RTX 4080 SUPER).

### Follow-up D: Why were `unseen_66` and the hospital `N = 21/32` / HeartFlow `FFR_CT` claims removed?
- **`unseen_66`:** All `66` cases (`[3..199]`) were present in the legacy `850`-case nnU-Net `dataset.json` (`LAB_CHECKS.md:295-330`) and suffered from unverified cross-dataset provenance (`D4`) and broken cache/default `0.45` threshold (`I2`).
- **Hospital `N = 21/32` & HeartFlow `FFR_CT`:** Excluded due to unfixed `AUDIT_FINDINGS.md` defects (`S1` diagnosis leakage, `S2` `10.0% LAD` fallback, `S5` area-vs-diameter CAD-RADS mismatch, `E5` parser zeroing, `D2` heuristic DICOM selection, `I5` Top-2 branch deletion) and missing segmentation mask provenance. Furthermore, anatomical CAD-RADS grading on `21` local scans cannot establish superiority or equivalence to physiological `FFR_CT` (`Cook et al., JAMA Cardiology 2017`).

### Follow-up E: Why did `clDice` (`p = 0.0352`) fail post-hoc Holm-8 (`p = 0.1760`), and why do its Wilcoxon `p` and bootstrap CI appear to disagree?
- **Different statistical targets:** The percentile bootstrap `95%` CI (`[-0.000589, +0.013532]`) targets the **mean paired difference**, which is sensitive to the magnitude of skewed tails; the two-sided Wilcoxon signed-rank test (`p = 0.0352`) targets **paired rank location / pseudomedian**. Because `clDice` is an unadjusted exploratory test among 8 metrics on `Val94`, applying post-hoc Holm-Bonferroni (`K = 8`) yields `p = 0.1760` (`table_epoch200_val94_posthoc_holm8_exploratory.md`), confirming that `clDice` is **not** confirmed topological superiority.
