# V1 Reporting Correction Report (`v1_reporting_correction_20261117_105000Z`)

**Date (UTC):** `2026-11-17`  
**Repository:** `C:\Thesis_RASNET` (baseline commit `8e9b303`)  
**Correction Package Directory:** `results/v1_reporting_correction_20261117_105000Z/`  
**External Manuscript / Slide Integration Status:** **`OUTSTANDING`** (no `.tex`, `.docx`, `.pptx`, or `.key` source exists in `C:\Thesis_RASNET`; ready-to-paste replacement text and slide outline are provided in `THESIS_REPLACEMENT_TEXT.md` and `DEFENSE_SLIDE_OUTLINE.md`).

---

## 1. Executive Summary & Defensible Central Claim (R7)

### Central Claim
> **"RASNet is a compact coronary segmentation model with observed precision advantages at specified operating points and competitive full-volume overlap. The epoch-200 comparison did not establish Dice superiority or equivalence; nnU-Net retained recall advantages."**

This reporting-correction package audits and reconciles all completed V1 ImageCAS coronary artery segmentation artifacts across training logs, checkpoints, the Val94 final epoch-200 endpoint evaluation (`20261117_041500Z`), the best-checkpoint Val94 runs (`val_full_94` Arm A and Option A grid), the retrospective Test150 3-arm benchmark (`test_retrospective_150`), and the symmetric 4-model Option A comparison (`optionA_v1_symmetric_sw050`).

Key verified conclusions:
1. **Training curves vs. full-volume evaluation (`R4`):** In the historical convergence figure (`results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.{png,svg}`), nnU-Net's plotted validation curve is **raw unsmoothed `Pseudo dice`** (`mean_fg_dice`, `r = 0.99999978`, peaking at `0.8376` at epoch `189`), not EMA (`ema_fg_dice`), and SegResNet/V-Net were mislabeled `DiceCE` instead of `DiceFocalLoss(lambda_dice=1.0, lambda_focal=2.0)`. Causal phrasing attributing nnU-Net's higher curve solely to patch sampling has been removed and replaced by: *"The logged metrics use different sampling, aggregation and spatial evaluation protocols and therefore cannot establish comparative full-volume performance."* Corrected figure: `fig3_corrected_training_and_val_convergence_200ep.{png,svg}`.
2. **Val94 Epoch-200 endpoint (`R5`):** Across `N = 94` exact matched Val94 cases (`0` empty predictions, `0` NaN exclusions, `188/188` verified mask headers), final epoch-200 RASNet (`0.7792 ± 0.0569`) and nnU-Net (`0.7767 ± 0.0562`) show a paired Dice difference of `+0.002473` (`95%` bootstrap CI `[-0.002747, +0.007545]`, `54W / 0T / 40L`, exploratory unadjusted Wilcoxon `p = 0.2401`). This establishes **neither Dice superiority nor Dice equivalence**. All 8 endpoint p-values are **exploratory / unadjusted**; under a separate **post-hoc exploratory (not preregistered)** Holm-Bonferroni (`K = 8`) check, RASNet Precision (`+0.0158`, `Holm-8 p = 7.47e-4`) and nnU-Net Recall (`-0.0138`, `Holm-8 p = 3.88e-9`) remain `< 0.05`, whereas `clDice` (`unadj p = 0.0352`, `Holm-8 p = 0.1760`) and `centerline_recall` (`unadj p = 0.0092`, `Holm-8 p = 0.0551`) do not.
3. **Symmetric Option A (`R6`):** Across the 4 sliding-window models (`3dunet` excluded), under the preregistered primary recipe `R4_tuned_tta8_top2` on retrospective Test150 (`IDs 851–1000`), nnU-Net achieved `0.8059 ± 0.0630` Dice vs. RASNet `0.8039 ± 0.0691` (mean diff `-0.0021`, `95%` CI `[-0.0088, +0.0030]`, **non-inferiority `False` at `-0.005` margin**, `superior = False`). RASNet won primary `FA` Precision (`+0.0218`, Holm `p = 7.23e-17`) and secondary `FB`/`FC` Dice over SegResNet and V-Net, while nnU-Net retained higher Recall (`+0.0244`, Holm `p = 2.40e-17`).
4. **Efficiency & scope discipline (`R6`):** Exact verified parameter counts follow `SegResNet (4.701M) < RASNet (4.707M) < 3D U-Net (4.806M) < nnU-Net (30.786M) < V-Net (45.599M)`. Valid timing separates single-patch `Table 4` metrics (`123.39` vs `1,119.34` native / `402.96` equal `96³` GFLOPs) from Option A `n = 10` full-volume runtime (`5.40s` 1-pass / `22.95s` TTA-8 vs `29.84s` / `74.37s` for nnU-Net, with higher MONAI VRAM `4,782–5,134 MB` vs `2,243–2,261 MB` due to `sw_batch_size=8`). Invalid `0.0s` (`manifest.json` `--resume` skip) and `~1.85s` (legacy hardcoded estimate) timings are excluded, as are `unseen_66` and hospital `N = 21/32` QCA / HeartFlow claims.

---

## 2. Complete Issue Matrix (R2–R7)

### 2.1 Reporting, Provenance, Statistical, and Documentation Issues

| ID | Source File / Line(s) | Direct Evidence Observed | Correction Applied / Handling | Verification Method | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V1R-01** | `CHECKPOINTS.md:9-13` | Lists `400` keys for RASNet, `153` SegResNet, `322` V-Net, `90` + `optimizer_state_dict` for 3D U-Net, `402` for nnU-Net. | Actual `torch.load` key counts verified: `114` (RASNet), `83` (SegResNet), `164` (V-Net), `63` without `optimizer_state_dict` (3D U-Net), `292` (nnU-Net); recorded in `table_checkpoint_provenance_verified.{csv,md}`. Pre-edit backup saved (`22624d08...`). | `torch.load(..., map_location='cpu')` + `len(state_dict)` | **Verified & tabulated; root doc backup captured** |
| **V1R-02** | `CHECKPOINTS.md:13,31,43`; `CURRENT_EVIDENCE_INDEX.md:25` | Lists nnU-Net plans `[128, 192, 192]` and `[0.5, 0.359375, 0.359375]` and `Best Epoch = 189` without distinguishing EMA selection. | Verified `plans.json` / `init_args['plans']`: `[96, 160, 160]` and `[0.5, 0.349609375, 0.349609375]`; `checkpoint_best.pth` records `current_epoch=199` (`_best_ema=0.816239`) while raw `mean_fg_dice` peaked at epoch `189` (`0.8376`). | `plans.json` SHA-256 `c67af4aa...` + `nnUNetTrainer.py` inspection | **Verified & tabulated; root doc backup captured** |
| **V1R-03** | `CHECKPOINTS.md:1-49`; `CURRENT_EVIDENCE_INDEX.md:20-28` | Missing `final_epoch200` checkpoints (`rasnet_last.pth`, `nnunet_last.pt`), Option A (`optionA_v1_symmetric_sw050`), Epoch-200 comparison (`v1_epoch200_comparison_20261117_041500Z`), and reporting correction dir. | Verified SHA-256 & provenance of both `final_epoch200` checkpoints in `table_checkpoint_provenance_verified.{csv,md}` (`rasnet_last.pth` `embedded_epoch_metadata_present=False`; `nnunet_last.pt` `current_epoch=200`). | `torch.load` + SHA-256 + `rasnet_200ep_log.csv` + ZIP timestamps | **Verified & tabulated** |
| **V1R-04** | `CHECKPOINTS.md:22` | Lists excluded `66` IDs as `{4, 15, 31, ..., 843}`. | `splits_final.json` excludes `66` IDs all within `1..200` (`[3..199]`); IDs `15` and `843` are actually in `train`. | Set difference on `splits_final.json` (`efe3f06c...`) | **Verified (`CLAIM_TO_EVIDENCE_MATRIX.csv`)** |
| **V1R-05** | `CHECKPOINTS.md:41` | Lists V-Net validation cadence as `Every 20 epochs`. | `logs/vnet_200ep_log.csv` has `13` evals (`[10, 20, 40, 50, ..., 200]`) due to ep 10 before ep 18–19 auto-resume; RASNet has `12` (`ep % 20 == 0 or ep % 50 == 0`). | Log CSV inspection (`cbcb31b2...`, `d3ad4d06...`) | **Verified & plotted in corrected Fig 3** |
| **V1R-06** | `results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.{png,svg}` (**PROTECTED**) | Subtitle says `EMA Pseudo-Dice`; Panel A says `SegResNet (DiceCE)`, `V-Net (DiceCE)`; Panel B conflates full-vol vs patch & clips y-axis at `0.50`/y-label. | Preserved protected original untouched. Generated `fig3_corrected_training_and_val_convergence_200ep.{png,svg}` with `DiceFocalLoss(1.0, 2.0)`, unshifted dual loss axes, explicit raw `Pseudo dice` (`r=0.99999978`, ep 189 star) vs `ema_fg_dice` (ep 199 diamond), full `0–0.96` y-axis, all 5 protocols, and required non-causal footer. | SVG path correlation (`r=0.99999978`), `train_*_200ep.py`, visual PNG check | **Protected original preserved; corrected Fig 3 generated & verified** |
| **V1R-07** | `results/v1_epoch200_comparison_20261117_041500Z/THESIS_CURVE_AND_ENDPOINT_CORRECTION.md:35` (**PROTECTED**) & `scripts/generate_v1_epoch200_report_and_figures.py:443` | Stated nnU-Net's higher `Pseudo dice` `"is an artifact of patch sampling rather than whole-scan generalization."` | Updated `scripts/generate_v1_epoch200_report_and_figures.py` with R4's exact non-causal sentence (backup `9cfb517c...` preserved); preserved historical `.md` untouched and recorded in `HISTORICAL_ERRATA.md` & `THESIS_REPLACEMENT_TEXT.md`. | `pytest tests/test_v1_reporting_corrections.py` (`72 passed`) | **Script fixed & tested; historical `.md` in errata** |
| **V1R-08** | `THESIS_CURVE_AND_ENDPOINT_CORRECTION.md:24-25`; `ENDPOINT_EVALUATION_REPORT.md:21`; `LAB_RUN_PLAN.md:198` | Listed 8 Val94 endpoint p-values without marking them exploratory/unadjusted, headlined `clDice p = 0.0352`, and omitted Wilcoxon vs bootstrap CI explanation. | Updated `generate_v1_epoch200_report_and_figures.py`; generated `table_epoch200_val94_summary_and_paired_corrected.{csv,md}`, `table_epoch200_val94_posthoc_holm8_exploratory.{csv,md}`, and `fig_v1_epoch200_val94_endpoint_corrected.{png,svg}`. | Recomputed via `benchmark.stats` (`<= 3.55e-15` diff) & `pytest` | **Corrected tables/figures generated & verified** |
| **V1R-09** | `results/v1_epoch200_comparison_20261117_041500Z/manifest.json:318` & `ENDPOINT_EVALUATION_REPORT.md:8` (**PROTECTED**) | Records `timing_seconds.rasnet: 0.0` (due to `--resume` skipping 94 completed cases in `benchmark/endpoint_eval.py:1020-1066`). | Updated `generate_v1_epoch200_report_and_figures.py` timing note; recorded in `table_v1_efficiency_and_timing_verified.{csv,md}` & `HISTORICAL_ERRATA.md` that `0.0s` is a resume-skip artifact, never latency. | Code inspection (`endpoint_eval.py:1020-1066`) & smoke manifest (`42.092s`) | **Clarified & excluded from latency claims** |
| **V1R-10** | `results/audits/2026-11-10_verification_audit/checkpoint_verification.csv:6` (**PROTECTED**) | Claims `best EMA pseudo-Dice=0.8376 at epoch 189 (current_epoch=190 in checkpoint)` and `duplicate log lines at resume boundaries 51/101/151`. | Preserved untouched; documented in `HISTORICAL_ERRATA.md` (`0.8376` = raw `mean_fg_dice` at ep 189; `current_epoch=199` under `_best_ema=0.8162`; CSV rows `51,101,151,201` are empty, not duplicates). | `torch.load('nnunet_best.pt')` + `nnunet_200ep_log.csv` check | **Preserved untouched; logged in `HISTORICAL_ERRATA.md`** |
| **V1R-11** | `results/thesis_outputs/Table1_validation_94_and_thresholds.{csv,md}:6` (**PROTECTED**) | Lists `Best Epoch = 189` alongside `0.8376 (Patch Pseudo-Dice (EMA=0.8162))`. | Preserved untouched; logged in `HISTORICAL_ERRATA.md` clarifying ep `189` = raw peak (`0.8376`) vs ep `199` = EMA checkpoint (`0.816239`). | `nnunet_last.pt['logging']` verification | **Preserved untouched; logged in `HISTORICAL_ERRATA.md`** |
| **V1R-12** | `results/thesis_outputs/THESIS_METHODS_RESULTS_AND_LIMITATIONS.md:26,50` (**PROTECTED**) | States `AdamW (lr = 10^-3)` for RASNet/SegResNet/V-Net; line 50 says `"0.8376 overestimates whole-scan precision"` (pseudo-Dice is not precision); predates Option A & Epoch-200. | Preserved untouched; corrected (`lr=2e-4` RASNet, `1e-4` SegResNet/V-Net, non-precision wording, all 4 regimes) in `THESIS_REPLACEMENT_TEXT.md` and logged in `HISTORICAL_ERRATA.md`. | `train_{rasnet,segresnet,vnet}_200ep.py:68` | **Preserved untouched; superseded in `THESIS_REPLACEMENT_TEXT.md`** |
| **V1R-13** | `README.md:7,141-142,210` | Claims `2,149.13 GFLOPs per patch` and `17.4x fewer GFLOPs` for nnU-Net, and stale test counts (`33/33` / `31`). | Actual `Table 4` nnU-Net GFLOPs = `1,119.34` (`96×160×160`, `9.07×`) or `402.96` (`96³`, `3.27×`); `tests/` has `72` passing tests. Pre-edit backup (`c1c8a63d...`) saved. | `Table4_computational_efficiency_rtx4080super.csv:2-3` & `pytest` | **Verified & tabulated (`table_v1_efficiency_and_timing_verified`)** |
| **V1R-14** | `Answers-thesis-def.md:6,11-32,52-54,62-73` | Claims `"100% bulletproof"`, equal-condition Arm A win, `alpha=0.30`, `75%`/`4x` smaller, `~1.85s` runtime, `81.0%` hospital `N=21` accuracy vs HeartFlow `FFR_CT`. | All errors refuted/qualified in `CLAIM_TO_EVIDENCE_MATRIX.csv` and `DEFENSE_QA_V1.md` (`alpha=0.4`, `6.54×`/`84.7%` vs nnU-Net, SegResNet smaller, `5.40s`/`22.95s` Option A runtime, hospital/HeartFlow excluded, Option A R4 NI=`False`). Pre-edit backup (`dca9d063...`) saved. | `rasnet_loss.py`, `timing_summary.csv`, `OPTION_A_RESULTS.md`, `AUDIT_FINDINGS.md` | **Verified; replacement Q&A authored in `DEFENSE_QA_V1.md`** |
| **V1R-15** | `CLAUDE.md:23,54-77` & `GEMINI.md:29,64-194` | Stale test count (`33`), stale pending status for Option A & Stage 2A/2B, internal A1 inconsistency (`GEMINI.md:154` vs `169`), missing Epoch-200 & Option A results. | Documented in `V1_REPORTING_CORRECTION_REPORT.md` & `table_v1_four_regimes_comparison.{csv,md}`; pre-edit backups (`ceece082...`, `23687f64...`) captured. | `pilot_lock_decision.md`, `OPTION_A_RESULTS.md`, `git log` | **Verified; root doc backups captured** |
| **V1R-16** | `AUDIT_FINDINGS.md:88-91` | Addendum implies broad resolution without explicitly distinguishing `E6` partial state (`fig3` uncorrected) or keeping `S1–S6`, `E5`, `D2`, `D4`, `I2`, `T1–T4` unclosed in original source pipelines. | Explicitly restated below (§2.2) that unrelated audit issues `S1–S6`, `E5`, `D2`, `D4`, `I2`, `T1–T4` remain **OPEN / ARCHIVED / NOT CLOSED**. Pre-edit backup (`19fc95db...`) captured. | `AUDIT_FINDINGS.md` + `train_*_200ep.py` + archive inspection | **Documented & enforced** |
| **V1R-17** | `LAB_CHECKS.md:477` (**PROTECTED**) | Dated test count (`31/31` at 2026-11-10; `33/33` at 2026-11-13). | Preserved untouched as historical forensic snapshot; current `72` passing tests documented in `HISTORICAL_ERRATA.md`. | `pytest tests/` (`72 passed`) | **Preserved untouched; logged in `HISTORICAL_ERRATA.md`** |

### 2.2 Unrelated `AUDIT_FINDINGS.md` Issues Explicitly Kept **OPEN / ARCHIVED / NOT CLOSED** (R2)

Per R2, the following defects documented in `AUDIT_FINDINGS.md` (2026-11-07) were **archived or left frozen in original source pipelines** and are **NOT closed** by V1 reporting corrections:

| Audit ID(s) | Original Source File(s) | Summary of Defect | Disposition / Status |
| :--- | :--- | :--- | :--- |
| **`S1–S6`** | `Phase3_Local_Integration/production_qca_engine.py` (archived) | Diagnosis leakage (`NORMAL`/`PATENT`), fabricated `10.0%`/`LAD` fallback, orientation-blind artery mapping, mean-voxel branch lengths, area-vs-diameter CAD-RADS mismatch, heuristic stenosis shrinkage. | **OPEN / ARCHIVED / NOT CLOSED** (hospital QCA cohort excluded from active claims) |
| **`E5`** | `Q1_Publication_Package/clinical_validation/07_stenosis_agreement.py` (archived) | Report regex silently returns `0.0` on unrecognized stenosis reports. | **OPEN / ARCHIVED / NOT CLOSED** |
| **`D2`** | `Phase3_Local_Integration/run_hospital_cohort_pipeline.py`, `dicom_to_nifti.py` (archived) | Heuristic `SeriesNumber` / `series_ids[0]` DICOM selection and z-only slice sorting without UID verification. | **OPEN / ARCHIVED / NOT CLOSED** |
| **`D4`** | `3d Cas Validations and doings/scripts/evaluate_unseen_66_models.py` (archived) | `QC_Excluded` label used as "unseen" without verified cross-dataset patient provenance (note: `unseen_66` IDs `[3..199]` were also in legacy 850-case nnU-Net `dataset.json`). | **OPEN / ARCHIVED / NOT CLOSED** (`unseen_66` excluded from active claims) |
| **`I2`** | `evaluate_unseen_66_models.py` (archived) | Unseen evaluation cache ignores overlap/dust/checkpoint settings and defaults to `0.45` threshold. | **OPEN / ARCHIVED / NOT CLOSED** |
| **`T1–T4`** | `Q1_Publication_Package/matched_200ep_benchmark/train_*_200ep.py` (frozen v1 trainers) | Incomplete AdamW/scaler/RNG `--resume` (`T1`), stale nnU-Net setup checks (`T2`), worker RNG determinism (`T3`), training vs historical evaluator orientation difference (`T4`; mitigated at inference by `benchmark/adapters/monai_models.py`, frozen trainers unchanged). | **OPEN (frozen v1 trainers unchanged) / NOT CLOSED** |

---

## 3. Checkpoint Provenance & `nnUNetTrainer` Epoch/EMA Conventions (R3)

Verified directly via `torch.load(..., map_location="cpu", weights_only=False)`, `plans.json`, and `nnunetv2` site-packages (`table_checkpoint_provenance_verified.{csv,md}`):

| Model & Role | File Path | SHA-256 (first 12) | State-Dict Keys | Exact Params | Epoch & Selection Provenance |
| :--- | :--- | :--- | :---: | ---: | :--- |
| **RASNet `best`** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/rasnet_best.pth` | `8d4c25cd12d0` | **`114`** (not 400) | `4,706,772` | Raw `OrderedDict`, no embedded metadata; `rasnet_200ep_log.csv` best at ep `160` (`0.77696`). |
| **RASNet `final_epoch200`** | `recovered_checkpoints/v1_epoch200_20261117_041500Z/rasnet_last.pth` | `71c9e5ef4a02` | **`114`** | `4,706,772` | Raw `OrderedDict`, `embedded_epoch_metadata_present=False`. Epoch-200 corroborated by `train_rasnet_200ep.py:268`, 200-row `rasnet_200ep_log.csv`, `CosineAnnealingLR` alignment (`5.42e-20`), split-ZIP timestamp delta (`5120.00s` vs `5119.97s`), and `114/114` tensor weight diff vs `rasnet_best.pth`. |
| **SegResNet `best`** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/segresnet_best.pt` | `b8ea61d95106` | **`83`** (not 153) | `4,700,914` | Raw `OrderedDict`, no embedded metadata; `segresnet_200ep_log.csv` best at ep `190` (`0.75212`). |
| **V-Net `best`** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/vnet_best.pt` | `a89d71bd41c3` | **`164`** (not 322) | `45,599,452` | Raw `OrderedDict` (`+5,340` BN buffer elements); `vnet_200ep_log.csv` best at ep `180` (`0.74905`). |
| **3D U-Net `best`** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/3dunet_best.pth` | `d8a8f5ec95b5` | **`63`** (not 90) | `4,805,534` | Wrapped dict `['epoch', 'model_state_dict', 'best_val_dice']` (**no** `optimizer_state_dict`), `epoch=180`, `best_val_dice=0.663835`. |
| **nnU-Net `best`** | `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/nnunet_best.pt` | `9accddf283b1` | **`292`** (not 402) | `30,785,994` | `dict` (`9` keys), `current_epoch=199`, `_best_ema=0.816239`, `logging` len `199`. |
| **nnU-Net `final_epoch200`** | `recovered_checkpoints/v1_epoch200_20261117_041500Z/nnunet_last.pt` | `2105bd7238ff` | **`292`** | `30,785,994` | `dict` (`9` keys), `current_epoch=200`, `_best_ema=0.816239`, `logging` len `200`. |

- **Verified `3d_fullres` plans (`plans.json` SHA-256 `c67af4aa...` & `init_args['plans']`):** `patch_size = [96, 160, 160]` (NOT `[128, 192, 192]`), `spacing = [0.5, 0.349609375, 0.349609375]` (NOT `[0.5, 0.359375, 0.359375]`), `CTNormalization`, `batch_size = 2`, `PlainConvUNet` (`n_stages=6`, `features=[32,64,128,256,320,320]`; 292 keys span 88 shared-storage aliases totaling `30,785,994` unique parameters).
- **`nnUNetTrainer` epoch & EMA mechanics (`nnUNetTrainer.py`, `nnunet_logger.py`):**
  1. Training loops `for epoch in range(0, 200)` using 0-based `self.current_epoch` (`0..199`).
  2. `on_epoch_end()` logs raw `mean_fg_dice` (`Pseudo dice` over 50 val batches), `MetaLogger.log()` computes `ema[0] = mean_fg_dice[0]` and `ema[t] = 0.9 * ema[t-1] + 0.1 * mean_fg_dice[t]` (verified to `max_abs_err = 0.0` against `nnunet_last.pt['logging']`), and `checkpoint_best.pth` is saved **only when `ema_fg_dice > _best_ema`**.
  3. `save_checkpoint()` writes `current_epoch = self.current_epoch + 1` (completed 1-based epoch). At 0-based index `188` (1-based epoch `189`), raw `mean_fg_dice` peaked at **`0.837630`** (`ema_fg_dice = 0.811305`). At 0-based index `198` (1-based epoch `199`), `ema_fg_dice` peaked at **`0.816239`** -> `checkpoint_best.pth` recorded **`current_epoch = 199`**. At 0-based index `199` (1-based epoch `200`), `mean_fg_dice = 0.814504`, `ema_fg_dice = 0.816066`, and `on_train_end()` saved `checkpoint_final.pth` with **`current_epoch = 200`**.

---

## 4. Training-Curve Corrections & Figures (R4)

- **Corrected Figure:** `results/v1_reporting_correction_20261117_105000Z/fig3_corrected_training_and_val_convergence_200ep.{png,svg}` (paired with preserved original `results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.{png,svg}`).
- **Verified Per-Model Training & Logged Validation Protocols:**

| Model | Loss Function (Source-Verified) | Optimizer & LR | Validation Subset & Spatial Grid | Cadence (`val_epochs`) | Aggregation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RASNet** | `StenosisAwareLoss(alpha=0.4, gamma=2.5)` (`0.4*Dice + 0.6*Focal`) + DS (`main + 0.4*aux2 + 0.2*aux3`) | `AdamW(lr=2e-4, wd=1e-5)`, Cosine | `Val94` (`N=94`), `0.5 mm` iso cropped, `96³` SW `overlap=0.25` | `12` evals (`20,40,50,60,80,100,120,140,150,160,180,200`) | Case-mean `DiceMetric` (`include_background=False`) |
| **SegResNet** | **`DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, lambda_dice=1.0, lambda_focal=2.0)`** | `AdamW(lr=1e-4, wd=1e-5)`, Cosine | `Val94` (`N=94`), `0.5 mm` iso cropped, `96³` SW `overlap=0.25` | `20` evals (`every 10 epochs`) | Case-mean `DiceMetric` |
| **V-Net** | **`DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, lambda_dice=1.0, lambda_focal=2.0)`** + grad clip `1.0` | `AdamW(lr=1e-4, wd=1e-5)`, Cosine | `Val94` (`N=94`), `0.5 mm` iso cropped, `96³` SW `overlap=0.25` | `13` evals (`[10,20,40,50,...,200]`) | Case-mean `DiceMetric` |
| **3D U-Net** | `DiceCELoss(sigmoid=True)` | `AdamW(lr=1e-4, wd=1e-5)`, Cosine | `Val94` (`N=94`), `128³` resized full volume, 1-shot | `21` evals (`[1,10,20,...,200]`) | Case-mean `DiceMetric` on `128³` grid |
| **nnU-Net V2** | `DC_and_CE_loss` + DS (negative scale) | `SGD(lr=1e-2, mom=0.99, nesterov)`, PolyLR | `Val94` patch pool, `96×160×160` (`[0.5, 0.3496, 0.3496]`), `50` batches/ep (`33.3%` FG oversampled) | `200` evals (`every epoch`) | Batch-pooled `2*TP/(2*TP+FP+FN)` = raw `Pseudo dice` (`mean_fg_dice`); `MetaLogger` computes `ema_fg_dice` |

- **Raw `Pseudo dice` vs `ema_fg_dice`:** SVG path extraction proved historical `fig3` plotted raw `mean_fg_dice` (`r = 0.99999978`, max residual `0.035 px`) while its subtitle falsely stated `EMA Pseudo-Dice`. The corrected figure plots both series distinctly (dashed light-blue raw `mean_fg_dice` with ep `189` star at `0.8376`; dotted dark-blue `ema_fg_dice` with ep `199` diamond at `0.8162`), keeps loss axes unshifted, and carries the required non-causal statement:
  > *"The logged metrics use different sampling, aggregation and spatial evaluation protocols and therefore cannot establish comparative full-volume performance."*

---

## 5. Val94 Epoch-200 Endpoint Statistics & Multiplicity Context (R5)

Recomputed directly from `results/v1_epoch200_comparison_20261117_041500Z/per_case_metrics_{rasnet,nnunet}_epoch200.csv` (`max_diff <= 3.55e-15` vs stored CSVs):
- **Integrity:** `94` unique cases per model matching `splits_final.json['val']` identically; `94/94` `'ok'`; `0` empty predictions/GT/skeleton; `0` NaN exclusions; `188/188` verified mask headers (`geometry_verification.csv`).
- **Corrected Figure & Tables:** `fig_v1_epoch200_val94_endpoint_corrected.{png,svg}`, `table_epoch200_val94_summary_and_paired_corrected.{csv,md}`, and separate `table_epoch200_val94_posthoc_holm8_exploratory.{csv,md}`.

| Metric (`N = 94`) | RASNet (`final_epoch200`) | nnU-Net (`final_epoch200`) | Mean Diff (`R − N`) [95% Boot CI] | W / T / L | Exploratory Unadj. Wilcoxon `p` | Post-Hoc Exploratory Holm-8 `p` *(NOT preregistered)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dice`** | `0.7792 ± 0.0569` | `0.7767 ± 0.0562` | `+0.002473 [-0.002747, +0.007545]` | `54 / 0 / 40` | `0.2401` | `0.9605` (n.s.) |
| **`iou`** | `0.6416 ± 0.0738` | `0.6383 ± 0.0736` | `+0.003373 [-0.003585, +0.010283]` | `54 / 0 / 40` | `0.2478` | `0.9605` (n.s.) |
| **`precision`** | `0.7586 ± 0.0838` | `0.7428 ± 0.0825` | `+0.015765 [+0.007244, +0.024537]` | `66 / 0 / 28` | `1.068e-4` | **`7.475e-4`** (favors RASNet) |
| **`recall`** | `0.8065 ± 0.0514` | `0.8203 ± 0.0535` | `-0.013787 [-0.017494, -0.009801]` | `16 / 0 / 78` | `4.849e-10` | **`3.879e-9`** (favors nnU-Net) |
| **`hd95` (mm)** | `17.32 ± 15.27` | `18.67 ± 16.35` | `-1.349 [-3.826, +0.947]` | `49 / 0 / 45` | `0.6742` | `0.9605` (n.s.) |
| **`asd` (mm)** | `2.619 ± 1.801` | `2.748 ± 2.061` | `-0.128 [-0.362, +0.110]` | `49 / 0 / 45` | `0.3372` | `0.9605` (n.s.) |
| **`cldice`** | `0.8413 ± 0.0633` | `0.8347 ± 0.0633` | `+0.006664 [-0.000589, +0.013532]` | `57 / 0 / 37` | `0.0352` | `0.1760` (n.s.) |
| **`centerline_recall`** | `0.9068 ± 0.0494` | `0.9142 ± 0.0476` | `-0.007412 [-0.012357, -0.002493]` | `37 / 2 / 55` | `0.0092` | `0.0551` (n.s.) |

- **Required statistical caveats:**
  1. All 8 p-values in `paired_stats_epoch200.{csv,json}` are **exploratory and unadjusted**. The Holm-8 column above comes from a separate **post-hoc exploratory (NOT preregistered)** table declaring all 8 metrics as its family (`table_epoch200_val94_posthoc_holm8_exploratory.{csv,md}`).
  2. **Why Wilcoxon `p` and Bootstrap 95% CI need not coincide:** The percentile bootstrap CI estimates uncertainty around the **mean paired difference** (sensitive to skewed tails/magnitudes), whereas the two-sided Wilcoxon signed-rank test evaluates **paired signed-rank location / pseudomedian**. Thus `clDice` shows unadjusted Wilcoxon `p = 0.0352` while its 95% mean-difference CI `[-0.000589, +0.013532]` spans zero—and under post-hoc Holm-8 (`p = 0.1760`), it is **not** confirmed topology superiority.
  3. **Endpoint scope:** This evaluates `final_epoch200` on development split `Val94` only (keeping `rasnet_last.pth` provenance qualification visible). It does not prove superiority or equivalence on Dice (`p = 0.2401`) and does not reconstruct epochs `50`, `100`, or `150`.

---

## 6. Strict Separation of the 4 Regimes, Option A, Exclusions & Efficiency (R6)

### 6.1 Four Strictly Separated Regimes (`table_v1_four_regimes_comparison.{csv,md}`)
1. **(A) Final Epoch-200 Endpoint (`Val94`, `N = 94`, `2` models):** Symmetric 1-pass, `0.50` Gauss SW, `τ = 0.50`, no CC (`final_epoch200` checkpoints). RASNet Dice `0.7792` vs nnU-Net `0.7767`.
2. **(B1 / B2) Best-Checkpoint Val94 (`N = 94`, `best` checkpoints: RASNet ep `160`, nnU-Net `current_epoch=199`):**
   - `B1` (`val_full_94` Arm A, `5` models, asymmetric SW `0.25` const vs `0.50` Gauss, `τ = 0.50`, no CC): RASNet `0.7789` vs nnU-Net `0.7761`.
   - `B2` (Option A Val94 grid, `4` models excluding 3D U-Net, symmetric `0.50` Gauss, `τ = 0.50`, Top-2 CC): RASNet `0.7953` vs nnU-Net `0.8070`.
3. **(C) Retrospective Test150 (`N = 150`, `IDs 851–1000`, `5` models, `best` checkpoints):** Asymmetric SW (`0.25` const vs `0.50` Gauss). Arm A (`0.7736` vs `0.7687`), Arm B (asymmetric TTA 4-pass vs 8-pass: `0.7807` vs `0.7760`), Arm C (Val94 `τ*` clamped at `0.30` + Top-2 CC: `0.7899` vs `0.8055`). **Retrospective** because `IDs 851–1000` were previously inspected during V1 threshold/loss exploration.
4. **(D) Option A Symmetric Test150 (`N = 150`, `IDs 851–1000` retrospective, `4` sliding-window models only — `3dunet` excluded):**
   - **Primary `R4_tuned_tta8_top2`** (`0.50` Gauss SW, 8-pass logit-avg TTA, Val-locked `τ*` `0.40/0.20`, Top-2 CC): nnU-Net `0.8059 ± 0.0630` vs RASNet `0.8039 ± 0.0691`; diff `-0.002075` (`95%` CI `[-0.008763, +0.002965]`); **`non_inferior = False` (margin `-0.005`), `superior = False`**. Primary `FA`: Precision `+0.0218` (Holm `p = 7.23e-17`, `132W/0T/18L` favoring RASNet); HD95 `+0.757 mm` (Holm `p = 0.489`); ASD `+0.180 mm` (Holm `p = 0.489`). Secondary `FB`/`FC`: RASNet beats SegResNet (`+0.0050` Dice, Holm `p = 4.27e-8`) and V-Net (`+0.0124` Dice, Holm `p = 3.58e-12`). Secondary `FD`: nnU-Net retains Recall advantage (`-0.0244`, Holm `p = 2.40e-17`, `17W/0T/133L`).
   - **Secondary `R1–R3` (kept secondary even where `R1`/`R2` favor RASNet):** `R1` (`0.7788` vs `0.7687`, diff `+0.0101`), `R2` (`0.7863` vs `0.7760`, diff `+0.0103`), `R3` (`0.8008` vs `0.8055`, diff `-0.0047`, NI=`False`).

### 6.2 Excluded Cohorts
- **`unseen_66` (`66` ImageCAS cases in `1..200`):** Excluded due to legacy nnU-Net 850-case training leakage, unverified cross-dataset provenance (`D4`), and broken cache/wrapper (`I2`).
- **Hospital QCA Cohort (`N = 21` verified / `N = 32` pre-quarantine):** Excluded due to unfixed clinical/DICOM audit findings (`S1–S6`, `E5`, `D2`, `I5`) and missing segmentation checkpoint provenance.

### 6.3 Verified Parameters, Compute & Runtime Scope (`table_v1_efficiency_and_timing_verified.{csv,md}`)
- **Exact Parameters:** `SegResNet` `4,700,914` (`4.701M`) < `RASNet` `4,706,772` (`4.707M`, `+5,858` params) < `3D U-Net` `4,805,534` (`4.806M`) < `nnU-Net` `30,785,994` (`30.786M`, `6.54×` larger than RASNet) < `V-Net` `45,599,452` (`45.599M`). Do NOT claim RASNet is smaller than every baseline or `4×`/`75%` smaller than nnU-Net.
- **Single-Patch Compute (`Table 4`, RTX 4080 SUPER):** `RASNet` (`96³`): `123.39 GFLOPs`, `15.68 ms` FP32 / `13.71 ms` AMP, `390.7 MB` patch VRAM vs `nnU-Net` (`96×160×160` native / `96³` equal): `1,119.34` (`9.07×`) / `402.96` (`3.27×`) GFLOPs, `59.56 ms` FP32 / `31.52 ms` AMP, `1,926.5 MB` patch VRAM. (`2,149.13 GFLOPs` / `17.4×` in `README.md` is erroneous.)
- **End-to-End Full-Volume Runtime (`Option A timing_summary.csv`):** RTX 4080 SUPER, `n = 10` Val cases after `1` warm-up, `0.50` Gauss SW, `sw_batch_size=8` (MONAI) vs `1` (nnU-Net): `RASNet` `5.40s` `[5.16, 5.72]` (1-pass, `4,782 MB` VRAM) / `22.95s` `[22.66, 23.06]` (TTA-8, `5,134 MB` VRAM) vs `nnU-Net` `29.84s` (1-pass, `2,243 MB` VRAM) / `74.37s` (TTA-8, `2,261 MB` VRAM).
- **Strictly Excluded Timings:** `manifest.json` `rasnet = 0.0s` (`--resume` skip) and `Answers-thesis-def.md` `~1.85s` (hardcoded estimate in archived `03_efficiency_benchmark.py`).

### 6.4 Prohibited / Removed Claims
- Universal superiority over nnU-Net V2 across all metrics or operating points.
- Statistical equivalence inferred from non-significant Dice (`p = 0.2401` or Option A `R4` `p = 0.99`).
- Zero overfitting inferred from similar aggregate validation and test scores.
- Causal generalization claims from parameter count, capacity, or attention gates alone.
- Clinical stenosis-grading, CAD-RADS, or HeartFlow/`FFR_CT` superiority inferred from segmentation or excluded hospital data.
- Claims that V1 corrections establish training-wide superiority (`ep 1–200`), or guarantee thesis acceptance.

---

## 7. Deliverables & Integration Status

- **Generated in `results/v1_reporting_correction_20261117_105000Z/`:** `V1_REPORTING_CORRECTION_REPORT.md`, `CLAIM_TO_EVIDENCE_MATRIX.csv`, `HISTORICAL_ERRATA.md`, `THESIS_REPLACEMENT_TEXT.md`, `DEFENSE_QA_V1.md`, `DEFENSE_SLIDE_OUTLINE.md`, `fig3_corrected_training_and_val_convergence_200ep.{png,svg}`, `fig_v1_epoch200_val94_endpoint_corrected.{png,svg}`, all 5 verified `table_*.{csv,md}` pairs, `pre_edit_backups/`, `pre_edit_hashes.json`, and `protected_baseline_hashes.json` (`548` protected files verified `0` mismatches).
- **External Manuscript & Slide Integration:** **`OUTSTANDING`** (no `.tex`, `.docx`, `.pptx`, or `.key` sources exist inside `C:\Thesis_RASNET`).
