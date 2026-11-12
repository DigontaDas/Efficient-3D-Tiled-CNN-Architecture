# Pre-Registered Retrospective Reassessment for ImageCAS (5-Model 200-Epoch Benchmark)

## 1. Epistemic Status & Retrospective Scope
- **Objective**: Assess how the five 200-epoch trained coronary artery segmentation models (`RASNet`, `SegResNet`, `nnU-Net V2`, `3D U-Net`, `V-Net`) actually compare under transparent, symmetric, and native inference rules, without trying to make `RASNet` win.
- **Retrospective Reassessment Notice**: As documented in `LAB_CHECKS.md` and `AUDIT_FINDINGS.md`, historical post-processing and threshold sweeps (`tau = 0.60`, top-2 connected components) inspected the 150-case ImageCAS test split (`IDs 851–1000`) and the 66-case unseen split. Therefore, **all evaluations on the 150-case cohort (`IDs 851–1000`) are explicitly retrospective reassessments**, not untouched confirmatory tests. A truly independent, previously untouched external cohort is required for a strong confirmatory claim.

---

## 2. Frozen Comparison Arms (Pre-Specified Before Examining New Test Results)

Each model retains its necessary training-consistent preprocessing (`CropForegroundd -> Spacingd(0.5, 0.5, 0.5) -> ScaleIntensityRanged(-100, 800, 0, 1)` for `RASNet`, `SegResNet`, `V-Net`; `ScaleIntensityRanged(-100, 800, 0, 1) -> Resized(128, 128, 128)` for `3D U-Net`; native `nnUNetPlans` `3d_fullres` preprocessing via `nnUNetPredictor` for `nnU-Net V2`). All continuous foreground probability maps are inverted/resampled onto the native SimpleITK `(Z, Y, X)` image grid prior to thresholding and post-processing.

### Arm A — Common Raw Inference (Single Pass, Fixed Threshold, No Post-Processing)
- **Passes**: Single forward pass (`tta=False` for all four MONAI models; `use_mirroring=False` for `nnUNetPredictor`).
- **Probability Threshold**: Fixed $\tau = 0.50$ (`prob >= 0.50`) for all 5 models.
- **Post-Processing**: None (`keep_top_k = None`).

### Arm B — Documented Native / Standard TTA (Fixed Threshold, No Post-Processing)
- **Passes / TTA Policy**:
  - **`nnU-Net V2`**: Native `nnUNetPredictor` 3D mirroring (`use_mirroring=True` across `inference_allowed_mirroring_axes = (0, 1, 2)`, i.e., 8-pass mirror averaging).
  - **`RASNet`**: Documented 4-pass spatial flip TTA (`[None, (2,), (3,), (4,)]` averaged in probability space), as implemented in `evaluate_matched_200ep_models.py:435-440` and `Phase3_Local_Integration/legacy_pre_200ep_scripts/eval_rasnet.py`.
  - **`SegResNet`, `V-Net`, `3D U-Net`**: Neither the repository training/evaluation scripts nor MONAI's network definitions specify a standard TTA for these baselines. Per requirement (*"If a model has no demonstrable standard TTA, specify what is used and why; do not invent one"*), they use their documented single-pass inference (`tta=False`) in Arm B.
- **Probability Threshold**: Fixed $\tau = 0.50$ (`prob >= 0.50`) for all 5 models.
- **Post-Processing**: None (`keep_top_k = None`).

### Arm C — Validation-Tuned Thresholds + Identical Connected-Component Post-Processing
- **Passes**: Single forward pass (`tta=False` / `use_mirroring=False`) across all 5 models so post-processing and threshold tuning are evaluated symmetrically without TTA confounding.
- **Validation Threshold Selection Rule (Frozen in `thresholds.json`)**:
  - **Selection Split**: Full validation split only (`N = 94` cases from `Phase3_Local_Integration/splits_final.json`). Test cases (`851–1000`) are strictly forbidden during threshold selection.
  - **Candidate Grid**: $\tau \in \{0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70\}$ (`9` levels, step `0.05`).
  - **Selection Objective**: Maximize mean validation Dice across all 94 validation cases *after* applying the identical Arm C connected-component filter (`keep_top_k = 2`, 26-connectivity).
  - **Tie-Breaking Rule**: If two thresholds achieve mean validation Dice within $10^{-12}$, select the threshold closest to $0.50$ (`min |\tau - 0.50|`), then the lower threshold (`min \tau`).
- **Identical Post-Processing**: Keep at most the top `2` largest 3D connected components (`cc3d.connected_components`, 26-connectivity) for every model. Log `components_total`, `components_retained`, `components_removed`, `voxels_retained`, `voxels_removed`, `volume_retained_mm3`, and `volume_removed_mm3` for every case.

---

## 3. Pre-Declared Hypothesis Families & Multiplicity Correction

All paired comparisons match exact unique `case_id`s (`N = 150` on the test split) using two-sided paired Wilcoxon signed-rank tests, rank-biserial correlation effect sizes, percentile bootstrap 95% confidence intervals ($B = 2000$, seed `42`), and win/tie/loss counts oriented by metric direction (`higher` for `dice, iou, precision, recall, cldice, centerline_recall`; `lower` for `hd95, asd`).

1. **Primary Pre-Specified Family (`K = 3`, Holm-Bonferroni corrected within each arm)**:
   - `RASNet` vs. `nnU-Net V2` on `precision` (`direction: higher`)
   - `RASNet` vs. `nnU-Net V2` on `hd95` (`direction: lower`)
   - `RASNet` vs. `nnU-Net V2` on `asd` (`direction: lower`)
2. **Full Secondary Benchmark Family (`K = 32`, Holm-Bonferroni corrected across all 4 baselines $\times$ 8 metrics within each arm)**:
   - `RASNet` vs. `{nnunet, segresnet, vnet, 3dunet}` across `{dice, iou, precision, recall, hd95, asd, cldice, centerline_recall}`.

---

## 4. Historical Comparison Tolerance
- **Pre-Declared Numerical Tolerance**: $\epsilon = 10^{-4}$ (`0.0001`) for per-case metric comparisons between new benchmark runs and the historical CSVs in `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_*_200ep.csv`.

---

## 5. Dated Amendments & Post-Audit Notes (`2026-11-10`)
1. **Paired Difference Bootstrap 95% CIs & Non-Parametric Summary Columns**:
   - During the `2026-11-10` verification audit (`results/audits/2026-11-10_verification_audit/`), `benchmark/stats.py` was extended so that `paired_analysis` and `analyze_family` also output percentile bootstrap 95% confidence intervals on the paired difference $A - B$ (`diff_ci95_lo`, `diff_ci95_hi`, $B=2000$, seed `42`) as well as `median_difference`, `q25_difference`, `q75_difference`, and `iqr_difference`. Full augmented tables for `test_retrospective_150` and `val_full_94` are stored in `results/audits/2026-11-10_verification_audit/paired_stats_*_with_ci.csv`.
2. **Validation Threshold Grid Boundary Observation (`results/thresholds.json`)**:
   - Under the frozen validation grid $\tau \in \{0.30, 0.35, \dots, 0.70\}$, `nnunet` ($\tau^*=0.35$, mean validation Dice `0.8075`) and `segresnet` ($\tau^*=0.35$, mean validation Dice `0.7901`) peaked strictly inside the grid, whereas `rasnet` ($\tau^*=0.30$, `0.7897`), `vnet` ($\tau^*=0.30$, `0.7702`), and `3dunet` ($\tau^*=0.30$, `0.6047`) selected the lower boundary `0.30`. To prevent post-hoc threshold snooping, the pre-registered grid was **not** expanded after inspecting validation/test results; this boundary behavior is explicitly reported as a methodological limitation and sensitivity curve (`fig2_validation_threshold_sensitivity_n94.png`).

---

## 6. Amendment: Option A — Retrospective Inference-Parity Supplement (`2026-11-13`)

- **Status:** pre-registered on `2026-11-13`, before any Option A code was executed on data and before any Option A result existed. Approved by the user.
- **Scope:** a **retrospective** supplement. Test IDs 851–1000 were already inspected in Arms A/B/C and Phase H.
- **What it changes:** it re-evaluates the frozen v1 checkpoints of RASNet, SegResNet, V-Net and nnU-Net V2 under identical inference, identical 8-pass mirroring TTA and an identical validation-only threshold search.
- **What it does not change:** no locked v1 table (Tables 1–4), Arm A/B/C file or Phase H file is modified. Outputs go only to `results/optionA_v1_symmetric_sw050/`.
- **Implementation:** `benchmark/symmetric_eval.py` and `scripts/run_optionA_symmetric_v1_eval.py`, tests in `tests/test_optionA_symmetric_eval.py`.

### 6.1 Protocol (copied verbatim from `docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md` §1)

| Item | Value (identical for every model) |
| :--- | :--- |
| Models | `rasnet` (`rasnet_best.pth`), `segresnet` (`segresnet_best.pt`), `vnet` (`vnet_best.pt`), `nnunet` (`fold_0/checkpoint_best.pth`). The 3D U-Net is **excluded**: it runs single-shot on a 128³ resize, so sliding window does not apply. Report its existing Table 2 rows with that caveat. |
| Preprocessing | Each model keeps the preprocessing it was trained with. This is unavoidable with frozen weights. MONAI models: `benchmark/adapters/monai_models.py:33-41`. nnU-Net: its native plans. |
| Patch size | Each model's trained patch: 96³ for MONAI, 96 × 160 × 160 for nnU-Net. |
| Sliding window | overlap 0.50, Gaussian weighting, σ = 0.125 × patch. This matches nnU-Net's `compute_gaussian(sigma_scale=1/8)`, verified in MIC-DKFZ/nnUNet `predict_from_raw_data.py`. |
| Mode `sw050_gauss_1pass` | One pass. |
| Mode `sw050_gauss_tta8` | 8-pass mirroring: identity plus all 7 non-empty combinations of the 3 spatial flips, **logits averaged, then softmax**. This is exactly what `nnUNetPredictor._internal_maybe_mirror_and_predict` does (all `itertools.combinations` of the mirror axes; raw logits summed, divided by the pass count). For nnU-Net the native predictor with `use_mirroring=True` is used. **Implementation difference to disclose:** nnU-Net mirrors each tile inside its sliding window, while the MONAI path flips the whole volume and then runs the sliding window. Every mirrored prediction is still a Gaussian-blended average of the same patch predictions, but tile borders fall at different positions, so the results are numerically close rather than bit-identical. State this in the Methods. |
| AMP | fp16 autocast on CUDA for MONAI (same as the original benchmark). nnU-Net uses its own autocast. |
| Threshold grid | τ ∈ {0.20, 0.25, …, 0.70} (11 values). The lower bound was widened from 0.30 to remove the boundary clamp. |
| Post-processing | 26-connected **Top-2 CC** for tuned recipes, none for raw recipes. No hysteresis and no Top-3, to keep the forking paths small and the same for everyone. |
| τ selection | Per model × mode, on **Val-94 only**: maximise mean Dice after Top-2 CC. Tie-break: closest to 0.50, then lower (same as `benchmark/thresholds.py`). |
| Test recipes | Four per model (below). Evaluated **once** on Test-150. |
| Primary recipe | **R4** (each model at its best symmetric pipeline). R1–R3 are secondary. |
| Metrics | `surface-v1` (`benchmark/metrics.py`): Dice, IoU, precision, recall, HD95, ASD, clDice, centreline recall, plus component counts. |
| Statistics | Two-sided Wilcoxon signed-rank, Holm within family, 2,000-bootstrap 95 % CI (`benchmark/stats.py`). |

Test recipes:

| ID | Mode | τ | CC |
| :--- | :--- | :--- | :--- |
| R1 | 1-pass | 0.50 | none |
| R2 | TTA-8 | 0.50 | none |
| R3 | 1-pass | τ\* locked for 1-pass | Top-2 |
| R4 | TTA-8 | τ\* locked for TTA-8 | Top-2 |

**Pre-declared hypothesis families.** Holm is applied within each family, evaluated on R4.

| Family | Comparison | Endpoints |
| :--- | :--- | :--- |
| F-A primary | RASNet vs nnU-Net | precision ↑, HD95 ↓, ASD ↓. This is the same K = 3 family as `config.lab.yaml → hypotheses`. |
| F-B | RASNet vs SegResNet (matched 4.7 M backbone) | Dice ↑, precision ↑, HD95 ↓, clDice ↑ |
| F-C | RASNet vs V-Net | Dice ↑, precision ↑, HD95 ↓, clDice ↑ |
| F-D secondary | RASNet vs nnU-Net | Dice, clDice, recall |
| NI | RASNet vs nnU-Net Dice non-inferiority | Lower bound of the 95 % bootstrap CI of the paired difference > **−0.005** (same margin as `Rasnet-v2/PREREGISTRATION_V2.md:53`) |
| Exploratory | Baseline vs baseline | Not used for claims |


