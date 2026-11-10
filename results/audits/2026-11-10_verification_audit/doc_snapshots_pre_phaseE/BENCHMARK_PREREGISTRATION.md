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
- **Pre-Declared Numerical Tolerance**: $\epsilon = 10^{-4}$ (`0.0001`) for per-case metric comparisons between new benchmark runs and the historical CSVs in `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_*_200ep.csv`.
