# V1 Thesis Defense Q&A — Evidence-Verified Reference (`2026-11-17`)

> **Authoritative Correction Package**: [`results/v1_reporting_correction_20261117_105000Z/`](results/v1_reporting_correction_20261117_105000Z/) (`V1_REPORTING_CORRECTION_REPORT.md`, `CLAIM_TO_EVIDENCE_MATRIX.csv`, `HISTORICAL_ERRATA.md`, `THESIS_REPLACEMENT_TEXT.md`, `DEFENSE_QA_V1.md`, `DEFENSE_SLIDE_OUTLINE.md`, and corrected figures/tables).
>
> **Central Defensible Claim**: RASNet is a compact coronary segmentation model (`4.707M` parameters) with observed precision advantages at specified operating points and competitive full-volume overlap. The epoch-200 Val94 comparison (`0.7792` vs `0.7767`, `p = 0.2401`) did not establish Dice superiority or equivalence, and Option A (`R4` primary symmetric evaluation on retrospective Test150) did not establish Dice non-inferiority (`0.8039` vs `0.8059`, margin `−0.005`, `NI = False`); nnU-Net V2 retained consistent recall advantages.

---

## 1. Strictly Separated V1 Evaluation Regimes (Do Not Mix)

| Regime | Cohort (`N`) | Checkpoint Role | Models | Inference / SW / TTA / $\tau$ / CC | RASNet Dice | nnU-Net V2 Dice | Source |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **(A) Final Epoch-200 Endpoint** | `Val94` (`94`, dev) | `final_epoch200` | 2 (`rasnet`, `nnunet`) | Symmetric 1-pass, `0.50` Gauss SW, $\tau=0.50$, No CC | `0.7792 ± 0.0569` | `0.7767 ± 0.0562` | [`table_epoch200_val94_summary_and_paired_corrected.md`](results/v1_reporting_correction_20261117_105000Z/table_epoch200_val94_summary_and_paired_corrected.md) |
| **(B1) Best-Ckpt Val94 (Arm A)** | `Val94` (`94`, dev) | `best` (R:160, N:199/189*) | 5 (all v1) | **Asymmetric SW** (MONAI `0.25` const vs nnU-Net `0.50` Gauss; 3D U-Net `128³`), 1-pass, $\tau=0.50$, No CC | `0.7789 ± 0.0521` | `0.7761 ± 0.0567` | [`val_full_94/summary_arm_A.csv`](results/benchmark_runs/val_full_94/summary_arm_A.csv) |
| **(B2) Best-Ckpt Val94 (Option A Grid)** | `Val94` (`94`, dev) | `best` (R:160, N:199/189*) | 4 (`3dunet` excluded) | Symmetric 1-pass, `0.50` Gauss SW, $\tau=0.50$, **+ Top-2 CC** | `0.7953 ± 0.0657` | `0.8070 ± 0.0585` | [`optionA_v1_symmetric_sw050/val_grid_scores.csv`](results/optionA_v1_symmetric_sw050/val_grid_scores.csv) |
| **(C) Retrospective Test150 (Arms A/B/C)** | `Test150` (`851–1000`, retrospec.) | `best` | 5 (all v1) | Arm A (asymm SW, 1-pass, `0.50`, no CC); Arm B (asymm TTA: R 4-pass, N 8-pass, others 1-pass); Arm C (asymm SW, 1-pass, Val94 $\tau^* \in [0.30..0.70]$, Top-2 CC) | A: `0.7736`<br>B: `0.7807`<br>C: `0.7899` | A: `0.7687`<br>B: `0.7760`<br>C: `0.8055` | [`test_retrospective_150/summary_arm_{A,B,C}.csv`](results/benchmark_runs/test_retrospective_150/summary_arm_A.csv) |
| **(D) Option A Symmetric Test150** | `Test150` (`851–1000`, retrospec.) | `best` | 4 (`3dunet` excluded) | Symmetric `0.50` Gauss SW: `R1` (1-pass, `0.50`, no CC), `R2` (TTA-8, `0.50`, no CC), `R3` (1-pass, $\tau^*$, Top-2 CC), **`R4` primary** (TTA-8, $\tau^*$, Top-2 CC) | R1: `0.7788`<br>R2: `0.7863`<br>R3: `0.8008`<br>**R4: `0.8039`** | R1: `0.7687`<br>R2: `0.7760`<br>R3: `0.8055`<br>**R4: `0.8059`** | [`optionA_v1_symmetric_sw050/summary_test.csv`](results/optionA_v1_symmetric_sw050/summary_test.csv) |

*\*Note on nnU-Net Best Epoch*: In `nnUNetTrainer`, `current_epoch` is 0-based (`0..199`) and `save_checkpoint` writes `current_epoch + 1`. Raw unsmoothed `mean_fg_dice` (`Pseudo dice`) peaks at epoch `189` (`0.8376`), whereas `checkpoint_best.pth` (`nnunet_best.pt`) is selected by `_best_ema = 0.816239` at `current_epoch = 199`, and `checkpoint_final.pth` (`nnunet_last.pt`) records `current_epoch = 200`.

### Option A (`R4_tuned_tta8_top2`, Pre-Registered Primary Symmetric Test150; `N = 150` Retrospective)

- **Primary Non-Inferiority (`RASNet − nnU-Net` Dice, margin `−0.005`):** mean diff `−0.0021`, `95% CI [−0.0088, +0.0030]`, Wilcoxon `p = 0.990`, `76W / 0T / 74L` → **`Non-inferior = False`, `Superior = False`**.
- **R4 `FA_primary` (`RASNet` vs `nnU-Net`, Holm within family):** Precision `+0.0218 [+0.0143, +0.0278]` (`Holm p = 7.23e-17`, `132W/0T/18L`, favors RASNet); HD95 `+0.757 mm [−0.181, +1.927]` (`Holm p = 0.489`, n.s.); ASD `+0.180 mm [−0.065, +0.556]` (`Holm p = 0.489`, n.s.).
- **R4 `FB`/`FC` (`RASNet` vs `SegResNet` / `V-Net`):** RASNet beats `SegResNet` (`0.8039` vs `0.7989`) on Dice (`+0.0050`, `Holm p = 4.27e-08`) and Precision (`+0.0102`, `Holm p = 3.17e-06`), and beats `V-Net` (`0.7915`) on Dice (`+0.0124`, `Holm p = 3.58e-12`), Precision (`+0.0085`, `Holm p = 0.0074`), HD95 (`−0.572 mm`, `Holm p = 0.0091`), and clDice (`+0.0072`, `Holm p = 0.0053`).
- **R4 `FD_secondary` (`RASNet` vs `nnU-Net`):** Recall `−0.0244 [−0.0319, −0.0183]` (`Holm p = 2.40e-17`, `17W/0T/133L`, favors nnU-Net); clDice `+0.0006` (`Holm p = 0.446`, n.s.).
- **Secondary Option A Recipes (`R1`–`R3`):** `R1` (`0.7788` vs `0.7687`, `NI=True`) and `R2` (`0.7863` vs `0.7760`, `NI=True`) show raw unpruned advantages for RASNet at $\tau=0.50$, whereas tuned Top-2 CC `R3` (`0.8008` vs `0.8055`, `NI=False`) matches `R4` in showing non-inferiority is not established.

---

## 2. Verified Architecture, Parameter Counts, & Timing Evidence

- **Loss & Architecture (`Phase3_Local_Integration/rasnet_model.py`, `rasnet_loss.py`, `train_rasnet_200ep.py`):** `RASNet` (`init_filters=16`) adds 3D additive attention gates and multi-scale deep supervision (`loss_main + 0.4 * aux2_loss + 0.2 * aux3_loss`) to MONAI's `SegResNet` backbone, trained with `StenosisAwareLoss` ($\alpha = 0.4$ Dice + $0.6$ Focal with $\gamma = 2.5$).
- **Exact Model Sizes (`table_v1_efficiency_and_timing_verified.md`):**
  - `SegResNet`: `4,700,914` (`4.701M`) — slightly **smaller** than RASNet (`−5,858` params)
  - `RASNet`: `4,706,772` (`4.707M`) — **`84.7%` fewer parameters (`6.54×` smaller)** than this `nnU-Net V2` (`30,785,994` / `30.786M`), and `89.7%` fewer than `V-Net` (`45,599,452` / `45.599M`)
  - `3D U-Net`: `4,805,534` (`4.806M`)
- **Single-Patch Compute (`Table4_computational_efficiency_rtx4080super.csv`, RTX 4080 SUPER):**
  - `RASNet` (`96³`): `123.39` GFLOPs, `15.68 ± 0.32 ms` (FP32), `13.71 ± 0.12 ms` (AMP), `390.7 MB` peak patch VRAM.
  - `nnU-Net V2` (`96×160×160` native / `96³` equal): `1,119.34` GFLOPs (`9.07×` more than RASNet) / `402.96` GFLOPs (`3.27×` more), `59.56 ± 0.41 ms` (FP32), `31.52 ± 0.30 ms` (AMP), `1,926.5 MB` peak patch VRAM.
- **End-to-End Full-Volume Runtime (`results/optionA_v1_symmetric_sw050/timing_summary.csv`, `n = 10` Val cases after 1 warm-up on idle RTX 4080 SUPER, `0.50` Gauss SW, `--sw-batch-size 8` for MONAI vs `1` for nnU-Net, incl. native inversion):**
  - `RASNet`: `5.40s` median `[5.16, 5.72]` (1-pass, peak VRAM `4,782 MB`) | `22.95s` `[22.66, 23.06]` (TTA-8, peak VRAM `5,134 MB`).
  - `SegResNet`: `5.04s` (1-pass, `4,782 MB`) | `19.83s` (TTA-8, `5,134 MB`).
  - `V-Net`: `6.05s` (1-pass, `4,941 MB`) | `27.69s` (TTA-8, `5,929 MB`).
  - `nnU-Net V2`: `29.84s` `[28.25, 31.69]` (1-pass, peak VRAM `2,243 MB`) | `74.37s` `[73.06, 76.28]` (TTA-8, peak VRAM `2,261 MB`).
  - *Important nuance*: Full-volume peak VRAM is higher for MONAI (`4.78–5.93 GB`) than nnU-Net (`2.24–2.26 GB`) under Option A because MONAI used `sw_batch_size = 8` with GPU tensor accumulation.
- **Excluded Invalid Timing & Out-of-Scope Cohorts:**
  - `results/v1_epoch200_comparison_20261117_041500Z/manifest.json` records `rasnet = 0.0s` because `--resume` skipped 94 pre-completed cases (`endpoint_eval.py:1020-1066`); never cite `0.0s` as inference latency. Legacy `~1.85s` estimates from archived `03_efficiency_benchmark.py` are also excluded.
  - Both `unseen_66` (`66` IDs in `[3..199]` excluded from `splits_final.json`) and historical hospital QCA cohorts (`N = 21` / `N = 32`, affected by `AUDIT_FINDINGS.md` `S1–S6`, `E5`, `D2`, `I5`) are **strictly excluded** from active supporting claims. No clinical stenosis-grading or HeartFlow/$\text{FFR}_{\text{CT}}$ superiority claim is made from segmentation alone.

---

## 3. Concise Defense Answers to the 8 Core Questions (R7)

### Q1. Why was nnU-Net's logged curve higher?
During training, nnU-Net logged raw unsmoothed `Pseudo dice` (`mean_fg_dice`) every epoch (`200` evals) on `50` foreground-oversampled (`33%`) `96 × 160 × 160` validation patches (`batch_size = 2`, `[0.5, 0.3496, 0.3496] mm`, `CTNormalization`) pooled via batch-level `2*TP / (2*TP + FP + FN)`, peaking at `0.8376` at epoch `189` (while EMA `_best_ema = 0.8162` selected `checkpoint_best.pth` at `current_epoch = 199`). By contrast, `RASNet` (`12` evals), `SegResNet` (`20` evals), and `V-Net` (`13` evals) evaluated full foreground-cropped `0.5 mm` volumes with `96³` sliding-window (`overlap = 0.25`) averaged per case (`3D U-Net` used `21` evals on `128³` resized volumes). **The logged metrics use different sampling, aggregation and spatial evaluation protocols and therefore cannot establish comparative full-volume performance.**

### Q2. What did the same-epoch full-volume evaluation establish?
Evaluating both `final_epoch200` checkpoints on `Val94` (`N = 94`, `0` empty predictions, `188/188` verified physical grids) under identical `1-pass`, `0.50` Gaussian SW, $\tau = 0.50$, no-CC settings yielded RASNet Dice `0.7792 ± 0.0569` vs nnU-Net `0.7767 ± 0.0562` (paired diff `+0.0025`, `95% CI [−0.0027, +0.0075]`, exploratory unadjusted Wilcoxon `p = 0.2401`, `54W/0T/40L`). This established **comparable full-volume Dice at epoch 200**, **higher observed Precision for RASNet** (`0.7586` vs `0.7428`, `+0.0158`, unadjusted `p = 1.07e-4`, post-hoc `Holm-8 p = 7.47e-4`), and **higher Recall for nnU-Net** (`0.8203` vs `0.8065`, `−0.0138`, unadjusted `p = 4.85e-10`, post-hoc `Holm-8 p = 3.88e-9`). It did **not** establish Dice superiority or equivalence (`p = 0.2401` ≠ equal). All 8 endpoint p-values are **exploratory / unadjusted**; `clDice` (`+0.0067`, `95% CI [−0.0006, +0.0135]`, unadjusted `p = 0.0352`) and `centerline_recall` (`−0.0074`, `95% CI [−0.0124, −0.0025]`, unadjusted `p = 0.0092`) do **not** remain `< 0.05` under post-hoc exploratory Holm-8 (`p = 0.1760` and `0.0551`), and Wilcoxon vs bootstrap CI target different quantities (rank location vs mean difference). Provenance qualification: `nnunet_last.pt` embeds `current_epoch = 200`, whereas `rasnet_last.pth` has no embedded epoch metadata and is corroborated by `train_rasnet_200ep.py:268`, `rasnet_200ep_log.csv`, `CosineAnnealingLR`, and archive timestamps.

### Q3. Why does this not prove RASNet won throughout training?
The endpoint comparison evaluates a single terminal checkpoint (`epoch 200`) on `Val94` (`N = 94`). Because intermediate checkpoints (`epochs 50, 100, 150`) were not saved or evaluated under a unified full-volume physical-grid protocol, epoch-200 performance cannot reconstruct earlier training trajectories or prove superiority throughout training.

### Q4. What are RASNet's supported contributions?
1. **Parameter & patch-compute efficiency vs heavier baselines:** `4.707M` parameters (`84.7%` / `6.54×` fewer than `nnU-Net V2`'s `30.786M`; `89.7%` fewer than `V-Net`'s `45.599M`; `9.07×` fewer GFLOPs per native patch than `nnU-Net V2` and `3.27×` fewer on equal `96³` patches), with `5.40s` median 1-pass / `22.95s` TTA-8 full-volume runtime on RTX 4080 SUPER (`n = 10` Val cases) vs `29.84s` / `74.37s` for `nnU-Net V2`. (Note `SegResNet` at `4.701M` is slightly smaller.)
2. **Consistent Precision advantage at tested operating points:** Higher Precision than `nnU-Net V2` at Epoch-200 Val94 (`+0.0158`, `Holm-8 p = 7.47e-4`) and under Option A `R4` (`+0.0218`, `Holm p = 7.23e-17`, `132W/0T/18L`) as well as `R1`–`R3`.
3. **Gains over matched MONAI baselines (`Option A R4`, retrospective Test150):** Higher Dice and Precision than `SegResNet` (`+0.0050` Dice, `Holm p = 4.27e-08`; `+0.0102` Precision, `Holm p = 3.17e-06`) and higher Dice/Precision/HD95/clDice than `V-Net` (`+0.0124` Dice, `Holm p = 3.58e-12`).

### Q5. Where does nnU-Net perform better?
- **Sensitivity / Recall:** Consistently higher voxel Recall (`0.8203` vs `0.8065` at Epoch-200 Val94, `Holm-8 p = 3.88e-9`; `0.8104` vs `0.7861` in Option A `R4`, `Holm p = 2.40e-17`, `133/150` wins) and higher observed `centerline_recall` on Val94 (`0.9142` vs `0.9068`, unadjusted `p = 0.0092`, `95% CI [−0.0124, −0.0025]`).
- **Tuned / Top-2 CC Overlap & Option A Non-Inferiority:** Higher observed Dice in Option A `R3` (`0.8055` vs `0.8008`) and `R4` (`0.8059` vs `0.8039`), where RASNet did not pass the `−0.005` non-inferiority margin (`NI = False`).
- **Lower Full-Volume Peak VRAM under Option A settings:** `2.24–2.26 GB` vs `4.78–5.93 GB` for MONAI (`sw_batch_size = 8`).

### Q6. Why are best-checkpoint and final-checkpoint results separate?
They come from different checkpoints, epochs, cohorts, and inference settings: `best` checkpoints were selected at `epoch 160` (RASNet) vs `current_epoch = 199` (nnU-Net, `_best_ema = 0.8162`), and evaluated across distinct protocols (`Val94 Arm A` used asymmetric `0.25` const vs `0.50` Gauss SW; `Option A Val94` added Top-2 CC; `Option A Test150 R4` used TTA-8 + Val94-locked $\tau^* \neq 0.50$ + Top-2 CC). By contrast, `final_epoch200` evaluated terminal epoch-200 weights on `Val94` with symmetric `1-pass`, `0.50` Gauss SW, $\tau = 0.50$, and no CC. Comparing best vs final across mismatched inference protocols conflates checkpoint age with TTA, thresholding, and CC pruning.

### Q7. Why are the Test150 results retrospective?
Prior historical evaluations (e.g., $\tau = 0.60$, loss calibration, and Phase H exploration) inspected ImageCAS test cases `851–1000` before `BENCHMARK_PREREGISTRATION.md` and Option A were locked. Although Option A froze thresholds on `Val94` (`locked_recipes.json` committed at `558e7eb`) before running `Test150`, the underlying test split (`851–1000`) was no longer untouched/naïve and must be disclosed as a **retrospective reassessment**, not untouched confirmatory evidence.

### Q8. What remains uncertain and what independent evaluation is still needed?
1. **True confirmatory generalization:** V1 lacks an untouched external multi-center/multi-scanner test cohort with verified provenance (`unseen_66` and hospital `N=21/32` cohorts are excluded due to audit defects `S1–S6`, `E5`, `D2`, `D4`, `I2`).
2. **Boundary/topology uncertainty:** At Epoch-200 Val94, HD95 (`17.32 ± 15.27` vs `18.67 ± 16.35 mm`, `p = 0.674`) and ASD (`p = 0.337`) had wide confidence intervals and did not differ significantly, nor did `clDice` survive Holm-8 (`p = 0.1760`); in Option A `R4`, HD95/ASD/clDice vs nnU-Net were also non-significant.
3. **Architecture vs training-pipeline separation & controlled multi-seed study:** V1 bakes training differences into weights (`0.5³` mm `96³` patch vs `0.5 × 0.35 × 0.35` mm `96 × 160 × 160` patch, normalization, augmentation, AdamW vs SGD). `Rasnet-v2` was designed to control these factors across 3 seeds, and its Stage 2A/2B pilots ended in `FALLBACK` (`pilot_lock_decision.md`, `2026-11-16`); a prospective external validation and controlled multi-seed evaluation remain open.
