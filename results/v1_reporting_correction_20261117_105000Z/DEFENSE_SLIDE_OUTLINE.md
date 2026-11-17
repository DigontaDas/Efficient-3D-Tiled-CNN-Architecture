# V1 Thesis Defense Slide Outline (`DEFENSE_SLIDE_OUTLINE.md`)

**Date (UTC):** `2026-11-17`  
**External Defense Slide Deck Integration Status:** **`OUTSTANDING`**  
> *Verification note:* A recursive search of `C:\Thesis_RASNET` found **zero** `.pptx` or `.key` slide deck source files. Incorporating these slides and corrected figures/tables into the external presentation deck remains **`OUTSTANDING`**.

---

## Deck Structure (14 Slides)

### Slide 1 — Title & Central Thesis Claim
- **Title:** *RASNet: Compact 3D Attention-Gated Coronary Artery Segmentation on CCTA — Audited V1 Evaluation*
- **Central Claim Banner (R7):** *"RASNet is a compact coronary segmentation model with observed precision advantages at specified operating points and competitive full-volume overlap. The epoch-200 comparison did not establish Dice superiority or equivalence; nnU-Net retained recall advantages."*
- **Scope Note:** ImageCAS `N = 1,000` (`690` train / `94` val / `150` retrospective test; `unseen_66` & hospital `N = 21/32` excluded).

### Slide 2 — Clinical Motivation & Scope Boundaries (Without Overclaiming)
- **Visual:** Clean coronary CCTA anatomy schematic / task formulation (`3D voxel lumen mask` -> downstream geometry).
- **Key Points:**
  - Why thin, tortuous coronary branches challenge 3D segmentation (foreground sparsity, venous/myocardial false positives, blooming).
  - Strict scientific boundary: V1 evaluates **3D lumen segmentation overlap, precision/recall trade-offs, boundary distances (`surface-v1`), and topology (`clDice`)**—not clinical CAD-RADS accuracy or physiological `FFR_CT` / HeartFlow superiority.

### Slide 3 — Dataset Splits & Pre-Registered V1 Audit Hygiene
- **Data / Table:** `Phase3_Local_Integration/splits_final.json` (`690` train, `94` val, `150` retrospective test `IDs 851–1000`).
- **Key Points:**
  - Zero overlap across `934` split IDs; `66` excluded ImageCAS cases (`[3..199]`) and hospital QCA cohort excluded due to legacy leakage/unfixed audit defects (`S1–S6`, `D2`, `D4`, `I2`).
  - Why `Test150` is explicitly labeled **retrospective** (earlier `τ = 0.60`, loss calibration, and Phase H inspected `851–1000`).

### Slide 4 — RASNet Architecture & Loss Formulation
- **Key Points (`rasnet_model.py`, `rasnet_loss.py`):**
  - MONAI `SegResNet` backbone (`init_filters = 16`, `4.701M` params) + 3D additive `AttentionGate3D` at 3 decoder levels + multi-scale deep supervision (`aux2` `0.4`, `aux3` `0.2`) = **`4,706,772` params (`4.707M`, only `+5,858` params over SegResNet)**.
  - `StenosisAwareLoss`: `0.4 * Dice + 0.6 * Focal (γ = 2.5)` (`α = 0.4`, NOT `0.30`).
  - Exact 5-model verified parameter order: `SegResNet (4.701M) < RASNet (4.707M) < 3D U-Net (4.806M) < nnU-Net (30.786M, 6.54×) < V-Net (45.599M)`.

### Slide 5 — Matched 200-Epoch Training Protocols & Checkpoint Provenance
- **Table:** Condensed from `table_checkpoint_provenance_verified.md`.
- **Key Points:**
  - Exact losses & LRs: RASNet (`StenosisAwareLoss`, `lr = 2e-4`), SegResNet & V-Net (`DiceFocalLoss(1.0, 2.0)`, `lr = 1e-4`), 3D U-Net (`DiceCELoss`, `128³` resized, `lr = 1e-4`), nnU-Net V2 (`DC_and_CE_loss`, `3d_fullres` `[96, 160, 160]` at `[0.5, 0.3496, 0.3496]`, `SGD lr = 1e-2`).
  - Provenance clarity: `nnunet_best.pt` (`292` keys, `current_epoch = 199` under `_best_ema = 0.816239`) vs. raw `Pseudo dice` peak at ep `189` (`0.8376`) vs. `nnunet_last.pt` (`current_epoch = 200`). `rasnet_last.pth` (`114` keys) lacks embedded epoch metadata and rests on script/log/ZIP provenance.

### Slide 6 — Resolving the Figure 3 Training-Curve Discrepancy (R4)
- **Figure:** **`results/v1_reporting_correction_20261117_105000Z/fig3_corrected_training_and_val_convergence_200ep.png`** (replacing protected `results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.png`).
- **Key Points:**
  - Panel A shows unshifted non-comparable training losses (`DiceFocalLoss` fixed for SegResNet/V-Net).
  - Panel B distinguishes MONAI full-volume `Val94` trajectories (`12–20` sparse evals on `0.5 mm` grid), `3D U-Net` `128³` grid (`21` evals), and nnU-Net raw `Pseudo dice` (`mean_fg_dice`, ep `189` star `0.8376`) vs `ema_fg_dice` (ep `199` diamond `0.8162`) over `50` foreground-oversampled patch batches/epoch.
  - **Verbatim takeaway:** *"The logged metrics use different sampling, aggregation and spatial evaluation protocols and therefore cannot establish comparative full-volume performance."*

### Slide 7 — Final Epoch-200 Full-Volume Endpoint on Val94 (Regime A, R5)
- **Figure:** **`results/v1_reporting_correction_20261117_105000Z/fig_v1_epoch200_val94_endpoint_corrected.png`** (Panels A & B) + **`table_epoch200_val94_summary_and_paired_corrected.md`**.
- **Key Points (`N = 94`, symmetric 1-pass `0.50` Gauss SW, `τ = 0.50`, no CC, `0` empties, `188/188` geom OK):**
  - Dice: RASNet `0.7792 ± 0.0569` vs. nnU-Net `0.7767 ± 0.0562`; paired diff `+0.0025` (`95%` CI `[-0.0027, +0.0075]`, `54W/0T/40L`, unadj `p = 0.2401` — **neither superiority nor equivalence**).
  - Does NOT reconstruct epochs `50/100/150` (no intermediate checkpoints exist).

### Slide 8 — Epoch-200 Precision–Recall Trade-Off & Post-Hoc Multiplicity (R5)
- **Table:** **`table_epoch200_val94_posthoc_holm8_exploratory.md`** (clearly labeled **POST-HOC EXPLORATORY — NOT PREREGISTERED**).
- **Key Points:**
  - All 8 endpoint p-values are **exploratory / unadjusted**.
  - Under post-hoc Holm-8 (`K = 8`), only two differences remain `< 0.05`: **Precision favors RASNet** (`0.7586` vs. `0.7428`, `+0.0158`, Holm-8 `p = 7.47e-4`) and **Recall favors nnU-Net** (`0.8203` vs. `0.8065`, `-0.0138`, Holm-8 `p = 3.88e-9`).
  - `clDice` (`unadj p = 0.0352`, `95%` CI `[-0.0006, +0.0135]`, Holm-8 `p = 0.1760`) and `Centerline Recall` (`unadj p = 0.0092`, Holm-8 `p = 0.0551`) do not survive Holm-8; bootstrap CI (mean) and Wilcoxon (rank location) target different quantities.

### Slide 9 — Strict Separation of the Four V1 Evaluation Regimes (R6)
- **Table / Visual:** **`table_v1_four_regimes_comparison.md`** + Panel C of **`fig_v1_epoch200_val94_endpoint_corrected.png`**.
- **Key Points:**
  - **Regime A:** Final Epoch-200 Val94 (`2` models, `0.50` Gauss, no CC).
  - **Regime B1 / B2:** Best-Checkpoint Val94 (`B1` asymmetric Arm A; `B2` Option A symmetric + Top-2 CC).
  - **Regime C:** Retrospective Test150 Arms A/B/C (`5` models, asymmetric SW/TTA).
  - **Regime D:** Option A Symmetric Test150 (`4` sliding-window models, `3dunet` excluded, `0.50` Gauss SW).
  - Never mix scores or infer best-vs-final deltas across unmatched recipes.

### Slide 10 — Retrospective 3-Arm Benchmark (`Regime C`, `Test150`)
- **Figure / Table:** `results/thesis_outputs/Table2_test_150_arms_A_B_C_summary.md` & `results/thesis_outputs/fig1_multi_arm_metric_distributions.png` / `fig4_precision_recall_boundary_tradeoff.png`.
- **Key Points:**
  - Raw `Arm A` & native-TTA `Arm B`: RASNet leads Precision (`0.8032` / `0.8138` vs `0.7391` / `0.7536`) and raw HD95/ASD (`14.83` vs `21.10 mm`).
  - `Arm C` (`τ*` + Top-2 CC): Top-2 CC eliminates nnU-Net's background islands (`0.7687 -> 0.8055` Dice, `8.26 mm` HD95), while RASNet retains Precision (`0.8252` vs `0.8207`, `p = 0.160`). Disclose SW/TTA asymmetries and `τ = 0.30` grid floor.

### Slide 11 — Symmetric Option A Evaluation (`Regime D`, 4 Models, `Test150`)
- **Figures / Source:** `results/optionA_v1_symmetric_sw050/fig_optionA_dice_distributions.png`, `fig_optionA_paired_diff_rasnet_vs_nnunet.png`, `OPTION_A_RESULTS.md`.
- **Key Points:**
  - **Preregistered Primary `R4_tuned_tta8_top2`** (8-pass logit-avg TTA, `0.50` Gauss, Val-locked `τ*`, Top-2 CC):
    - Dice: nnU-Net `0.8059 ± 0.0630` vs. RASNet `0.8039 ± 0.0691` (diff `-0.0021`, `95%` CI `[-0.0088, +0.0030]`, **Non-inferiority `False` at `-0.005` margin**).
    - Primary `FA`: RASNet wins **Precision** (`+0.0218`, Holm `p = 7.23e-17`, `132W/0T/18L`); HD95/ASD n.s.
    - Secondary `FB/FC`: RASNet beats `SegResNet` (`+0.0050` Dice, `p = 4.27e-8`) and `V-Net` (`+0.0124` Dice, `p = 3.58e-12`).
    - Secondary `FD`: nnU-Net retains **Recall** (`+0.0244`, Holm `p = 2.40e-17`).
  - `R1–R3` kept strictly secondary (`R1`/`R2` favor RASNet, `R3` `NI = False`).

### Slide 12 — Computational Efficiency, Runtime & VRAM Scope (R6)
- **Figure / Table:** **`table_v1_efficiency_and_timing_verified.md`** + `results/optionA_v1_symmetric_sw050/fig_optionA_dice_vs_params.png`.
- **Key Points (RTX 4080 SUPER, explicitly scoped):**
  - **Model Size & Patch Compute (`Table 4`):** `RASNet` `4.707M` (`6.54×` smaller than `nnU-Net` `30.786M`; `SegResNet` is `4.701M`), `123.39` vs. `1,119.34` native (`9.07×`) / `402.96` equal-`96³` (`3.27×`) GFLOPs, `390.7` vs. `1,926.5 MB` patch VRAM.
  - **Full-Volume End-to-End Runtime (`Option A n = 10` Val cases):** `5.40s` vs. `29.84s` (`1-pass`); `22.95s` vs. `74.37s` (`TTA-8`). Caveat: Option A full-volume peak VRAM is higher for MONAI (`4,782–5,134 MB`) than nnU-Net (`2,243–2,261 MB`) due to `sw_batch_size = 8`.
  - Explicitly excluded invalid numbers: `manifest.json` `0.0s` (resume skip) and legacy `~1.85s` (hardcoded estimate).

### Slide 13 — Limitations & Non-Claimed Hypotheses
- **Key Points:**
  - `Test150` is retrospective (prior development inspection); `Val94` is a development split.
  - Non-inferiority not established under Option A `R4`; epoch-200 `p = 0.2401` is neither superiority nor equivalence.
  - V1 training differences (patch/spacing/normalization/LR) prevent causal attribution to attention gates or capacity across frameworks.
  - No untouched external multi-center cohort (`unseen_66` & hospital `N = 21/32` excluded); no clinical stenosis or `FFR_CT` / HeartFlow claims.

### Slide 14 — Conclusions & Future Work (`Rasnet-v2` & External Validation)
- **Key Points:**
  - **V1 Takeaway:** RASNet offers a compact `4.707M`-parameter architecture with verified Precision advantages at `τ = 0.50` / Option A `R4`, faster end-to-end inference under Option A, and higher Dice/Precision than matched MONAI `SegResNet`/`V-Net`, while `nnU-Net V2` retains Recall advantages and slightly higher post-processed `R4` Dice (`0.8059` vs. `0.8039`).
  - **Next Steps:** Controlled multi-seed symmetric training (`Rasnet-v2`; note Stage 2A/2B pilots ended in `FALLBACK` on `2026-11-16`, requiring careful pre-registered resolution before Stage 3) and prospective independent multi-center external evaluation.

---
**Integration Status Reminder:** External defense slide deck (`.pptx`/`.key`) integration remains **`OUTSTANDING`**.
