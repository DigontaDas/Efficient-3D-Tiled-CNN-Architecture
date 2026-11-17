# Corrected Thesis & Defense Text: Training Curves vs. Epoch-200 Full-Volume Val94 Endpoint

> **Integration Status (`OUTSTANDING`):**
> - Protected manuscript file [`results/thesis_outputs/THESIS_METHODS_RESULTS_AND_LIMITATIONS.md`](../../results/thesis_outputs/THESIS_METHODS_RESULTS_AND_LIMITATIONS.md) and [`results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.svg`](../../results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.svg) are preserved untouched per repository protection rules.
> - [`Answers-thesis-def.md`](../../Answers-thesis-def.md) was not overwritten, and external thesis manuscript/slide sources (`.tex`/`.docx`/`.pptx`) are not present in this repository. Integrating the replacement paragraphs below into the external thesis manuscript/defense deck remains **OUTSTANDING**.

---

## 1. Why the Original Logged Training Curves Are Not Directly Comparable

In `Q1_Publication_Package/matched_200ep_benchmark/`, the logged validation curves plotted in `results/thesis_outputs/fig3_training_and_val_n94_convergence_200ep.svg` measure **different quantities under different protocols**:
- **RASNet (and MONAI baselines):** `train_rasnet_200ep.py` computed full-volume sliding-window validation Dice (`96x96x96`, overlap `0.25`) across all `N = 94` validation scans on the `0.5 mm` isotropic foreground-cropped grid at 12 sparse checkpoints (every `20` epochs plus epochs `50`, `100`, `150`, and `200`).
- **nnU-Net V2:** `nnUNetTrainer_200epochs` logged `mean_fg_dice` (`Pseudo dice`) and `ema_fg_dice` at every epoch (`1..200`) over `50` validation batches of single `96x160x160` patches with `33.3%` forced foreground oversampling and batch-pooled Dice (`batch_dice`).
- **Loss axis note:** Panel A of `fig3` also plots non-comparable training objectives (`StenosisAwareLoss` positive values vs. nnU-Net negative Dice+CE loss; and note that Panel A's historical legend mislabels SegResNet and V-Net as `DiceCE` even though `train_segresnet_200ep.py` and `train_vnet_200ep.py` actually trained with `DiceFocalLoss(lambda_dice=1.0, lambda_focal=2.0)`).
- Because foreground-oversampled patches exclude most non-coronary thoracic background, nnU-Net's logged epoch-200 `Pseudo dice` (`0.8145`; `ema_fg_dice = 0.8161`) cannot be compared numerically against RASNet's logged full-volume Val Dice (`0.7761` at epoch 200). No intermediate epoch checkpoints (`50`, `100`, `150`) exist in the recovered archive, so intermediate trajectories on a common full-volume protocol were not measured.

---

## 2. What the Completed Epoch-200 Endpoint Evaluation Actually Shows (Val94, N = 94)

Evaluating both recovered epoch-200 final checkpoints (`rasnet_last.pth` via script/log/archive provenance, `nnunet_last.pt` with embedded `current_epoch = 200`) on the exact same `N = 94` native-grid validation scans under a frozen symmetric 1-pass protocol (`0.50` Gaussian sliding-window overlap, threshold `tau = 0.50`, no TTA, no connected-component pruning; `run_id = 20261117_041500Z`):

- **Primary descriptive endpoint (Dice):** RASNet achieved **`0.7792 ± 0.0569`** (median `0.7921`) and nnU-Net V2 achieved **`0.7767 ± 0.0562`** (median `0.7824`). The paired case-level difference (`RASNet − nnU-Net`) was **`+0.0025`** (95% bootstrap CI `[-0.0027, +0.0075]`, `54W / 0T / 40L`, Wilcoxon `p = 0.2401`). This difference is not statistically significant and **does not establish equivalence**.
- **Precision vs. Recall trade-off (including nnU-Net advantages):** Under this unpruned raw threshold (`tau = 0.50`), RASNet achieved higher **Precision** (`0.7586` vs. `0.7428`, paired diff `+0.0158`, 95% CI `[+0.0072, +0.0245]`, `66W / 0T / 28L`, `p = 1.07e-04`), whereas **nnU-Net V2 achieved higher voxel Recall (`0.8203` vs. `0.8065`, diff `-0.0138`, 95% CI `[-0.0175, -0.0098]`, `16W / 0T / 78L`, `p = 4.85e-10`) and higher Centerline Recall (`0.9142` vs. `0.9068`, diff `-0.0074`, 95% CI `[-0.0124, -0.0025]`, `37W / 2T / 55L`, `p = 0.0092`)**.
- **Surface and topological descriptors:** Mean `HD95` (`17.32` vs. `18.67 mm`, diff `-1.35 mm`, 95% CI `[-3.83, +0.95]`, `p = 0.6742`), `ASD` (`2.619` vs. `2.748 mm`, diff `-0.128 mm`, `p = 0.3372`), and `clDice` (`0.8413` vs. `0.8347`, diff `+0.0067`, 95% CI `[-0.0006, +0.0135]`, `p = 0.0352`) showed mixed or non-significant paired differences.
- **Strict scope limits:** Because only the final epoch-200 endpoint was re-evaluated (and Val94 was previously used during V1 development), these results do **not** show that RASNet outperformed nnU-Net throughout epochs `1–200`, do **not** imply equal training losses or absence of overfitting, and do **not** establish a causal effect of parameter count or capacity.

---

## 3. Ready-to-Paste Replacement Text for Thesis Section 2.1.1 & Defense Q&A

### Replacement Paragraph for Thesis Results (§2.1.1)

> **Reconciling Training-Log Validation Telemetry and the Completed Epoch-200 Endpoint (`N = 94`):**
> The training-log trajectories in Figure 3 reflect fundamentally different validation definitions: `train_rasnet_200ep.py` logged full-volume sliding-window Dice on the `0.5 mm` grid across all `94` validation scans every 20 epochs (plus epochs 50, 100, 150, and 200), whereas `nnUNetTrainer_200epochs` logged patch-level `Pseudo dice` (and its exponential moving average `ema_fg_dice`) over 50 foreground-oversampled patches per epoch. Consequently, nnU-Net's higher logged `Pseudo dice` (`0.8145` at epoch 200; peak `0.8376` at epoch 189) is an artifact of patch sampling rather than whole-scan generalization. When both recovered epoch-200 final checkpoints (`rasnet_last.pth` supported by script/log/archive provenance; `nnunet_last.pt` with embedded `current_epoch = 200`) are evaluated on the identical `94` full validation volumes under a single-pass, `0.50`-overlap Gaussian sliding-window protocol (`tau = 0.50`, no post-processing), both models reach comparable overall Dice (`0.7792 ± 0.0569` for RASNet vs. `0.7767 ± 0.0562` for nnU-Net V2; paired difference `+0.0025`, 95% bootstrap CI `[-0.0027, +0.0075]`, `p = 0.240`; not statistically significant, which does not establish equivalence). At this unpruned `tau = 0.50` operating point, RASNet exhibits higher Precision (`0.7586` vs. `0.7428`, `p = 1.07e-4`), while nnU-Net V2 exhibits higher voxel Recall (`0.8203` vs. `0.8065`, `p = 4.85e-10`) and Centerline Recall (`0.9142` vs. `0.9068`, `p = 0.0092`). Because intermediate epoch checkpoints (`50`, `100`, `150`) are absent from the archive, this single-endpoint check characterizes epoch 200 only and does not reconstruct a full-volume trajectory across earlier epochs. Best-checkpoint comparisons (`rasnet_best.pth` from epoch 160; `nnunet_best.pt` carrying `current_epoch = 199` under `_best_ema`, vs. raw `Pseudo dice` peak at epoch 189) use separate evaluation configurations (`val_full_94` Arm A and `Option A` Val94) and are reported separately.

### Defense Talking Points (Response to Examiner Question on Figure 3 Curves)

- **Why did nnU-Net's curve sit ~4–6% higher in Figure 3B?** Because nnU-Net logs `Pseudo dice` on 50 validation batches of foreground-oversampled `96x160x160` patches (`33.3%` forced foreground), excluding most thoracic background where false positives reduce full-volume precision.
- **What happens when we compare both models fairly at epoch 200 on the full `Val94` volumes?** Under symmetric single-pass `0.50` Gaussian sliding-window inference at `tau = 0.50` without pruning, full-volume Dice is `0.7792` (RASNet) vs. `0.7767` (nnU-Net), paired difference `+0.0025` (`95% CI [-0.0027, +0.0075]`, `p = 0.240`). RASNet is higher on Precision (`+0.0158`), while nnU-Net is higher on Recall (`+0.0138`) and Centerline Recall (`+0.0074`).
- **What can we NOT claim?** We cannot claim RASNet beat nnU-Net throughout training (no epoch 50/100/150 checkpoints exist), we cannot interpret `p = 0.240` as statistical equivalence, and we must keep best-checkpoint runs (`val_full_94` Arm A and `Option A` Val94 with Top-2 CC) distinct from the epoch-200 unpruned endpoint.
