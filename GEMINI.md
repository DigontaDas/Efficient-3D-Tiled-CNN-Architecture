# 🧠 RASNet Thesis Project — Master Context & Verified Benchmark Briefing

> **Auto-Loaded Project Memory**: Antigravity automatically reads this file upon launching in this workspace.
> **Canonical Index**: Always consult [`CURRENT_EVIDENCE_INDEX.md`](CURRENT_EVIDENCE_INDEX.md) for the exact file paths and SHA-256 digests of all verified benchmark runs, audit tables, and thesis outputs.

---

## 📌 Project Overview
- **Repository**: `Efficient-3D-Tiled-CNN-Architecture` / `Thesis_RASNET`
- **Topic**: RASNet (Residual Attention Segmentation Network with Multi-Scale Deep Supervision & `StenosisAwareLoss`) — A Symmetric 3-Arm Retrospective Reassessment of 5 Matched 200-Epoch 3D Coronary Artery CCTA Segmentation Models on ImageCAS.
- **Author**: Nafis Mehedi (`Rytnix786 <nafismehedi37@gmail.com>`)
- **Methodological Status**: **Retrospective Reassessment** (`BENCHMARK_PREREGISTRATION.md`). Because historical development sweeps ($\tau = 0.60$, $\alpha=0.4, \gamma=2.5$) previously inspected the 150-case ImageCAS test split (`IDs 851–1000`), all evaluations on `851–1000` are reported as retrospective reassessments rather than untouched confirmatory tests.

---

## 🟢 Verified 200-Epoch 5-Model ImageCAS Benchmark (`N = 150` Test, `N = 94` Val)

All five models were trained for 200 epochs on the identical `N = 690` ImageCAS training split (`Phase3_Local_Integration/splits_final.json`, `sha256: efe3f06cfe...`) and evaluated on the `N = 94` validation split (`results/benchmark_runs/val_full_94/`) and `N = 150` test split (`results/benchmark_runs/test_retrospective_150/`) using true 6-connected physical surface distances (`surface-v1`) and native SimpleITK `(Z, Y, X)` coordinate inversion (`INFERENCE_SPEC.md`).

### 1. Arm A — Raw Single Pass ($\tau = 0.50$, No TTA, No Post-Processing; `N = 150`)
*Measures intrinsic network calibration and raw segmentation quality without heuristic cleanup:*
- **RASNet (Ours, 4.71M params)**: **Dice `0.7736 ± 0.0616`**, **IoU `0.6348 ± 0.0787`**, **Precision `0.8032 ± 0.0830`**, Recall `0.7521 ± 0.0718`, HD95 `14.83 ± 13.89 mm`, ASD `2.431 ± 1.916 mm`, **clDice `0.8389 ± 0.0686`**, Centerline Recall `0.8655 ± 0.0648`.
- **nnU-Net V2 (30.79M params)**: Dice `0.7687 ± 0.0668`, IoU `0.6289 ± 0.0856`, Precision `0.7391 ± 0.0967`, **Recall `0.8106 ± 0.0680`**, HD95 `21.10 ± 17.10 mm`, ASD `3.168 ± 2.636 mm`, clDice `0.8202 ± 0.0765`, **Centerline Recall `0.9111 ± 0.0556`**.
- **V-Net (45.60M params)**: Dice `0.7492 ± 0.0693`, IoU `0.6037 ± 0.0856`, Precision `0.7524 ± 0.1037`, Recall `0.7555 ± 0.0694`, HD95 `22.10 ± 18.62 mm`, ASD `3.335 ± 2.762 mm`, clDice `0.8164 ± 0.0780`, Centerline Recall `0.8561 ± 0.0627`.
- **SegResNet (4.70M params)**: Dice `0.7481 ± 0.0654`, IoU `0.6018 ± 0.0809`, Precision `0.7253 ± 0.0945`, Recall `0.7811 ± 0.0640`, HD95 `30.41 ± 18.11 mm`, ASD `4.571 ± 2.685 mm`, clDice `0.7941 ± 0.0790`, Centerline Recall `0.8950 ± 0.0561`.
- **3D U-Net (4.81M params)**: Dice `0.5517 ± 0.0480`, IoU `0.3824 ± 0.0449`, Precision `0.7269 ± 0.0722`, Recall `0.4484 ± 0.0526`, **HD95 `10.88 ± 7.25 mm`**, **ASD `1.999 ± 1.036 mm`**, clDice `0.7103 ± 0.0652`, Centerline Recall `0.6311 ± 0.0821`.

### 2. Arm B — Documented Native / Standard TTA ($\tau = 0.50$, No Post-Processing; `N = 150`)
*`RASNet` uses documented 4-pass spatial flip TTA; `nnU-Net V2` uses native 8-pass 3D mirroring; `SegResNet`, `V-Net`, and `3D U-Net` have no documented TTA (`1-pass`):*
- **RASNet (4-pass TTA)**: **Dice `0.7807 ± 0.0606`**, **IoU `0.6442 ± 0.0783`**, **Precision `0.8138 ± 0.0813`**, Recall `0.7563 ± 0.0724`, HD95 `13.69 ± 13.88 mm`, ASD `2.270 ± 1.863 mm`, **clDice `0.8453 ± 0.0677`**, Centerline Recall `0.8715 ± 0.0644`.
- **nnU-Net V2 (8-pass Mirroring)**: Dice `0.7760 ± 0.0641`, IoU `0.6384 ± 0.0833`, Precision `0.7536 ± 0.0936`, **Recall `0.8090 ± 0.0681`**, HD95 `19.25 ± 16.93 mm`, ASD `2.914 ± 2.487 mm`, clDice `0.8264 ± 0.0750`, **Centerline Recall `0.9096 ± 0.0562`**.

### 3. Arm C — Validation-Tuned Thresholds ($\tau^*$) + Identical Top-2 CC Pruning (`N = 150`)
*Single pass, validation-selected threshold $\tau^* \in [0.30..0.70]$ (`results/thresholds.json`: `0.30` for RASNet/V-Net/3D U-Net, `0.35` for nnU-Net/SegResNet), and identical 26-connected top-2 connected-component pruning across all 5 models:*
- **nnU-Net V2 ($\tau^*=0.35$, Top-2 CC)**: **Dice `0.8055 ± 0.0633`**, **IoU `0.6786 ± 0.0821`**, Precision `0.8207 ± 0.0676`, **Recall `0.7977 ± 0.0913`**, **HD95 `8.26 ± 9.81 mm`**, **ASD `1.395 ± 1.862 mm`**, **clDice `0.8739 ± 0.0679`**, **Centerline Recall `0.8752 ± 0.0908`**.
- **RASNet (Ours, $\tau^*=0.30$, Top-2 CC)**: Dice `0.7899 ± 0.0677`, IoU `0.6575 ± 0.0859`, **Precision `0.8252 ± 0.0650`**, Recall `0.7649 ± 0.0982`, HD95 `9.94 ± 11.25 mm`, ASD `1.671 ± 2.382 mm`, clDice `0.8646 ± 0.0759`, Centerline Recall `0.8256 ± 0.1049`.
- **SegResNet ($\tau^*=0.35$, Top-2 CC)**: Dice `0.7884 ± 0.0700`, IoU `0.6556 ± 0.0868`, Precision `0.8095 ± 0.0750`, Recall `0.7751 ± 0.0949`, HD95 `9.84 ± 11.35 mm`, ASD `1.717 ± 2.688 mm`, clDice `0.8664 ± 0.0776`, Centerline Recall `0.8463 ± 0.0994`.
- **V-Net ($\tau^*=0.30$, Top-2 CC)**: Dice `0.7710 ± 0.0736`, IoU `0.6328 ± 0.0914`, Precision `0.8054 ± 0.0765`, Recall `0.7488 ± 0.1048`, HD95 `12.38 ± 13.20 mm`, ASD `2.032 ± 2.490 mm`, clDice `0.8508 ± 0.0795`, Centerline Recall `0.8009 ± 0.1050`.
- **3D U-Net ($\tau^*=0.30$, Top-2 CC)**: Dice `0.6064 ± 0.0532`, IoU `0.4371 ± 0.0532`, Precision `0.5877 ± 0.0615`, Recall `0.6336 ± 0.0752`, HD95 `11.85 ± 8.74 mm`, ASD `2.158 ± 1.442 mm`, clDice `0.7664 ± 0.0705`, Centerline Recall `0.7206 ± 0.0947`.

*(Historical Asymmetric Recipe Diagnostic — `RASNet` with 4-pass TTA, $\tau=0.60$, Top-2 CC (`diagnostic_rasnet_historical_recipe_tau060_tta_top2.csv`): Dice `0.7769 ± 0.0694`, Precision `0.8805 ± 0.0528`, Recall `0.7019 ± 0.0977`, HD95 `10.30 ± 10.67 mm`, ASD `1.564 ± 1.812 mm`, clDice `0.8594 ± 0.0720`.)*

### 4. Paired Statistical Testing (`K = 3` Primary Family & `K = 32` Secondary Family)
- **Arm A (`RASNet` vs. `nnU-Net V2`, $N=150$, Holm-Bonferroni $K=3$)**:
  - `Precision`: $\Delta = +0.0642$ (95% CI $[+0.0565, +0.0728]$), $p = 6.42 \times 10^{-25}$, $p_{\text{Holm}} = 1.93 \times 10^{-24}$ (`139W / 0T / 11L`, $r_{rb} = +0.970$).
  - `HD95`: $\Delta = -6.27\text{ mm}$ (95% CI $[-8.43, -4.38]$), $p = 1.98 \times 10^{-9}$, $p_{\text{Holm}} = 3.96 \times 10^{-9}$ (`107W / 0T / 43L`, $r_{rb} = -0.565$).
  - `ASD`: $\Delta = -0.737\text{ mm}$ (95% CI $[-0.995, -0.503]$), $p = 2.37 \times 10^{-8}$, $p_{\text{Holm}} = 2.37 \times 10^{-8}$ (`107W / 0T / 43L`, $r_{rb} = -0.525$).
- **Arm B (`RASNet` vs. `nnU-Net V2`, $N=150$, Holm-Bonferroni $K=3$)**:
  - `Precision`: $\Delta = +0.0602$ (95% CI $[+0.0527, +0.0680]$), $p_{\text{Holm}} = 5.39 \times 10^{-24}$ (`141W / 0T / 9L`).
  - `HD95`: $\Delta = -5.56\text{ mm}$ (95% CI $[-7.67, -3.63]$), $p_{\text{Holm}} = 4.34 \times 10^{-8}$ (`103W / 0T / 47L`).
  - `ASD`: $\Delta = -0.644\text{ mm}$ (95% CI $[-0.885, -0.422]$), $p_{\text{Holm}} = 4.17 \times 10^{-7}$ (`98W / 0T / 52L`).
- **Arm C (`RASNet` vs. `nnU-Net V2`, $N=150$, Holm-Bonferroni $K=3$)**:
  - `Precision`: $\Delta = +0.0046$ (95% CI $[-0.0046, +0.0124]$), $p_{\text{Holm}} = 0.160$ (`84W / 0T / 66L`, no significant difference; `RASNet` numerically higher).
  - `HD95`: $\Delta = +1.68\text{ mm}$ (95% CI $[+0.50, +3.08]$), $p_{\text{Holm}} = 4.20 \times 10^{-4}$ (`55W / 1T / 94L` in favor of `nnU-Net V2`).
  - `ASD`: $\Delta = +0.277\text{ mm}$ (95% CI $[-0.058, +0.713]$), $p_{\text{Holm}} = 1.54 \times 10^{-4}$ (`51W / 0T / 99L` in favor of `nnU-Net V2`).
- **Arm C (`RASNet` vs. `SegResNet` Backbone, $N=150$, $K=32$ Secondary Family)**:
  - `Precision`: $\Delta = +0.0158$ (95% CI $[+0.0109, +0.0209]$), $p = 3.22 \times 10^{-9}$, $p_{\text{Holm}} = 4.50 \times 10^{-8}$ (`107W / 0T / 43L`).
  - `Dice`: $\Delta = +0.0016$ (95% CI $[-0.0020, +0.0053]$), $p = 0.0887$ (`88W / 0T / 62L`).

---

## 📂 Key Active File Locations
- **Canonical Evidence Index**: `CURRENT_EVIDENCE_INDEX.md`
- **Verified Benchmark Package**: `benchmark/` (`run_benchmark.py`, `metrics.py`, `postprocess.py`, `stats.py`, `thresholds.py`, `adapters/`)
- **Unit & Synthetic Dry-Run Tests (`31/31` passing)**: `tests/`
- **Core RASNet Architecture & Loss**: `Phase3_Local_Integration/rasnet_model.py`, `Phase3_Local_Integration/rasnet_loss.py`
- **Canonical Split Manifest**: `Phase3_Local_Integration/splits_final.json` (`690 train`, `94 val`, `150 test`)
- **200-Epoch Trainers & Verified Checkpoints**: `Q1_Publication_Package/matched_200ep_benchmark/` (`CHECKPOINTS.md`)
- **Verified Benchmark Runs**: `results/benchmark_runs/{val_full_94,test_retrospective_150,smoke_val_all5,smoke_verified_v2}/`
- **Frozen Validation Thresholds**: `results/thresholds.json`
- **2026-11-10 Verification Audit**: `results/audits/2026-11-10_verification_audit/`
- **Thesis Tables, Figures & Narrative**: `results/thesis_outputs/`
- **Archived Historical & Outside-Scope Assets**: `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/` (`ARCHIVE_INDEX.csv`)
