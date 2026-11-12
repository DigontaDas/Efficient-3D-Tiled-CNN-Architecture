# Read-Only Forensic & Provenance Diagnostics (`LAB_CHECKS.md`)

**Generated**: `2026-11-07T13:41:30+06:00`  
**Workstation**: Windows 11 / PowerShell, `NVIDIA GeForce RTX 4080 SUPER (16,376 MiB VRAM)`  
**Python Environment Used**: `E:\SANPO_v2\.venv\Scripts\python.exe` (`PyTorch 2.14.1+cu126`)  
**Workspace Constraint**: Strictly read-only diagnostics. Zero code, configuration, checkpoint, or dataset files in `C:\Thesis_RASNET` were modified.

---

## Executive Summary of the 7 Checks

| Check | Diagnostic Topic | Summary Finding |
| :--- | :--- | :--- |
| **1** | **Threshold ($\tau=0.60$) & `StenosisAwareLoss` ($\alpha=0.4, \gamma=2.5$) Provenance** | **Direct test-set (`851–1000`) and unseen-cohort (`unseen_66`) selection.** No validation-set (`val`, $N=94$) sweep script or log exists anywhere in the repository. $\tau=0.60$ vs $\tau=0.50$ was compared directly on `splits["test"]` (`851–1000`, $N=150$) in `train_curriculum.py` / `eval_curriculum.py` and tracked every 5 epochs on `851–1000` in `train_and_track_v3.py`. On the 66 unseen cases, `evaluate_unseen_66_models.py` / `master_unseen_66_runner.py` swept $\tau \in \{0.60, 0.50, 0.45\}$ directly on the 66 unseen test cases. `StenosisAwareLoss` ($\alpha=0.4, \gamma=2.5$) was tuned from $\gamma=3.0 \to 2.5$ directly against the 150 test cases (`851–1000`), as explicitly stated in `rasnet_loss.py` lines 14–41 and `RASNet_Problems_and_Adaptations_Documentation.md` §3.1. |
| **2** | **ImageCAS Orientation (`nibabel.aff2axcodes`) & `RAS` Canonicalization Impact** | **All 1,000 ImageCAS cases (`1–1000`) have native `LAS` orientation** (`train`: 690, `val`: 94, `test`: 150, `unseen_66`: 66; `0` image/label mismatches). Across 3 `LAS` cases (`851, 852, 853`), when `Orientationd(axcodes="RAS")` is paired with `Invertd`, Dice differences vs native `LAS` are small ($\|\Delta\text{Dice}\| \le 0.0125$) because all models were trained with 50% `RandFlipd` on axes 0, 1, 2. **However**: (a) applying `RAS` canonicalization without inverting the X-axis flip collapses Dice to `0.0221–0.0345` (explaining legacy 3D U-Net `Dice = 0.0298`), and (b) `04_evaluate_models_3d_cas.py` and `evaluate_unseen_66_models.py` bypassed `nnUNetPredictor` and fed `(X,Y,Z)` tensors into `PlainConvUNet` without nnU-Net's `transpose_forward: [2, 1, 0]` (`(Z,Y,X)`), dropping nnU-Net V2 Dice on cases `851–853` from **`0.8251`** (native `nnUNetPredictor`) to **`0.6908`** (`-0.1343`). |
| **3** | **Provenance Table for All Thesis Results CSVs** | Complete provenance table with generating script, CLI arguments, file `LastWriteTime`, and checkpoint SHA-256 hashes established below. **Major provenance breaks identified**: (a) `Table8_component_ablation.csv` / `ablation_table.csv` / `ablation_results.csv` contain hardcoded numbers (`Step 2: 0.7565, Step 3: 0.7640, Step 4: 0.7715, Step 5: 0.7790`) that **contradict** the actual 150-case evaluation CSVs in `Q1_Publication_Package/ablation/evaluation_results/` (`Step 2: 0.7440, Step 3: 0.7535, Step 4: 0.7681, Step 5: 0.7745`); (b) `results/FINAL_MASTER_RESULTS.csv` hardcodes earlier Unseen-66 numbers (`RASNet 0.7376`, `SegResNet 0.7546`, `nnU-Net 0.4704`) that no longer match `unseen_66_*_case_metrics.csv` on disk (`RASNet 0.7497`, `SegResNet 0.7369`, `nnU-Net 0.6580`); (c) `3d_cas_3dunet_case_metrics.csv` has `Dice = 0.0000` across all 134 cases because `04_evaluate_models_3d_cas.py` ran sliding-window `96^3` on `0.5 mm` spacing instead of `128^3` `Resized` on `3D U-Net`; (d) `3d_cas_nnunet_case_metrics.csv` only evaluated `N=35` cases (`1–54`). |
| **4** | **Hospital Results CSVs: S2 Fallback (`10.0%`, `LAD`) & `NORMAL`/`PATENT` Count** | Across the verified `N=21` production CSVs (`qca_production_results_targeted.csv`, `qca_production_results_autonomous.csv`, `verified_cohort_baseline_results.csv`), **0 cases** have `rasnet_stenosis_pct == 10.0` (with `artery == "LAD"`) and **0 cases** have `NORMAL` or `PATENT` in `lesion_location`. In the pre-quarantine `N=32` CSVs (`hospital_cohort_clinical_agreement.csv` and `Phase3_Local_Integration/hospital_cohort_automated_stenosis.csv`), **0 cases** have `rasnet_stenosis_pct == 10.0`, while **3 cases** (`CT1`, `CT62`, `CT82`) have `radiologist_stenosis_pct == 10.0` and `lesion_location == "LAD Patent"` (all 3 were later excluded into `quarantined_cohort_cases.csv`). |
| **5** | **nnU-Net Inventory (`nnUNet_raw` / `nnUNet_preprocessed`)** | **Two distinct `Dataset501_CoronarySeg` inventories exist**: (1) In `_Archive_Historical_Experiments_and_Logs/all_four_validations/nnUNet_raw/Dataset501_CoronarySeg/dataset.json`, `numTraining = 850` (all cases `1–850`), which contains `690` train + `94` val + **all `66` unseen cases (`unseen_66`) as extras** (`0` missing from `train+val`, `66` extras = the entire `unseen_66` cohort; trained `2026-02-03`). (2) In `Q1_Publication_Package/matched_200ep_benchmark/`, `prepare_and_run_nnunet_200ep.py` linked exactly `784` cases (`690` train + `94` val from `splits_final.json`; `0` extras, `0` missing) into `Dataset501_CoronarySeg` (`predictions/nnu_net_v2/dataset.json` and `nnunet_best.pt['init_args']['dataset_json']` both confirm `numTraining: 784`), ran preprocessing and 200-epoch training on `2026-09-08 06:11:31`–`09:20:06` (`epoch_start_timestamps` in `nnunet_best.pt`), and then deleted `nnUNet_preprocessed` on line 269 via `shutil.rmtree(prep_data_dir)`. |
| **6** | **Unseen-66 Identity & De-duplication Check** | Across all 1,000 ImageCAS cases (`66` unseen vs `690` train, `94` val, `150` test): NIfTI header string fields (`db_name`, `descrip`, `aux_file`, `intent_name`) are anonymized/empty (`''`) in all 1,000 files; **0 exact compressed file SHA-256 duplicates**, **0 decompressed voxel-prefix SHA-256 duplicates**, and **0 geometry (`shape + zooms + qoffset`) duplicates** exist between the 66 unseen cases and any training, validation, or test case. |
| **7** | **`train_and_track_v3.py` Checkpoint Audit** | `train_and_track_v3.py` saved `rasnet_v3_best.pth`, `rasnet_v3_temp.pth`, `rasnet_v3_best_10ep.pth` and wrote `metrics_rasnet_v3_10ep.csv` (`N=150`, Dice `0.7665 ± 0.0707`, IoU `0.6264 ± 0.0868`, Precision `0.9002 ± 0.0535`, Recall `0.6737 ± 0.0948`, HD95 `10.4251 ± 10.6424 mm`). None of those `.pth` files remain on disk, and **none of the primary 200-epoch or Unseen-66 thesis tables use `metrics_rasnet_v3_10ep.csv`**, **EXCEPT** legacy table `_Archive_Historical_Experiments_and_Logs/all_four_validations/unified_comparison_table.csv` (Row: `RASNet v3 (10ep, +100 cases)`, `Dice 0.7665 ± 0.0705`) and duplicate file `Final_Generated_assets/imagecas_pipeline_validation/rasnet_development/metrics_rasnet_v3_10ep.csv`. |

---

## Check 1: Threshold ($\tau=0.60$) & `StenosisAwareLoss` ($\alpha=0.4, \gamma=2.5$) Provenance

### 1.1 Command Run
```powershell
& "E:\SANPO_v2\.venv\Scripts\python.exe" "C:\Users\user3\.gemini\antigravity\brain\fee83eef-8ec2-4394-83ae-6d9cc55b4667\scratch\run_checks_1_3_4_5_7.py"
```

### 1.2 Raw Output (Check 1 Section)
```text
================================================================================
CHECK 1: THRESHOLD (tau=0.60) & STENOSISAWARELOSS (alpha=0.4, gamma=2.5) PROVENANCE
================================================================================
splits_final.json summary:
  train: 690 cases (min=1, max=850)
  val  : 94 cases (min=14, max=845)
  test : 150 cases (min=851, max=1000)
  unseen_66 (in 1..200 but not in train/val/test): 66 cases -> ['3', '4', '5', '6', '7', '10', '13', '16', '21', '23'] ... ['185', '186', '189', '192', '199']

--- Code occurrences of threshold / StenosisAwareLoss ---
  3d Cas Validations and doings\scripts\01_run_experiment_audit.py:84: - Stenosis-aware compound loss modulation (`StenosisAwareLoss` with focal weighting $\\alpha=0.4, \\gamma=2.5$)
  3d Cas Validations and doings\scripts\04_evaluate_models_3d_cas.py:253: pred_mask = (pred_probs > 0.6).cpu().numpy().astype(np.uint8)
  3d Cas Validations and doings\scripts\10_ablation_and_reproducibility.py:31: # Step 4: + StenosisAwareLoss (Raw Single Pass)
  3d Cas Validations and doings\scripts\10_ablation_and_reproducibility.py:92: "Configuration": "+ StenosisAwareLoss (Raw Single Pass)",
  3d Cas Validations and doings\scripts\10_ablation_and_reproducibility.py:160: 2. **Boundary Error Compression**: Hausdorff Distance (HD95) drops by **over 3×** (from 31.48 mm down to 10.29 mm), demonstrating the sharp boundary delineation imparted by StenosisAwareLoss and Deep Supervision.
  3d Cas Validations and doings\scripts\11_related_work_and_clinical.py:82: "Paper Title": "RASNet: Residual Attention Segmentation Network with StenosisAwareLoss (Ours)",
  3d Cas Validations and doings\scripts\11_related_work_and_clinical.py:123: - **RASNet's Contribution**: RASNet demonstrates that combining an efficient residual backbone with 3D Attention Gates and targeted `StenosisAwareLoss` achieves state-of-the-art boundary accuracy (HD95: 10.29 mm) and clDice (0.8592) with only **4.71M parameters** and 123.39 GFLOPs.
  3d Cas Validations and doings\scripts\evaluate_unseen_66_models.py:183: thresh_val = 0.45
  3d Cas Validations and doings\scripts\train_rasnet_recall_tuned.py:6: 1. Compound Loss: StenosisAwareLoss (Dice + Focal) + Soft-Tversky Loss (alpha=0.3, beta=0.7).
  3d Cas Validations and doings\scripts\train_rasnet_recall_tuned.py:47: from rasnet_loss import StenosisAwareLoss
  3d Cas Validations and doings\scripts\train_rasnet_recall_tuned.py:69: Combines StenosisAwareLoss (Dice + Focal) with TverskyLoss (alpha=0.3, beta=0.7).
  3d Cas Validations and doings\scripts\train_rasnet_recall_tuned.py:75: self.stenosis_loss = StenosisAwareLoss(alpha=0.4, gamma=2.5)
  Phase3_Local_Integration\rasnet_loss.py:23: class StenosisAwareLoss(nn.Module):
  Phase3_Local_Integration\rasnet_loss.py:85: # Not part of the verified StenosisAwareLoss formulation.  Import directly
  Phase3_Local_Integration\rasnet_loss.py:98: NOTE: This is NOT included in StenosisAwareLoss.  It is preserved
  Phase3_Local_Integration\rasnet_loss.py:133: print("Testing StenosisAwareLoss (α=0.4, γ=2.5)...")
  Phase3_Local_Integration\rasnet_loss.py:134: loss_fn = StenosisAwareLoss()
  Q1_Publication_Package\ablation\02_ablation_study.py:11: Row 4: + StenosisAwareLoss (Dice + Focal loss, gamma=2.5, replacing plain loss) [Raw Model]
  Q1_Publication_Package\ablation\02_ablation_study.py:168: "Configuration": "+ StenosisAwareLoss (Raw Model)",
  Q1_Publication_Package\ablation\02_ablation_study.py:341: f.write(r"3. **StenosisAwareLoss (+0.0075 DSC, +0.0370 Precision)**: Dynamic focal modulation ($\gamma=2.5$) prevents over-penalization of sparse vessel voxels and sharpens narrow stenosis lumens." + "\n")
  Q1_Publication_Package\ablation\evaluate_ablation_suite.py:114: pred_mask = (pred_probs > 0.6).cpu().numpy().astype(np.uint8)
  Q1_Publication_Package\ablation\evaluate_ablation_suite.py:202: "name": "Step 4 (+ StenosisAwareLoss Raw Model)",
  Q1_Publication_Package\matched_200ep_benchmark\evaluate_matched_200ep_models.py:447: pred_bin = (fg_prob > 0.6).astype(np.uint8)
  Q1_Publication_Package\matched_200ep_benchmark\train_rasnet_200ep.py:8: - Loss: StenosisAwareLoss (Dice + Focal, alpha=0.4, gamma=2.5) with Multi-Scale Deep Supervision
  Q1_Publication_Package\matched_200ep_benchmark\train_rasnet_200ep.py:39: from rasnet_loss import StenosisAwareLoss
  Q1_Publication_Package\matched_200ep_benchmark\train_rasnet_200ep.py:152: criterion = StenosisAwareLoss(alpha=0.4, gamma=2.5)
  Q1_Publication_Package\matched_200ep_benchmark\train_rasnet_200ep.py:193: print(f"[*] Loss Function: StenosisAwareLoss (alpha=0.4, gamma=2.5) with Deep Supervision")
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\eval_curriculum.py:95: threshold: float = 0.6
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\eval_curriculum.py:348: threshold=0.6
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\eval_rasnet.py:246: pred_mask = (fg_prob > 0.6).cpu().numpy().astype(np.uint8)
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\finalize_curriculum.py:123: pred_mask = (fg_prob > 0.6).cpu().numpy().astype(np.uint8)
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\generalization_test.py:200: pred_mask = (fg_prob > 0.6).cpu().numpy().astype(np.uint8)
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\run_evaluation_after_fix.py:200: pred_mask = (fg_prob > 0.6).cpu().numpy().astype(np.uint8)
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\run_generalization_after_fix.py:173: pred_mask = (fg_prob > 0.6).cpu().numpy().astype(np.uint8)
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\run_training_after_fix.py:160: loss_fn = StenosisAwareLoss()  # α=0.4, γ=2.5
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\train_and_track_v3.py:180: pred_mask = (fg_prob > 0.6).cpu().numpy().astype(np.uint8)
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\train_and_track_v3.py:249: loss_fn = StenosisAwareLoss()  # Uses verified defaults: α=0.4, γ=2.5
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\train_curriculum.py:137: loss_fn = StenosisAwareLoss(alpha=0.5, beta=0.3, delta=0.2, gamma=2.5)
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\train_curriculum.py:268: s1_metrics = evaluate_checkpoint(s1_ckpt, s1_csv, s1_pred_dir, use_tta=True, threshold=0.6)
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\train_curriculum.py:313: s2_metrics = evaluate_checkpoint(s2_ckpt, s2_csv, s2_pred_dir, use_tta=True, threshold=0.6)
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\train_curriculum.py:356: final_metrics = evaluate_checkpoint(final_ckpt, final_csv_local, final_pred_dir, use_tta=True, threshold=0.6)
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\train_rasnet.py:154: loss_fn = StenosisAwareLoss()  # Uses verified defaults: α=0.4, γ=2.5
  _Archive_Historical_Experiments_and_Logs\legacy_pre_200ep_scripts\train_rasnet_v3.py:161: loss_fn = StenosisAwareLoss()  # Uses verified defaults: α=0.4, γ=2.5
```

### 1.3 Detailed Breakdown of Every Script/Log Selecting $\tau=0.60$ (and $\tau=0.45$) and $\alpha=0.4, \gamma=2.5$

| Script / Log File | Parameter(s) | Case IDs Used | In `851–1000` Test Range? | In `66` Unseen Cases? | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Phase3_Local_Integration/rasnet_loss.py` (Lines 14–41) & `_Archive_Historical_Experiments_and_Logs/RASNet_Problems_and_Adaptations_Documentation.md` (§3.1, Lines 69–74) | `StenosisAwareLoss` ($\alpha=0.4, \gamma=2.5$, reduced from $\gamma=3.0 \to 2.5$) | `splits["test"]` (`851–1000`, $N=150$) | **YES (all 150 test cases `851–1000`)** | No | Docstring explicitly states: *"This two-component loss was validated on the ImageCAS 150-case test set to achieve Precision 0.8533 and HD95 8.12 mm"* and *"Reducing $\gamma$ from $3.0 \to 2.5$ is critical"*. |
| `_Archive_Historical_Experiments_and_Logs/legacy_pre_200ep_scripts/train_curriculum.py` (Lines 137, 268, 313, 356, 420–421) & `eval_curriculum.py` (Lines 95, 348) | `threshold=0.6` vs `0.5`; `StenosisAwareLoss(alpha=0.5, beta=0.3, delta=0.2, gamma=2.5)` | Trained on `splits["train"]` (`1–850` subset, $N=690$); evaluated stages & compared `0.6 threshold` vs `0.5 threshold` on `splits["test"]` (`851–1000`, $N=150$) | **YES (all 150 test cases `851–1000`)** | No | `train_curriculum.py` Lines 420–421 explicitly log `RASNet v2 (0.6 threshold)` vs `RASNet v2 (0.5 threshold)` on the `851–1000` test set (`unified_comparison_table.csv`). |
| `_Archive_Historical_Experiments_and_Logs/legacy_pre_200ep_scripts/train_and_track_v3.py` (Lines 5, 180, 249, 259) | `fg_prob > 0.6`, `StenosisAwareLoss(alpha=0.4, gamma=2.5)` | Trained on `splits["train"][:100]`; evaluated every 5 epochs on `splits["test"]` (`851–1000`, $N=150$) to save `rasnet_v3_best.pth`! | **YES (all 150 test cases `851–1000`)** | No | Model checkpoint selection (`best_dice`) and $\tau=0.60$ evaluation occurred directly on the 150 test cases (`851–1000`) every 5 epochs during training. |
| `_Archive_Historical_Experiments_and_Logs/legacy_pre_200ep_scripts/eval_rasnet.py` (Line 246) & `run_evaluation_after_fix.py` (Line 200) | `fg_prob > 0.6` | `splits["test"]` (`851–1000`, $N=150$) and `--single-case 934` | **YES (`851–1000` and `934`)** | No | Evaluates `rasnet_best.pth` on `851–1000` using `fg_prob > 0.6`. |
| `Q1_Publication_Package/matched_200ep_benchmark/train_rasnet_200ep.py` (Line 152) & `evaluate_matched_200ep_models.py` (Line 447) | `StenosisAwareLoss(alpha=0.4, gamma=2.5)`; `fg_prob > 0.6` (for RASNet only; baselines use `0.5` / `argmax`) | Trained on `splits["train"]` ($N=690$), validated on `splits["val"]` ($N=94$, using `AsDiscrete(argmax=True)`, i.e. $\tau=0.50$), evaluated on `splits["test"]` (`851–1000`, $N=150$) with $\tau=0.60$ | **YES (`851–1000` at test time)** | No | During 200-epoch training, validation Dice used `argmax` ($\tau=0.50$), while test evaluation applied $\tau=0.60$ exclusively to RASNet. |
| `Q1_Publication_Package/ablation/evaluate_ablation_suite.py` (Line 114) | `pred_probs > 0.6` | `splits["test"]` (`851–1000`, $N=150$) | **YES (all 150 test cases `851–1000`)** | No | Applied $\tau=0.60$ across ablation steps on `851–1000`. |
| `3d Cas Validations and doings/scripts/04_evaluate_models_3d_cas.py` (Line 253) | `pred_probs > 0.6` (for RASNet; `0.5` for others) | Cases `1–200` (`134` in `train+val`, `66` in `unseen_66`) | No (`1–200`) | **YES (all 66 unseen cases in `1–200`)** | First evaluation on `unseen_66` with $\tau=0.60$ gave RASNet Dice `0.7376` vs SegResNet `0.7546`. |
| `3d Cas Validations and doings/scripts/evaluate_unseen_66_models.py` (Line 183) & `master_unseen_66_runner.py` (Lines 54–62) | `thresh_val = 0.45` (default) and `--threshold 0.5` | `unseen_66` (`3, 4, 5, ..., 199`, $N=66$) | No | **YES (all 66 unseen cases)** | Threshold was lowered from `0.60` $\to$ `0.50` (`Dice 0.7493`) $\to$ `0.45` (`Dice 0.7497`, with `overlap=0.7` and `dust_threshold=50`) directly on the 66 unseen cases. |

---

## Check 2: ImageCAS Orientation (`nibabel.aff2axcodes`) & 3-Case Inference With vs. Without `RAS` Canonicalization

### 2.1 Commands Run
```powershell
& "E:\SANPO_v2\.venv\Scripts\python.exe" "C:\Users\user3\.gemini\antigravity\brain\fee83eef-8ec2-4394-83ae-6d9cc55b4667\scratch\run_check_2a_and_6.py"
& "E:\SANPO_v2\.venv\Scripts\python.exe" "C:\Users\user3\.gemini\antigravity\brain\fee83eef-8ec2-4394-83ae-6d9cc55b4667\scratch\run_check_2b_inference.py"
```

### 2.2 Raw Output — Part A: Orientation Code (`nibabel.aff2axcodes`) Across All 1,000 ImageCAS Cases
```text
================================================================================
CHECK 2 (PART A): ORIENTATION CODES (nibabel.aff2axcodes) ACROSS ALL 1,000 CASES
================================================================================
Total cases per orientation (nibabel.aff2axcodes):
  LAS: 1000 cases
    - train     : 690 cases (first 5: [1, 2, 8, 9, 11], last 5: [846, 847, 848, 849, 850])
    - val       :  94 cases (first 5: [14, 45, 47, 51, 53], last 5: [830, 835, 838, 841, 845])
    - test      : 150 cases (first 5: [851, 852, 853, 854, 855], last 5: [996, 997, 998, 999, 1000])
    - unseen_66 :  66 cases (first 5: [3, 4, 5, 6, 7], last 5: [185, 186, 189, 192, 199])

Image vs Label orientation mismatches: 0

Orientation LAS (1000 cases):
  ALL 1,000 cases (1..1000) have orientation: LAS
```

### 2.3 Raw Output — Part B: 3 Cases (`851, 852, 853`, Orientation `LAS`) — Each Model's Dice vs. Ground Truth With and Without `RAS` Canonicalization
```text
Using device: cuda (NVIDIA GeForce RTX 4080 SUPER)

====================================================================================================
CHECK 2 (PART B): 3 CASES PER ORIENTATION (LAS: 851, 852, 853) — DICE WITH vs WITHOUT RAS CANONICALIZATION
====================================================================================================

--- Case 851 (Native Orientation: LAS, Shape: (512, 512, 217)) ---
  RASNet (1-pass, tau=0.6)  | WITH RAS (Invertd): 0.8268 | WITHOUT RAS (Native LAS): 0.8244 | Delta: +0.0024 | (RAS w/o orient invert: 0.0279)
  RASNet (4-pass TTA, 0.6)  | WITH RAS (Invertd): 0.8316 | WITHOUT RAS (Native LAS): 0.8339 | Delta: -0.0023 | (RAS w/o orient invert: 0.0276)
  SegResNet (1-pass, 0.5)   | WITH RAS (Invertd): 0.8209 | WITHOUT RAS (Native LAS): 0.8226 | Delta: -0.0017 | (RAS w/o orient invert: 0.0305)
  V-Net (1-pass, 0.5)       | WITH RAS (Invertd): 0.8180 | WITHOUT RAS (Native LAS): 0.8163 | Delta: +0.0017 | (RAS w/o orient invert: 0.0260)
  3D U-Net (Resized 128^3)  | WITH RAS (Invertd): 0.5992 | WITHOUT RAS (Native LAS): 0.5964 | Delta: +0.0028 | (RAS w/o orient invert: 0.0265)
  nnU-Net V2 (MONAI wrap)   | WITH RAS (Invertd): 0.7339 | WITHOUT RAS (Native LAS): 0.7337 | Delta: +0.0002 | (RAS w/o orient invert: 0.0027)

--- Case 852 (Native Orientation: LAS, Shape: (512, 512, 275)) ---
  RASNet (1-pass, tau=0.6)  | WITH RAS (Invertd): 0.7553 | WITHOUT RAS (Native LAS): 0.7757 | Delta: -0.0204 | (RAS w/o orient invert: 0.0614)
  RASNet (4-pass TTA, 0.6)  | WITH RAS (Invertd): 0.7686 | WITHOUT RAS (Native LAS): 0.7684 | Delta: +0.0002 | (RAS w/o orient invert: 0.0594)
  SegResNet (1-pass, 0.5)   | WITH RAS (Invertd): 0.7173 | WITHOUT RAS (Native LAS): 0.7135 | Delta: +0.0037 | (RAS w/o orient invert: 0.0535)
  V-Net (1-pass, 0.5)       | WITH RAS (Invertd): 0.7323 | WITHOUT RAS (Native LAS): 0.7444 | Delta: -0.0121 | (RAS w/o orient invert: 0.0584)
  3D U-Net (Resized 128^3)  | WITH RAS (Invertd): 0.5913 | WITHOUT RAS (Native LAS): 0.5708 | Delta: +0.0205 | (RAS w/o orient invert: 0.0585)
  nnU-Net V2 (MONAI wrap)   | WITH RAS (Invertd): 0.5766 | WITHOUT RAS (Native LAS): 0.5623 | Delta: +0.0143 | (RAS w/o orient invert: 0.0489)

--- Case 853 (Native Orientation: LAS, Shape: (512, 512, 275)) ---
  RASNet (1-pass, tau=0.6)  | WITH RAS (Invertd): 0.8479 | WITHOUT RAS (Native LAS): 0.8439 | Delta: +0.0039 | (RAS w/o orient invert: 0.0141)
  RASNet (4-pass TTA, 0.6)  | WITH RAS (Invertd): 0.8478 | WITHOUT RAS (Native LAS): 0.8462 | Delta: +0.0016 | (RAS w/o orient invert: 0.0139)
  SegResNet (1-pass, 0.5)   | WITH RAS (Invertd): 0.8196 | WITHOUT RAS (Native LAS): 0.8087 | Delta: +0.0109 | (RAS w/o orient invert: 0.0131)
  V-Net (1-pass, 0.5)       | WITH RAS (Invertd): 0.8394 | WITHOUT RAS (Native LAS): 0.8423 | Delta: -0.0029 | (RAS w/o orient invert: 0.0145)
  3D U-Net (Resized 128^3)  | WITH RAS (Invertd): 0.6075 | WITHOUT RAS (Native LAS): 0.6047 | Delta: +0.0029 | (RAS w/o orient invert: 0.0149)
  nnU-Net V2 (MONAI wrap)   | WITH RAS (Invertd): 0.7619 | WITHOUT RAS (Native LAS): 0.7388 | Delta: +0.0231 | (RAS w/o orient invert: 0.0147)

--- Mean across the 3 LAS cases (851, 852, 853) ---
  RASNet (1-pass, tau=0.6)  | Mean With RAS: 0.8100 | Mean Without RAS: 0.8147 | Diff (RAS - NoRAS): -0.0047 | Mean RAS w/o Invert: 0.0345
  RASNet (4-pass TTA, 0.6)  | Mean With RAS: 0.8160 | Mean Without RAS: 0.8162 | Diff (RAS - NoRAS): -0.0002 | Mean RAS w/o Invert: 0.0336
  SegResNet                 | Mean With RAS: 0.7859 | Mean Without RAS: 0.7816 | Diff (RAS - NoRAS): +0.0043 | Mean RAS w/o Invert: 0.0324
  V-Net                     | Mean With RAS: 0.7966 | Mean Without RAS: 0.8010 | Diff (RAS - NoRAS): -0.0044 | Mean RAS w/o Invert: 0.0330
  3D U-Net                  | Mean With RAS: 0.5993 | Mean Without RAS: 0.5906 | Diff (RAS - NoRAS): +0.0087 | Mean RAS w/o Invert: 0.0333
  nnU-Net V2 (MONAI wrap)   | Mean With RAS: 0.6908 | Mean Without RAS: 0.6783 | Diff (RAS - NoRAS): +0.0125 | Mean RAS w/o Invert: 0.0221
```

> **Note on `nnU-Net V2`**: In `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_nnu_net_v2_200ep.csv` (which used native `nnUNetPredictor` with `transpose_forward: [2, 1, 0]` and native target spacing `[0.5, 0.3496, 0.3496]`), cases `851, 852, 853` achieved Dice **`0.8315, 0.8068, 0.8369` (Mean `0.8251`)**. When `04_evaluate_models_3d_cas.py` and `evaluate_unseen_66_models.py` loaded `nnunet_best.pt` into a MONAI `Spacingd(0.5, 0.5, 0.5)` wrapper without transposing `(X, Y, Z) -> (Z, Y, X)`, nnU-Net's mean Dice on those same 3 cases dropped from **`0.8251` to `0.6908`** (`-0.1343`).

---

## Check 3: Provenance Table for All Thesis Results CSVs & Checkpoints

### 3.1 Checkpoint Inventory & SHA-256 Hashes

```text
--- Checkpoints directly on disk ---
  Q1_Publication_Package\matched_200ep_benchmark\checkpoints\ablation_attngate_best.pth | Size=18,873,080 B | MTime=2026-11-01 15:15:18 | SHA256=15be641cd4d693a5ce4b3a28c920c232c9d1aa8661af579fc3c701cddf8bb481
  Q1_Publication_Package\matched_200ep_benchmark\checkpoints\ablation_deepsup_best.pth  | Size=18,872,898 B | MTime=2026-11-01 15:15:19 | SHA256=a4f508f596cffb9b4e4b11f2880c804ed6a6d4d2c45c0c9507c807a2dd0a6a8f

--- Checkpoints inside multipart archive (checkpoints_200ep.zip.001 .. .010) ---
  [ZIP] segresnet_best.pt | Size=18,836,926 B  | ZipDate=2026-09-08 04:06:46 | SHA256=b8ea61d95106a0bbe15e1ce9f9e5af1a38d9508e59d0639d32bc883d93c6e109
  [ZIP] segresnet_last.pt | Size=18,836,926 B  | ZipDate=2026-09-08 04:29:44 | SHA256=b7181b2db171f8624c48d1cccdb5d9e1dc178647391cc68e8b8731f1245a70a5
  [ZIP] 3dunet_best.pth   | Size=19,253,157 B  | ZipDate=2026-09-08 13:03:48 | SHA256=d8a8f5ec95b5f34e2afe0b807ac49bb36d136f9d864238c27a763a29a800a60f
  [ZIP] 3dunet_last.pth   | Size=19,253,157 B  | ZipDate=2026-09-08 13:19:46 | SHA256=c652928e935c96a76894770dc11277c0bf2e78ead75929abb9010f9e4acea54f
  [ZIP] vnet_best.pt      | Size=182,479,290 B | ZipDate=2026-09-08 19:25:48 | SHA256=a89d71bd41c3e773c68e3f4b5e27c5a067f959154aa049eead76f1642c5c1063
  [ZIP] vnet_last.pt      | Size=182,479,290 B | ZipDate=2026-09-08 20:02:36 | SHA256=8dbecefe8f39e3dce08200c7a22308cee42cbabedf9f33d211bf3135f5fc94e5
  [ZIP] nnunet_best.pt    | Size=246,472,150 B | ZipDate=2026-09-08 09:19:08 | SHA256=9accddf283b1c480e386f30f426b9200ca8d4aa1ecd70d5b5af0b8aad6058711
  [ZIP] nnunet_last.pt    | Size=246,472,688 B | ZipDate=2026-09-08 09:20:06 | SHA256=2105bd7238fff7485f1a8d72622f977f635f441e6d44863920c85709b02ffec2
  [ZIP] rasnet_best.pth   | Size=18,871,718 B  | ZipDate=2026-09-09 02:06:02 | SHA256=8d4c25cd12d07c40423062b27b1d36836e5fd6cecb30c20c34055eeac976d1b8
  [ZIP] rasnet_last.pth   | Size=18,871,718 B  | ZipDate=2026-09-09 03:31:22 | SHA256=71c9e5ef4a02849686c99ccb2d1544ff78c78187e1c0d53e7b22884c708c1b95
```

### 3.2 Provenance Table of Results CSVs Used in Thesis Tables

> **Note on File Modified Times (`MTime`)**: All repository files cloned/extracted onto this workstation carry the checkout timestamp `2026-11-01 15:15:17` – `2026-11-01 15:15:21`. Internal checkpoint timestamps (`ZipDate` and `epoch_start_timestamps`) record the actual training dates (`2026-09-08` to `2026-09-09`).

| Results CSV Path | Rows | Mean Dice | Generating Script & Arguments | File Modified Time (`MTime`) | Checkpoint Used & SHA-256 Hash | Provenance Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_rasnet_200ep.csv` | 150 | `0.7765` | `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py` (no flags; 4-pass TTA, $\tau=0.60$, `cc3d` top-2) | `2026-11-01 15:15:21` | `rasnet_best.pth` (`8d4c25cd12d07c40...`) | **ESTABLISHED** |
| `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_nnu_net_v2_200ep.csv` | 150 | `0.7687` | `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py` (via `nnUNetPredictor`, `fold_0`, `checkpoint_best.pth`, `use_mirroring=False`) | `2026-11-01 15:15:21` | `nnunet_best.pt` (`9accddf283b1c480...`) | **ESTABLISHED** |
| `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_v_net_200ep.csv` | 150 | `0.7474` | `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py` (1-pass, `argmax`) | `2026-11-01 15:15:21` | `vnet_best.pt` (`a89d71bd41c3e773...`) | **ESTABLISHED** |
| `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_segresnet_200ep.csv` | 150 | `0.7469` | `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py` (1-pass, `argmax`) | `2026-11-01 15:15:21` | `segresnet_best.pt` (`b8ea61d95106a0bb...`) | **ESTABLISHED** |
| `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/metrics_3d_u_net_200ep.csv` | 150 | `0.5561` | `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py` (`Resized(128,128,128)`, `probs > 0.5`) | `2026-11-01 15:15:21` | `3dunet_best.pth` (`d8a8f5ec95b5f34e...`) | **ESTABLISHED** |
| `Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/benchmark_200ep_summary_table.csv` & `benchmark_200ep_significance_tests.csv` | 5 / 32 | — | `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py` (aggregates the 5 CSVs above) | `2026-11-01 15:15:21` | All 5 checkpoints above | **ESTABLISHED** |
| `Q1_Publication_Package/stats/unified_benchmark_comparison_table.csv`, `stats_significance_table.csv`, `bootstrap_CI_table.csv` | 5 / 32 / 8 | — | `Q1_Publication_Package/stats/01_significance_testing.py` (reads the 5 `metrics_*_200ep.csv` files above) | `2026-11-01 15:15:21` | All 5 checkpoints above | **ESTABLISHED** |
| `Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step2_attngate_200ep.csv` | 150 | `0.7440` | `Q1_Publication_Package/ablation/evaluate_ablation_suite.py` (1-pass, $\tau=0.60$) | `2026-11-01 15:15:18` | `ablation_attngate_best.pth` (`15be641cd4d693a5...`) | **ESTABLISHED (Case-Level CSV)** |
| `Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step3_deepsup_200ep.csv` | 150 | `0.7535` | `Q1_Publication_Package/ablation/evaluate_ablation_suite.py` (1-pass, $\tau=0.60$) | `2026-11-01 15:15:18` | `ablation_deepsup_best.pth` (`a4f508f596cffb9b...`) | **ESTABLISHED (Case-Level CSV)** |
| `Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step4_raw_model_200ep.csv` | 150 | `0.7681` | `Q1_Publication_Package/ablation/evaluate_ablation_suite.py` (1-pass, $\tau=0.60$, no TTA, no `cc3d`) | `2026-11-01 15:15:18` | `rasnet_best.pth` (`8d4c25cd12d07c40...`) | **ESTABLISHED (Case-Level CSV)** |
| `Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step5_tta_200ep.csv` | 150 | `0.7745` | `Q1_Publication_Package/ablation/evaluate_ablation_suite.py` (4-pass TTA, $\tau=0.60$, no `cc3d`) | `2026-11-01 15:15:18` | `rasnet_best.pth` (`8d4c25cd12d07c40...`) | **ESTABLISHED (Case-Level CSV)** |
| `Q1_Publication_Package/ablation/ablation_table.csv`, `3d Cas Validations and doings/results/ablation_results.csv`, & `final_tables/Table8_component_ablation.csv` | 6 | `0.7606` / `0.7650` | `Q1_Publication_Package/ablation/02_ablation_study.py` and `3d Cas Validations and doings/scripts/10_ablation_and_reproducibility.py` (Lines 64–105) | `2026-11-01 15:15:17` | **NONE (Hardcoded dictionaries in scripts)** | **UNESTABLISHED / CONTRADICTS CASE CSVs** (`10_ablation_and_reproducibility.py` hardcodes `0.7565, 0.7640, 0.7715, 0.7790`; `ablation_table.csv` has `0.7440, 0.7535, 0.7681, 0.7745` in `Q1_Publication_Package/ablation/ablation_table.csv` vs `0.7565` in `results/ablation_results.csv`) |
| `Q1_Publication_Package/efficiency/efficiency_table.csv`, `3d Cas Validations and doings/results/computational_cost.csv`, & `Table7_computational_cost.csv` | 5 / 4 | — | `Q1_Publication_Package/efficiency/03_efficiency_benchmark.py` and `3d Cas Validations and doings/scripts/09_computational_cost_benchmark.py` | `2026-11-01 15:15:18` | Architecture graph profiling (unweighted instantiation) | **ESTABLISHED (Synthetic tensor profiling)** |
| `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_rasnet_case_metrics.csv` | 66 | `0.7497` | `3d Cas Validations and doings/scripts/evaluate_unseen_66_models.py --model rasnet --sw-batch-size 8 --overlap 0.7 --dust-threshold 50 --overwrite` (default $\tau=0.45$, 1-pass) | `2026-11-01 15:15:17` | `rasnet_best.pth` (`8d4c25cd12d07c40...`) | **ESTABLISHED** |
| `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_rasnet_thresh05_case_metrics.csv` | 66 | `0.7493` | `evaluate_unseen_66_models.py --model rasnet --threshold 0.5 --sw-batch-size 8 --overlap 0.7 --dust-threshold 50 --overwrite` | `2026-11-01 15:15:17` | `rasnet_best.pth` (`8d4c25cd12d07c40...`) | **ESTABLISHED** |
| `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_segresnet_case_metrics.csv` | 66 | `0.7369` | `evaluate_unseen_66_models.py --model segresnet --sw-batch-size 8 --overlap 0.7 --dust-threshold 50 --overwrite` | `2026-11-01 15:15:18` | `segresnet_best.pt` (`b8ea61d95106a0bb...`) | **ESTABLISHED** |
| `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_vnet_case_metrics.csv` | 66 | `0.7272` | `evaluate_unseen_66_models.py --model vnet --sw-batch-size 8 --overlap 0.7 --dust-threshold 50 --overwrite` | `2026-11-01 15:15:18` | `vnet_best.pt` (`a89d71bd41c3e773...`) | **ESTABLISHED** |
| `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_nnunet_case_metrics.csv` | 66 | `0.6580` | `evaluate_unseen_66_models.py --model nnunet --sw-batch-size 8 --overlap 0.7 --dust-threshold 50 --overwrite` (Note: broken MONAI wrapper without axis transpose) | `2026-11-01 15:15:17` | `nnunet_best.pt` (`9accddf283b1c480...`) | **ESTABLISHED (Broken MONAI wrapper)** |
| `3d Cas Validations and doings/results/unseen_66_cohort/metrics/unseen_66_3dunet_case_metrics.csv` | 66 | `0.5428` | `evaluate_unseen_66_models.py --model 3dunet --sw-batch-size 8 --overlap 0.7 --dust-threshold 50 --overwrite` | `2026-11-01 15:15:17` | `3dunet_best.pth` (`d8a8f5ec95b5f34e...`) | **ESTABLISHED** |
| `3d Cas Validations and doings/results/unseen_66_cohort/final_tables/Table_Unseen66_Model_Comparison.csv`, `Table_Unseen66_Wilcoxon_Significance.csv`, `Table_Unseen66_95CI_Bootstrap.csv` | 6 / 45 / 54 | — | `3d Cas Validations and doings/scripts/master_unseen_66_runner.py` (aggregates the 6 `unseen_66_*_case_metrics.csv` files above) | `2026-11-01 15:15:17` | All 5 checkpoints above | **ESTABLISHED** |
| `3d Cas Validations and doings/results/3d_cas_rasnet_case_metrics.csv`, `segresnet`, `vnet`, `3dunet` (`N=134`), `nnunet` (`N=35`) & `Table3_3d_cas_model_comparison.csv` | 134 / 35 / 10 | `0.7765` (RASNet), `0.0000` (3DUNet) | `3d Cas Validations and doings/scripts/04_evaluate_models_3d_cas.py` and `06_aggregate_and_compare.py` | `2026-11-01 15:15:17` | 200-epoch checkpoints (`rasnet_best.pth`, etc.) | **PARTIAL / CORRUPTED** (`134` cases are `train+val` leak; `3dunet` has `0.0000` Dice due to missing `Resized(128^3)`; `nnunet` aborted at `N=35`; `66` unseen rows in `Table3` were overwritten when `unseen_66` CSVs were regenerated) |
| `results/FINAL_MASTER_RESULTS.csv` & `results/case_level_model_comparison.csv` | 17 / 77 | — | `scratch/build_master_results.py` | `2026-11-01 15:15:21` | Mixed / Hardcoded fallback strings in `build_master_results.py` | **UNESTABLISHED FOR UNSEEN-66 ROWS** (Hardcodes earlier `unseen_66` run: `RASNet 0.7376 @ 0.60`, `0.7397 @ 0.50`, `SegResNet 0.7546`, `nnU-Net 0.4704` whose source CSVs were overwritten by `master_unseen_66_runner.py`) |
| `results-after-hallucin-fix/metrics_rasnet.csv` | 150 | `0.7862` | `_Archive_Historical_Experiments_and_Logs/legacy_pre_200ep_scripts/run_evaluation_after_fix.py` | `2026-11-01 15:15:21` | Historical pre-200ep `rasnet_best.pth` (not on disk; superseded by 200ep `8d4c25cd12d07c40...`) | **UNESTABLISHED CHECKPOINT HASH (Legacy checkpoint overwritten)** |
| `Q1_Publication_Package/clinical_validation/qca_production_results_targeted.csv`, `qca_production_results_autonomous.csv`, & `production_dual_mode_comparison.csv` | 21 / 21 / 2 | — | `Phase3_Local_Integration/production_qca_engine.py` (evaluates pre-existing `pred_mask.nii.gz` in `Phase3_Local_Integration/segresnet_finetuned/inference_results/CT{id}/pred_mask.nii.gz`) | `2026-11-01 15:15:18` | Pre-computed masks in `segresnet_finetuned/inference_results/` (no checkpoint hash recorded at mask generation time) | **PARTIAL (QCA script verified; mask generation checkpoint unrecorded)** |
| `Q1_Publication_Package/clinical_validation/hospital_cohort_clinical_agreement.csv` & `verified_cohort_baseline_results.csv` | 32 / 21 | — | Legacy scripts in `Phase3_Local_Integration/_archive_legacy_postprocess/` (`07_clinical_stenosis_validation.py` / `provenance_audit.py`) | `2026-11-01 15:15:18` | Pre-computed masks + legacy postprocessing | **LEGACY / QUARANTINED** |

---

## Check 4: Hospital Results CSVs — S2 Fallback Signature (`10.0%` & `LAD`) and `NORMAL`/`PATENT` Cases

### 4.1 Code Origin of the S2 Fallback Signature
In `Phase3_Local_Integration/production_qca_engine.py` (Line 259):
```python
if best_lesion is None:
    mean_d = float(np.mean([b["mean_d"] for b in branch_ candidates])) if branch_candidates else 2.5
    best_lesion = {"ds": 10.0, "as": 19.0, "mld": mean_d, "ref_d": mean_d, "artery": "LAD"}
```

### 4.2 Exact Counts & Case Lists Across All Hospital CSVs

1. **`Q1_Publication_Package/clinical_validation/qca_production_results_targeted.csv` (`N=21` verified cases)**:
   - Cases with `rasnet_stenosis_pct == 10.0` and `artery == "LAD"`: **0 cases** (lowest `rasnet_stenosis_pct` is `CT78 = 7.1%`, `artery = LCx`).
   - Cases with `radiologist_stenosis_pct == 10.0` and `artery == "LAD"`: **0 cases**.
   - Cases whose `lesion_location` contains `NORMAL` or `PATENT`: **0 cases**.

2. **`Q1_Publication_Package/clinical_validation/qca_production_results_autonomous.csv` (`N=21` verified cases)**:
   - Cases with `rasnet_stenosis_pct == 10.0` and `artery == "LAD"`: **0 cases** (lowest `rasnet_stenosis_pct` is `CT78 = 29.6%`, `artery = RCA`).
   - Cases with `radiologist_stenosis_pct == 10.0` and `artery == "LAD"`: **0 cases**.
   - Cases whose `lesion_location` contains `NORMAL` or `PATENT`: **0 cases**.

3. **`Q1_Publication_Package/clinical_validation/verified_cohort_baseline_results.csv` (`N=21` verified cases)**:
   - Cases with `rasnet_stenosis_pct == 10.0`: **0 cases**.
   - Cases with `radiologist_stenosis_pct == 10.0`: **0 cases**.
   - Cases whose `lesion_location` contains `NORMAL` or `PATENT`: **0 cases**.

4. **`Q1_Publication_Package/clinical_validation/hospital_cohort_clinical_agreement.csv` (`N=32` pre-quarantine cohort)**:
   - Cases with `rasnet_stenosis_pct == 10.0`: **0 cases** (`CT1 = 0.0%`, `CT62 = 20.4%`, `CT82 = 7.3%`).
   - Cases with `radiologist_stenosis_pct == 10.0` (and `lesion_location` in `LAD`): **3 cases** (`CT1`, `CT62`, `CT82`).
   - Cases whose `lesion_location` contains `NORMAL` or `PATENT`: **3 cases**:
     ```text
     case_id  rasnet_stenosis_pct  radiologist_stenosis_pct        rasnet_cad_rads  radiologist_cad_rads  lesion_location  difference
         CT1                  0.0                      10.0      CAD-RADS 0 (None)  CAD-RADS 1 (Minimal)       LAD Patent       -10.0
        CT62                 20.4                      10.0   CAD-RADS 1 (Minimal)  CAD-RADS 1 (Minimal)       LAD Patent        10.4
        CT82                  7.3                      10.0   CAD-RADS 1 (Minimal)  CAD-RADS 1 (Minimal)       LAD Patent        -2.7
     ```

5. **`Phase3_Local_Integration/hospital_cohort_automated_stenosis.csv` (`N=32` pre-quarantine cohort)**:
   - Cases with `rasnet_stenosis_pct == 10.0`: **0 cases** (closest is `CT67 = 10.4%`, `lesion_location = LAD/LCx`).
   - Cases with `radiologist_stenosis_pct == 10.0`: **3 cases** (`CT1`, `CT62`, `CT82`).
   - Cases whose `lesion_location` contains `NORMAL` or `PATENT`: **3 cases** (`CT1`, `CT62`, `CT82`, all with `lesion_location = "LAD Patent"`):
     ```text
     case_id,centerline_points,mean_diameter_mm,min_diameter_mm,ref_diameter_mm,rasnet_stenosis_pct,rasnet_cad_rads,cad_rads_grade,radiologist_stenosis_pct,radiologist_cad_rads,lesion_location,difference
     CT1,2010,3.33,1.663,2.66,37.5,CAD-RADS 2 (Mild),CAD-RADS 2 (Mild),10.0,CAD-RADS 1 (Minimal),LAD Patent,27.5
     CT62,961,2.996,1.215,2.409,49.5,CAD-RADS 2 (Mild),CAD-RADS 2 (Mild),10.0,CAD-RADS 1 (Minimal),LAD Patent,39.5
     CT82,669,3.258,3.289,4.971,33.8,CAD-RADS 2 (Mild),CAD-RADS 2 (Mild),10.0,CAD-RADS 1 (Minimal),LAD Patent,23.8
     ```

---

## Check 5: nnU-Net Inventory (`nnUNet_raw` / `nnUNet_preprocessed`) vs. Expected Case List

### 5.1 Commands Run
```powershell
& "E:\SANPO_v2\.venv\Scripts\python.exe" -c "import os, json, glob, zipfile, io, torch ..." # (Full command in task-128 & task-134 logs)
```

### 5.2 Raw Output
```text
=== FILE: Q1_Publication_Package\matched_200ep_benchmark\predictions\nnu_net_v2\dataset.json
  Content: {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'vessel': 1}, 'numTraining': 784, 'file_ending': '.nii.gz'}

=== FILE: _Archive_Historical_Experiments_and_Logs\all_four_validations\nnUNet_raw\Dataset501_CoronarySeg\dataset.json
numTraining: 850 len(training): 850 len(test): 0
  Unique IDs in dataset.json training: 850
  In splits train (690): 690
  In splits val (94): 94
  In splits test (150): 0
  In unseen_66 (66): 66
  Extras vs train+val: 66 ['3', '4', '5', '6', '7', '10', '13', '16', '21', '23', '26', '27', '28', '31', '32', '33', '35', '38', '44', '48', '54', '62', '64', '66', '73', '74', '75', '82', '83', '86', ...]
  Missing vs train+val: 0 []

=== FILE: _Archive_Historical_Experiments_and_Logs\all_four_validations\mandatory_artifacts_nnunet\checkpoints\fold_0\debug.json
  configuration_name: 3d_fullres
  current_epoch: 0
  fold: 0
  gpu_name: NVIDIA RTX A6000
  initial_lr: 0.01
  log_file: C:\T2520694\nnunet_results\Dataset501_CoronarySeg\nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres\fold_0\training_log_2026_2_3_20_49_03.txt
  num_epochs: 100
  output_folder: C:\T2520694\nnunet_results\Dataset501_CoronarySeg\nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres\fold_0
  preprocessed_dataset_folder: C:\T2520694\nnunet_preprocessed\Dataset501_CoronarySeg\nnUNetPlans_3d_fullres

=== ZIP CKPT: nnunet_best.pt type: <class 'dict'>
  keys: ['network_weights', 'optimizer_state', 'grad_scaler_state', 'logging', '_best_ema', 'current_epoch', 'init_args', 'trainer_name', 'inference_allowed_mirroring_axes']
  current_epoch: 199
  _best_ema: 0.8162392377853394
  init_args keys: ['plans', 'configuration', 'fold', 'dataset_json', 'device']
  init_args.dataset_json numTraining: 784 keys: ['channel_names', 'labels', 'numTraining', 'file_ending']
  init_args.plans dataset_name: Dataset501_CoronarySeg fg_props: {'0': {'max': 3069.0, 'mean': 147.97113037109375, 'median': 109.0, 'min': -1015.0, 'percentile_00_5': -166.0, 'percentile_99_5': 727.0, 'std': 179.68487548828125}}
  epoch_start_timestamps: len=199, first=1788826291.0572646 (2026-09-08 06:11:31), last=1788837493.4687836 (2026-09-08 09:18:13)
  epoch_end_timestamps: len=199, first=1788826352.9632611 (2026-09-08 06:12:32), last=1788837549.5037923 (2026-09-08 09:19:09)
```

### 5.3 Summary of Both nnU-Net Inventories Found
1. **Archive / Legacy 100-Epoch Inventory (`_Archive_Historical_Experiments_and_Logs/all_four_validations/nnUNet_raw/Dataset501_CoronarySeg/dataset.json`)**:
   - **Cases present**: `850` (`case_0001` through `case_0850`).
   - **Expected (`splits_final.json` `train + val`)**: `784` (`690` train + `94` val).
   - **Extras**: **`66` cases** — specifically the entire `unseen_66` cohort (`[3, 4, 5, 6, 7, 10, 13, 16, 21, 23, 26, 27, 28, 31, 32, 33, 35, 38, 44, 48, 54, 62, 64, 66, 73, 74, 75, 82, 83, 86, 87, 94, 96, 97, 98, 100, 103, 104, 107, 108, 117, 129, 131, 134, 137, 141, 142, 144, 146, 155, 156, 158, 160, 163, 165, 166, 170, 171, 176, 177, 178, 184, 185, 186, 189, 192, 199]`).
   - **Missing**: `0` cases.
   - **Preprocessing / Training Timestamp**: `2026-02-03 12:09:59` – `2026-02-03 20:49:03` (`training_log_2026_2_3_*.txt`).
2. **Matched 200-Epoch Benchmark Inventory (`Q1_Publication_Package/matched_200ep_benchmark/`)**:
   - **Cases present**: `784` (`numTraining: 784` verified in both `predictions/nnu_net_v2/dataset.json` and `nnunet_best.pt['init_args']['dataset_json']`, generated by `prepare_and_run_nnunet_200ep.py` Lines 46–50 from `splits_final.json["train"] + splits_final.json["val"]`).
   - **Extras vs `train + val`**: **`0` cases** (Note: the foreground intensity fingerprint in `plans.json` changed from `mean: 148.2419, median: 110.0, p99_5: 733.0` on the 850-case run to `mean: 147.9711, median: 109.0, p99_5: 727.0` on the 784-case run, confirming `nnUNetv2_plan_and_preprocess` ran cleanly on the 784 cases).
   - **Missing vs `train + val`**: **`0` cases**.
   - **Why `nnUNet_preprocessed` is absent on disk**: `prepare_and_run_nnunet_200ep.py` Line 269 explicitly executed `shutil.rmtree(prep_data_dir, ignore_errors=True)` after epoch 200 completed to reclaim ~125 GB of disk space.
   - **Preprocessing / Training Timestamp**: Preprocessing completed immediately before Epoch 0 started at Unix timestamp `1788826291.06` (**`2026-09-08 06:11:31`**), and Epoch 200 finished at `1788837606.10` (**`2026-09-08 09:20:06`**).

---

## Check 6: Unseen-66 Identity — Image Hashes & Header Patient IDs vs. Training (`N=690`) and Validation (`N=94`)

### 6.1 Command Run
```powershell
& "E:\SANPO_v2\.venv\Scripts\python.exe" "C:\Users\user3\.gemini\antigravity\brain\fee83eef-8ec2-4394-83ae-6d9cc55b4667\scratch\run_check_2a_and_6.py"
```

### 6.2 Raw Output (Check 6 Section)
```text
================================================================================
CHECK 6: UNSEEN-66 IDENTITY & DUPLICATE AUDIT VS TRAIN (690) & VAL (94) & TEST (150)
================================================================================
Unseen-66 count: 66
Train + Val count: 784 (train=690, val=94)

NIfTI Header Patient ID / Metadata Strings across all 1,000 cases:
  db_name unique values    : {''}
  descrip unique values    : {''}
  aux_file unique values   : {''}
  intent_name unique values: {''}

Exact compressed file SHA-256 duplicates across all 1,000 cases: 0

Decompressed voxel-prefix (first 256KB) SHA-256 duplicates across all 1,000 cases: 0

Exact geometry (shape + zooms + qoffset) matches across all 1,000 cases: 0

Full 66 Unseen Cases Inventory (ID, Shape, Zooms, QOffset, File_SHA256[:16], VoxelPrefix_SHA256[:16]):
  Case   3 | shape=(512, 512, 275) | zooms=(0.361328, 0.361328, 0.5) | qoffset=(62.3193, 78.6807, 65.45) | file_sha256=b704c865ae430211... | vox_sha256=6da571a9bb96f418...
  Case   4 | shape=(512, 512, 259) | zooms=(0.341797, 0.341797, 0.5) | qoffset=(56.3291, 66.1709, 423.95) | file_sha256=6ce313cf2461979c... | vox_sha256=811eb3ce78a89d09...
  Case   5 | shape=(512, 512, 206) | zooms=(0.347656, 0.347656, 0.5) | qoffset=(60.8262, 73.6738, 59.95) | file_sha256=c797e536278a4fcf... | vox_sha256=fde3eb7d31d29e95...
  Case   6 | shape=(512, 512, 275) | zooms=(0.353516, 0.353516, 0.5) | qoffset=(56.3232, 91.1768, -305.05) | file_sha256=715812e12f49fc22... | vox_sha256=765bbe51465e66a5...
  Case   7 | shape=(512, 512, 206) | zooms=(0.34375, 0.34375, 0.5) | qoffset=(60.8281, 76.1719, -299.55) | file_sha256=4795e936a8534e10... | vox_sha256=2964e2396b24d583...
  Case  10 | shape=(512, 512, 275) | zooms=(0.351562, 0.351562, 0.5) | qoffset=(57.8242, 72.6758, 9.95) | file_sha256=88fd69257dfc1544... | vox_sha256=ade264fcbc6ee06f...
  Case  13 | shape=(512, 512, 241) | zooms=(0.347656, 0.347656, 0.5) | qoffset=(64.8262, 47.1738, 221.95) | file_sha256=e8426cc95babc215... | vox_sha256=20486bbf16350e63...
  Case  16 | shape=(512, 512, 275) | zooms=(0.390625, 0.390625, 0.5) | qoffset=(63.8047, 62.6953, -282.05) | file_sha256=9629cd122ac9b41c... | vox_sha256=3cba0a4fbb59a172...
  Case  21 | shape=(512, 512, 275) | zooms=(0.345703, 0.345703, 0.5) | qoffset=(46.3271, 74.6729, -51.05) | file_sha256=e19474c8e49cc6fc... | vox_sha256=ab1e8a07ae09ea1e...
  Case  23 | shape=(512, 512, 275) | zooms=(0.380859, 0.380859, 0.5) | qoffset=(47.3096, 65.1904, 42.45) | file_sha256=b014a4c59927b0fc... | vox_sha256=9e794230c2c3acf7...
  Case  26 | shape=(512, 512, 275) | zooms=(0.349609, 0.349609, 0.5) | qoffset=(71.3252, 73.1748, 66.45) | file_sha256=99bfd2d50844328f... | vox_sha256=d6f870a30ffe78c5...
  Case  27 | shape=(512, 512, 275) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(44.3408, 45.1592, 100.45) | file_sha256=30f3983f5f553e7b... | vox_sha256=004bbcee31a9cfcf...
  Case  28 | shape=(512, 512, 275) | zooms=(0.335938, 0.335938, 0.5) | qoffset=(53.832, 69.168, -300.55) | file_sha256=52254c53979581bf... | vox_sha256=9421d060cf91737e...
  Case  31 | shape=(512, 512, 275) | zooms=(0.345703, 0.345703, 0.5) | qoffset=(53.3271, 64.6729, -242.05) | file_sha256=6e733c9a2c665365... | vox_sha256=13d60df4609672bc...
  Case  32 | shape=(512, 512, 275) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(45.3408, 54.6592, 94.95) | file_sha256=f131c889c03e1933... | vox_sha256=84beeebd68bff537...
  Case  33 | shape=(512, 512, 275) | zooms=(0.326172, 0.326172, 0.5) | qoffset=(57.3369, 77.1631, -23.55) | file_sha256=56f317f6ef47b30b... | vox_sha256=f455b323dcd8df1b...
  Case  35 | shape=(512, 512, 275) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(37.3408, 81.1592, 349.45) | file_sha256=806ce1b9f49f09fe... | vox_sha256=1e5695887e2ef214...
  Case  38 | shape=(512, 512, 275) | zooms=(0.376953, 0.376953, 0.5) | qoffset=(59.3115, 66.1885, 37.95) | file_sha256=c24c914b1e53e044... | vox_sha256=d140c36da3190446...
  Case  44 | shape=(512, 512, 275) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(65.3408, 70.6592, 56.45) | file_sha256=0cbeb6eb66b47302... | vox_sha256=a6a42fa238ec2e23...
  Case  48 | shape=(512, 512, 206) | zooms=(0.382812, 0.382812, 0.5) | qoffset=(54.3086, 86.1914, 15.95) | file_sha256=977cf8e3d9705c22... | vox_sha256=ef0d8225a67ef368...
  Case  54 | shape=(512, 512, 275) | zooms=(0.335938, 0.335938, 0.5) | qoffset=(37.832, 74.168, -267.05) | file_sha256=9398f103d007091f... | vox_sha256=d72e3a4c32e89d1d...
  Case  62 | shape=(512, 512, 275) | zooms=(0.351562, 0.351562, 0.5) | qoffset=(49.3242, 65.6758, 63.45) | file_sha256=e986e566334fa253... | vox_sha256=9cd95f861d87f910...
  Case  64 | shape=(512, 512, 241) | zooms=(0.292969, 0.292969, 0.5) | qoffset=(34.8535, 39.6465, -242.55) | file_sha256=e863df3c4cc1bc9c... | vox_sha256=7aa351a24b4d2a1b...
  Case  66 | shape=(512, 512, 206) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(39.3408, 60.1592, 41.95) | file_sha256=c8ae7dc91089d8a1... | vox_sha256=7bc515fd7e6bf864...
  Case  73 | shape=(512, 512, 275) | zooms=(0.363281, 0.363281, 0.5) | qoffset=(57.8184, 71.1816, 48.95) | file_sha256=5ee03329a1f16565... | vox_sha256=d9818ee14f3c8ae6...
  Case  74 | shape=(512, 512, 275) | zooms=(0.361328, 0.361328, 0.5) | qoffset=(53.3193, 66.1807, -304.05) | file_sha256=c67a02ce6b6b0846... | vox_sha256=02bd631d52546148...
  Case  75 | shape=(512, 512, 247) | zooms=(0.371094, 0.371094, 0.5) | qoffset=(46.3145, 79.1855, 81.95) | file_sha256=f74011ef1f3c6814... | vox_sha256=73a13b93bd6c216e...
  Case  82 | shape=(512, 512, 275) | zooms=(0.355469, 0.355469, 0.5) | qoffset=(50.8223, 81.1777, 17.45) | file_sha256=086bc5239150b211... | vox_sha256=93cd9ae829523295...
  Case  83 | shape=(512, 512, 255) | zooms=(0.396484, 0.396484, 0.5) | qoffset=(65.3018, 69.1982, 52.95) | file_sha256=664350482369c39b... | vox_sha256=550df96071ca5541...
  Case  86 | shape=(512, 512, 206) | zooms=(0.369141, 0.369141, 0.5) | qoffset=(59.3154, 71.1846, 30.95) | file_sha256=8fb2327a57805d67... | vox_sha256=c4c17a54965cc31d...
  Case  87 | shape=(512, 512, 206) | zooms=(0.376953, 0.376953, 0.5) | qoffset=(69.3115, 73.1885, 67.95) | file_sha256=b8d21bc629713b60... | vox_sha256=342e283b80594173...
  Case  94 | shape=(512, 512, 257) | zooms=(0.361328, 0.361328, 0.5) | qoffset=(59.8193, 62.6807, -279.05) | file_sha256=7bc9dc949b76063e... | vox_sha256=890a41345107381f...
  Case  96 | shape=(512, 512, 275) | zooms=(0.351562, 0.351562, 0.5) | qoffset=(46.3242, 65.1758, 70.45) | file_sha256=d55e1429a319320a... | vox_sha256=4988a556a2c709e5...
  Case  97 | shape=(512, 512, 243) | zooms=(0.333984, 0.333984, 0.5) | qoffset=(46.333, 64.167, -238.05) | file_sha256=3254151e34101f44... | vox_sha256=dc430f270fd7355a...
  Case  98 | shape=(512, 512, 206) | zooms=(0.337891, 0.337891, 0.5) | qoffset=(45.8311, 59.6689, -155.55) | file_sha256=5ab3651d1a26c873... | vox_sha256=48ef5ff670933b2e...
  Case 100 | shape=(512, 512, 206) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(48.3408, 67.6592, 41.95) | file_sha256=e6b4f21bd8c46981... | vox_sha256=9398f88b88c0eb73...
  Case 103 | shape=(512, 512, 275) | zooms=(0.376953, 0.376953, 0.5) | qoffset=(57.3115, 76.1885, 80.45) | file_sha256=396c2dc92817d4a5... | vox_sha256=5e54a188b0312141...
  Case 104 | shape=(512, 512, 275) | zooms=(0.357422, 0.357422, 0.5) | qoffset=(61.3213, 62.6787, 36.45) | file_sha256=20438079ec70a233... | vox_sha256=1e965571eb48b847...
  Case 107 | shape=(512, 512, 206) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(48.3408, 56.6592, 10.45) | file_sha256=388f1ef2ac6cf635... | vox_sha256=495f1145dc09e92a...
  Case 108 | shape=(512, 512, 275) | zooms=(0.373047, 0.373047, 0.5) | qoffset=(50.3135, 74.6865, 74.95) | file_sha256=b6bf4595db127235... | vox_sha256=9f1709acded9e70d...
  Case 117 | shape=(512, 512, 275) | zooms=(0.382812, 0.382812, 0.5) | qoffset=(49.8086, 70.6914, -329.05) | file_sha256=9c31b99120a1fb29... | vox_sha256=10a45313727ea458...
  Case 129 | shape=(512, 512, 275) | zooms=(0.373047, 0.373047, 0.5) | qoffset=(47.3135, 67.1865, 111.45) | file_sha256=197e7d3f8e95e161... | vox_sha256=c0779df7ec7dd4bc...
  Case 131 | shape=(512, 512, 247) | zooms=(0.390625, 0.390625, 0.5) | qoffset=(64.8047, 60.1953, -253.55) | file_sha256=f4642677c008ae39... | vox_sha256=a3fbf7d01b4a640d...
  Case 134 | shape=(512, 512, 206) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(32.3408, 55.1592, 325.95) | file_sha256=affcddcd56b7a902... | vox_sha256=6b34f433116215a5...
  Case 137 | shape=(512, 512, 275) | zooms=(0.386719, 0.386719, 0.5) | qoffset=(51.8066, 63.6934, -352.05) | file_sha256=63eb730c08472210... | vox_sha256=8ad99eea35065e17...
  Case 141 | shape=(512, 512, 275) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(54.3408, 70.1592, 87.95) | file_sha256=dff21c4ebc613a3f... | vox_sha256=45e0e423b3c7406f...
  Case 142 | shape=(512, 512, 275) | zooms=(0.388672, 0.388672, 0.5) | qoffset=(43.3057, 85.1943, 64.45) | file_sha256=cb0299a03cf4f0d7... | vox_sha256=d8508cd602934299...
  Case 144 | shape=(512, 512, 275) | zooms=(0.351562, 0.351562, 0.5) | qoffset=(55.8242, 71.1758, -254.05) | file_sha256=714a139977f2f556... | vox_sha256=ff6e3abacfacef04...
  Case 146 | shape=(512, 512, 275) | zooms=(0.353516, 0.353516, 0.5) | qoffset=(49.3232, 44.1768, 96.95) | file_sha256=443556057a215095... | vox_sha256=cddf99f18aecc020...
  Case 155 | shape=(512, 512, 253) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(44.3408, 81.1592, -255.55) | file_sha256=963bb9dc7d627ba7... | vox_sha256=07e0fcf99635ae84...
  Case 156 | shape=(512, 512, 206) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(51.3408, 85.6592, -173.05) | file_sha256=8ee93a379daa4e74... | vox_sha256=5f90eba3a45af2cc...
  Case 158 | shape=(512, 512, 275) | zooms=(0.382812, 0.382812, 0.5) | qoffset=(60.8086, 57.1914, 43.45) | file_sha256=f325ef9e944eee79... | vox_sha256=c4c5f7ddfee55c48...
  Case 160 | shape=(512, 512, 219) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(40.3408, 74.1592, 94.95) | file_sha256=57a776a192424b43... | vox_sha256=92e9041568f0c493...
  Case 163 | shape=(512, 512, 275) | zooms=(0.392578, 0.392578, 0.5) | qoffset=(70.3037, 86.1963, 3.45) | file_sha256=6a69b3a7b8a79f3d... | vox_sha256=4c0f981c61652694...
  Case 165 | shape=(512, 512, 275) | zooms=(0.398438, 0.398438, 0.5) | qoffset=(53.8008, 84.1992, -290.05) | file_sha256=5a915cdef24cdf94... | vox_sha256=fec2838029bd02fb...
  Case 166 | shape=(512, 512, 259) | zooms=(0.396484, 0.396484, 0.5) | qoffset=(66.3018, 38.6982, -239.55) | file_sha256=f5781a790b76848b... | vox_sha256=7c30f2a178231d58...
  Case 170 | shape=(512, 512, 275) | zooms=(0.333984, 0.333984, 0.5) | qoffset=(52.333, 77.167, 31.95) | file_sha256=c3c63cf47b40dee3... | vox_sha256=f513c2e62e74273a...
  Case 171 | shape=(512, 512, 261) | zooms=(0.386719, 0.386719, 0.5) | qoffset=(63.8066, 84.1934, -292.55) | file_sha256=9bd80b392719816c... | vox_sha256=899263540a6f46fb...
  Case 176 | shape=(512, 512, 249) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(62.3408, 73.6592, -252.05) | file_sha256=b48d07b1942d6a92... | vox_sha256=e6d253305bdff9be...
  Case 177 | shape=(512, 512, 275) | zooms=(0.447266, 0.447266, 0.5) | qoffset=(71.2764, 74.2236, -267.05) | file_sha256=ba4f5bff7d4de4a7... | vox_sha256=e0efb8ed2af911ea...
  Case 178 | shape=(512, 512, 206) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(47.3408, 61.1592, -215.05) | file_sha256=13507e7f37330b41... | vox_sha256=2cf8aae5c3aeb673...
  Case 184 | shape=(512, 512, 275) | zooms=(0.369141, 0.369141, 0.5) | qoffset=(50.3154, 67.6846, 95.45) | file_sha256=1e1d912c60189310... | vox_sha256=c3cefd0ec1cadee1...
  Case 185 | shape=(512, 512, 275) | zooms=(0.34375, 0.34375, 0.5) | qoffset=(57.8281, 81.1719, 306.45) | file_sha256=017847270c047c88... | vox_sha256=ecfe42ca0b0778c0...
  Case 186 | shape=(512, 512, 206) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(42.3408, 59.1592, 32.95) | file_sha256=4dabbb6d86e3a477... | vox_sha256=f916f5f712457498...
  Case 189 | shape=(512, 512, 275) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(37.3408, 81.1592, -5.55) | file_sha256=0e36fae9a872a87f... | vox_sha256=50017219a59c74a8...
  Case 192 | shape=(512, 512, 215) | zooms=(0.341797, 0.341797, 0.5) | qoffset=(51.3291, 86.1709, 99.95) | file_sha256=dbe552d0298244a8... | vox_sha256=3d25bf6296d55aa0...
  Case 199 | shape=(512, 512, 263) | zooms=(0.318359, 0.318359, 0.5) | qoffset=(61.3408, 48.6592, -184.55) | file_sha256=c9eec1b548e6560f... | vox_sha256=f2a3b162fb96447c...
```

---

## Check 7: `train_and_track_v3.py` Checkpoint Audit

### 7.1 What `train_and_track_v3.py` Produced
In `_Archive_Historical_Experiments_and_Logs/legacy_pre_200ep_scripts/train_and_track_v3.py`:
- Lines 235–237 define checkpoint paths:
  - `best_ckpt = "rasnet_v3_best.pth"`
  - `temp_ckpt = "rasnet_v3_temp.pth"`
  - And in `_Archive_Historical_Experiments_and_Logs/legacy_pre_200ep_scripts/train_rasnet_v3.py` (Lines 146, 206):
    - `best_ckpt = "rasnet_v3_best_10ep.pth"`
    - `csv_out = "metrics_rasnet_v3_10ep.csv"`
- Summary statistics of `_Archive_Historical_Experiments_and_Logs/legacy_pre_200ep_scripts/metrics_rasnet_v3_10ep.csv` (and its byte-for-byte duplicate `Final_Generated_assets/imagecas_pipeline_validation/rasnet_development/metrics_rasnet_v3_10ep.csv`):
  ```text
  metrics_rasnet_v3_10ep.csv: 150 rows, MTime=2026-11-01 15:15:21
  Summary stats (mean ± std):
    case_id: 925.5000 ± 43.4454
    dice: 0.7665 ± 0.0707
    iou: 0.6264 ± 0.0868
    precision: 0.9002 ± 0.0535
    recall: 0.6737 ± 0.0948
    hd95: 10.4251 ± 10.6424
  ```

### 7.2 Does the Checkpoint from `train_and_track_v3.py` Appear in Any Reported Result?
1. **Checkpoints on disk**: Neither `rasnet_v3_best.pth`, `rasnet_v3_temp.pth`, nor `rasnet_v3_best_10ep.pth` exists anywhere on disk or inside `checkpoints_200ep.zip`.
2. **Primary 200-epoch benchmark, Ablation suite, Unseen-66 cohort, and Clinical validation**: **NO.** None of the active tables (`Q1_Publication_Package/matched_200ep_benchmark/evaluation_results/*`, `Q1_Publication_Package/stats/*`, `Q1_Publication_Package/ablation/*`, `3d Cas Validations and doings/results/*`, `results/FINAL_MASTER_RESULTS.csv`) use `rasnet_v3_best.pth` or `metrics_rasnet_v3_10ep.csv` (the 200-epoch RASNet checkpoint `rasnet_best.pth` has SHA-256 `8d4c25cd12d07c40423062b27b1d36836e5fd6cecb30c20c34055eeac976d1b8` and achieves `Dice = 0.7765 ± 0.0695`, `Precision = 0.8801 ± 0.0530`, `Recall = 0.7016 ± 0.0976`, `HD95 = 10.29 ± 10.49 mm`).
3. **Historical / Legacy Reported Results Where It DOES Appear**:
   - `_Archive_Historical_Experiments_and_Logs/all_four_validations/unified_comparison_table.csv` (Row 5):
     ```text
     Model,N,Dice,IoU,Precision,Recall,HD95 (mm)
     "RASNet v3 (10ep, +100 cases)",150,0.7665 ± 0.0705,0.6264 ± 0.0865,0.9002 ± 0.0533,0.6737 ± 0.0945,10.43 ± 10.61
     ```
   - `Final_Generated_assets/imagecas_pipeline_validation/rasnet_development/metrics_rasnet_v3_10ep.csv` (exact copy of `_Archive_Historical_Experiments_and_Logs/legacy_pre_200ep_scripts/metrics_rasnet_v3_10ep.csv`).

---

## Post-Archival Update (`2026-11-10`)
- All historical scripts, legacy evaluation CSVs, ablation scripts, Unseen-66 / 3D-CAS folders, `Final_Generated_assets/`, and clinical QCA scripts referenced in Checks 1–7 above were archived with 100% SHA-256 verification into `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/` (see `ARCHIVE_INDEX.csv`).
- The active repository now retains only the verified 5-model 200-epoch ImageCAS trainers, model definitions (`Phase3_Local_Integration/rasnet_model.py`, `rasnet_loss.py`), split manifest (`Phase3_Local_Integration/splits_final.json`), the fail-closed `benchmark/` package, `tests/` (`31/31` passing at the 2026-11-10 audit; `33/33` as of 2026-11-13), and verified run outputs (`results/benchmark_runs/{val_full_94,test_retrospective_150,smoke_val_all5,smoke_verified_v2}`, `results/audits/2026-11-10_verification_audit/`, `results/thesis_outputs/`). See `CURRENT_EVIDENCE_INDEX.md`.

