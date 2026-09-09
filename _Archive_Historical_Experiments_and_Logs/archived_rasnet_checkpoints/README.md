# Archived RASNet Historical Checkpoints

This directory contains previous versions of RASNet checkpoints that have been superseded by the **matched 200-epoch champion checkpoint**.

---

## Authoritative Active Checkpoint
The authoritative, publication-ready champion model is located at:
- **`H:\Thesis_Trainings\Q1_Publication_Package\matched_200ep_benchmark\checkpoints\rasnet_best.pth`**
  - **Trained**: 200 Epochs (Matched protocol)
  - **Performance (N=150)**: Dice `0.7765 ± 0.0701`, IoU `0.6396 ± 0.0863`, Precision `0.8801 ± 0.0531`, HD95 `10.29 ± 10.49 mm`, clDice `0.8592 ± 0.0717`

---

## Archived Checkpoints in this Directory

### 1. `hallucin_fix_aug31/`
- **File**: `rasnet_best.pth`
- **Timestamp**: August 31, 2026
- **History**: Post-hallucination-fix checkpoint trained on 690 cases for 70 epochs after fixing false-positive vascular hallucination bugs. Used in the initial Phase 3 local hospital cohort ($N=21$) validation.

### 2. `70ep_development_run/`
- **Files**: `rasnet_best.pth`, `rasnet_epoch_5.pth` through `rasnet_epoch_70.pth`
- **Timestamp**: June 24, 2026
- **History**: Original preliminary development runs on ImageCAS before hallucination fixes and before the standardized 200-epoch matched benchmark protocol was established.
