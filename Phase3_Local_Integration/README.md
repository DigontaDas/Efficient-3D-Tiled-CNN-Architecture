# Phase 3 Local Integration — Core RASNet Model, Loss & Split Manifest

> **Archival Notice (`2026-11-10`)**: To keep the active thesis repository strictly focused on the verified 5-model 200-epoch ImageCAS segmentation benchmark, all historical pre-200-epoch evaluation scripts (`00_dataset_audit.py`..`10_plot_comparison_bar.py`), clinical hospital QCA scripts (`production_qca_engine.py`, `refine_clinical_postprocess.py`, `run_hospital_cohort_pipeline.py`, `outlier_audit/`), and legacy scratch files previously located in `Phase3_Local_Integration/` have been archived to `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/Phase3_Local_Integration/` (indexed with SHA-256 hashes in `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/ARCHIVE_INDEX.csv`).

---

## Active Files in `Phase3_Local_Integration/`

1. **`rasnet_model.py`** (`SHA-256: 4f2372ec25081f483d21150129730bdc98f4edfa2714d4493735187a744a305c`):
   - Defines `AttentionGate3D` (3D additive attention gate with `GroupNorm(1, 1)` before `Sigmoid`) and `RASNet` (subclass of MONAI `SegResNet` with `init_filters=16`, encoder blocks `[1, 2, 2, 4]`, decoder additive skip fusion after `AttentionGate3D`, and multi-scale deep supervision heads `aux2` at Level 2 [`1/4` resolution] and `aux3` at Level 1 [`1/2` resolution]).
   - Used directly by `Q1_Publication_Package/matched_200ep_benchmark/train_rasnet_200ep.py` and `benchmark/adapters/monai_models.py`.
2. **`rasnet_loss.py`**:
   - Defines `StenosisAwareLoss(alpha=0.4, gamma=2.5)`, combining `0.4 * DiceLoss(smooth_nr=0.0, smooth_dr=1e-5, to_onehot_y=True, softmax=True)` and `0.6 * FocalLoss(gamma=2.5, to_onehot_y=True)`.
3. **`splits_final.json`** (`SHA-256: efe3f06cfe9c7dff646224378ad4d0ea5ce50fb4bf920807b17054af4bdf953c`):
   - Canonical ImageCAS split manifest (`train = 690`, `val = 94`, `test = 150` [`IDs 851–1000`]; `0` overlaps).
4. **`dataset_paths.py`**:
   - Path resolution helper for local ImageCAS NIfTI volumes.

For running the verified 5-model benchmark, see [`LAB_RUN_PLAN.md`](../LAB_RUN_PLAN.md) and [`CURRENT_EVIDENCE_INDEX.md`](../CURRENT_EVIDENCE_INDEX.md).

