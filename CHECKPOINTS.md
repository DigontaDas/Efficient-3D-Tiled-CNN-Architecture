# Checkpoint Record & Provenance

All five 200-epoch **BEST** checkpoints have been verified on the lab machine (`C:\Thesis_RASNET`) from the multi-part archive `Q1_Publication_Package/matched_200ep_benchmark/checkpoints_best_and_last/checkpoints_200ep.zip.001`..`.010` and extracted into the gitignored directory `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/` (and for `nnU-Net V2`, also linked into the native trained-model folder `Q1_Publication_Package/matched_200ep_benchmark/nnUNet_results/Dataset501_CoronarySeg/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth`).

## 1. Primary 200-Epoch Best Checkpoints

| Model | Filename | Size (Bytes) | Archive Timestamp | Best Epoch | Best Val Metric (Training Log) | State-Dict Format & Verified Counts | SHA-256 Digest | Status |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| **RASNet** | `rasnet_best.pth` | `18,871,718` | `2026-09-09 02:06:02` | `160` | `0.776960` (`val_dice`) | Raw `OrderedDict` (`114` keys, `4,706,772` params; no embedded epoch metadata; strict `load_state_dict` into `RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)`) | `8d4c25cd12d07c40423062b27b1d36836e5fd6cecb30c20c34055eeac976d1b8` | **VERIFIED** |
| **SegResNet** | `segresnet_best.pt` | `18,836,926` | `2026-09-08 04:06:46` | `190` | `0.752116` (`val_dice`) | Raw `OrderedDict` (`83` keys, `4,700,914` params; no embedded epoch metadata; strict `load_state_dict` into `SegResNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)`) | `b8ea61d95106a0bbe15e1ce9f9e5af1a38d9508e59d0639d32bc883d93c6e109` | **VERIFIED** |
| **V-Net** | `vnet_best.pt` | `182,479,290` | `2026-09-08 19:25:48` | `180` | `0.749052` (`val_dice`) | Raw `OrderedDict` (`164` keys, `45,599,452` params + `5,340` BN buffer elements; no embedded epoch metadata; strict `load_state_dict` into `VNet(spatial_dims=3, in_channels=1, out_channels=2, act="elu", dropout_prob_down=0.1, dropout_prob_up=(0.1, 0.1), bias=True)`) | `a89d71bd41c3e773c68e3f4b5e27c5a067f959154aa049eead76f1642c5c1063` | **VERIFIED** |
| **3D U-Net** | `3dunet_best.pth` | `19,253,157` | `2026-09-08 13:03:48` | `180` | `0.663835` (`best_val_dice`) | Dict (`['epoch', 'model_state_dict', 'best_val_dice']` — no `optimizer_state_dict`; `epoch=180`, `model_state_dict` with `63` keys, `4,805,534` params vs `19,253,157` file bytes; strict `load_state_dict` into `UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm="instance", dropout=0.15)`) | `d8a8f5ec95b5f34e2afe0b807ac49bb36d136f9d864238c27a763a29a800a60f` | **VERIFIED** |
| **nnU-Net V2** | `nnunet_best.pt` / `fold_0/checkpoint_best.pth` | `246,472,150` | `2026-09-09 20:51:38` | `199` (EMA) / `189` (raw peak) | `0.816239` (`_best_ema` at `current_epoch=199`) / `0.837600` (raw `Pseudo dice` peak at ep `189`) | Native `nnUNetv2` checkpoint dict (`network_weights` with `292` keys — `88` shared-storage aliases, `30,785,994` unique params; `current_epoch=199`, `trainer_name='nnUNetTrainer_200epochs'`, `inference_allowed_mirroring_axes=(0, 1, 2)`, `init_args` containing `plans` and `dataset_json`) | `9accddf283b1c480e386f30f426b9200ca8d4aa1ecd70d5b5af0b8aad6058711` | **VERIFIED** |

## 1A. Verified Final Epoch-200 Checkpoints (`final_epoch200`)

Recovered from `checkpoints_200ep.zip.001`..`.010` into `recovered_checkpoints/v1_epoch200_20261117_041500Z/` (`PROVENANCE_REPORT.md`):

| Model | Filename & Path | Size (Bytes) | Embedded Epoch Metadata | State-Dict Format & Verified Counts | Provenance Basis | SHA-256 Digest | Status |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| **RASNet** | `recovered_checkpoints/v1_epoch200_20261117_041500Z/rasnet_last.pth` | `18,871,718` | None (`embedded_epoch_metadata_present=False`) | Raw `OrderedDict` (`114` keys, `4,706,772` params; strict `load_state_dict` OK) | Supported by `train_rasnet_200ep.py:268` + contiguous 200-epoch `logs/rasnet_200ep_log.csv` + `CosineAnnealingLR` alignment + archive timestamp alignment (`2026-09-09 03:31:22`, `5120.0 s` delta from ep 160) | `71c9e5ef4a02849686c99ccb2d1544ff78c78187e1c0d53e7b22884c708c1b95` | **VERIFIED** |
| **nnU-Net V2** | `recovered_checkpoints/v1_epoch200_20261117_041500Z/nnunet_last.pt` (also `nnunet_native_model/.../fold_0/checkpoint_final.pth`) | `246,472,688` | `current_epoch=200` (`logging` len `200`, `_best_ema=0.816239`) | Native `nnUNetv2` checkpoint dict (`network_weights` with `292` keys — `88` shared-storage aliases, `30,785,994` unique params) | Embedded checkpoint metadata (`current_epoch=200`, `trainer_name='nnUNetTrainer_200epochs'`) + `logs/nnunet_200ep_log.csv` | `2105bd7238fff7485f1a8d72622f977f635f441e6d44863920c85709b02ffec2` | **VERIFIED** |

---

## 2. Training Scripts, Splits & nnU-Net Metadata Provenance

- **Split Manifest**: `Phase3_Local_Integration/splits_final.json`
  - **SHA-256**: `efe3f06cfe9c7dff646224378ad4d0ea5ce50fb4bf920807b17054af4bdf953c`
  - **Counts**: `train = 690` (`IDs 1–850` subset), `val = 94` (`IDs 1–850` subset), `test = 150` (`IDs 851–1000` contiguous). All 934 split IDs are unique and mutually disjoint (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`).
  - **Excluded 66 Cases (`unseen_66`)**: `66` IDs in `1..200` (`[3..199]`, e.g., `3, 4, 5, ..., 199`) are absent from `splits_final.json` (`690 + 94 + 150 + 66 = 1000`; note `15` and `843` are in `train`, correcting an earlier erroneous `{4, 15, 31, ..., 843}` description).
- **Model Construction & Training Scripts**:
  - **RASNet**: `Q1_Publication_Package/matched_200ep_benchmark/train_rasnet_200ep.py` (`Phase3_Local_Integration.rasnet_model.RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)`; trained with `StenosisAwareLoss(alpha=0.4, gamma=2.5)` and explicit deep supervision weighting `total_loss = loss_main + 0.4 * aux2_loss + 0.2 * aux3_loss` in `train_rasnet_200ep.py:215`, where `aux2` is Level 2 at `1/4` resolution and `aux3` is Level 1 at `1/2` resolution).
  - **SegResNet**: `Q1_Publication_Package/matched_200ep_benchmark/train_segresnet_200ep.py` (`SegResNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)`; trained with `DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, lambda_dice=1.0, lambda_focal=2.0)` (`train_segresnet_200ep.py:153-159`; corrected 2026-11-12, previously misstated as `DiceCELoss`)).
  - **V-Net**: `Q1_Publication_Package/matched_200ep_benchmark/train_vnet_200ep.py` (`VNet(spatial_dims=3, in_channels=1, out_channels=2, act="elu", dropout_prob_down=0.1, dropout_prob_up=(0.1, 0.1), bias=True)`; trained with `DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, lambda_dice=1.0, lambda_focal=2.0)` (`train_vnet_200ep.py:154-160`; corrected 2026-11-12, previously misstated as `DiceCELoss`) and `clip_grad_norm_(max_norm=1.0)` (`train_vnet_200ep.py:224`)).
  - **3D U-Net**: `Q1_Publication_Package/matched_200ep_benchmark/train_3dunet_200ep.py` (`UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm="instance", dropout=0.15)`; trained on `Resized(128, 128, 128)` volumes with `DiceCELoss(sigmoid=True)`).
  - **nnU-Net V2**: `Q1_Publication_Package/matched_200ep_benchmark/prepare_and_run_nnunet_200ep.py`
    - **Trained Model Folder**: `Q1_Publication_Package/matched_200ep_benchmark/nnUNet_results/Dataset501_CoronarySeg/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres`
    - **Fold**: `0` (`fold_0/checkpoint_best.pth`)
    - **Trainer & Epoch Convention**: `nnUNetTrainer_200epochs` (`PlainConvUNet`, `3d_fullres`, patch size `[96, 160, 160]`, target spacing `[0.5, 0.349609375, 0.349609375]`, `CTNormalization`; registered dynamically at runtime by `prepare_and_run_nnunet_200ep.py` and `benchmark/adapters/nnunet_adapter.py`). In `nnUNetTrainer`, `self.current_epoch` is 0-based (`0..199`) during the training loop and `save_checkpoint` writes `'current_epoch': self.current_epoch + 1` (1-based completed epoch count). Raw unsmoothed `mean_fg_dice` (`Pseudo dice`) peaks at epoch `189` (`0.8376`, 0-based index `188`), whereas `checkpoint_best.pth` (`nnunet_best.pt`) is selected by `_best_ema = 0.816239` at `current_epoch = 199` (0-based index `198`), and `checkpoint_final.pth` (`nnunet_last.pt`) records `current_epoch = 200`.
    - **`plans.json` SHA-256**: `c67af4aaeb24f5a72b58c83e351a9c62cf026f8baf1bbb727e3022491d5733d8` (matches `ckpt['init_args']['plans']` identically)
    - **`dataset.json` SHA-256**: `d81f40253d7c2d88b9d1cdbd4bd7319a7a0206fce5b2350287317bd34f9c7389` (`numTraining: 784` = `690 train + 94 val`; matches `ckpt['init_args']['dataset_json']` identically)

### 2.1 Training Protocol Parameters & Validation Metric Reconciliation (`N = 94`)

| Model | Patch / Volume Grid | `RandCropByPosNegLabeld` `num_samples` | Validation Cadence (`val_epochs` in log) | Best Epoch | Training-Log Validation Metric (Resampled Grid) | Native-Grid Val Dice (Arm A, $\tau=0.50$) | Native-Grid Val Dice (Arm C, $\tau^*$, Top-2 CC) |
| :--- | :--- | :---: | :--- | ---: | :--- | :---: | :---: |
| **RASNet** | `96 × 96 × 96` (`0.5 mm` iso) | `2` | `12` evals (`ep % 20 == 0 or ep % 50 == 0`) | `160` | `0.7770` (full-vol `0.5 mm` grid) | `0.7789 ± 0.0521` | `0.7897` ($\tau^* = 0.30$) |
| **SegResNet** | `96 × 96 × 96` (`0.5 mm` iso) | `2` | `20` evals (every `10` epochs) | `190` | `0.7521` (full-vol `0.5 mm` grid) | `0.7532 ± 0.0591` | `0.7901` ($\tau^* = 0.35$) |
| **V-Net** | `96 × 96 × 96` (`0.5 mm` iso) | `1` | `13` evals (`[10, 20, 40, 50, 60, 80, 100, 120, 140, 150, 160, 180, 200]`) | `180` | `0.7491` (full-vol `0.5 mm` grid) | `0.7511 ± 0.0619` | `0.7702` ($\tau^* = 0.30$) |
| **3D U-Net** | `128 × 128 × 128` (resized) | Full resized vol | `21` evals (`[1, 10, 20, ..., 200]`, `128³` resized) | `180` | `0.6638` (`128³` resized grid) | `0.5516 ± 0.0480` | `0.6047` ($\tau^* = 0.30$) |
| **nnU-Net V2** | `96 × 160 × 160` (`[0.5, 0.3496, 0.3496]`) | Oversampled patch | `200` evals (`50` batches/epoch, `33%` foreground) | `199` (EMA) / `189` (raw) | `0.8162` (`_best_ema`, `current_epoch=199`); raw `Pseudo dice` peak `0.8376` (ep `189`) | `0.7761 ± 0.0567` | `0.8075` ($\tau^* = 0.35$) |

- **Note on Historical Audit CSV Errata**: Protected `results/audits/2026-11-10_verification_audit/checkpoint_verification.csv` (preserved untouched) contains historical provenance errata (e.g., listing nnU-Net `current_epoch=190` instead of `199` and conflating raw `0.8376` at ep `189` with EMA `_best_ema = 0.8162`); see `results/v1_reporting_correction_20261117_105000Z/HISTORICAL_ERRATA.md`.

- **Note on Auto-Resume (`T1` in `AUDIT_FINDINGS.md`)**: In the four MONAI training scripts, `--resume` reloads model weights and steps the cosine scheduler to `start_epoch`, without restoring AdamW moment buffers (`nnU-Net V2` saves and restores full `optimizer_state` and `grad_scaler_state`). Inspection of the training logs (`results/audits/2026-11-10_verification_audit/checkpoint_verification.csv`) shows `rasnet` and `3dunet` ran 200 epochs in a single uninterrupted session, while `segresnet` (`ep 86, 109, 133`) and `vnet` (`ep 18–19`) used auto-resume and subsequently converged cleanly to their best validation checkpoints at epochs `190` and `180`, respectively.
- **Lab Runtime Environment Used for Benchmark Verification**:
  - **GPU**: `NVIDIA GeForce RTX 4080 SUPER` (`16376 MiB`, Driver `591.86`)
  - **Python**: `3.12.10` (`C:\Thesis_RASNET_env\bench\Scripts\python.exe`)
  - **Key Libraries**: `torch 2.14.1+cu126`, `monai 1.6.1`, `nnunetv2 2.8.1`, `SimpleITK 2.5.7`, `nibabel 5.4.2`, `scikit-image 0.26.0`, `scipy 1.18.1`, `connected-components-3d 4.1.1`, `numpy 2.5.4`, `pandas 3.0.6`
