# V1 Epoch-200 Checkpoint Recovery & Provenance Report

- **Run ID:** `20261117_041500Z`
- **Generated (UTC):** `2026-11-17T04:29:45Z`
- **Overall Status:** `PASS`
- **Git HEAD:** `b3ca1f1a5553654d1b31c73ec50f36c2d6dc8ea3` (tracked dirty: `False`)
- **Environment:** Python `3.12.3`, torch `2.14.1+cu126`, MONAI `1.6.1`, nnunetv2 `2.8.1`

## 1. Seekable Split-Archive Recovery Integrity

Recovered **only** `rasnet_last.pth` and `nnunet_last.pt` from `C:\Thesis_RASNET\Q1_Publication_Package\matched_200ep_benchmark\checkpoints_best_and_last` (`10` parts, `901,178,285` bytes) using `SplitZipReader(io.RawIOBase)` + `io.BufferedReader` + `zipfile.ZipFile` (no 900 MB combined ZIP created, no `extractall()` into historical folders).

| Member | Action | Size (B) | CRC-32 (CD/LH/Computed) | SHA-256 | ZIP Timestamp |
|---|---|---:|---|---|---|
| `rasnet_last.pth` | `verified_existing` | 18,871,718 | `0x382cc853` | `71c9e5ef4a02849686c99ccb2d1544ff78c78187e1c0d53e7b22884c708c1b95` | `2026-09-09 03:31:22` |
| `nnunet_last.pt` | `verified_existing` | 246,472,688 | `0xae7f79e9` | `2105bd7238fff7485f1a8d72622f977f635f441e6d44863920c85709b02ffec2` | `2026-09-08 09:20:06` |

## 2. nnU-Net V2 Epoch-200 Provenance & Isolated Native Folder

- **Checkpoint Role:** `final_epoch200` (`embedded_checkpoint_metadata_and_trainer_convention`)
- **Embedded `current_epoch`:** `200` (0-based internal current_epoch (0..199); on_train_end() decrements by 1 and save_checkpoint() writes current_epoch + 1 == 200)
- **Trainer / Config / Fold:** `nnUNetTrainer_200epochs` / `3d_fullres` / fold `0`
- **Mirroring Axes:** `(0, 1, 2)`
- **Architecture:** `PlainConvUNet` (`30,785,994` unique params, `292` keys)
- **Logging Verification:** all 8 logging keys have length `200`, `0` inter-epoch gaps > 1s; matches `nnunet_200ep_log.csv` (`200` valid rows + `4` empty resume-marker rows `[51, 101, 151, 201]`) with `max_raw_diff=4.95e-05` (4/5-decimal log rounding; `max_rounded_diff=0.00e+00`)
- **Diff vs `nnunet_best.pt` (`current_epoch=199`):** `290/292` tensors differ, `max_abs_diff=0.001956`, `total_l2_diff=0.094291`
- **Isolated Native Folder:** `C:\Thesis_RASNET\recovered_checkpoints\v1_epoch200_20261117_041500Z\nnunet_native_model\Dataset501_CoronarySeg\nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres`
  - `plans.json` (`verified_existing`): `c67af4aaeb24f5a72b58c83e351a9c62cf026f8baf1bbb727e3022491d5733d8` (equals checkpoint `init_args['plans']`: `True`)
  - `dataset.json` (`verified_existing`): `d81f40253d7c2d88b9d1cdbd4bd7319a7a0206fce5b2350287317bd34f9c7389` (equals checkpoint `init_args['dataset_json']`: `True`)
  - `fold_0/checkpoint_final.pth` (`verified_existing`): `2105bd7238fff7485f1a8d72622f977f635f441e6d44863920c85709b02ffec2`
  - **`nnUNetPredictor` CPU Initialization:** `True` (`patch_size=[96, 160, 160]`, `spacing=[0.5, 0.349609375, 0.349609375]`, `normalization=['CTNormalization']`)

## 3. RASNet V1 Epoch-200 Provenance & Strict Loading

- **Checkpoint Role:** `final_epoch200` (`script_log_archive_provenance`)
- **Embedded Metadata Note:** **No embedded epoch/optimizer metadata is present in `rasnet_last.pth`** (`embedded_epoch_metadata_present=False`). Raw OrderedDict state_dict saved via torch.save(model.state_dict(), last_ckpt_path) in train_rasnet_200ep.py:268; epoch-200 identity is established by script saving logic, uninterrupted 200-epoch rasnet_200ep_log.csv, and ZIP archive timestamp alignment rather than embedded checkpoint metadata.
- **Training Script & Log Provenance:** `train_rasnet_200ep.py` (SHA-256 `6d73bdb1c055fa343a8145adb7742cb46cbb08c736ef8bf3021893e252b86a0d`) saves `rasnet_best.pth` on validation improvement (line `265`) and unconditionally saves `rasnet_last.pth` after every epoch (line `268`). `rasnet_200ep_log.csv` records `200` contiguous epochs (`1..200`), `12` validation evaluations, best epoch `160` (`val_dice=0.7769595980644226`), final epoch `200` (`val_dice=0.7761295437812805`), and exact `CosineAnnealingLR` alignment (`max_abs_err=5.42e-20`).
- **Archive Timestamp Alignment:** ZIP timestamp delta (`rasnet_last.pth` minus `rasnet_best.pth`) = `5120.00 s` vs log elapsed time ep `160->200` = `5119.97 s` (discrepancy `0.0293 s`, within 2s DOS ZIP timestamp granularity).
- **Strict Model Loading:** `OrderedDict` with `114` `float32` tensors (all finite), loaded with `strict=True` into `RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)` (`4,706,772` params). `eval()` returns `torch.Tensor`; `train()` returns `tuple[torch.Tensor, list[torch.Tensor]]`.
- **Diff vs `rasnet_best.pth`:** `114/114` tensors differ, `max_abs_diff=0.005901`, `total_l2_diff=1.376970`.

## 4. Non-Interference & Historical Doc Discrepancies

- **Protected Paths Unmodified (Before == After):** `True` (`19` anchor files and `8` protected directories verified).
- **Known Historical Doc Discrepancies (read-only observation; `CHECKPOINTS.md` left untouched):**
  - CHECKPOINTS.md:9 lists 400 keys for rasnet_best.pth; verified key count in both rasnet_best.pth and rasnet_last.pth is 114.
  - CHECKPOINTS.md:13,31 lists 402 keys, patch_size=[128,192,192], spacing=[0.5,0.359375,0.359375] for nnU-Net V2; verified plans.json and checkpoints have 292 keys, patch_size=[96,160,160], spacing=[0.5,0.349609375,0.349609375], and nnunet_best.pt has current_epoch=199.

## 5. Reproducibility Command

```powershell
C:\Thesis_RASNET_env\bench\Scripts\python.exe C:\Thesis_RASNET\recovered_checkpoints\v1_epoch200_20261117_041500Z\recover_and_verify.py
```
