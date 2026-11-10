# Checkpoint Record & Provenance

All five 200-epoch **BEST** checkpoints have been verified on the lab machine (`C:\Thesis_RASNET`) from the multi-part archive `Q1_Publication_Package/matched_200ep_benchmark/checkpoints_best_and_last/checkpoints_200ep.zip.001`..`.010` and extracted into the gitignored directory `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/` (and for `nnU-Net V2`, also linked into the native trained-model folder `Q1_Publication_Package/matched_200ep_benchmark/nnUNet_results/Dataset501_CoronarySeg/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth`).

## 1. Primary 200-Epoch Best Checkpoints

| Model | Filename | Size (Bytes) | Archive Timestamp | Best Epoch | Best Val Metric (Training Log) | State-Dict Format | SHA-256 Digest | Status |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| **RASNet** | `rasnet_best.pth` | `18,871,718` | `2026-09-09 02:06:02` | `160` | `0.776960` (`val_dice`) | Raw `OrderedDict` (`400` keys; strict `load_state_dict` into `RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)`) | `8d4c25cd12d07c40423062b27b1d36836e5fd6cecb30c20c34055eeac976d1b8` | **VERIFIED** |
| **SegResNet** | `segresnet_best.pt` | `18,836,926` | `2026-09-08 04:06:46` | `190` | `0.752116` (`val_dice`) | Raw `OrderedDict` (`153` keys; strict `load_state_dict` into `SegResNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)`) | `b8ea61d95106a0bbe15e1ce9f9e5af1a38d9508e59d0639d32bc883d93c6e109` | **VERIFIED** |
| **V-Net** | `vnet_best.pt` | `182,479,290` | `2026-09-08 19:25:48` | `180` | `0.749052` (`val_dice`) | Raw `OrderedDict` (`322` keys; strict `load_state_dict` into `VNet(spatial_dims=3, in_channels=1, out_channels=2, act="elu", dropout_prob_down=0.1, dropout_prob_up=(0.1, 0.1), bias=True)`) | `a89d71bd41c3e773c68e3f4b5e27c5a067f959154aa049eead76f1642c5c1063` | **VERIFIED** |
| **3D U-Net** | `3dunet_best.pth` | `19,253,157` | `2026-09-08 13:03:48` | `180` | `0.663835` (`best_val_dice`) | Dict (`epoch=180`, `model_state_dict` with `90` keys, `optimizer_state_dict`, `best_val_dice`; strict `load_state_dict` into `UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm="instance", dropout=0.15)`) | `d8a8f5ec95b5f34e2afe0b807ac49bb36d136f9d864238c27a763a29a800a60f` | **VERIFIED** |
| **nnU-Net V2** | `nnunet_best.pt` / `fold_0/checkpoint_best.pth` | `246,472,150` | `2026-09-09 20:51:38` | `189` | `0.816239` (`_best_ema`) / `0.837600` (`Pseudo dice`) | Native `nnUNetv2` checkpoint dict (`network_weights` with `402` keys, `trainer_name='nnUNetTrainer_200epochs'`, `inference_allowed_mirroring_axes=(0, 1, 2)`, `init_args` containing `plans` and `dataset_json`) | `9accddf283b1c480e386f30f426b9200ca8d4aa1ecd70d5b5af0b8aad6058711` | **VERIFIED** |

---

## 2. Training Scripts, Splits & nnU-Net Metadata Provenance

- **Split Manifest**: `Phase3_Local_Integration/splits_final.json`
  - **SHA-256**: `efe3f06cfe9c7dff646224378ad4d0ea5ce50fb4bf920807b17054af4bdf953c`
  - **Counts**: `train = 690` (`IDs 1–850` subset), `val = 94` (`IDs 1–850` subset), `test = 150` (`IDs 851–1000` contiguous). All 934 split IDs are unique and mutually disjoint (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`).
  - **Excluded 66 Cases**: `66` IDs in `1–850` (`{4, 15, 31, ..., 843}`) are absent from `splits_final.json` (`690 + 94 + 150 + 66 = 1000`).
- **Model Construction & Training Scripts**:
  - **RASNet**: `Q1_Publication_Package/matched_200ep_benchmark/train_rasnet_200ep.py` (`Phase3_Local_Integration.rasnet_model.RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)`; trained with `StenosisAwareLoss(alpha=0.4, gamma=2.5, ds_weights=[1.0, 0.5, 0.25])`).
  - **SegResNet**: `Q1_Publication_Package/matched_200ep_benchmark/train_segresnet_200ep.py` (`SegResNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)`).
  - **V-Net**: `Q1_Publication_Package/matched_200ep_benchmark/train_vnet_200ep.py` (`VNet(spatial_dims=3, in_channels=1, out_channels=2, act="elu", dropout_prob_down=0.1, dropout_prob_up=(0.1, 0.1), bias=True)`).
  - **3D U-Net**: `Q1_Publication_Package/matched_200ep_benchmark/train_3dunet_200ep.py` (`UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm="instance", dropout=0.15)`).
  - **nnU-Net V2**: `Q1_Publication_Package/matched_200ep_benchmark/prepare_and_run_nnunet_200ep.py`
    - **Trained Model Folder**: `Q1_Publication_Package/matched_200ep_benchmark/nnUNet_results/Dataset501_CoronarySeg/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres`
    - **Fold**: `0` (`fold_0/checkpoint_best.pth`)
    - **Trainer**: `nnUNetTrainer_200epochs` (`PlainConvUNet`, `3d_fullres`, patch size `[128, 192, 192]`, target spacing `[0.5, 0.359375, 0.359375]`, `CTNormalization`)
    - **`plans.json` SHA-256**: `c67af4aaeb24f5a72b58c83e351a9c62cf026f8baf1bbb727e3022491d5733d8` (matches `ckpt['init_args']['plans']` identically)
    - **`dataset.json` SHA-256**: `d81f40253d7c2d88b9d1cdbd4bd7319a7a0206fce5b2350287317bd34f9c7389` (`numTraining: 784` = `690 train + 94 val`; matches `ckpt['init_args']['dataset_json']` identically)
- **Lab Runtime Environment Used for Benchmark Verification**:
  - **GPU**: `NVIDIA GeForce RTX 4080 SUPER` (`16376 MiB`, Driver `591.86`)
  - **Python**: `3.12.10` (`C:\Thesis_RASNET_env\bench\Scripts\python.exe`)
  - **Key Libraries**: `torch 2.14.1+cu126`, `monai 1.6.1`, `nnunetv2 2.8.1`, `SimpleITK 2.5.7`, `nibabel 5.4.2`, `scikit-image 0.26.0`, `scipy 1.18.1`, `connected-components-3d 4.1.1`, `numpy 2.5.4`, `pandas 3.0.6`
