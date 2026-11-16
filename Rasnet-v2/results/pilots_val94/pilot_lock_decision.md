# Stage 2 Validation-Only Pilot Lock Decision (`Val N = 94`)

- **Timestamp**: `2026-11-16T16:02:46Z`
- **Validation Cases Evaluated**: `N = 94` (`IDs 851–1000` strictly untouched)
- **Locked LRs (`locked_lrs.json`, SHA-256 `94f14392c316...`)**: `{"segresnet_f16": 0.0001, "segresnet_f32": 0.001, "rasnet_v2_f16": 0.0001, "rasnet_v2_f32": 0.0003, "vnet": 0.0003, "unet3d": 0.0001}`
- **Gate Status (`Val Precision >= 0.8000` & `<=` matched `SegResNet-v2` component fragmentation)**: **`FALLBACK (NO CANDIDATE PASSED BOTH GATES)`**
- **Locked Lead Design**: `F*=16`, `gate_mode=conv_bias`, `w_cldice=0.0` (Val Dice `0.7340`, Precision `0.6966`, Components `33.96` vs SegResNet `28.05`)

## Stage 2B Design Sweep (`Val N = 94`, 0.50 Gauss, 1-Pass, τ = 0.50)

| candidate_name | init_filters | gate_mode | w_cldice | dice_mean | precision_mean | recall_mean | hd95_mean | cldice_mean | components_total_mean | segresnet_ref_components_total | pass_precision_ge_080 | pass_frag_le_segresnet | qualifies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot2b_rasnet_v2_f16_convbias_cl020 | 16 | conv_bias | 0.2000 | 0.7095 | 0.6432 | 0.8043 | 29.1400 | 0.8165 | 21.6489 | 28.0532 | False | True | False |
| pilot2b_rasnet_v2_f16_gnorig_cl020 | 16 | groupnorm_orig | 0.2000 | 0.6983 | 0.6249 | 0.8056 | 33.5156 | 0.8158 | 27.1064 | 28.0532 | False | True | False |
| pilot2b_rasnet_v2_f16_convbias_cl000 | 16 | conv_bias | 0.0000 | 0.7340 | 0.6966 | 0.7857 | 27.9414 | 0.8081 | 33.9574 | 28.0532 | False | False | False |
| pilot2b_rasnet_v2_f32_convbias_cl020 | 32 | conv_bias | 0.2000 | 0.7027 | 0.6390 | 0.7933 | 31.8190 | 0.8116 | 23.7340 | 23.7766 | False | True | False |
| pilot2b_rasnet_v2_f32_gnorig_cl020 | 32 | groupnorm_orig | 0.2000 | 0.7167 | 0.6657 | 0.7881 | 26.3249 | 0.8247 | 22.5426 | 23.7766 | False | True | False |

## Stage 2A Learning-Rate Sweep (`Val N = 94`)

| run_name | lr_key | arch | init_filters | lr | dice_mean | precision_mean | hd95_mean | components_total_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot2a_segresnet_f16_lr1e-4_seed42 | segresnet_f16 | segresnet | 16 | 0.0001 | 0.6659 | 0.5800 | 43.4893 | 28.0532 |
| pilot2a_segresnet_f16_lr3e-4_seed42 | segresnet_f16 | segresnet | 16 | 0.0003 | 0.6337 | 0.5338 | 50.8342 | 37.1702 |
| pilot2a_segresnet_f16_lr1e-3_seed42 | segresnet_f16 | segresnet | 16 | 0.0010 | 0.4466 | 0.3624 | 56.0858 | 83.5319 |
| pilot2a_rasnet_v2_f16_lr1e-4_seed42 | rasnet_v2_f16 | rasnet_v2 | 16 | 0.0001 | 0.7095 | 0.6432 | 29.1400 | 21.6489 |
| pilot2a_rasnet_v2_f16_lr3e-4_seed42 | rasnet_v2_f16 | rasnet_v2 | 16 | 0.0003 | 0.6940 | 0.6237 | 35.6209 | 30.2340 |
| pilot2a_rasnet_v2_f16_lr1e-3_seed42 | rasnet_v2_f16 | rasnet_v2 | 16 | 0.0010 | 0.7039 | 0.6452 | 30.4200 | 28.0213 |
| pilot2a_vnet_f16_lr1e-4_seed42 | vnet | vnet | 16 | 0.0001 | 0.1598 | 0.0892 | 94.4730 | 117.4043 |
| pilot2a_vnet_f16_lr3e-4_seed42 | vnet | vnet | 16 | 0.0003 | 0.7134 | 0.6804 | 25.8184 | 25.3191 |
| pilot2a_vnet_f16_lr1e-3_seed42 | vnet | vnet | 16 | 0.0010 | 0.7054 | 0.6999 | 23.8642 | 33.3511 |
| pilot2a_unet3d_f16_lr1e-4_seed42 | unet3d | unet3d | 32 | 0.0001 | 0.6992 | 0.6315 | 32.5362 | 34.4149 |
| pilot2a_unet3d_f16_lr3e-4_seed42 | unet3d | unet3d | 32 | 0.0003 | 0.6895 | 0.6207 | 34.6263 | 35.7021 |
| pilot2a_unet3d_f16_lr1e-3_seed42 | unet3d | unet3d | 32 | 0.0010 | 0.6404 | 0.5480 | 52.6440 | 76.5319 |
| pilot2a_segresnet_f32_lr3e-4_seed42 | segresnet_f32 | segresnet | 32 | 0.0003 | 0.6422 | 0.5480 | 47.7026 | 37.5851 |
| pilot2a_segresnet_f32_lr1e-3_seed42 | segresnet_f32 | segresnet | 32 | 0.0010 | 0.6717 | 0.6138 | 41.9301 | 23.7766 |
| pilot2a_rasnet_v2_f32_lr3e-4_seed42 | rasnet_v2_f32 | rasnet_v2 | 32 | 0.0003 | 0.7027 | 0.6390 | 31.8190 | 23.7340 |
| pilot2a_rasnet_v2_f32_lr1e-3_seed42 | rasnet_v2_f32 | rasnet_v2 | 32 | 0.0010 | 0.6998 | 0.6368 | 30.7093 | 24.8085 |
| pilot2a_segresnet_f32_lr1e-4_seed42 | segresnet_f32 | segresnet | 32 | 0.0001 | 0.6336 | 0.5296 | 49.2825 | 31.3936 |
| pilot2a_rasnet_v2_f32_lr1e-4_seed42 | rasnet_v2_f32 | rasnet_v2 | 32 | 0.0001 | 0.6981 | 0.6234 | 34.0939 | 26.7340 |
