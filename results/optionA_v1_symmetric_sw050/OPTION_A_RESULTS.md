# Option A results (generated; do not hand-edit)

Source files: `summary_test.csv`, `paired_stats_families.csv`, `noninferiority_rasnet_vs_nnunet_dice.csv`.

## R1_raw_1pass

| Model | τ | Dice | Precision | Recall | HD95 (mm) | ASD (mm) | clDice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| rasnet | 0.50 | 0.7788 ± 0.0623 | 0.7764 ± 0.0908 | 0.7888 ± 0.0666 | 15.9693 ± 14.0348 | 2.5313 ± 1.9612 | 0.8394 ± 0.0711 |
| nnunet | 0.50 | 0.7687 ± 0.0668 | 0.7391 ± 0.0967 | 0.8106 ± 0.0680 | 21.1013 ± 17.1005 | 3.1683 ± 2.6358 | 0.8201 ± 0.0766 |
| segresnet | 0.50 | 0.7545 ± 0.0668 | 0.7197 ± 0.0991 | 0.8030 ± 0.0618 | 27.1973 ± 17.8312 | 4.2431 ± 2.5647 | 0.8080 ± 0.0775 |
| vnet | 0.50 | 0.7483 ± 0.0715 | 0.7151 ± 0.1076 | 0.7964 ± 0.0630 | 25.1259 ± 18.3880 | 3.7499 ± 2.8651 | 0.8151 ± 0.0802 |

## R2_raw_tta8

| Model | τ | Dice | Precision | Recall | HD95 (mm) | ASD (mm) | clDice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| rasnet | 0.50 | 0.7863 ± 0.0619 | 0.7876 ± 0.0889 | 0.7923 ± 0.0677 | 14.8818 ± 14.3976 | 2.3450 ± 1.9452 | 0.8469 ± 0.0710 |
| nnunet | 0.50 | 0.7760 ± 0.0641 | 0.7536 ± 0.0936 | 0.8090 ± 0.0681 | 19.2464 ± 16.9328 | 2.9143 ± 2.4871 | 0.8264 ± 0.0750 |
| segresnet | 0.50 | 0.7644 ± 0.0663 | 0.7369 ± 0.0992 | 0.8037 ± 0.0616 | 24.6099 ± 18.1593 | 3.7960 ± 2.5379 | 0.8182 ± 0.0774 |
| vnet | 0.50 | 0.7574 ± 0.0711 | 0.7320 ± 0.1085 | 0.7960 ± 0.0624 | 22.2000 ± 18.0916 | 3.3559 ± 2.8355 | 0.8249 ± 0.0794 |

## R3_tuned_1pass_top2

| Model | τ | Dice | Precision | Recall | HD95 (mm) | ASD (mm) | clDice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nnunet | 0.35 | 0.8055 ± 0.0633 | 0.8207 ± 0.0676 | 0.7977 ± 0.0913 | 8.2618 ± 9.8067 | 1.3946 ± 1.8622 | 0.8739 ± 0.0679 |
| rasnet | 0.40 | 0.8008 ± 0.0661 | 0.8270 ± 0.0716 | 0.7833 ± 0.0927 | 9.1844 ± 10.7069 | 1.5260 ± 2.1615 | 0.8724 ± 0.0727 |
| segresnet | 0.40 | 0.7891 ± 0.0758 | 0.7979 ± 0.0866 | 0.7882 ± 0.0964 | 10.1067 ± 11.6118 | 1.7939 ± 2.9087 | 0.8631 ± 0.0860 |
| vnet | 0.40 | 0.7875 ± 0.0677 | 0.8058 ± 0.0786 | 0.7781 ± 0.0926 | 10.1413 ± 11.1828 | 1.7016 ± 2.1810 | 0.8638 ± 0.0744 |

## R4_tuned_tta8_top2

| Model | τ | Dice | Precision | Recall | HD95 (mm) | ASD (mm) | clDice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nnunet | 0.20 | 0.8059 ± 0.0630 | 0.8084 ± 0.0669 | 0.8104 ± 0.0917 | 8.1727 ± 9.8609 | 1.3899 ± 1.8662 | 0.8744 ± 0.0706 |
| rasnet | 0.40 | 0.8039 ± 0.0691 | 0.8302 ± 0.0718 | 0.7861 ± 0.0955 | 8.9299 ± 10.9696 | 1.5700 ± 2.4873 | 0.8750 ± 0.0777 |
| segresnet | 0.45 | 0.7989 ± 0.0692 | 0.8200 ± 0.0788 | 0.7855 ± 0.0904 | 9.1244 ± 10.6884 | 1.5750 ± 2.3510 | 0.8728 ± 0.0774 |
| vnet | 0.45 | 0.7915 ± 0.0677 | 0.8216 ± 0.0769 | 0.7711 ± 0.0923 | 9.5019 ± 10.9470 | 1.6161 ± 2.1687 | 0.8678 ± 0.0745 |

## Pre-declared families (Holm within family)

| Recipe | Family | a vs b | Metric | Mean diff (a−b) | 95% CI | p | Holm p | W/T/L |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| R1_raw_1pass | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | precision | +0.0373 | [+0.0302, +0.0452] | 3.83e-16 | 1.15e-15 | 121/0/29 |
| R1_raw_1pass | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | hd95 | -5.1320 | [-7.2682, -3.1640] | 1.66e-06 | 3.32e-06 | 103/0/47 |
| R1_raw_1pass | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | asd | -0.6369 | [-0.8870, -0.4014] | 7.75e-06 | 7.75e-06 | 97/0/53 |
| R1_raw_1pass | FB_rasnet_vs_segresnet | rasnet vs segresnet | dice | +0.0243 | [+0.0210, +0.0276] | 2.86e-24 | 5.71e-24 | 138/0/12 |
| R1_raw_1pass | FB_rasnet_vs_segresnet | rasnet vs segresnet | precision | +0.0567 | [+0.0517, +0.0618] | 2.54e-26 | 1.02e-25 | 148/0/2 |
| R1_raw_1pass | FB_rasnet_vs_segresnet | rasnet vs segresnet | hd95 | -11.2280 | [-13.1932, -9.4157] | 2.01e-25 | 6.02e-25 | 143/1/6 |
| R1_raw_1pass | FB_rasnet_vs_segresnet | rasnet vs segresnet | cldice | +0.0314 | [+0.0268, +0.0360] | 7.44e-23 | 7.44e-23 | 136/0/14 |
| R1_raw_1pass | FC_rasnet_vs_vnet | rasnet vs vnet | dice | +0.0305 | [+0.0260, +0.0354] | 2.21e-21 | 6.62e-21 | 133/0/17 |
| R1_raw_1pass | FC_rasnet_vs_vnet | rasnet vs vnet | precision | +0.0613 | [+0.0541, +0.0687] | 1.72e-25 | 6.89e-25 | 142/0/8 |
| R1_raw_1pass | FC_rasnet_vs_vnet | rasnet vs vnet | hd95 | -9.1565 | [-11.3967, -7.1730] | 4.05e-17 | 8.1e-17 | 125/1/24 |
| R1_raw_1pass | FC_rasnet_vs_vnet | rasnet vs vnet | cldice | +0.0243 | [+0.0187, +0.0306] | 2.37e-12 | 2.37e-12 | 106/0/44 |
| R1_raw_1pass | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | dice | +0.0101 | [+0.0058, +0.0146] | 0.000292 | 0.000292 | 88/0/62 |
| R1_raw_1pass | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | cldice | +0.0193 | [+0.0138, +0.0249] | 2.05e-09 | 4.1e-09 | 107/0/43 |
| R1_raw_1pass | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | recall | -0.0218 | [-0.0250, -0.0187] | 7.59e-22 | 2.28e-21 | 18/0/132 |
| R1_raw_1pass | EX_baselines | nnunet vs segresnet | dice | +0.0142 | [+0.0101, +0.0183] | 1.6e-10 | 3.21e-10 | 112/0/38 |
| R1_raw_1pass | EX_baselines | nnunet vs vnet | dice | +0.0204 | [+0.0165, +0.0246] | 6.78e-16 | 2.03e-15 | 121/0/29 |
| R1_raw_1pass | EX_baselines | segresnet vs vnet | dice | +0.0062 | [+0.0020, +0.0104] | 0.0149 | 0.0149 | 88/0/62 |
| R2_raw_tta8 | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | precision | +0.0341 | [+0.0270, +0.0411] | 3.83e-16 | 1.15e-15 | 126/0/24 |
| R2_raw_tta8 | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | hd95 | -4.3646 | [-6.3957, -2.5388] | 4.03e-06 | 8.06e-06 | 100/1/49 |
| R2_raw_tta8 | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | asd | -0.5693 | [-0.8004, -0.3509] | 4.47e-06 | 8.06e-06 | 100/0/50 |
| R2_raw_tta8 | FB_rasnet_vs_segresnet | rasnet vs segresnet | dice | +0.0219 | [+0.0188, +0.0251] | 1.96e-23 | 3.92e-23 | 140/0/10 |
| R2_raw_tta8 | FB_rasnet_vs_segresnet | rasnet vs segresnet | precision | +0.0508 | [+0.0461, +0.0557] | 2.35e-26 | 9.39e-26 | 149/0/1 |
| R2_raw_tta8 | FB_rasnet_vs_segresnet | rasnet vs segresnet | hd95 | -9.7281 | [-11.7327, -7.8595] | 2.27e-25 | 6.81e-25 | 145/0/5 |
| R2_raw_tta8 | FB_rasnet_vs_segresnet | rasnet vs segresnet | cldice | +0.0287 | [+0.0244, +0.0329] | 4.75e-23 | 4.75e-23 | 134/0/16 |
| R2_raw_tta8 | FC_rasnet_vs_vnet | rasnet vs vnet | dice | +0.0290 | [+0.0244, +0.0336] | 3.46e-22 | 1.04e-21 | 135/0/15 |
| R2_raw_tta8 | FC_rasnet_vs_vnet | rasnet vs vnet | precision | +0.0556 | [+0.0485, +0.0629] | 1.56e-25 | 6.24e-25 | 145/0/5 |
| R2_raw_tta8 | FC_rasnet_vs_vnet | rasnet vs vnet | hd95 | -7.3182 | [-9.5034, -5.3473] | 3.73e-14 | 7.47e-14 | 117/1/32 |
| R2_raw_tta8 | FC_rasnet_vs_vnet | rasnet vs vnet | cldice | +0.0220 | [+0.0165, +0.0278] | 1e-10 | 1e-10 | 108/0/42 |
| R2_raw_tta8 | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | dice | +0.0103 | [+0.0062, +0.0145] | 8.24e-06 | 8.24e-06 | 97/0/53 |
| R2_raw_tta8 | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | cldice | +0.0205 | [+0.0153, +0.0257] | 1.02e-11 | 2.03e-11 | 107/0/43 |
| R2_raw_tta8 | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | recall | -0.0167 | [-0.0198, -0.0136] | 2.48e-18 | 7.44e-18 | 21/0/129 |
| R2_raw_tta8 | EX_baselines | nnunet vs segresnet | dice | +0.0116 | [+0.0074, +0.0157] | 1.97e-08 | 3.94e-08 | 113/0/37 |
| R2_raw_tta8 | EX_baselines | nnunet vs vnet | dice | +0.0187 | [+0.0149, +0.0226] | 5.39e-15 | 1.62e-14 | 122/0/28 |
| R2_raw_tta8 | EX_baselines | segresnet vs vnet | dice | +0.0070 | [+0.0031, +0.0112] | 0.00109 | 0.00109 | 96/0/54 |
| R3_tuned_1pass_top2 | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | precision | +0.0063 | [-0.0027, +0.0136] | 0.000195 | 0.000585 | 94/0/56 |
| R3_tuned_1pass_top2 | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | hd95 | +0.9226 | [-0.1077, +2.1706] | 0.262 | 0.525 | 72/6/72 |
| R3_tuned_1pass_top2 | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | asd | +0.1314 | [-0.1668, +0.5369] | 0.553 | 0.553 | 73/0/77 |
| R3_tuned_1pass_top2 | FB_rasnet_vs_segresnet | rasnet vs segresnet | dice | +0.0116 | [+0.0064, +0.0176] | 6.21e-10 | 1.86e-09 | 116/0/34 |
| R3_tuned_1pass_top2 | FB_rasnet_vs_segresnet | rasnet vs segresnet | precision | +0.0291 | [+0.0227, +0.0361] | 7.86e-23 | 3.15e-22 | 138/0/12 |
| R3_tuned_1pass_top2 | FB_rasnet_vs_segresnet | rasnet vs segresnet | hd95 | -0.9223 | [-2.0478, +0.0569] | 0.0322 | 0.0322 | 83/4/63 |
| R3_tuned_1pass_top2 | FB_rasnet_vs_segresnet | rasnet vs segresnet | cldice | +0.0093 | [+0.0026, +0.0169] | 0.0115 | 0.0231 | 85/0/65 |
| R3_tuned_1pass_top2 | FC_rasnet_vs_vnet | rasnet vs vnet | dice | +0.0133 | [+0.0055, +0.0202] | 4.67e-09 | 1.4e-08 | 108/0/42 |
| R3_tuned_1pass_top2 | FC_rasnet_vs_vnet | rasnet vs vnet | precision | +0.0212 | [+0.0118, +0.0295] | 2.71e-12 | 1.09e-11 | 115/0/35 |
| R3_tuned_1pass_top2 | FC_rasnet_vs_vnet | rasnet vs vnet | hd95 | -0.9569 | [-2.4174, +0.4002] | 0.0289 | 0.0289 | 90/3/57 |
| R3_tuned_1pass_top2 | FC_rasnet_vs_vnet | rasnet vs vnet | cldice | +0.0086 | [-0.0007, +0.0169] | 0.0057 | 0.0114 | 91/0/59 |
| R3_tuned_1pass_top2 | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | dice | -0.0047 | [-0.0122, +0.0012] | 0.0362 | 0.0724 | 66/0/84 |
| R3_tuned_1pass_top2 | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | cldice | -0.0015 | [-0.0110, +0.0064] | 0.986 | 0.986 | 78/0/72 |
| R3_tuned_1pass_top2 | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | recall | -0.0144 | [-0.0223, -0.0080] | 2.03e-07 | 6.1e-07 | 52/0/98 |
| R3_tuned_1pass_top2 | EX_baselines | nnunet vs segresnet | dice | +0.0163 | [+0.0098, +0.0245] | 5.51e-11 | 1.1e-10 | 115/0/35 |
| R3_tuned_1pass_top2 | EX_baselines | nnunet vs vnet | dice | +0.0180 | [+0.0125, +0.0241] | 1.37e-13 | 4.12e-13 | 119/0/31 |
| R3_tuned_1pass_top2 | EX_baselines | segresnet vs vnet | dice | +0.0017 | [-0.0075, +0.0100] | 0.0792 | 0.0792 | 84/0/66 |
| R4_tuned_tta8_top2 | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | precision | +0.0218 | [+0.0143, +0.0278] | 2.41e-17 | 7.23e-17 | 132/0/18 |
| R4_tuned_tta8_top2 | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | hd95 | +0.7572 | [-0.1813, +1.9270] | 0.245 | 0.489 | 65/4/81 |
| R4_tuned_tta8_top2 | FA_primary_rasnet_vs_nnunet | rasnet vs nnunet | asd | +0.1801 | [-0.0654, +0.5562] | 0.375 | 0.489 | 67/0/83 |
| R4_tuned_tta8_top2 | FB_rasnet_vs_segresnet | rasnet vs segresnet | dice | +0.0050 | [-0.0024, +0.0121] | 1.07e-08 | 4.27e-08 | 112/0/38 |
| R4_tuned_tta8_top2 | FB_rasnet_vs_segresnet | rasnet vs segresnet | precision | +0.0102 | [+0.0024, +0.0179] | 1.06e-06 | 3.17e-06 | 101/0/49 |
| R4_tuned_tta8_top2 | FB_rasnet_vs_segresnet | rasnet vs segresnet | hd95 | -0.1945 | [-1.3017, +0.9282] | 0.233 | 0.383 | 77/9/64 |
| R4_tuned_tta8_top2 | FB_rasnet_vs_segresnet | rasnet vs segresnet | cldice | +0.0022 | [-0.0073, +0.0108] | 0.191 | 0.383 | 83/0/67 |
| R4_tuned_tta8_top2 | FC_rasnet_vs_vnet | rasnet vs vnet | dice | +0.0124 | [+0.0056, +0.0183] | 8.96e-13 | 3.58e-12 | 115/0/35 |
| R4_tuned_tta8_top2 | FC_rasnet_vs_vnet | rasnet vs vnet | precision | +0.0085 | [-0.0000, +0.0162] | 0.00371 | 0.00743 | 93/0/57 |
| R4_tuned_tta8_top2 | FC_rasnet_vs_vnet | rasnet vs vnet | hd95 | -0.5720 | [-1.8114, +0.6466] | 0.00909 | 0.00909 | 90/3/57 |
| R4_tuned_tta8_top2 | FC_rasnet_vs_vnet | rasnet vs vnet | cldice | +0.0072 | [-0.0013, +0.0143] | 0.00177 | 0.0053 | 91/0/59 |
| R4_tuned_tta8_top2 | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | dice | -0.0021 | [-0.0088, +0.0030] | 0.99 | 0.99 | 76/0/74 |
| R4_tuned_tta8_top2 | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | cldice | +0.0006 | [-0.0085, +0.0078] | 0.223 | 0.446 | 81/0/69 |
| R4_tuned_tta8_top2 | FD_secondary_rasnet_vs_nnunet | rasnet vs nnunet | recall | -0.0244 | [-0.0319, -0.0183] | 7.99e-18 | 2.4e-17 | 17/0/133 |
| R4_tuned_tta8_top2 | EX_baselines | nnunet vs segresnet | dice | +0.0071 | [+0.0019, +0.0135] | 0.000803 | 0.000803 | 96/0/54 |
| R4_tuned_tta8_top2 | EX_baselines | nnunet vs vnet | dice | +0.0145 | [+0.0095, +0.0200] | 8.8e-12 | 2.64e-11 | 117/0/33 |
| R4_tuned_tta8_top2 | EX_baselines | segresnet vs vnet | dice | +0.0074 | [+0.0003, +0.0141] | 9.81e-06 | 1.96e-05 | 97/0/53 |

## Dice non-inferiority RASNet vs nnU-Net (margin −0.005)

| Recipe | Mean diff | 95% CI | Non-inferior | Superior |
| :--- | :---: | :---: | :---: | :---: |
| R1_raw_1pass | +0.0101 | [+0.0058, +0.0146] | True | True |
| R2_raw_tta8 | +0.0103 | [+0.0062, +0.0145] | True | True |
| R3_tuned_1pass_top2 | -0.0047 | [-0.0122, +0.0012] | False | False |
| R4_tuned_tta8_top2 | -0.0021 | [-0.0088, +0.0030] | False | False |

## End-to-end inference time per case (Val cases, RTX 4080 SUPER; `timing_summary.csv`)

| Model | Mode | n | Median s/case [IQR] | Peak allocated VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: |
| nnunet | sw050_gauss_1pass | 10 | 29.8 [28.3, 31.7] | 2243 |
| nnunet | sw050_gauss_tta8 | 10 | 74.4 [73.1, 76.3] | 2261 |
| rasnet | sw050_gauss_1pass | 10 | 5.4 [5.2, 5.7] | 4782 |
| rasnet | sw050_gauss_tta8 | 10 | 22.9 [22.7, 23.1] | 5134 |
| segresnet | sw050_gauss_1pass | 10 | 5.0 [4.7, 5.1] | 4782 |
| segresnet | sw050_gauss_tta8 | 10 | 19.8 [19.5, 19.9] | 5134 |
| vnet | sw050_gauss_1pass | 10 | 6.1 [6.0, 6.9] | 4941 |
| vnet | sw050_gauss_tta8 | 10 | 27.7 [27.4, 27.8] | 5929 |
