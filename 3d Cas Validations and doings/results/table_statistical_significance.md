# 📈 Statistical Significance Testing (Wilcoxon Signed-Rank Tests)
**Generated in Phase 6**  
**Methodology**: Paired two-sided Wilcoxon signed-rank tests across per-case metric pairs with Holm-Bonferroni step-down multiple-testing correction.

---

### Dataset A: Primary Benchmark Test Set ($N=150$ ImageCAS)

| Comparison          | Metric    |   Sample_Size_N |   Champ_Mean |   Comp_Mean |   Wilcoxon_Stat |   P_Value_Raw |   P_Value_Holm_Bonferroni | Significant_p_005   | Effect_Direction            |
|:--------------------|:----------|----------------:|-------------:|------------:|----------------:|--------------:|--------------------------:|:--------------------|:----------------------------|
| RASNet vs nnU-Net   | PRECISION |             150 |       0.8801 |      0.7391 |               0 |    2.2996e-26 |                1.6097e-25 | YES                 | RASNet Superior (+0.1410)   |
| RASNet vs nnU-Net   | RECALL    |             150 |       0.7016 |      0.8106 |               0 |    2.2996e-26 |                1.3798e-25 | YES                 | nnU-Net Superior            |
| RASNet vs nnU-Net   | ASD       |             150 |       1.5981 |      3.0966 |            1879 |    1.2589e-12 |                6.2944e-12 | YES                 | RASNet Better (Lower Error) |
| RASNet vs nnU-Net   | HD95      |             150 |      10.2924 |     21.0983 |            2227 |    1.1498e-10 |                4.5991e-10 | YES                 | RASNet Better (Lower Error) |
| RASNet vs nnU-Net   | CLDICE    |             150 |       0.8592 |      0.8201 |            2430 |    1.32e-09   |                3.96e-09   | YES                 | RASNet Superior (+0.0391)   |
| RASNet vs nnU-Net   | IOU       |             150 |       0.6396 |      0.6289 |            4818 |    0.1131     |                0.2262     | NO                  | RASNet Superior (+0.0106)   |
| RASNet vs nnU-Net   | DICE      |             150 |       0.7765 |      0.7687 |            4823 |    0.1152     |                0.1152     | NO                  | RASNet Superior (+0.0078)   |
| RASNet vs V-Net     | PRECISION |             150 |       0.8801 |      0.7545 |               0 |    2.2996e-26 |                1.6097e-25 | YES                 | RASNet Superior (+0.1256)   |
| RASNet vs V-Net     | RECALL    |             150 |       0.7016 |      0.7499 |             433 |    1.0017e-22 |                6.0103e-22 | YES                 | V-Net Superior              |
| RASNet vs V-Net     | CLDICE    |             150 |       0.8592 |      0.8075 |            1454 |    2.8764e-15 |                1.4382e-14 | YES                 | RASNet Superior (+0.0517)   |
| RASNet vs V-Net     | ASD       |             150 |       1.5981 |      3.1758 |            1933 |    2.6069e-12 |                1.0428e-11 | YES                 | RASNet Better (Lower Error) |
| RASNet vs V-Net     | IOU       |             150 |       0.6396 |      0.6006 |            2007 |    6.9542e-12 |                2.0863e-11 | YES                 | RASNet Superior (+0.0389)   |
| RASNet vs V-Net     | DICE      |             150 |       0.7765 |      0.7474 |            2116 |    2.8507e-11 |                5.7014e-11 | YES                 | RASNet Superior (+0.0291)   |
| RASNet vs V-Net     | HD95      |             150 |      10.2924 |     21.805  |            2289 |    2.4597e-10 |                2.4597e-10 | YES                 | RASNet Better (Lower Error) |
| RASNet vs SegResNet | PRECISION |             150 |       0.8801 |      0.7313 |               0 |    2.2996e-26 |                1.6097e-25 | YES                 | RASNet Superior (+0.1488)   |
| RASNet vs SegResNet | RECALL    |             150 |       0.7016 |      0.7713 |               0 |    2.2996e-26 |                1.3798e-25 | YES                 | SegResNet Superior          |
| RASNet vs SegResNet | ASD       |             150 |       1.5981 |      4.6614 |             445 |    1.2518e-22 |                6.2592e-22 | YES                 | RASNet Better (Lower Error) |
| RASNet vs SegResNet | CLDICE    |             150 |       0.8592 |      0.7769 |             649 |    5.128e-21  |                2.0512e-20 | YES                 | RASNet Superior (+0.0823)   |
| RASNet vs SegResNet | HD95      |             150 |      10.2924 |     31.4775 |             650 |    5.2203e-21 |                1.5661e-20 | YES                 | RASNet Better (Lower Error) |
| RASNet vs SegResNet | IOU       |             150 |       0.6396 |      0.6001 |            1819 |    5.5409e-13 |                1.1082e-12 | YES                 | RASNet Superior (+0.0395)   |
| RASNet vs SegResNet | DICE      |             150 |       0.7765 |      0.7469 |            1929 |    2.4709e-12 |                2.4709e-12 | YES                 | RASNet Superior (+0.0296)   |
| RASNet vs 3D U-Net  | DICE      |             150 |       0.7765 |      0.5561 |               0 |    2.2996e-26 |                1.6097e-25 | YES                 | RASNet Superior (+0.2205)   |
| RASNet vs 3D U-Net  | IOU       |             150 |       0.6396 |      0.3865 |               0 |    2.2996e-26 |                1.3798e-25 | YES                 | RASNet Superior (+0.2531)   |
| RASNet vs 3D U-Net  | PRECISION |             150 |       0.8801 |      0.6069 |               0 |    2.2996e-26 |                1.1498e-25 | YES                 | RASNet Superior (+0.2732)   |
| RASNet vs 3D U-Net  | CLDICE    |             150 |       0.8592 |      0.7027 |              38 |    4.9248e-26 |                1.9699e-25 | YES                 | RASNet Superior (+0.1566)   |
| RASNet vs 3D U-Net  | RECALL    |             150 |       0.7016 |      0.5178 |              43 |    5.4418e-26 |                1.6326e-25 | YES                 | RASNet Superior (+0.1838)   |
| RASNet vs 3D U-Net  | ASD       |             150 |       1.5981 |      1.6903 |            3821 |    0.0005501  |                0.0011     | YES                 | RASNet Better (Lower Error) |
| RASNet vs 3D U-Net  | HD95      |             150 |      10.2924 |      9.878  |            4975 |    0.1971     |                0.1971     | NO                  | 3D U-Net Better             |

---

### Dataset B: 3D CAS External Dataset ($N=200$)

| Comparison          | Metric    |   Sample_Size_N |   Champ_Mean |   Comp_Mean |   Wilcoxon_Stat |   P_Value_Raw |   P_Value_Holm_Bonferroni | Significant_p_005   | Effect_Direction            |
|:--------------------|:----------|----------------:|-------------:|------------:|----------------:|--------------:|--------------------------:|:--------------------|:----------------------------|
| RASNet vs nnU-Net   | DICE      |             134 |       0.7765 |      0.5574 |               5 |    1.0972e-23 |                7.6807e-23 | YES                 | RASNet Superior (+0.2190)   |
| RASNet vs nnU-Net   | IOU       |             134 |       0.6387 |      0.4001 |               5 |    1.0972e-23 |                6.5834e-23 | YES                 | RASNet Superior (+0.2386)   |
| RASNet vs nnU-Net   | CLDICE    |             134 |       0.8585 |      0.5693 |               8 |    1.1738e-23 |                5.8691e-23 | YES                 | RASNet Superior (+0.2893)   |
| RASNet vs nnU-Net   | ASD       |             134 |       1.5091 |      7.187  |              43 |    2.5707e-23 |                1.0283e-22 | YES                 | RASNet Better (Lower Error) |
| RASNet vs nnU-Net   | RECALL    |             134 |       0.7031 |      0.4559 |              54 |    3.2848e-23 |                9.8544e-23 | YES                 | RASNet Superior (+0.2472)   |
| RASNet vs nnU-Net   | HD95      |             134 |      29.3828 |     58.1327 |             102 |    9.5073e-23 |                1.9015e-22 | YES                 | RASNet Better (Lower Error) |
| RASNet vs nnU-Net   | PRECISION |             134 |       0.8759 |      0.7925 |             364 |    2.5777e-20 |                2.5777e-20 | YES                 | RASNet Superior (+0.0834)   |
| RASNet vs V-Net     | PRECISION |             134 |       0.8759 |      0.8424 |             313 |    8.8922e-21 |                6.2245e-20 | YES                 | RASNet Superior (+0.0335)   |
| RASNet vs V-Net     | ASD       |             134 |       1.5091 |      1.8849 |            2979 |    0.00060847 |                0.0037     | YES                 | RASNet Better (Lower Error) |
| RASNet vs V-Net     | CLDICE    |             134 |       0.8585 |      0.8446 |            3027 |    0.00089627 |                0.0045     | YES                 | RASNet Superior (+0.0139)   |
| RASNet vs V-Net     | RECALL    |             134 |       0.7031 |      0.7101 |            3103 |    0.0016     |                0.0065     | YES                 | V-Net Superior              |
| RASNet vs V-Net     | IOU       |             134 |       0.6387 |      0.6266 |            3176 |    0.0028     |                0.0084     | YES                 | RASNet Superior (+0.0121)   |
| RASNet vs V-Net     | DICE      |             134 |       0.7765 |      0.7669 |            3216 |    0.0037     |                0.0074     | YES                 | RASNet Superior (+0.0095)   |
| RASNet vs V-Net     | HD95      |             134 |      29.3828 |     30.9232 |            3810 |    0.1472     |                0.1472     | NO                  | RASNet Better (Lower Error) |
| RASNet vs SegResNet | PRECISION |             134 |       0.8759 |      0.8409 |               1 |    1.0028e-23 |                7.0194e-23 | YES                 | RASNet Superior (+0.0351)   |
| RASNet vs SegResNet | RECALL    |             134 |       0.7031 |      0.7461 |             287 |    5.1433e-21 |                3.086e-20  | YES                 | SegResNet Superior          |
| RASNet vs SegResNet | IOU       |             134 |       0.6387 |      0.6548 |            2238 |    3.9072e-07 |                1.9536e-06 | YES                 | SegResNet Superior          |
| RASNet vs SegResNet | DICE      |             134 |       0.7765 |      0.7883 |            2245 |    4.2394e-07 |                1.6958e-06 | YES                 | SegResNet Superior          |
| RASNet vs SegResNet | HD95      |             134 |      29.3828 |     27.0431 |            2914 |    0.00035404 |                0.0011     | YES                 | SegResNet Better            |
| RASNet vs SegResNet | ASD       |             134 |       1.5091 |      1.3672 |            3026 |    0.00088917 |                0.0018     | YES                 | SegResNet Better            |
| RASNet vs SegResNet | CLDICE    |             134 |       0.8585 |      0.868  |            3305 |    0.0069     |                0.0069     | YES                 | SegResNet Superior          |
| RASNet vs 3D U-Net  | DICE      |             134 |       0.7765 |      0      |               0 |    9.8044e-24 |                6.8631e-23 | YES                 | RASNet Superior (+0.7765)   |
| RASNet vs 3D U-Net  | IOU       |             134 |       0.6387 |      0      |               0 |    9.8044e-24 |                5.8827e-23 | YES                 | RASNet Superior (+0.6387)   |
| RASNet vs 3D U-Net  | PRECISION |             134 |       0.8759 |      0      |               0 |    9.8044e-24 |                4.9022e-23 | YES                 | RASNet Superior (+0.8759)   |
| RASNet vs 3D U-Net  | RECALL    |             134 |       0.7031 |      0      |               0 |    9.8044e-24 |                3.9218e-23 | YES                 | RASNet Superior (+0.7031)   |
| RASNet vs 3D U-Net  | CLDICE    |             134 |       0.8585 |      0      |               0 |    9.8044e-24 |                2.9413e-23 | YES                 | RASNet Superior (+0.8585)   |
| RASNet vs 3D U-Net  | ASD       |             134 |       1.5091 |     15.7315 |             885 |    6.5744e-16 |                1.3149e-15 | YES                 | RASNet Better (Lower Error) |
| RASNet vs 3D U-Net  | HD95      |             134 |      29.3828 |     37.0049 |            2851 |    0.00020557 |                0.00020557 | YES                 | RASNet Better (Lower Error) |
