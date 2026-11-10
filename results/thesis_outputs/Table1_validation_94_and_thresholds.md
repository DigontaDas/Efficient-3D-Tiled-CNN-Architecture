# Table 1: Validation Split Performance (`N = 94`) & Frozen Threshold Selection (`thresholds.json`)

| Model | Best Epoch | Training-Log Val Metric | Val Arm A Dice (Mean ± SD) | Val Arm A Precision | Val Arm A Recall | Val Arm A HD95 (mm) | Val Arm A ASD (mm) | Val Arm A clDice | Selected $\tau^*$ | Grid Status | Val Arm C Dice ($\tau^*$, Top-2 CC) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **RASNet (Ours)** | `160` | `0.7770` (Full-vol 0.5mm grid Dice) | `0.7789 ± 0.0521` | `0.8165 ± 0.0692` | `0.7483 ± 0.0612` | `13.07 ± 13.53` | `2.066 ± 1.375` | `0.8480 ± 0.0617` | `0.30` | Lower Bound (0.30) | `0.7897` |
| **nnU-Net V2** | `189` | `0.8376` (Patch Pseudo-Dice (EMA=0.8162)) | `0.7761 ± 0.0567` | `0.7467 ± 0.0834` | `0.8144 ± 0.0541` | `18.76 ± 16.43` | `2.766 ± 2.076` | `0.8339 ± 0.0628` | `0.35` | Interior Peak (0.35) | `0.8075` |
| **SegResNet** | `190` | `0.7521` (Full-vol 0.5mm grid Dice) | `0.7532 ± 0.0591` | `0.7338 ± 0.0848` | `0.7794 ± 0.0559` | `28.24 ± 17.27` | `4.201 ± 1.917` | `0.8058 ± 0.0700` | `0.35` | Interior Peak (0.35) | `0.7901` |
| **V-Net** | `180` | `0.7491` (Full-vol 0.5mm grid Dice) | `0.7511 ± 0.0619` | `0.7553 ± 0.0925` | `0.7547 ± 0.0630` | `20.94 ± 17.26` | `3.076 ± 2.344` | `0.8225 ± 0.0710` | `0.30` | Lower Bound (0.30) | `0.7702` |
| **3D U-Net** | `180` | `0.6638` (128^3 resized grid Dice) | `0.5516 ± 0.0480` | `0.7263 ± 0.0618` | `0.4470 ± 0.0508` | `9.74 ± 5.58` | `1.852 ± 0.780` | `0.7110 ± 0.0674` | `0.30` | Lower Bound (0.30) | `0.6047` |
