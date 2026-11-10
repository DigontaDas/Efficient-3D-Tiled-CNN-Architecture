# Table 4: Measured Computational Efficiency & Parameter Footprint (`NVIDIA GeForce RTX 4080 SUPER`)

| Model | Parameters (M) | Checkpoint Size (MB) | Inference Patch Grid | GFLOPs / Patch | Single-Patch Latency FP32 (ms) | Single-Patch Latency AMP FP16 (ms) | Peak Patch VRAM (MB) | Test Arm A Dice (Raw) | Test Arm C Dice (Top-2 CC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RASNet (Ours)** | `4.707M` (`4,706,772`) | `18.00` | `96x96x96` | `123.39` | `15.64 ± 0.36` | `13.55 ± 0.06` | `390.7` | `0.7736` | `0.7899` |
| **nnU-Net V2** | `30.786M` (`30,785,994`) | `235.05` | `128x192x192` | `2149.13` | `115.74 ± 0.19` | `63.04 ± 0.18` | `2381.1` | `0.7687` | `0.8055` |
| **SegResNet** | `4.701M` (`4,700,914`) | `17.96` | `96x96x96` | `122.56` | `13.64 ± 0.08` | `11.81 ± 0.10` | `402.0` | `0.7481` | `0.7884` |
| **V-Net** | `45.599M` (`45,599,452`) | `174.03` | `96x96x96` | `617.23` | `25.64 ± 0.19` | `13.77 ± 0.13` | `1028.9` | `0.7492` | `0.7710` |
| **3D U-Net** | `4.806M` (`4,805,534`) | `18.36` | `128x128x128` | `23.48` | `4.20 ± 0.16` | `4.29 ± 0.19` | `756.3` | `0.5517` | `0.6064` |
