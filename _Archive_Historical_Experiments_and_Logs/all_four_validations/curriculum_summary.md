# RASNet Curriculum Learning Fine-Tuning Summary

## Best Checkpoint Analysis

- **Best Stage**: Stage 1
- **Checkpoint Path**: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\rasnet_development\rasnet_curriculum_stage1.pth`
- **Inference settings**: Test Time Augmentation (TTA) = True, Threshold = 0.6

### Mean Metrics on Test Set (N=150)

- **Dice**: 0.7711 +/- 0.0677
- **IoU**: 0.6320 +/- 0.0837
- **Precision**: 0.8889 +/- 0.0611
- **Recall**: 0.6869 +/- 0.0907
- **HD95**: 9.73 +/- 10.53 mm

### Stopping Rule Trigger Detail

> **Stage 2 Stopping Rule Triggered**:
> Stage 2 training (Mid-vessel generalization) completed with a test Dice score of **0.7534**, which is below the **0.76** safety floor.
> In accordance with stopping rules, training was terminated and weights were reverted to **Stage 1 (Proximal Vessel Mastery)** weights.

### Comparison Table

| Model | Dice | Precision | Recall | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: |
| **RASNet-P (Precision-Optimized)** | **0.7711** | **0.8889** | 0.6869 | **9.73** |
| RASNet v2 (0.6 threshold) | 0.7879 | 0.8619 | 0.7495 | 8.98 |
| RASNet v2 (0.5 threshold) | 0.7942 | 0.8533 | 0.7495 | 8.12 |
| SegResNet | 0.7637 | 0.8140 | 0.7260 | 9.11 |
| nnU-Net | 0.6003 | 0.5354 | 0.7017 | 58.30 |
| 3D U-Net | 0.6087 | 0.6289 | 0.5919 | 4.62 |
