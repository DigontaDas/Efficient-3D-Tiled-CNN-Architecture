# nnUNet Model Artifacts

## Model Information
- **Model Type**: nnUNet (3D U-Net)
- **Dataset**: Dataset501_CoronarySeg
- **Configuration**: 3d_fullres
- **Trainer**: nnUNetTrainer_100epochs
- **Cross-validation**: 5-fold

## Folder Structure
```
mandatory_artifacts_nnunet/
├── checkpoints/
│   ├── fold_0/
│   │   ├── checkpoint_best.pth      # Best model weights
│   │   ├── debug.json               # Training configuration
│   │   ├── progress.png             # Training curves
│   │   ├── progress_original.png    # Original progress plot (fold 0)
│   │   └── progress_complete.png    # Complete progress plot (folds 1-4)
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
├── configuration/
│   ├── dataset.json                # Dataset metadata and labels
│   └── plans.json                  # Architecture and preprocessing config
└── training_logs/
    └── training_log_*.txt          # Detailed training logs per fold
```

## Total Size
- **Files**: 52
- **Total Size**: ~1.24 GB

## Usage
These artifacts contain all necessary components for:
1. Model inference on test data
2. Reproducing training results
3. Cross-validation performance analysis
4. Comparison with other models

## Requirements
- nnUNet v2 framework
- PyTorch
- Compatible dataset preprocessing pipeline
