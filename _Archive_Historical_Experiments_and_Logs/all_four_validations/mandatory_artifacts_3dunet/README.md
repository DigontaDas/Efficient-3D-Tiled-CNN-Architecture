# 3D U-Net Model Export - IMGcas Dataset

## Training Summary
- **Total Epochs Trained**: 35
- **Best Performance**: Epoch 15 (Validation Dice: 0.5219)
- **Final Checkpoint**: Epoch 35
- **Dataset**: IMGcas coronary artery segmentation

## Model Architecture
- **Architecture**: 3D U-Net
- **Input Patch Size**: [96, 96, 96]
- **Channels**: (16, 32, 64, 128, 256)
- **Normalization**: Instance Norm
- **Dropout**: 0.15

## Training Configuration
- **Learning Rate**: 1e-4
- **Batch Size**: 1
- **Optimizer**: Adam (implied)
- **Loss Function**: Dice + Cross-entropy (implied)
- **Augmentation**: 70% probability
- **Early Stopping Patience**: 30 epochs

## Files Description

### Checkpoints
- `best_model.pt`: Model weights from epoch 15 (best validation performance)
- `last_checkpoint.pt`: Complete training state from epoch 35 (final)

### Configuration
- `config_exp02_strong_baseline.py`: Complete training hyperparameters
- `splits.json`: Train/val/test data splits
- `requirements.txt`: Python environment dependencies

### Analysis Tools
- `reconstruct_training_curves.py`: Script to extract full training history from checkpoints
- `training_log_resume.txt`: Training resume log (epoch 30+ evidence)

### Predictions
- `predictions/val/`: Validation set predictions (2 cases)
- `predictions/test/`: Test set predictions (300 cases)
  - `*_pred_mask.nii.gz`: Binary segmentation masks
  - `*_pred_prob.nii.gz`: Probability maps

## Usage Instructions

### Loading the Best Model
```python
import torch
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
model_state = checkpoint['model_state_dict']
best_epoch = checkpoint['epoch']  # 15
best_dice = checkpoint['best_val_dice']  # 0.5219
```

### Loading Final Model
```python
import torch
checkpoint = torch.load('checkpoints/last_checkpoint.pt', map_location='cpu')
model_state = checkpoint['model_state_dict']
final_epoch = checkpoint['epoch']  # 35
optimizer_state = checkpoint['optimizer_state_dict']
```

### Reconstructing Training Curves
```bash
python reconstruct_training_curves.py
```
This will extract the complete 35-epoch training history from the checkpoints.

## Data Splits
- **Train**: 700 volumes
- **Validation**: 150 volumes  
- **Test**: 150 volumes

## Model Performance
- **Best Validation Dice@0.3**: 0.5219 (epoch 15)
- **Final Validation Dice@0.3**: Available in checkpoint
- **Training Duration**: 35 epochs (stopped manually)

## Notes
- Training was configured for 120 epochs but manually stopped at 35
- Best model saved at epoch 15, training continued to epoch 35
- All metrics extracted from checkpoint files
- Compatible with standard PyTorch + MONAI inference pipelines
