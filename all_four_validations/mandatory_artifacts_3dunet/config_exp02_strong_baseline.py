"""
Configuration for exp02_strong_baseline
======================================

This configuration implements a strong baseline for coronary artery segmentation with:
- Extended training (120 epochs)
- Enhanced regularization and augmentation
- Careful monitoring for optimal generalization
"""

# Training configuration
TRAIN_CONFIG = {
    'num_epochs': 120,
    'early_stopping_patience': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-3,  # Increased regularization
    'patch_size': [96, 96, 96],  # Primary patch size with fallback to [80, 80, 80] if needed
    'batch_size': 1,
    'num_samples': 4,  # Number of patches per volume per iteration
    'pos_neg_ratio': (3, 1),  # Positive to negative sampling ratio
    'augmentation_probability': 0.7,  # Increased augmentation
    'dropout': 0.15,  # Slightly increased dropout
}

# Model configuration
MODEL_CONFIG = {
    'channels': (16, 32, 64, 128, 256),
    'strides': (2, 2, 2, 2),
    'norm': 'instance',
    'dropout': 0.15,  # Match the training config
}

# Transform configuration
TRANSFORM_CONFIG = {
    'augmentation_probability': 0.7,  # Higher augmentation probability
    'spacing': (0.5, 0.5, 0.8),  # Standard CTA spacing
    'axcodes': 'RAS',  # Reorientation target
}

# Validation and monitoring configuration
VALIDATION_CONFIG = {
    'monitor_metric': 'val_dice',  # Metric to monitor for early stopping
    'scheduler_factor': 0.5,  # Factor for ReduceLROnPlateau
    'scheduler_patience': 10,  # Patience for LR scheduler
    'save_best_by': 'val_dice',  # Save model based on validation Dice
}