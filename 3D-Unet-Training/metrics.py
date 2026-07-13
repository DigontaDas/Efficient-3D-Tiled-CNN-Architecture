import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    
    Dice Loss = 1 - Dice Coefficient
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Apply sigmoid to predictions if they are logits
        if not (pred > 0).all() and not (pred <= 1).all():
            pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        # Calculate dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return dice loss
        return 1.0 - dice

class CombinedLoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross-Entropy Loss
    
    This combination often provides better training stability
    """
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice Coefficient
    
    Args:
        pred (torch.Tensor): Predictions (after sigmoid)
        target (torch.Tensor): Ground truth labels
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Dice coefficient
    """
    # Apply sigmoid if predictions are logits
    if not (pred > 0).all() and not (pred <= 1).all():
        pred = torch.sigmoid(pred)
    
    # Threshold predictions to binary (0 or 1)
    pred = (pred > 0.5).float()
    
    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    # Calculate dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()

def iou_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) / Jaccard Index
    
    Args:
        pred (torch.Tensor): Predictions (after sigmoid)
        target (torch.Tensor): Ground truth labels
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: IoU coefficient
    """
    # Apply sigmoid if predictions are logits
    if not (pred > 0).all() and not (pred <= 1).all():
        pred = torch.sigmoid(pred)
    
    # Threshold predictions to binary (0 or 1)
    pred = (pred > 0.5).float()
    
    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()

def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate multiple segmentation metrics
    
    Args:
        pred (torch.Tensor): Predictions (logits)
        target (torch.Tensor): Ground truth labels
        threshold (float): Threshold for binary conversion
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Apply sigmoid to get probabilities
    pred_prob = torch.sigmoid(pred)
    
    # Apply threshold to get binary predictions
    pred_binary = (pred_prob > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Calculate basic metrics
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    
    # Dice coefficient
    dice = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
    
    # IoU coefficient
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Precision
    precision = (intersection + 1e-6) / (pred_sum + 1e-6)
    
    # Recall (Sensitivity)
    recall = (intersection + 1e-6) / (target_sum + 1e-6)
    
    # Specificity
    true_negatives = ((1 - pred_flat) * (1 - target_flat)).sum()
    total_negatives = (1 - target_flat).sum()
    specificity = (true_negatives + 1e-6) / (total_negatives + 1e-6)
    
    # Accuracy
    correct_predictions = (pred_flat == target_flat).sum()
    total_predictions = pred_flat.numel()
    accuracy = correct_predictions.float() / total_predictions
    
    metrics = {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'specificity': specificity.item(),
        'accuracy': accuracy.item()
    }
    
    return metrics

def batch_metrics(pred_batch, target_batch, threshold=0.5):
    """
    Calculate metrics for a batch of predictions
    
    Args:
        pred_batch (torch.Tensor): Batch of predictions (B, C, D, H, W)
        target_batch (torch.Tensor): Batch of targets (B, C, D, H, W)
        threshold (float): Threshold for binary conversion
    
    Returns:
        dict: Average metrics across the batch
    """
    batch_size = pred_batch.shape[0]
    all_metrics = []
    
    for i in range(batch_size):
        metrics = calculate_metrics(pred_batch[i], target_batch[i], threshold)
        all_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics

class MetricsTracker:
    """
    Class to track metrics during training and validation
    """
    
    def __init__(self):
        self.train_metrics = []
        self.val_metrics = []
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.best_epoch = 0
    
    def update_train(self, metrics):
        """Update training metrics"""
        self.train_metrics.append(metrics)
    
    def update_val(self, metrics):
        """Update validation metrics and check for new best"""
        self.val_metrics.append(metrics)
        
        # Check for new best model
        if metrics['dice'] > self.best_dice:
            self.best_dice = metrics['dice']
            self.best_iou = metrics['iou']
            self.best_epoch = len(self.val_metrics) - 1
            return True  # New best model
        
        return False
    
    def get_best_metrics(self):
        """Get the best validation metrics"""
        return {
            'best_dice': self.best_dice,
            'best_iou': self.best_iou,
            'best_epoch': self.best_epoch
        }
    
    def get_metrics_history(self):
        """Get the complete metrics history"""
        return {
            'train': self.train_metrics,
            'val': self.val_metrics
        }

if __name__ == "__main__":
    # Test the metrics
    print("Testing segmentation metrics...")
    
    # Create dummy data
    batch_size = 2
    channels = 1
    depth, height, width = 64, 64, 64
    
    # Random predictions (logits) and targets
    pred = torch.randn(batch_size, channels, depth, height, width)
    target = torch.randint(0, 2, (batch_size, channels, depth, height, width)).float()
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"Target unique values: {torch.unique(target)}")
    
    # Test individual metrics
    dice = dice_coefficient(pred, target)
    iou = iou_coefficient(pred, target)
    
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"IoU Coefficient: {iou:.4f}")
    
    # Test batch metrics
    batch_metrics_result = batch_metrics(pred, target)
    print(f"Batch metrics: {batch_metrics_result}")
    
    # Test loss functions
    dice_loss = DiceLoss()
    combined_loss = CombinedLoss()
    
    loss_dice = dice_loss(pred, target)
    loss_combined = combined_loss(pred, target)
    
    print(f"Dice Loss: {loss_dice:.4f}")
    print(f"Combined Loss: {loss_combined:.4f}")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update_train(batch_metrics_result)
    is_best = tracker.update_val(batch_metrics_result)
    
    print(f"Is new best: {is_best}")
    print(f"Best metrics: {tracker.get_best_metrics()}")
    
    print("Metrics test successful!")
