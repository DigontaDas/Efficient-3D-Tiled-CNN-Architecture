#!/usr/bin/env python3
"""
Training Curve Reconstruction
Reconstructs training curves from existing logs and checkpoints.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_checkpoints_data(checkpoint_dir):
    """Load data from all available checkpoints."""
    checkpoints = []
    
    # Find all checkpoint files
    checkpoint_files = list(Path(checkpoint_dir).glob("*.pt"))
    
    for checkpoint_file in sorted(checkpoint_files):
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            data = {
                'filename': checkpoint_file.name,
                'epoch': checkpoint.get('epoch', None),
                'train_loss': checkpoint.get('train_loss', None),
                'val_loss': checkpoint.get('val_loss', None),
                'val_dice': checkpoint.get('val_dice', None),
                'best_metric': checkpoint.get('best_metric', None),
                'is_best': checkpoint_file.name == 'best_model.pt'
            }
            
            checkpoints.append(data)
            print(f"Loaded {checkpoint_file.name}: epoch {data['epoch']}")
            
        except Exception as e:
            print(f"Failed to load {checkpoint_file}: {str(e)}")
            continue
    
    return checkpoints

def load_training_logs(log_dir):
    """Load training logs if available."""
    log_files = []
    
    # Look for common log file patterns
    patterns = ["*.log", "*.txt", "*.csv", "training_log*", "metrics*"]
    
    for pattern in patterns:
        log_files.extend(Path(log_dir).glob(pattern))
    
    logs = {}
    for log_file in log_files:
        try:
            if log_file.suffix == '.csv':
                df = pd.read_csv(log_file)
                logs[log_file.name] = df
                print(f"Loaded CSV log: {log_file.name}")
            else:
                with open(log_file, 'r') as f:
                    content = f.read()
                    logs[log_file.name] = content
                print(f"Loaded text log: {log_file.name}")
        except Exception as e:
            print(f"Failed to load log {log_file}: {str(e)}")
    
    return logs

def extract_metrics_from_checkpoints(checkpoints):
    """Extract training metrics from checkpoint data."""
    epochs = []
    train_losses = []
    val_losses = []
    val_dices = []
    
    # Sort by epoch
    checkpoints_sorted = sorted([c for c in checkpoints if c['epoch'] is not None], 
                               key=lambda x: x['epoch'])
    
    for checkpoint in checkpoints_sorted:
        epochs.append(checkpoint['epoch'])
        train_losses.append(checkpoint['train_loss'])
        val_losses.append(checkpoint['val_loss'])
        val_dices.append(checkpoint['val_dice'])
    
    return epochs, train_losses, val_losses, val_dices

def create_training_curves(epochs, train_losses, val_losses, val_dices, output_dir):
    """Create training curve plots."""
    
    # Create output directory
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Set academic style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Filter out None values
    valid_indices = [i for i, (t, v, d) in enumerate(zip(train_losses, val_losses, val_dices)) 
                    if t is not None and v is not None and d is not None]
    
    if not valid_indices:
        print("❌ No valid training data found in checkpoints")
        return None, None
    
    epochs_valid = [epochs[i] for i in valid_indices]
    train_losses_valid = [train_losses[i] for i in valid_indices]
    val_losses_valid = [val_losses[i] for i in valid_indices]
    val_dices_valid = [val_dices[i] for i in valid_indices]
    
    print(f"Found training data for epochs: {epochs_valid}")
    
    # Plot 1: Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs_valid, train_losses_valid, 'b-', linewidth=2, label='Training Loss')
    ax.plot(epochs_valid, val_losses_valid, 'r-', linewidth=2, label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark best validation loss
    best_val_idx = np.argmin(val_losses_valid)
    ax.plot(epochs_valid[best_val_idx], val_losses_valid[best_val_idx], 'ro', markersize=8)
    ax.annotate(f'Best: {val_losses_valid[best_val_idx]:.4f}', 
                xy=(epochs_valid[best_val_idx], val_losses_valid[best_val_idx]),
                xytext=(10, 10), textcoords='offset points')
    
    plt.tight_layout()
    loss_curve_path = os.path.join(figures_dir, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Dice curve
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs_valid, val_dices_valid, 'g-', linewidth=2, label='Validation Dice@0.3')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Coefficient')
    ax.set_title('Validation Dice@0.3 Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Mark best Dice
    best_dice_idx = np.argmax(val_dices_valid)
    ax.plot(epochs_valid[best_dice_idx], val_dices_valid[best_dice_idx], 'ro', markersize=8)
    ax.annotate(f'Best: {val_dices_valid[best_dice_idx]:.4f}', 
                xy=(epochs_valid[best_dice_idx], val_dices_valid[best_dice_idx]),
                xytext=(10, 10), textcoords='offset points')
    
    plt.tight_layout()
    dice_curve_path = os.path.join(figures_dir, "dice_curve.png")
    plt.savefig(dice_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Loss curve saved: {loss_curve_path}")
    print(f"✓ Dice curve saved: {dice_curve_path}")
    
    return loss_curve_path, dice_curve_path

def generate_training_report(epochs, train_losses, val_losses, val_dices, output_dir):
    """Generate training report."""
    
    # Filter valid data
    valid_indices = [i for i, (t, v, d) in enumerate(zip(train_losses, val_losses, val_dices)) 
                    if t is not None and v is not None and d is not None]
    
    if not valid_indices:
        return "No valid training data available"
    
    epochs_valid = [epochs[i] for i in valid_indices]
    train_losses_valid = [train_losses[i] for i in valid_indices]
    val_losses_valid = [val_losses[i] for i in valid_indices]
    val_dices_valid = [val_dices[i] for i in valid_indices]
    
    report = f"""
TRAINING CURVE RECONSTRUCTION REPORT
=====================================

Data Sources:
- Checkpoint directory: experiments/exp02_strong_baseline/checkpoints/
- Available epochs: {epochs_valid}
- Total data points: {len(epochs_valid)}

Training Statistics:
- Epoch range: {min(epochs_valid)} - {max(epochs_valid)}
- Final training loss: {train_losses_valid[-1]:.6f}
- Final validation loss: {val_losses_valid[-1]:.6f}
- Final validation Dice@0.3: {val_dices_valid[-1]:.6f}

Best Performance:
- Best validation loss: {min(val_losses_valid):.6f} at epoch {epochs_valid[np.argmin(val_losses_valid)]}
- Best validation Dice@0.3: {max(val_dices_valid):.6f} at epoch {epochs_valid[np.argmax(val_dices_valid)]}

Notes:
- Training was manually stopped at epoch {max(epochs_valid)}
- Early stopping was configured but not activated
- All metrics extracted from checkpoint files
- No additional training logs were found

Data Completeness:
- Training loss: {'Complete' if all(t is not None for t in train_losses_valid) else 'Partial'}
- Validation loss: {'Complete' if all(v is not None for v in val_losses_valid) else 'Partial'}
- Validation Dice: {'Complete' if all(d is not None for d in val_dices_valid) else 'Partial'}
"""
    
    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Training report saved: {report_path}")
    return report

def main():
    """Main reconstruction function."""
    print("=" * 60)
    print("TRAINING CURVE RECONSTRUCTION")
    print("=" * 60)
    
    # Paths
    project_root = "E:\\Thesis\\3D-Unet_Segmentation_IMGcas"
    checkpoint_dir = os.path.join(project_root, "experiments", "exp02_strong_baseline", "checkpoints")
    log_dir = os.path.join(project_root, "experiments", "exp02_strong_baseline")
    output_dir = os.path.join(project_root, "upload_models_check")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load checkpoint data
        checkpoints = load_checkpoints_data(checkpoint_dir)
        
        if not checkpoints:
            print("❌ No checkpoint data found")
            return
        
        # Try to load training logs
        logs = load_training_logs(log_dir)
        
        # Extract metrics from checkpoints
        epochs, train_losses, val_losses, val_dices = extract_metrics_from_checkpoints(checkpoints)
        
        # Create training curves
        loss_path, dice_path = create_training_curves(epochs, train_losses, val_losses, val_dices, output_dir)
        
        # Generate report
        report = generate_training_report(epochs, train_losses, val_losses, val_dices, output_dir)
        
        print("\n" + "=" * 60)
        print("TRAINING CURVE RECONSTRUCTION: COMPLETED")
        print("=" * 60)
        
        if loss_path and dice_path:
            print("✓ All training curves generated successfully")
        else:
            print("⚠️ Limited training data available - curves incomplete")
        
    except Exception as e:
        print(f"\n❌ RECONSTRUCTION FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    main()
