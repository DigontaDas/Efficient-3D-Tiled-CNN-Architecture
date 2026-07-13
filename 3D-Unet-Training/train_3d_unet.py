import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_3d_unet import get_unet3d_model
from dataset_3d import create_data_loaders, get_dataset_info
from metrics import CombinedLoss, batch_metrics, MetricsTracker

class Trainer3D:
    """
    3D U-Net Trainer with checkpointing and resume capability
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory
        self.results_dir = config['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.model = get_unet3d_model(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            base_filters=config['base_filters'],
            bilinear=config['bilinear']
        )
        self.model = self.model.to(self.device)
        
        # Initialize loss, optimizer, and scheduler
        self.criterion = CombinedLoss(dice_weight=config['dice_weight'], bce_weight=config['bce_weight'])
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=config['lr_factor'], 
            patience=config['lr_patience'],
            verbose=True
        )
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.training_history = []
        
        # Check for existing checkpoint
        self.checkpoint_path = os.path.join(self.results_dir, 'checkpoints', 'latest_checkpoint.pth')
        self.best_model_path = os.path.join(self.results_dir, 'checkpoints', 'best_model.pth')
        
        logging.info(f"Trainer initialized with device: {self.device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.results_dir, 'logs', 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            logging.info(f"New best model saved with Dice: {self.best_val_dice:.4f}")
    
    def load_checkpoint(self):
        """Load model checkpoint"""
        if os.path.exists(self.checkpoint_path):
            logging.info(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_dice = checkpoint['best_val_dice']
            self.training_history = checkpoint.get('training_history', [])
            
            logging.info(f"Resuming training from epoch {self.current_epoch}")
            logging.info(f"Best validation Dice: {self.best_val_dice:.4f}")
            return True
        else:
            logging.info("No checkpoint found. Starting fresh training.")
            return False
    
    def train_epoch(self, train_loader, total_epochs):
        """Train for one epoch with AMP (Automatic Mixed Precision)."""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = []
        scaler = getattr(self, '_scaler', None)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            metrics = batch_metrics(outputs.detach(), labels)
            metrics['loss'] = loss.item()
            epoch_loss += loss.item()
            epoch_metrics.append(metrics)

            if batch_idx % self.config['log_interval'] == 0:
                progress_pct = (batch_idx + 1) / len(train_loader) * 100
                msg = (f"Epoch {self.current_epoch}/{total_epochs-1} [{progress_pct:.1f}%] "
                       f"Batch {batch_idx+1}/{len(train_loader)} - "
                       f"Loss: {loss.item():.4f}, Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
                logging.info(msg)
                print(msg)

        avg_loss = epoch_loss / len(train_loader)
        avg_metrics = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]}
        return avg_loss, avg_metrics
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_metrics = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate metrics
                metrics = batch_metrics(outputs, labels)
                metrics['loss'] = loss.item()
                
                epoch_loss += loss.item()
                epoch_metrics.append(metrics)
        
        # Calculate epoch averages
        avg_loss = epoch_loss / len(val_loader)
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
        return avg_loss, avg_metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop with AMP, resume, and per-5-epoch checkpointing."""
        # Initialise AMP scaler (no-op on CPU)
        if self.device.type == 'cuda':
            self._scaler = torch.cuda.amp.GradScaler()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            self._scaler = None

        self.load_checkpoint()

        start_epoch = self.current_epoch
        end_epoch   = num_epochs           # absolute target epoch count

        if start_epoch >= end_epoch:
            logging.info(f"Already trained to epoch {start_epoch}. Nothing to do.")
            return

        logging.info(f"Training from epoch {start_epoch} to {end_epoch - 1} (total {end_epoch} epochs).")

        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            t0 = time.time()

            train_loss, train_metrics = self.train_epoch(train_loader, end_epoch)
            val_loss,   val_metrics   = self.validate_epoch(val_loader)

            self.scheduler.step(val_metrics['dice'])
            self.metrics_tracker.update_train(train_metrics)
            self.metrics_tracker.update_val(val_metrics)

            is_best = val_metrics['dice'] >= self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']

            epoch_data = {
                'epoch':         epoch,
                'train_loss':    train_loss,
                'val_loss':      val_loss,
                'train_dice':    train_metrics['dice'],
                'val_dice':      val_metrics['dice'],
                'train_iou':     train_metrics['iou'],
                'val_iou':       val_metrics['iou'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time':    time.time() - t0,
            }
            self.training_history.append(epoch_data)

            elapsed = time.time() - t0
            print(f"\n" + "="*60)
            print(f"EPOCH {epoch}/{end_epoch-1} | Time: {elapsed:.1f}s")
            print(f"Train  Loss: {train_loss:.4f}  Dice: {train_metrics['dice']:.4f}  IoU: {train_metrics['iou']:.4f}")
            print(f"Val    Loss: {val_loss:.4f}  Dice: {val_metrics['dice']:.4f}  IoU: {val_metrics['iou']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}   Best Val Dice: {self.best_val_dice:.4f}")
            print("="*60 + "\n")

            logging.info(f"Epoch {epoch} | train_dice={train_metrics['dice']:.4f} "
                         f"val_dice={val_metrics['dice']:.4f} time={elapsed:.1f}s")

            # Always save latest + best
            self.save_checkpoint(is_best)

            # Per-5-epoch milestone checkpoint (like RASNet)
            if (epoch + 1) % 5 == 0:
                milestone_path = os.path.join(
                    self.results_dir, 'checkpoints', f'epoch_{epoch+1:04d}.pth'
                )
                torch.save({
                    'epoch':              epoch,
                    'model_state_dict':   self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_dice':      self.best_val_dice,
                }, milestone_path)
                logging.info(f"Milestone checkpoint saved: {milestone_path}")

            self.save_metrics_csv()

        logging.info("Training completed!")
        self.save_summary()
        self.plot_training_curves()
    
    def save_metrics_csv(self):
        """Save training metrics to CSV file"""
        df = pd.DataFrame(self.training_history)
        csv_path = os.path.join(self.results_dir, 'logs', 'training_log.csv')
        df.to_csv(csv_path, index=False)
    
    def save_summary(self):
        """Save training summary"""
        best_metrics = self.metrics_tracker.get_best_metrics()
        
        summary = f"""
3D U-Net Training Summary
========================
Dataset: {self.config['data_dir']}
Model: 3D U-Net with {self.config['base_filters']} base filters
Device: {self.device}

Training Configuration:
- Epochs trained: {len(self.training_history)}
- Initial learning rate: {self.config['learning_rate']}
- Batch size: {self.config['batch_size']}
- Loss function: Combined Dice + BCE
- Optimizer: AdamW

Best Results:
- Best validation Dice: {best_metrics['best_dice']:.4f}
- Best validation IoU: {best_metrics['best_iou']:.4f}
- Best epoch: {best_metrics['best_epoch']}

Training was {'resumed' if self.current_epoch > 0 else 'started fresh'}.

Files Generated:
- Checkpoints: {os.path.join(self.results_dir, 'checkpoints')}
- Training log: {os.path.join(self.results_dir, 'logs', 'training_log.csv')}
- Plots: {os.path.join(self.results_dir, 'plots')}
"""
        
        summary_path = os.path.join(self.results_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logging.info(f"Training summary saved to {summary_path}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        if not self.training_history:
            return
        
        df = pd.DataFrame(self.training_history)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice curves
        axes[0, 1].plot(df['epoch'], df['train_dice'], label='Train Dice', color='blue')
        axes[0, 1].plot(df['epoch'], df['val_dice'], label='Val Dice', color='red')
        axes[0, 1].set_title('Training and Validation Dice')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU curves
        axes[1, 0].plot(df['epoch'], df['train_iou'], label='Train IoU', color='blue')
        axes[1, 0].plot(df['epoch'], df['val_iou'], label='Val IoU', color='red')
        axes[1, 0].set_title('Training and Validation IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curve
        axes[1, 1].plot(df['epoch'], df['learning_rate'], color='green')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plots
        plots_dir = os.path.join(self.results_dir, 'plots')
        plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        
        # Save individual plots
        plt.figure(figsize=(8, 6))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.plot(df['epoch'], df['train_dice'], label='Train Dice', color='blue')
        plt.plot(df['epoch'], df['val_dice'], label='Val Dice', color='red')
        plt.title('Dice Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'dice_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.plot(df['epoch'], df['train_iou'], label='Train IoU', color='blue')
        plt.plot(df['epoch'], df['val_iou'], label='Val IoU', color='red')
        plt.title('IoU Curve')
        plt.xlabel('Epoch')
        plt.ylabel('IoU Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'iou_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

def get_default_config():
    """Get default training configuration."""
    return {
        'results_dir':   r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\3D-Unet-Training\results",
        'target_size':   (128, 128, 128),
        'batch_size':    8,
        'num_workers':   4,
        # Model
        'n_channels':    1,
        'n_classes':     1,
        'base_filters':  16,
        'bilinear':      True,
        # Training — 70 epochs to match RASNet protocol
        'num_epochs':    70,
        'learning_rate': 1e-4,
        'weight_decay':  1e-5,
        'dice_weight':   0.5,
        'bce_weight':    0.5,
        # Scheduler
        'lr_factor':     0.5,
        'lr_patience':   10,
        # Logging
        'log_interval':  10,
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='3D U-Net Training for Medical Image Segmentation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--base_filters', type=int, default=32, help='Base filters for U-Net')
    parser.add_argument('--data_dir', type=str, default=r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Dataset_Main", help='Dataset directory')
    parser.add_argument('--results_dir', type=str, default=r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\3D-Unet-Training\results", help='Results directory')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_default_config()
    config.update({
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'base_filters': args.base_filters,
        'data_dir': args.data_dir,
        'results_dir': args.results_dir,
    })
    
    # Print configuration
    print("=" * 60)
    print("3D U-Net Training Configuration")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key:20}: {value}")
    print("=" * 60)
    
    # Check CUDA availability — no blocking prompt, safe for background runs
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will run on CPU (slow).")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create data loaders (uses splits_final.json via updated dataset_3d)
    print("\nCreating data loaders from splits_final.json...")
    try:
        from dataset_3d import create_data_loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            target_size=config['target_size'],
        )
        print(f"Data loaders ready — train: {len(train_loader)} batches, "
              f"val: {len(val_loader)} batches, test: {len(test_loader)} batches.")
    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        return
    
    # Create trainer and start training
    print("\nInitializing trainer...")
    trainer = Trainer3D(config)
    
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, config['num_epochs'])
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {config['results_dir']}")

if __name__ == "__main__":
    main()
