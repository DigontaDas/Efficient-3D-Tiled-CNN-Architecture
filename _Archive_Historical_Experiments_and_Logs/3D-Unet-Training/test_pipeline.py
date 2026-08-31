import os
import sys
import torch
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_3d_unet import get_unet3d_model
from metrics import CombinedLoss, batch_metrics
from dataset_3d import create_data_loaders, get_dataset_info

def test_pipeline():
    """Test the complete 3D U-Net training pipeline"""
    
    print("=" * 60)
    print("3D U-Net Pipeline Test")
    print("=" * 60)
    
    # Test GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Test model
    print("Testing 3D U-Net model...")
    model = get_unet3d_model(n_channels=1, n_classes=1, base_filters=32)
    model = model.to(device)
    
    # Test with dummy data
    test_input = torch.randn(1, 1, 64, 64, 64).to(device)
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✓ Model input shape: {test_input.shape}")
    print(f"✓ Model output shape: {output.shape}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Test metrics
    print("Testing metrics...")
    criterion = CombinedLoss()
    target = torch.randint(0, 2, (1, 1, 64, 64, 64)).float().to(device)
    
    loss = criterion(output, target)
    metrics = batch_metrics(output, target)
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Dice: {metrics['dice']:.4f}")
    print(f"✓ IoU: {metrics['iou']:.4f}")
    print()
    
    # Test dataset
    print("Testing dataset...")
    data_dir = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Dataset_Main"
    
    try:
        info = get_dataset_info(data_dir)
        print(f"✓ Dataset directory: {data_dir}")
        print(f"✓ Number of images: {info['num_images']}")
        print(f"✓ Number of labels: {info['num_labels']}")
        print(f"✓ Sample shape: {info['sample_shape']}")
        print()
        
        # Test data loader creation
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir, 
            batch_size=1, 
            num_workers=0,  # Set to 0 for testing
            target_size=(64, 64, 64)
        )
        
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")
        print()
        
        # Test loading one batch
        print("Testing data loading...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            print(f"✓ Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  Labels unique: {torch.unique(labels)}")
            
            # Test forward pass
            with torch.no_grad():
                pred = model(images)
                loss = criterion(pred, labels)
                metrics = batch_metrics(pred, labels)
                
                print(f"  Prediction shape: {pred.shape}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Dice: {metrics['dice']:.4f}")
            
            if batch_idx >= 1:  # Only test first 2 batches
                break
        
        print()
        print("✓ All tests passed successfully!")
        print("✓ 3D U-Net pipeline is ready for training!")
        
    except Exception as e:
        print(f"✗ Error in dataset test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        print("\n" + "=" * 60)
        print("PIPELINE TEST COMPLETE - READY FOR TRAINING")
        print("=" * 60)
        print("\nTo start training, run:")
        print("python train_3d_unet.py --epochs 100 --batch_size 2")
    else:
        print("\n" + "=" * 60)
        print("PIPELINE TEST FAILED - FIX ERRORS BEFORE TRAINING")
        print("=" * 60)
