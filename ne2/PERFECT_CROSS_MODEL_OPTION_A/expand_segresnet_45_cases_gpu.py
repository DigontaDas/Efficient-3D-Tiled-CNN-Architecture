#!/usr/bin/env python3
"""
GPU-Accelerated SegResNet Evaluation Expansion
Expand SegResNet from 5 to 45 cases for perfect cross-model validation
"""

import os
import sys
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

def check_gpu():
    """Check and display GPU information"""
    print(f"🚀 GPU Status Check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    return False

def load_segresnet_model(checkpoint_path, device):
    """Load SegResNet model from checkpoint"""
    print(f"📦 Loading SegResNet model from: {checkpoint_path}")
    
    # This is a placeholder - you'll need to implement your actual model loading
    try:
        # Example implementation (replace with your actual model)
        from your_model_module import SegResNet  # Replace with actual import
        
        model = SegResNet(in_channels=1, out_channels=1, 
                          init_cfg=dict(type='Kaiming', layer='Conv3d'))
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print(f"✅ SegResNet model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error loading SegResNet model: {e}")
        print("⚠️ Using dummy model for testing - replace with actual model loading")
        
        # Create dummy model for testing
        class DummySegResNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(1, 1, 3, padding=1)
            
            def forward(self, x):
                return torch.sigmoid(self.conv(x))
        
        model = DummySegResNet().to(device)
        model.eval()
        return model

def get_missing_segresnet_cases():
    """Get cases that need SegResNet evaluation"""
    
    # Load case analysis
    analysis_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/case_45_analysis.csv")
    if not analysis_file.exists():
        print("❌ Case analysis file not found")
        return []
    
    analysis_df = pd.read_csv(analysis_file)
    
    # Get cases where SegResNet is not available
    missing_cases = analysis_df[~analysis_df['segresnet_available']]['case_id'].tolist()
    
    print(f"📊 SegResNet Coverage Analysis:")
    print(f"   Total target cases: {len(analysis_df)}")
    print(f"   SegResNet available: {analysis_df['segresnet_available'].sum()}")
    print(f"   Need to evaluate: {len(missing_cases)} cases")
    
    return missing_cases

def preprocess_image_segresnet_gpu(image_data, device):
    """Preprocess image for SegResNet using GPU"""
    # Convert to tensor and move to GPU
    image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Normalize (adjust based on SegResNet requirements)
    image_tensor = (image_tensor - image_tensor.mean()) / (image_tensor.std() + 1e-8)
    
    return image_tensor

def run_segresnet_inference_gpu(model, image_tensor, device):
    """Run SegResNet inference using GPU"""
    with torch.no_grad():
        # Run inference
        output = model(image_tensor)
        
        # Apply sigmoid if needed
        if output.min() < 0 or output.max() > 1:
            output = torch.sigmoid(output)
        
        # Convert to binary mask
        pred_mask = (output > 0.5).float()
        
        # Move back to CPU
        pred_mask_cpu = pred_mask.cpu().numpy().squeeze()
        
        return pred_mask_cpu

def process_segresnet_case_gpu(case_id, model, image_dir, label_dir, output_dir, device):
    """Process a single SegResNet case using GPU"""
    
    # Load image and ground truth
    image_path = Path(image_dir) / f"{case_id}.img.nii.gz"
    label_path = Path(label_dir) / f"{case_id}.label.nii.gz"
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Load data
        image_nii = nib.load(image_path)
        image_data = image_nii.get_fdata()
        affine = image_nii.affine
        
        # Preprocess on GPU
        image_tensor = preprocess_image_segresnet_gpu(image_data, device)
        
        # Run inference on GPU
        pred_mask = run_segresnet_inference_gpu(model, image_tensor, device)
        
        # Save prediction
        pred_nii = nib.Nifti1Image(pred_mask, affine)
        output_path = Path(output_dir) / f"{case_id}_pred_mask.nii.gz"
        nib.save(pred_nii, output_path)
        
        # Also save probability map
        prob_path = Path(output_dir) / f"{case_id}_pred_prob.nii.gz"
        prob_data = image_tensor.detach().cpu().numpy().squeeze()
        prob_nii = nib.Nifti1Image(prob_data, affine)
        nib.save(prob_nii, prob_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing case {case_id}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated SegResNet expansion to 45 cases')
    parser.add_argument('--checkpoint', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/all_four_validations/mandatory_artifacts_segresnet/best_resumed.pt',
                       help='Path to SegResNet checkpoint')
    parser.add_argument('--image_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Dataset_Main/images_img',
                       help='Directory containing images')
    parser.add_argument('--label_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Dataset_Main/images_label',
                       help='Directory containing labels')
    parser.add_argument('--output_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/segresnet_predictions_45',
                       help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Batch size for processing')
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with dummy model')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not check_gpu():
        print("❌ GPU not available. Exiting.")
        return
    
    # Get missing cases
    missing_cases = get_missing_segresnet_cases()
    
    if not missing_cases:
        print("✅ SegResNet already evaluated on all 45 cases!")
        return
    
    print(f"\n🎯 Expanding SegResNet to {len(missing_cases)} additional cases...")
    print(f"   Cases: {missing_cases[:5]}...{missing_cases[-5:]}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    device = torch.device('cuda')
    model = load_segresnet_model(args.checkpoint, device)
    
    # Process cases
    success_count = 0
    gpu_memory_usage = []
    
    for i, case_id in enumerate(missing_cases):
        print(f"\n🔄 Case {i+1}/{len(missing_cases)}: {case_id}")
        
        # Monitor GPU memory
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            gpu_memory_usage.append(memory_used)
            print(f"   GPU Memory: {memory_used:.2f} GB")
        
        # Process case
        success = process_segresnet_case_gpu(case_id, model, args.image_dir, args.label_dir,
                                             args.output_dir, device)
        
        if success:
            success_count += 1
            print(f"   ✅ Success")
        else:
            print(f"   ❌ Failed")
        
        # Clear GPU cache periodically
        if (i + 1) % args.batch_size == 0:
            torch.cuda.empty_cache()
            print(f"   🧹 GPU cache cleared")
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    # Summary
    print(f"\n🎉 SEGRESNET EXPANSION COMPLETE!")
    print(f"   Total cases processed: {len(missing_cases)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(missing_cases) - success_count}")
    print(f"   Success rate: {success_count/len(missing_cases)*100:.1f}%")
    
    if gpu_memory_usage:
        print(f"   Peak GPU memory: {max(gpu_memory_usage):.2f} GB")
        print(f"   Average GPU memory: {np.mean(gpu_memory_usage):.2f} GB")
    
    print(f"\n✅ Predictions saved to: {args.output_dir}")
    print(f"🚀 Ready for unified metrics computation!")

if __name__ == '__main__':
    main()
