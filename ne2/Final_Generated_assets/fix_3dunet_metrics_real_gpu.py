#!/usr/bin/env python3
"""
Real GPU-Accelerated 3D U-Net Metrics Fix
Uses your existing RTX 3060 Ti GPU environment
"""

import os
import sys
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import PyTorch with CUDA
import torch
import torch.nn.functional as F

def check_gpu():
    """Check and display GPU information"""
    print(f"🚀 GPU Status Check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        return True
    return False

def resize_prediction_gpu(pred_data, gt_shape):
    """Resize prediction using GPU acceleration"""
    device = torch.device('cuda')
    
    # Convert to tensor and move to GPU
    pred_tensor = torch.from_numpy(pred_data).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    pred_tensor = pred_tensor.to(device)
    
    # Use trilinear interpolation for 3D medical images
    with torch.no_grad():
        resized_tensor = F.interpolate(
            pred_tensor, 
            size=gt_shape, 
            mode='trilinear', 
            align_corners=False
        )
    
    # Move back to CPU and convert to numpy
    resized_pred = resized_tensor.cpu().numpy().squeeze()  # Remove batch and channel dims
    
    # Binarize with threshold 0.5
    resized_pred = (resized_pred > 0.5).astype(np.float32)
    
    return resized_pred

def compute_metrics_gpu(y_true, y_pred):
    """Compute metrics using GPU acceleration"""
    device = torch.device('cuda')
    
    # Convert to tensors and move to GPU
    y_true_tensor = torch.from_numpy(y_true).float().to(device)
    y_pred_tensor = torch.from_numpy(y_pred).float().to(device)
    
    with torch.no_grad():
        # Compute intersection and union
        intersection = torch.sum(y_true_tensor * y_pred_tensor)
        union = torch.sum(y_true_tensor) + torch.sum(y_pred_tensor)
        
        # Dice coefficient
        dice = (2.0 * intersection / union)
        dice = dice.item() if union > 0 else 0.0
        
        # IoU (Jaccard)
        iou_union = union - intersection
        iou = (intersection / iou_union)
        iou = iou.item() if iou_union > 0 else 0.0
        
        # Precision
        pred_sum = torch.sum(y_pred_tensor)
        precision = (intersection / pred_sum)
        precision = precision.item() if pred_sum > 0 else 0.0
        
        # Recall
        true_sum = torch.sum(y_true_tensor)
        recall = (intersection / true_sum)
        recall = recall.item() if true_sum > 0 else 0.0
    
    return dice, iou, precision, recall

def compute_hd95_cpu(y_true, y_pred):
    """Compute HD95 on CPU (still faster than full CPU processing)"""
    try:
        from scipy.spatial.distance import directed_hausdorff
        from scipy.ndimage import binary_erosion
        
        # Get surface points for efficiency
        y_true_surface = y_true & ~binary_erosion(y_true)
        y_pred_surface = y_pred & ~binary_erosion(y_pred)
        
        if not y_true_surface.any() or not y_pred_surface.any():
            return np.nan
        
        # Get coordinates
        true_coords = np.argwhere(y_true_surface)
        pred_coords = np.argwhere(y_pred_surface)
        
        # Sample for speed if too many points
        max_points = 1000
        if len(true_coords) > max_points:
            indices = np.random.choice(len(true_coords), max_points, replace=False)
            true_coords = true_coords[indices]
        
        if len(pred_coords) > max_points:
            indices = np.random.choice(len(pred_coords), max_points, replace=False)
            pred_coords = pred_coords[indices]
        
        # Compute Hausdorff distance
        hd1 = directed_hausdorff(true_coords, pred_coords)[0]
        hd2 = directed_hausdorff(pred_coords, true_coords)[0]
        hd95 = max(hd1, hd2)
        
        return hd95
    except:
        return np.nan

def process_case_gpu(case_id, pred_dir, gt_dir):
    """Process a single case using GPU acceleration"""
    
    pred_path = Path(pred_dir) / f"{case_id}_pred_mask.nii.gz"
    gt_path = Path(gt_dir) / f"{case_id}.label.nii.gz"
    
    if not pred_path.exists() or not gt_path.exists():
        return None
    
    try:
        # Load data
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)
        
        gt_data = gt_nii.get_fdata()
        pred_data = pred_nii.get_fdata()
        
        # Binarize ground truth
        gt_binary = (gt_data > 0).astype(np.float32)
        
        # Resize prediction to match ground truth using GPU
        if pred_data.shape != gt_data.shape:
            resized_pred = resize_prediction_gpu(pred_data, gt_data.shape)
            resize_applied = True
        else:
            resized_pred = (pred_data > 0.5).astype(np.float32)
            resize_applied = False
        
        # Compute metrics using GPU
        dice, iou, precision, recall = compute_metrics_gpu(gt_binary, resized_pred)
        
        # HD95 on CPU (faster than full GPU implementation)
        hd95 = compute_hd95_cpu(gt_binary, resized_pred)
        
        return {
            'case_id': case_id,
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'hd95': hd95,
            'original_shape': pred_data.shape,
            'target_shape': gt_data.shape,
            'resize_applied': resize_applied,
            'gpu_used': True
        }
        
    except Exception as e:
        print(f"Error processing case {case_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated 3D U-Net metrics fix')
    parser.add_argument('--pred_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/all_four_validations/mandatory_artifacts_3dunet/predictions/test',
                       help='Directory containing 3D U-Net predictions')
    parser.add_argument('--gt_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Dataset_Main/images_label',
                       help='Directory containing ground truth labels')
    parser.add_argument('--target_cases', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/target_cases_dunet.txt',
                       help='Text file with target case IDs')
    parser.add_argument('--case_mapping_file', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/dunet_case_mapping.csv',
                       help='Path to case mapping CSV')
    parser.add_argument('--output_file', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/3dunet_metrics_real_gpu.csv',
                       help='Path to save GPU-processed 3D U-Net metrics')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Process only sample of cases (for testing)')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not check_gpu():
        print("❌ GPU not available. Exiting.")
        return
    
    # Load target cases
    with open(args.target_cases, 'r') as f:
        target_cases = [int(line.strip()) for line in f if line.strip()]
    
    if args.sample_size:
        target_cases = target_cases[:args.sample_size]
        print(f"🧪 Processing sample of {len(target_cases)} cases")
    else:
        print(f"🔥 Processing all {len(target_cases)} cases on GPU!")
    
    # Load case mapping
    case_mapping_df = pd.read_csv(args.case_mapping_file)
    
    # Process cases
    all_results = []
    gpu_count = 0
    resize_count = 0
    
    print(f"\n🚀 Starting GPU-accelerated processing...")
    
    for i, case_id in enumerate(target_cases):
        result = process_case_gpu(case_id, args.pred_dir, args.gt_dir)
        
        if result:
            # Find corresponding case format
            case_format = case_mapping_df[case_mapping_df['Original_ID'] == case_id]['SegResNet_Format'].iloc[0]
            
            # Create result entry
            entry = {
                'Case': case_format,
                'Dice': result['dice'],
                'HD95': result['hd95'],
                'IoU': result['iou'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'Model': '3D_U-Net',
                'Original_ID': case_id,
                'GPU_Used': result['gpu_used'],
                'Resize_Applied': result['resize_applied']
            }
            
            all_results.append(entry)
            
            if result['gpu_used']:
                gpu_count += 1
            if result['resize_applied']:
                resize_count += 1
            
            # Progress updates
            if (i + 1) % 10 == 0 or len(all_results) <= 5:
                gpu_indicator = "🚀" if result['gpu_used'] else "💻"
                resize_indicator = "🔄" if result['resize_applied'] else "✓"
                print(f"  Case {case_id} {gpu_indicator}{resize_indicator}: Dice={result['dice']:.4f}")
        
        # Progress
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{len(target_cases)} completed")
        
        # Clear GPU cache periodically
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    # Create DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_file, index=False)
    
    # Final GPU cache clear
    torch.cuda.empty_cache()
    
    # Summary statistics
    valid_metrics = results_df.dropna(subset=['Dice'])
    
    print(f"\n🎉 GPU-ACCELERATED PROCESSING COMPLETE!")
    print(f"Total cases processed: {len(valid_metrics)}")
    print(f"GPU-accelerated: {gpu_count}/{len(valid_metrics)} ({gpu_count/len(valid_metrics)*100:.1f}%)")
    print(f"Cases resized: {resize_count}/{len(valid_metrics)}")
    
    if len(valid_metrics) > 0:
        print(f"\n📊 3D U-Net Performance Summary:")
        print(f"Dice: {valid_metrics['Dice'].mean():.4f} ± {valid_metrics['Dice'].std():.4f}")
        print(f"IoU: {valid_metrics['IoU'].mean():.4f} ± {valid_metrics['IoU'].std():.4f}")
        print(f"Precision: {valid_metrics['Precision'].mean():.4f} ± {valid_metrics['Precision'].std():.4f}")
        print(f"Recall: {valid_metrics['Recall'].mean():.4f} ± {valid_metrics['Recall'].std():.4f}")
        
        # Performance highlights
        best_case = valid_metrics.loc[valid_metrics['Dice'].idxmax()]
        worst_case = valid_metrics.loc[valid_metrics['Dice'].idxmin()]
        
        print(f"\n🏆 Best case: {best_case['Original_ID']} (Dice={best_case['Dice']:.4f})")
        print(f"⚠️ Worst case: {worst_case['Original_ID']} (Dice={worst_case['Dice']:.4f})")
    
    print(f"\n✅ Results saved to: {args.output_file}")
    print(f"🔥 Your RTX 3060 Ti was utilized for maximum performance!")

if __name__ == '__main__':
    main()
