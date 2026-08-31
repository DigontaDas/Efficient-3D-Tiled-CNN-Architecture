#!/usr/bin/env python3
"""
GPU-Accelerated 3D U-Net Metrics Fix
Uses GPU for faster resizing and metrics computation
"""

import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU libraries
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("✅ PyTorch available for GPU acceleration")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not available, falling back to CPU")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    print("❌ CuPy not available")

def resize_prediction_gpu_torch(pred_data, gt_shape):
    """Resize prediction using PyTorch GPU acceleration"""
    if not TORCH_AVAILABLE:
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensor and move to GPU
    pred_tensor = torch.from_numpy(pred_data).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    pred_tensor = pred_tensor.to(device)
    
    # Calculate output size
    output_size = gt_shape
    
    # Use bilinear interpolation for 3D
    with torch.no_grad():
        resized_tensor = F.interpolate(
            pred_tensor, 
            size=output_size, 
            mode='trilinear', 
            align_corners=False
        )
    
    # Move back to CPU and convert to numpy
    resized_pred = resized_tensor.cpu().numpy().squeeze()  # Remove batch and channel dims
    
    # Binarize
    resized_pred = (resized_pred > 0.5).astype(np.float32)
    
    return resized_pred

def resize_prediction_gpu_cupy(pred_data, gt_shape):
    """Resize prediction using CuPy GPU acceleration"""
    if not CUPY_AVAILABLE:
        return None
    
    # Move data to GPU
    pred_gpu = cp.asarray(pred_data)
    
    # Calculate zoom factors
    zoom_factors = [gt_dim / pred_dim for gt_dim, pred_dim in zip(gt_shape, pred_data.shape)]
    
    # Use CuPy for zoom operation (if available)
    try:
        from cupyx.scipy.ndimage import zoom as cupy_zoom
        resized_gpu = cupy_zoom(pred_gpu, zoom_factors, order=1, mode='nearest')
        resized_pred = cp.asnumpy(resized_gpu)
    except ImportError:
        print("CuPy zoom not available, falling back to CPU")
        return None
    
    # Binarize
    resized_pred = (resized_pred > 0.5).astype(np.float32)
    
    return resized_pred

def compute_metrics_gpu(y_true, y_pred):
    """Compute metrics using GPU acceleration"""
    
    # Try PyTorch GPU computation first
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.device('cuda')
        
        # Convert to tensors and move to GPU
        y_true_tensor = torch.from_numpy(y_true).float().to(device)
        y_pred_tensor = torch.from_numpy(y_pred).float().to(device)
        
        with torch.no_grad():
            # Compute metrics on GPU
            intersection = torch.sum(y_true_tensor * y_pred_tensor)
            union = torch.sum(y_true_tensor) + torch.sum(y_pred_tensor)
            
            dice = (2.0 * intersection / union).cpu().numpy() if union > 0 else 0.0
            
            # IoU
            iou_union = union - intersection
            iou = (intersection / iou_union).cpu().numpy() if iou_union > 0 else 0.0
            
            # Precision
            pred_sum = torch.sum(y_pred_tensor)
            precision = (intersection / pred_sum).cpu().numpy() if pred_sum > 0 else 0.0
            
            # Recall
            true_sum = torch.sum(y_true_tensor)
            recall = (intersection / true_sum).cpu().numpy() if true_sum > 0 else 0.0
        
        return float(dice), float(iou), float(precision), float(recall)
    
    # Fallback to CPU computation
    else:
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)
        
        dice = (2.0 * intersection / union) if union > 0 else 0.0
        iou = (intersection / (union - intersection)) if (union - intersection) > 0 else 0.0
        precision = (intersection / np.sum(y_pred)) if np.sum(y_pred) > 0 else 0.0
        recall = (intersection / np.sum(y_true)) if np.sum(y_true) > 0 else 0.0
        
        return dice, iou, precision, recall

def compute_metrics_for_case_gpu(gt_path, pred_path):
    """Compute metrics with GPU acceleration"""
    try:
        # Load ground truth and prediction
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)
        
        gt_data = gt_nii.get_fdata()
        pred_data = pred_nii.get_fdata()
        
        # Binarize ground truth
        gt_binary = (gt_data > 0).astype(np.float32)
        
        # Resize prediction to match ground truth
        if pred_data.shape != gt_data.shape:
            print(f"  Resizing: {pred_data.shape} -> {gt_data.shape}")
            
            # Try GPU resizing methods
            resized_pred = None
            
            # Method 1: PyTorch GPU
            if TORCH_AVAILABLE and torch.cuda.is_available():
                resized_pred = resize_prediction_gpu_torch(pred_data, gt_data.shape)
                if resized_pred is not None:
                    print(f"  ✅ Used PyTorch GPU resizing")
            
            # Method 2: CuPy GPU
            if resized_pred is None and CUPY_AVAILABLE:
                resized_pred = resize_prediction_gpu_cupy(pred_data, gt_data.shape)
                if resized_pred is not None:
                    print(f"  ✅ Used CuPy GPU resizing")
            
            # Method 3: CPU fallback
            if resized_pred is None:
                from scipy.ndimage import zoom
                zoom_factors = [gt_dim / pred_dim for gt_dim, pred_dim in zip(gt_data.shape, pred_data.shape)]
                resized_pred = zoom(pred_data, zoom_factors, order=1, mode='nearest')
                resized_pred = (resized_pred > 0.5).astype(np.float32)
                print(f"  ⚠️ Used CPU resizing (GPU not available)")
        else:
            resized_pred = (pred_data > 0.5).astype(np.float32)
            print(f"  No resize needed")
        
        # Compute metrics with GPU
        dice, iou, precision, recall = compute_metrics_gpu(gt_binary, resized_pred)
        
        # HD95 still needs scipy (CPU-based for now)
        try:
            from scipy.spatial.distance import directed_hausdorff
            true_coords = np.array(np.where(gt_binary > 0)).T
            pred_coords = np.array(np.where(resized_pred > 0)).T
            
            if len(true_coords) > 0 and len(pred_coords) > 0:
                hd95 = max(directed_hausdorff(true_coords, pred_coords)[0],
                          directed_hausdorff(pred_coords, true_coords)[0])
            else:
                hd95 = np.nan
        except:
            hd95 = np.nan
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'hd95': hd95,
            'original_shape': pred_data.shape,
            'target_shape': gt_data.shape,
            'resize_applied': pred_data.shape != gt_data.shape,
            'gpu_used': (TORCH_AVAILABLE and torch.cuda.is_available()) or CUPY_AVAILABLE
        }
        
    except Exception as e:
        print(f"Error computing metrics for {pred_path}: {e}")
        return {
            'dice': np.nan,
            'iou': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'hd95': np.nan,
            'original_shape': None,
            'target_shape': None,
            'resize_applied': False,
            'gpu_used': False
        }

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
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/3dunet_metrics_gpu.csv',
                       help='Path to save GPU-processed 3D U-Net metrics')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif CUPY_AVAILABLE:
        print(f"✅ CuPy GPU available")
    else:
        print("⚠️ No GPU acceleration available, using CPU")
    
    # Load target cases and case mapping
    with open(args.target_cases, 'r') as f:
        target_cases = [int(line.strip()) for line in f if line.strip()]
    
    case_mapping_df = pd.read_csv(args.case_mapping_file)
    
    print(f"\nProcessing {len(target_cases)} cases with GPU acceleration...")
    
    # Process cases in batches
    all_results = []
    gpu_used_count = 0
    
    for i in range(0, len(target_cases), args.batch_size):
        batch_cases = target_cases[i:i+args.batch_size]
        print(f"\nBatch {i//args.batch_size + 1}: Cases {batch_cases[0]}-{batch_cases[-1]}")
        
        batch_results = []
        for case_id in batch_cases:
            pred_path = Path(args.pred_dir) / f"{case_id}_pred_mask.nii.gz"
            gt_path = Path(args.gt_dir) / f"{case_id}.label.nii.gz"
            
            if pred_path.exists() and gt_path.exists():
                metrics = compute_metrics_for_case_gpu(gt_path, pred_path)
                
                if metrics['gpu_used']:
                    gpu_used_count += 1
                
                # Create result entry
                case_format = case_mapping_df[case_mapping_df['Original_ID'] == case_id]['SegResNet_Format'].iloc[0]
                
                result = {
                    'Case': case_format,
                    'Dice': metrics['dice'],
                    'HD95': metrics['hd95'],
                    'IoU': metrics['iou'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'Model': '3D_U-Net',
                    'Original_ID': case_id,
                    'GPU_Used': metrics['gpu_used']
                }
                
                batch_results.append(result)
                
                if len(all_results) < 5:  # Show first few results
                    gpu_indicator = "🚀" if metrics['gpu_used'] else "💻"
                    print(f"  Case {case_id} {gpu_indicator}: Dice={metrics['dice']:.4f}")
        
        all_results.extend(batch_results)
        
        # Clear GPU cache if using PyTorch
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_file, index=False)
    
    # Summary
    valid_metrics = results_df.dropna(subset=['Dice'])
    print(f"\n🎉 GPU-ACCELERATED PROCESSING COMPLETE!")
    print(f"Total cases processed: {len(valid_metrics)}")
    print(f"GPU-accelerated cases: {gpu_used_count}")
    print(f"GPU usage rate: {gpu_used_count/len(valid_metrics)*100:.1f}%")
    
    if len(valid_metrics) > 0:
        print(f"\n3D U-Net Performance Summary:")
        print(f"Dice: {valid_metrics['Dice'].mean():.4f} ± {valid_metrics['Dice'].std():.4f}")
        print(f"IoU: {valid_metrics['IoU'].mean():.4f} ± {valid_metrics['IoU'].std():.4f}")
        print(f"Precision: {valid_metrics['Precision'].mean():.4f} ± {valid_metrics['Precision'].std():.4f}")
        print(f"Recall: {valid_metrics['Recall'].mean():.4f} ± {valid_metrics['Recall'].std():.4f}")
    
    print(f"\n✅ Results saved to: {args.output_file}")

if __name__ == '__main__':
    main()
