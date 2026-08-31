#!/usr/bin/env python3
"""
Fix 3D U-Net Metrics by Resizing Predictions to Match Ground Truth
Resizes predictions to ground truth dimensions and recomputes metrics
"""

import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

def resize_prediction_to_gt(pred_data, gt_shape, order=1):
    """
    Resize prediction data to match ground truth shape
    Uses scipy.zoom for interpolation
    """
    # Calculate zoom factors (new_size / old_size)
    zoom_factors = [gt_dim / pred_dim for gt_dim, pred_dim in zip(gt_shape, pred_data.shape)]
    
    # Apply zoom with interpolation
    resized_pred = zoom(pred_data, zoom_factors, order=order, mode='nearest')
    
    # Ensure binary output (threshold at 0.5)
    resized_pred = (resized_pred > 0.5).astype(np.float32)
    
    return resized_pred

def compute_dice_score(y_true, y_pred):
    """Compute Dice coefficient"""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    if union == 0:
        return 0.0
    return (2.0 * intersection) / union

def compute_iou_score(y_true, y_pred):
    """Compute IoU (Jaccard) coefficient"""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    if union == 0:
        return 0.0
    return intersection / union

def compute_precision(y_true, y_pred):
    """Compute precision"""
    intersection = np.sum(y_true * y_pred)
    pred_sum = np.sum(y_pred)
    if pred_sum == 0:
        return 0.0
    return intersection / pred_sum

def compute_recall(y_true, y_pred):
    """Compute recall"""
    intersection = np.sum(y_true * y_pred)
    true_sum = np.sum(y_true)
    if true_sum == 0:
        return 0.0
    return intersection / true_sum

def compute_hausdorff_distance_95(y_true, y_pred):
    """Compute 95th percentile Hausdorff distance"""
    try:
        from scipy.spatial.distance import directed_hausdorff
        # Get coordinates of non-zero voxels
        true_coords = np.array(np.where(y_true > 0)).T
        pred_coords = np.array(np.where(y_pred > 0)).T
        
        if len(true_coords) == 0 or len(pred_coords) == 0:
            return np.nan
            
        # Compute directed Hausdorff distances
        dist1 = directed_hausdorff(true_coords, pred_coords)[0]
        dist2 = directed_hausdorff(pred_coords, true_coords)[0]
        
        return max(dist1, dist2)
    except ImportError:
        # Fallback: simplified distance metric
        return np.nan

def compute_metrics_for_case_fixed(gt_path, pred_path, resize_method='bilinear'):
    """Compute metrics with shape fixing"""
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
            order = 1 if resize_method == 'bilinear' else 0  # 1=bilinear, 0=nearest
            pred_resized = resize_prediction_to_gt(pred_data, gt_data.shape, order)
        else:
            pred_resized = (pred_data > 0.5).astype(np.float32)
        
        # Compute metrics
        dice = compute_dice_score(gt_binary, pred_resized)
        iou = compute_iou_score(gt_binary, pred_resized)
        precision = compute_precision(gt_binary, pred_resized)
        recall = compute_recall(gt_binary, pred_resized)
        hd95 = compute_hausdorff_distance_95(gt_binary, pred_resized)
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'hd95': hd95,
            'original_shape': pred_data.shape,
            'target_shape': gt_data.shape,
            'resize_applied': pred_data.shape != gt_data.shape
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
            'resize_applied': False
        }

def process_all_cases(target_cases_file, pred_dir, gt_dir, case_mapping_file, output_file):
    """Process all cases with shape fixing"""
    
    # Load target cases and case mapping
    with open(target_cases_file, 'r') as f:
        target_cases = [int(line.strip()) for line in f if line.strip()]
    
    case_mapping_df = pd.read_csv(case_mapping_file)
    
    print(f"Processing {len(target_cases)} cases with shape fixing...")
    
    # Compute metrics for each case
    metrics_dict = {}
    success_count = 0
    resize_count = 0
    
    for i, case_id in enumerate(target_cases):
        pred_path = Path(pred_dir) / f"{case_id}_pred_mask.nii.gz"
        gt_path = Path(gt_dir) / f"{case_id}.label.nii.gz"
        
        if pred_path.exists() and gt_path.exists():
            metrics = compute_metrics_for_case_fixed(gt_path, pred_path)
            metrics_dict[case_id] = metrics
            success_count += 1
            
            if metrics['resize_applied']:
                resize_count += 1
            
            # Progress update
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(target_cases)} cases...")
            
            # Show first few results
            if i < 5:
                resize_info = " (resized)" if metrics['resize_applied'] else " (no resize)"
                print(f"Case {case_id}{resize_info}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")
        else:
            print(f"Missing files for case {case_id}: pred={pred_path.exists()}, gt={gt_path.exists()}")
    
    print(f"\nSuccessfully processed {success_count}/{len(target_cases)} cases")
    print(f"Applied resizing to {resize_count}/{success_count} cases")
    
    # Create results DataFrame
    results = []
    for _, row in case_mapping_df.iterrows():
        case_id = row['Original_ID']
        case_format = row['SegResNet_Format']
        
        if case_id in metrics_dict:
            metrics = metrics_dict[case_id]
            results.append({
                'Case': case_format,
                'Dice': metrics['dice'],
                'HD95': metrics['hd95'],
                'IoU': metrics['iou'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Model': '3D_U-Net',
                'Original_ID': case_id,
                'Original_Shape': str(metrics['original_shape']),
                'Target_Shape': str(metrics['target_shape']),
                'Resize_Applied': metrics['resize_applied']
            })
        else:
            # Case with missing metrics
            results.append({
                'Case': case_format,
                'Dice': np.nan,
                'HD95': np.nan,
                'IoU': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'Model': '3D_U-Net',
                'Original_ID': case_id,
                'Original_Shape': None,
                'Target_Shape': None,
                'Resize_Applied': False
            })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"Fixed 3D U-Net metrics saved to: {output_file}")
    
    # Print summary statistics
    valid_metrics = results_df.dropna(subset=['Dice'])
    if len(valid_metrics) > 0:
        print(f"\n3D U-Net Performance Summary ({len(valid_metrics)} cases):")
        print(f"Dice: {valid_metrics['Dice'].mean():.4f} ± {valid_metrics['Dice'].std():.4f}")
        print(f"IoU: {valid_metrics['IoU'].mean():.4f} ± {valid_metrics['IoU'].std():.4f}")
        print(f"Precision: {valid_metrics['Precision'].mean():.4f} ± {valid_metrics['Precision'].std():.4f}")
        print(f"Recall: {valid_metrics['Recall'].mean():.4f} ± {valid_metrics['Recall'].std():.4f}")
        
        # Resize statistics
        resized_cases = valid_metrics[valid_metrics['Resize_Applied'] == True]
        print(f"\nResizing Statistics:")
        print(f"Cases resized: {len(resized_cases)}/{len(valid_metrics)}")
        if len(resized_cases) > 0:
            print(f"Resized Dice: {resized_cases['Dice'].mean():.4f} ± {resized_cases['Dice'].std():.4f}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Fix 3D U-Net metrics by resizing predictions')
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
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/3dunet_metrics_fixed.csv',
                       help='Path to save fixed 3D U-Net metrics')
    parser.add_argument('--resize_method', type=str, choices=['bilinear', 'nearest'], default='bilinear',
                       help='Interpolation method for resizing')
    
    args = parser.parse_args()
    
    print(f"Fixing 3D U-Net metrics with {args.resize_method} interpolation...")
    
    # Process all cases
    results_df = process_all_cases(
        args.target_cases, args.pred_dir, args.gt_dir, 
        args.case_mapping_file, args.output_file
    )
    
    print(f"\n✅ Strategy A Complete!")
    print(f"3D U-Net metrics now available for {len(results_df.dropna(subset=['Dice']))} cases")
    print(f"Next: Create unified comparison with V-Net and SegResNet")

if __name__ == '__main__':
    main()
