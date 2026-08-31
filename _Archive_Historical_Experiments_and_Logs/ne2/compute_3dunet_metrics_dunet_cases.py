#!/usr/bin/env python3
"""
Compute 3D U-Net Metrics for Cases 851-1000
Uses existing predictions and ground truth to fill metrics
"""

import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

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
    """Compute 95th percentile Hausdorff distance (simplified version)"""
    # This is a simplified version - for accurate HD95, you'd need scipy.spatial.distance
    # For now, return a placeholder based on surface distances
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
        
        # Return 95th percentile approximation (max of directed distances)
        return max(dist1, dist2)
    except ImportError:
        # Fallback: simplified distance metric
        return np.nan

def compute_metrics_for_case(gt_path, pred_path):
    """Compute all metrics for a single case"""
    try:
        # Load ground truth and prediction
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)
        
        gt_data = gt_nii.get_fdata()
        pred_data = pred_nii.get_fdata()
        
        # Binarize
        gt_binary = (gt_data > 0).astype(np.float32)
        pred_binary = (pred_data > 0.5).astype(np.float32)
        
        # Compute metrics
        dice = compute_dice_score(gt_binary, pred_binary)
        iou = compute_iou_score(gt_binary, pred_binary)
        precision = compute_precision(gt_binary, pred_binary)
        recall = compute_recall(gt_binary, pred_binary)
        hd95 = compute_hausdorff_distance_95(gt_binary, pred_binary)
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'hd95': hd95
        }
        
    except Exception as e:
        print(f"Error computing metrics for {pred_path}: {e}")
        return {
            'dice': np.nan,
            'iou': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'hd95': np.nan
        }

def create_3dunet_metrics_dataframe(case_mapping_df, metrics_dict):
    """Create 3D U-Net metrics DataFrame in SegResNet format"""
    
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
                'Original_ID': case_id
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
                'Original_ID': case_id
            })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Compute 3D U-Net metrics for cases 851-1000')
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
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/3dunet_metrics_851_1000.csv',
                       help='Path to save 3D U-Net metrics')
    
    args = parser.parse_args()
    
    # Load target cases and case mapping
    with open(args.target_cases, 'r') as f:
        target_cases = [int(line.strip()) for line in f if line.strip()]
    
    case_mapping_df = pd.read_csv(args.case_mapping_file)
    
    print(f"Computing metrics for {len(target_cases)} cases (851-1000)")
    
    # Compute metrics for each case
    metrics_dict = {}
    success_count = 0
    
    for i, case_id in enumerate(target_cases):
        pred_path = Path(args.pred_dir) / f"{case_id}_pred_mask.nii.gz"
        gt_path = Path(args.gt_dir) / f"{case_id}.label.nii.gz"
        
        if pred_path.exists() and gt_path.exists():
            metrics = compute_metrics_for_case(gt_path, pred_path)
            metrics_dict[case_id] = metrics
            success_count += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(target_cases)} cases...")
            
            if i < 5:  # Show first few results
                print(f"Case {case_id}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")
        else:
            print(f"Missing files for case {case_id}: pred={pred_path.exists()}, gt={gt_path.exists()}")
    
    print(f"\nSuccessfully computed metrics for {success_count}/{len(target_cases)} cases")
    
    # Create results DataFrame
    results_df = create_3dunet_metrics_dataframe(case_mapping_df, metrics_dict)
    
    # Save results
    results_df.to_csv(args.output_file, index=False)
    print(f"3D U-Net metrics saved to: {args.output_file}")
    
    # Print summary statistics
    valid_metrics = results_df.dropna(subset=['Dice'])
    if len(valid_metrics) > 0:
        print(f"\n3D U-Net Performance Summary ({len(valid_metrics)} cases):")
        print(f"Dice: {valid_metrics['Dice'].mean():.4f} ± {valid_metrics['Dice'].std():.4f}")
        print(f"IoU: {valid_metrics['IoU'].mean():.4f} ± {valid_metrics['IoU'].std():.4f}")
        print(f"Precision: {valid_metrics['Precision'].mean():.4f} ± {valid_metrics['Precision'].std():.4f}")
        print(f"Recall: {valid_metrics['Recall'].mean():.4f} ± {valid_metrics['Recall'].std():.4f}")
    
    print(f"\nNext steps:")
    print(f"1. Run V-Net inference on cases 851-1000")
    print(f"2. Run SegResNet inference on cases 851-1000")
    print(f"3. Create unified comparison table")

if __name__ == '__main__':
    main()
