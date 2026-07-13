#!/usr/bin/env python3
"""
Compute 3D U-Net Metrics for Target Cases
Fills the empty metrics in the comparison CSV with actual values
"""

import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics import f1_score, jaccard_score

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
    # For now, return a placeholder
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

def update_3dunet_results(metrics_df, case_mapping_df, metrics_dict):
    """Update the 3D U-Net results DataFrame with computed metrics"""
    
    for case_id, metrics in metrics_dict.items():
        # Find the corresponding row in the 3D U-Net results
        mask = case_mapping_df['Original_ID'] == case_id
        if mask.any():
            segresnet_format = case_mapping_df.loc[mask, 'SegResNet_Format'].iloc[0]
            
            # Update the metrics in the DataFrame
            idx = metrics_df[metrics_df['Case'] == segresnet_format].index
            if len(idx) > 0:
                metrics_df.loc[idx, 'Dice'] = metrics['dice']
                metrics_df.loc[idx, 'IoU'] = metrics['iou']
                metrics_df.loc[idx, 'Precision'] = metrics['precision']
                metrics_df.loc[idx, 'Recall'] = metrics['recall']
                metrics_df.loc[idx, 'HD95'] = metrics['hd95']
                
                print(f"Updated metrics for case {case_id} ({segresnet_format})")
            else:
                print(f"Case {case_id} ({segresnet_format}) not found in results DataFrame")
        else:
            print(f"Case {case_id} not found in case mapping")
    
    return metrics_df

def main():
    parser = argparse.ArgumentParser(description='Compute 3D U-Net metrics for target cases')
    parser.add_argument('--pred_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/3dunet_predictions_target',
                       help='Directory containing 3D U-Net predictions')
    parser.add_argument('--gt_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Dataset_Main/images_label',
                       help='Directory containing ground truth labels')
    parser.add_argument('--target_cases', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/target_cases.txt',
                       help='Text file with target case IDs')
    parser.add_argument('--results_file', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/_archive/20260204_2301/evaluation_results_3dunet/3dunet_results_segresnet_format.csv',
                       help='Path to 3D U-Net results CSV')
    parser.add_argument('--case_mapping_file', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/_archive/20260204_2301/evaluation_results_3dunet/case_mapping.csv',
                       help='Path to case mapping CSV')
    parser.add_argument('--output_file', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/_archive/20260204_2301/evaluation_results_3dunet/3dunet_results_segresnet_format_updated.csv',
                       help='Path to save updated results')
    
    args = parser.parse_args()
    
    # Load target cases
    with open(args.target_cases, 'r') as f:
        target_cases = [int(line.strip()) for line in f if line.strip()]
    
    print(f"Computing metrics for {len(target_cases)} target cases")
    
    # Load existing results and case mapping
    metrics_df = pd.read_csv(args.results_file)
    case_mapping_df = pd.read_csv(args.case_mapping_file)
    
    # Compute metrics for each case
    metrics_dict = {}
    
    for case_id in target_cases:
        pred_path = Path(args.pred_dir) / f"{case_id}_pred_mask.nii.gz"
        gt_path = Path(args.gt_dir) / f"{case_id}.label.nii.gz"
        
        if pred_path.exists() and gt_path.exists():
            metrics = compute_metrics_for_case(gt_path, pred_path)
            metrics_dict[case_id] = metrics
            print(f"Case {case_id}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")
        else:
            print(f"Missing files for case {case_id}: pred={pred_path.exists()}, gt={gt_path.exists()}")
    
    # Update the results DataFrame
    updated_df = update_3dunet_results(metrics_df, case_mapping_df, metrics_dict)
    
    # Save updated results
    updated_df.to_csv(args.output_file, index=False)
    print(f"Updated results saved to: {args.output_file}")
    
    # Also update the comparison CSV
    comparison_file = Path(args.results_file).parent / "comparison_3dunet_vs_segresnet.csv"
    if comparison_file.exists():
        print(f"Updating comparison file: {comparison_file}")
        # This would require additional logic to update the comparison table
        # For now, just note that it needs to be updated
        print("NOTE: You'll need to manually update the comparison CSV or run the comparison script")

if __name__ == '__main__':
    main()
