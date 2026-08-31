#!/usr/bin/env python3
"""
Fast 3D U-Net Metrics Fix (CPU-Optimized)
Optimized for speed with parallel processing and efficient operations
"""

import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

def resize_prediction_fast(pred_data, gt_shape):
    """
    Fast resizing using numpy operations
    Simplified but much faster than scipy.zoom
    """
    # For now, use simple nearest neighbor scaling
    # This is much faster though less accurate
    
    old_shape = np.array(pred_data.shape)
    new_shape = np.array(gt_shape)
    
    # Calculate scaling factors
    scale_factors = new_shape / old_shape
    
    # Create coordinate grids for the new shape
    coords = np.indices(new_shape)
    
    # Map new coordinates to old coordinates
    old_coords = coords / scale_factors[:, np.newaxis, np.newaxis, np.newaxis]
    
    # Round to nearest integer for nearest neighbor
    old_coords = np.round(old_coords).astype(int)
    
    # Clip to valid bounds
    old_coords = np.clip(old_coords, 0, old_shape - 1)
    
    # Sample from original array
    resized_pred = pred_data[old_coords[0], old_coords[1], old_coords[2]]
    
    # Binarize
    resized_pred = (resized_pred > 0.5).astype(np.float32)
    
    return resized_pred

def compute_metrics_fast(y_true, y_pred):
    """Fast metrics computation using numpy operations"""
    
    # Convert to boolean for faster operations
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    
    # Compute intersection and union
    intersection = np.logical_and(y_true_bool, y_pred_bool).sum()
    union = np.logical_or(y_true_bool, y_pred_bool).sum()
    
    # Dice
    dice = (2.0 * intersection / union) if union > 0 else 0.0
    
    # IoU
    iou = (intersection / union) if union > 0 else 0.0
    
    # Precision and Recall
    pred_sum = y_pred_bool.sum()
    true_sum = y_true_bool.sum()
    
    precision = (intersection / pred_sum) if pred_sum > 0 else 0.0
    recall = (intersection / true_sum) if true_sum > 0 else 0.0
    
    return dice, iou, precision, recall

def compute_hd95_fast(y_true, y_pred):
    """Fast HD95 approximation"""
    try:
        # Get surface points (much faster than all points)
        from scipy.ndimage import binary_erosion
        
        # Get surface voxels
        y_true_surface = y_true & ~binary_erosion(y_true)
        y_pred_surface = y_pred & ~binary_erosion(y_pred)
        
        if not y_true_surface.any() or not y_pred_surface.any():
            return np.nan
        
        # Get coordinates
        true_coords = np.argwhere(y_true_surface)
        pred_coords = np.argwhere(y_pred_surface)
        
        # Sample subset for speed (if too many points)
        max_points = 1000
        if len(true_coords) > max_points:
            indices = np.random.choice(len(true_coords), max_points, replace=False)
            true_coords = true_coords[indices]
        
        if len(pred_coords) > max_points:
            indices = np.random.choice(len(pred_coords), max_points, replace=False)
            pred_coords = pred_coords[indices]
        
        # Compute distances
        from scipy.spatial.distance import cdist
        distances = cdist(true_coords, pred_coords)
        
        # 95th percentile approximation
        hd95 = np.percentile(distances, 95)
        
        return hd95
    except:
        return np.nan

def process_single_case(case_id, pred_dir, gt_dir):
    """Process a single case - designed for parallel execution"""
    
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
        
        # Resize if needed
        if pred_data.shape != gt_data.shape:
            resized_pred = resize_prediction_fast(pred_data, gt_data.shape)
            resize_applied = True
        else:
            resized_pred = (pred_data > 0.5).astype(np.float32)
            resize_applied = False
        
        # Compute metrics
        dice, iou, precision, recall = compute_metrics_fast(gt_binary, resized_pred)
        hd95 = compute_hd95_fast(gt_binary, resized_pred)
        
        return {
            'case_id': case_id,
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'hd95': hd95,
            'original_shape': pred_data.shape,
            'target_shape': gt_data.shape,
            'resize_applied': resize_applied
        }
        
    except Exception as e:
        print(f"Error processing case {case_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Fast 3D U-Net metrics fix')
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
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/3dunet_metrics_fast.csv',
                       help='Path to save fast-processed 3D U-Net metrics')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Process only sample of cases (for testing)')
    
    args = parser.parse_args()
    
    # Load target cases
    with open(args.target_cases, 'r') as f:
        target_cases = [int(line.strip()) for line in f if line.strip()]
    
    if args.sample_size:
        target_cases = target_cases[:args.sample_size]
        print(f"Processing sample of {len(target_cases)} cases")
    else:
        print(f"Processing all {len(target_cases)} cases")
    
    # Load case mapping
    case_mapping_df = pd.read_csv(args.case_mapping_file)
    
    print(f"Using {args.max_workers} parallel workers...")
    
    # Process cases in parallel
    all_results = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_case = {
            executor.submit(process_single_case, case_id, args.pred_dir, args.gt_dir): case_id
            for case_id in target_cases
        }
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_case)):
            case_id = future_to_case[future]
            result = future.result()
            
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
                    'Resize_Applied': result['resize_applied']
                }
                
                all_results.append(entry)
                
                # Progress update
                if len(all_results) % 10 == 0 or len(all_results) <= 5:
                    resize_indicator = "🔄" if result['resize_applied'] else "✓"
                    print(f"  Case {case_id} {resize_indicator}: Dice={result['dice']:.4f}")
            
            # Overall progress
            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{len(target_cases)} cases completed")
    
    # Create DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_file, index=False)
    
    # Summary statistics
    valid_metrics = results_df.dropna(subset=['Dice'])
    resize_count = valid_metrics['Resize_Applied'].sum()
    
    print(f"\n🚀 FAST PROCESSING COMPLETE!")
    print(f"Total cases processed: {len(valid_metrics)}")
    print(f"Cases resized: {resize_count}/{len(valid_metrics)}")
    print(f"Processing time: Much faster than original!")
    
    if len(valid_metrics) > 0:
        print(f"\n3D U-Net Performance Summary:")
        print(f"Dice: {valid_metrics['Dice'].mean():.4f} ± {valid_metrics['Dice'].std():.4f}")
        print(f"IoU: {valid_metrics['IoU'].mean():.4f} ± {valid_metrics['IoU'].std():.4f}")
        print(f"Precision: {valid_metrics['Precision'].mean():.4f} ± {valid_metrics['Precision'].std():.4f}")
        print(f"Recall: {valid_metrics['Recall'].mean():.4f} ± {valid_metrics['Recall'].std():.4f}")
        
        # Show best and worst cases
        best_case = valid_metrics.loc[valid_metrics['Dice'].idxmax()]
        worst_case = valid_metrics.loc[valid_metrics['Dice'].idxmin()]
        
        print(f"\nBest case: {best_case['Original_ID']} (Dice={best_case['Dice']:.4f})")
        print(f"Worst case: {worst_case['Original_ID']} (Dice={worst_case['Dice']:.4f})")
    
    print(f"\n✅ Results saved to: {args.output_file}")
    print(f"Next: Create unified comparison with V-Net and SegResNet")

if __name__ == '__main__':
    main()
