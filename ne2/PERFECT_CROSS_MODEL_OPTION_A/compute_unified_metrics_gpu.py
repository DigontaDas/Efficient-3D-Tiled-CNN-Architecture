#!/usr/bin/env python3
"""
GPU-Accelerated Unified Metrics Computation
Perfect cross-model validation - Step 3
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def resize_prediction_gpu(pred_data, gt_shape, device):
    """Resize prediction using GPU acceleration"""
    # Convert to tensor and move to GPU
    pred_tensor = torch.from_numpy(pred_data).float().unsqueeze(0).unsqueeze(0)
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
    resized_pred = resized_tensor.cpu().numpy().squeeze()
    
    # Binarize with threshold 0.5
    resized_pred = (resized_pred > 0.5).astype(np.float32)
    
    return resized_pred

def compute_metrics_gpu(y_true, y_pred, device):
    """Compute metrics using GPU acceleration"""
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
    """Compute HD95 on CPU (faster than full GPU implementation)"""
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

def compute_model_metrics_gpu(model_name, case_id, pred_dir, gt_dir, device):
    """Compute metrics for a specific model and case using GPU"""
    
    pred_path = Path(pred_dir) / f"{case_id}_pred_mask.nii.gz"
    gt_path = Path(gt_dir) / f"{case_id}.label.nii.gz"
    
    if not pred_path.exists() or not gt_path.exists():
        return None
    
    try:
        # Load ground truth and prediction
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)
        
        gt_data = gt_nii.get_fdata()
        pred_data = pred_nii.get_fdata()
        
        # Binarize ground truth
        gt_binary = (gt_data > 0).astype(np.float32)
        
        # Resize prediction to match ground truth using GPU
        if pred_data.shape != gt_data.shape:
            resized_pred = resize_prediction_gpu(pred_data, gt_data.shape, device)
            resize_applied = True
        else:
            resized_pred = (pred_data > 0.5).astype(np.float32)
            resize_applied = False
        
        # Compute metrics using GPU
        dice, iou, precision, recall = compute_metrics_gpu(gt_binary, resized_pred, device)
        
        # HD95 on CPU (faster than full GPU implementation)
        hd95 = compute_hd95_cpu(gt_binary, resized_pred)
        
        return {
            'case_id': case_id,
            'model': model_name,
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'hd95': hd95,
            'resize_applied': resize_applied,
            'gpu_used': True
        }
        
    except Exception as e:
        print(f"Error computing metrics for {model_name} case {case_id}: {e}")
        return None

def load_existing_vnet_metrics():
    """Load existing V-Net metrics"""
    print("📊 Loading existing V-Net metrics...")
    
    metrics_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/_archive/20260204_2301/evaluation_results/metrics_summary.csv")
    if not metrics_file.exists():
        print("❌ V-Net metrics file not found")
        return []
    
    metrics_df = pd.read_csv(metrics_file)
    vnet_metrics = metrics_df[metrics_df['model'] == 'V-Net'].copy()
    
    # Standardize format
    vnet_results = []
    for _, row in vnet_metrics.iterrows():
        vnet_results.append({
            'case_id': row['case_id'],
            'model': 'V-Net',
            'dice': row['dice'],
            'iou': row['iou'],
            'precision': row['precision'],
            'recall': row['recall'],
            'hd95': row['hd95'],
            'resize_applied': False,
            'gpu_used': False
        })
    
    print(f"✅ Loaded {len(vnet_results)} V-Net metrics")
    return vnet_results

def load_segresnet_metrics_csv():
    """Load SegResNet metrics from CSV file (no predictions needed)"""
    print("📊 Loading SegResNet metrics from CSV...")
    
    metrics_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/segresnet_metrics_45.csv")
    if not metrics_file.exists():
        print("❌ SegResNet metrics CSV file not found")
        return []
    
    metrics_df = pd.read_csv(metrics_file)
    
    # Standardize format
    segresnet_results = []
    for _, row in metrics_df.iterrows():
        segresnet_results.append({
            'case_id': row['case_id'],
            'model': 'SegResNet',
            'dice': row['dice'],
            'iou': row['iou'],
            'precision': row['precision'],
            'recall': row['recall'],
            'hd95': row.get('hd95_mm', np.nan),  # Handle column name variation
            'resize_applied': False,
            'gpu_used': False,
            'source': 'csv_metrics'
        })
    
    print(f"✅ Loaded {len(segresnet_results)} SegResNet metrics from CSV")
    return segresnet_results

def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated unified metrics computation')
    parser.add_argument('--target_cases', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/target_45_cases.txt',
                       help='Text file with target case IDs')
    parser.add_argument('--gt_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Dataset_Main/images_label',
                       help='Directory containing ground truth labels')
    parser.add_argument('--dunet_pred_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/3dunet_predictions_45',
                       help='Directory containing 3D U-Net predictions')
    parser.add_argument('--segresnet_pred_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/segresnet_predictions_45',
                       help='Directory containing SegResNet predictions')
    parser.add_argument('--output_file', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/perfect_unified_metrics.csv',
                       help='Path to save unified metrics')
    parser.add_argument('--max_workers', type=int, default=2,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not check_gpu():
        print("❌ GPU not available. Exiting.")
        return
    
    # Load target cases
    with open(args.target_cases, 'r') as f:
        target_cases = [int(line.strip()) for line in f if line.strip()]
    
    print(f"\n🎯 Computing unified metrics for {len(target_cases)} cases...")
    
    # Initialize device
    device = torch.device('cuda')
    
    # Load existing V-Net metrics
    all_results = load_existing_vnet_metrics()
    
    # Compute 3D U-Net metrics
    print(f"\n🔄 Computing 3D U-Net metrics...")
    dunet_pred_dir = Path(args.dunet_pred_dir)
    if dunet_pred_dir.exists():
        for case_id in target_cases:
            result = compute_model_metrics_gpu('3D U-Net', case_id, dunet_pred_dir, args.gt_dir, device)
            if result:
                all_results.append(result)
        
        print(f"✅ 3D U-Net metrics computed")
    else:
        print(f"⚠️ 3D U-Net predictions not found at {dunet_pred_dir}")
    
    # Compute SegResNet metrics (from CSV - no predictions needed)
    print(f"\n🔄 Loading SegResNet metrics from CSV...")
    segresnet_results = load_segresnet_metrics_csv()
    if segresnet_results:
        all_results.extend(segresnet_results)
        print(f"✅ SegResNet metrics loaded from CSV (no predictions needed)")
    else:
        print(f"⚠️ SegResNet metrics CSV not found")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Create DataFrame and save
    results_df = pd.DataFrame(all_results)
    
    # Validate case coverage
    print(f"\n🔍 VALIDATING CASE COVERAGE:")
    model_case_counts = results_df.groupby('model')['case_id'].nunique()
    target_case_set = set(target_cases)
    
    missing_cases = {}
    for model in results_df['model'].unique():
        model_cases = set(results_df[results_df['model'] == model]['case_id'])
        missing = target_case_set - model_cases
        if missing:
            missing_cases[model] = sorted(list(missing))
    
    if missing_cases:
        print("❌ MISSING CASES DETECTED:")
        for model, cases in missing_cases.items():
            print(f"   {model}: {len(cases)} missing cases - {cases[:5]}{'...' if len(cases) > 5 else ''}")
        print("⚠️ STOPPING - All models must have the same 45 cases for fair comparison")
        return
    
    # Filter to only intersection cases (all 45)
    intersection_cases = []
    for case_id in target_cases:
        if all(case_id in results_df[results_df['model'] == model]['case_id'].values 
               for model in results_df['model'].unique()):
            intersection_cases.append(case_id)
    
    if len(intersection_cases) != len(target_cases):
        print(f"⚠️ Only {len(intersection_cases)}/{len(target_cases)} cases have all models")
        # Filter to intersection
        results_df = results_df[results_df['case_id'].isin(intersection_cases)]
    
    results_df.to_csv(args.output_file, index=False)
    
    # Summary statistics
    print(f"\n📊 UNIFIED METRICS SUMMARY:")
    model_counts = results_df['model'].value_counts()
    for model, count in model_counts.items():
        model_data = results_df[results_df['model'] == model]
        print(f"\n🎯 {model}:")
        print(f"   Cases: {count}")
        if len(model_data) > 0:
            print(f"   Dice: {model_data['dice'].mean():.4f} ± {model_data['dice'].std():.4f}")
            print(f"   IoU: {model_data['iou'].mean():.4f} ± {model_data['iou'].std():.4f}")
            print(f"   Precision: {model_data['precision'].mean():.4f} ± {model_data['precision'].std():.4f}")
            print(f"   Recall: {model_data['recall'].mean():.4f} ± {model_data['recall'].std():.4f}")
    
    # Rankings
    print(f"\n🏆 MODEL RANKINGS (by Dice Score):")
    model_performance = results_df.groupby('model')['dice'].mean().sort_values(ascending=False)
    for rank, (model, score) in enumerate(model_performance.items(), 1):
        print(f"   {rank}. {model}: {score:.4f}")
    
    # Final status
    print(f"\n🎉 FINAL STATUS:")
    print(f"   ✅ 3D U-Net: {model_counts.get('3D U-Net', 0)} cases (GPU inference)")
    print(f"   ✅ V-Net: {model_counts.get('V-Net', 0)} cases (existing metrics)")
    print(f"   ✅ SegResNet: {model_counts.get('SegResNet', 0)} cases (CSV metrics, no predictions)")
    print(f"   📊 Total cases per model: {len(intersection_cases)}")
    print(f"   🎯 Perfect cross-model validation: {'✅ COMPLETE' if len(intersection_cases) == len(target_cases) else '⚠️ INCOMPLETE'}")
    
    print(f"\n✅ Unified metrics saved to: {args.output_file}")
    print(f"🚀 Perfect cross-model analysis ready!")

if __name__ == '__main__':
    main()
