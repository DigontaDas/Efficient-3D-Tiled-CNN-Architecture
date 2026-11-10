#!/usr/bin/env python3
"""
Investigate 3D U-Net Shape Mismatches
Analyze the preprocessing differences between predictions and ground truth
"""

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def investigate_case_shapes(case_id, pred_dir, gt_dir):
    """Investigate shape differences for a specific case"""
    
    pred_path = Path(pred_dir) / f"{case_id}_pred_mask.nii.gz"
    gt_path = Path(gt_dir) / f"{case_id}.label.nii.gz"
    
    if not pred_path.exists() or not gt_path.exists():
        return None
    
    # Load files
    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)
    
    pred_data = pred_nii.get_fdata()
    gt_data = gt_nii.get_fdata()
    
    # Get shapes and properties
    pred_shape = pred_data.shape
    gt_shape = gt_data.shape
    pred_affine = pred_nii.affine
    gt_affine = gt_nii.affine
    
    # Calculate volume statistics
    pred_voxels = np.sum(pred_data > 0)
    gt_voxels = np.sum(gt_data > 0)
    
    return {
        'case_id': case_id,
        'pred_shape': pred_shape,
        'gt_shape': gt_shape,
        'shape_diff': tuple(p - g for p, g in zip(pred_shape, gt_shape)),
        'pred_voxels': pred_voxels,
        'gt_voxels': gt_voxels,
        'pred_affine_det': np.linalg.det(pred_affine),
        'gt_affine_det': np.linalg.det(gt_affine),
        'spacing_diff': np.abs(pred_affine[:3, :3] - gt_affine[:3, :3]).sum()
    }

def analyze_shape_patterns(investigations):
    """Analyze patterns in shape differences"""
    
    df = pd.DataFrame(investigations)
    
    print("=== SHAPE ANALYSIS SUMMARY ===")
    print(f"Total cases analyzed: {len(df)}")
    
    # Shape difference patterns
    shape_diffs = df['shape_diff'].value_counts()
    print(f"\nShape difference patterns:")
    for diff, count in shape_diffs.head(10).items():
        print(f"  {diff}: {count} cases")
    
    # Prediction shapes
    pred_shapes = df['pred_shape'].value_counts()
    print(f"\nPrediction shapes:")
    for shape, count in pred_shapes.head(5).items():
        print(f"  {shape}: {count} cases")
    
    # Ground truth shapes
    gt_shapes = df['gt_shape'].value_counts()
    print(f"\nGround truth shapes:")
    for shape, count in gt_shapes.head(5).items():
        print(f"  {shape}: {count} cases")
    
    # Volume statistics
    print(f"\nVolume statistics:")
    print(f"  Prediction voxels: {df['pred_voxels'].mean():.0f} ± {df['pred_voxels'].std():.0f}")
    print(f"  Ground truth voxels: {df['gt_voxels'].mean():.0f} ± {df['gt_voxels'].std():.0f}")
    
    return df

def suggest_solutions(df):
    """Suggest solutions based on shape analysis"""
    
    print("\n=== SOLUTION RECOMMENDATIONS ===")
    
    # Check if there's a consistent pattern
    shape_diffs = df['shape_diff'].value_counts()
    most_common_diff = shape_diffs.index[0] if len(shape_diffs) > 0 else None
    
    if most_common_diff:
        diff_count = shape_diffs.iloc[0]
        total_cases = len(df)
        consistency = diff_count / total_cases
        
        print(f"Most common shape difference: {most_common_diff} ({consistency:.1%} of cases)")
        
        if consistency > 0.8:
            print("RECOMMENDATION: Consistent preprocessing detected")
            print("SOLUTION: Apply uniform resizing/resampling to all predictions")
        elif consistency > 0.5:
            print("RECOMMENDATION: Semi-consistent preprocessing detected")
            print("SOLUTION: Case-by-case resizing with fallback strategies")
        else:
            print("RECOMMENDATION: Inconsistent preprocessing detected")
            print("SOLUTION: Investigate multiple preprocessing pipelines")
    
    # Check spacing differences
    high_spacing_diff = df[df['spacing_diff'] > 0.1]
    if len(high_spacing_diff) > 0:
        print(f"\nWARNING: {len(high_spacing_diff)} cases have significant spacing differences")
        print("SOLUTION: Resample to common spacing before resizing")
    
    # Check for extreme shape differences
    extreme_cases = df[df['shape_diff'].apply(lambda x: any(abs(d) > 200 for d in x))]
    if len(extreme_cases) > 0:
        print(f"\nWARNING: {len(extreme_cases)} cases have extreme shape differences")
        print("SOLUTION: Manual inspection required for these cases")
        print(f"Cases: {extreme_cases['case_id'].tolist()}")

def main():
    parser = argparse.ArgumentParser(description='Investigate 3D U-Net shape mismatches')
    parser.add_argument('--pred_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/all_four_validations/mandatory_artifacts_3dunet/predictions/test',
                       help='Directory containing 3D U-Net predictions')
    parser.add_argument('--gt_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Dataset_Main/images_label',
                       help='Directory containing ground truth labels')
    parser.add_argument('--sample_size', type=int, default=20,
                       help='Number of cases to investigate (for quick analysis)')
    parser.add_argument('--output_file', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/shape_investigation.csv',
                       help='Path to save investigation results')
    
    args = parser.parse_args()
    
    # Get sample cases
    pred_dir = Path(args.pred_dir)
    pred_files = list(pred_dir.glob("*_pred_mask.nii.gz"))
    sample_cases = sorted([int(f.name.split('_')[0]) for f in pred_files])[:args.sample_size]
    
    print(f"Investigating {len(sample_cases)} sample cases...")
    
    # Investigate each case
    investigations = []
    for case_id in sample_cases:
        result = investigate_case_shapes(case_id, args.pred_dir, args.gt_dir)
        if result:
            investigations.append(result)
            print(f"Case {case_id}: GT {result['gt_shape']} vs Pred {result['pred_shape']} (diff: {result['shape_diff']})")
    
    if not investigations:
        print("No valid cases found for investigation")
        return
    
    # Analyze patterns
    df = analyze_shape_patterns(investigations)
    
    # Save results
    df.to_csv(args.output_file, index=False)
    print(f"\nInvestigation results saved to: {args.output_file}")
    
    # Suggest solutions
    suggest_solutions(df)

if __name__ == '__main__':
    main()
