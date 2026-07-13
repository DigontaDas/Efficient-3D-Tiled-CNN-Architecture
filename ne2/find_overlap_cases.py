#!/usr/bin/env python3
"""
Find Overlapping Cases Between All Three Models
Identify cases where we have predictions for 3D U-Net and metrics for V-Net/SegResNet
"""

import pandas as pd
from pathlib import Path
import argparse

def find_overlapping_cases():
    """Find cases that exist in all three model evaluations"""
    
    # Load V-Net and SegResNet metrics
    metrics_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/_archive/20260204_2301/evaluation_results/metrics_summary.csv")
    metrics_df = pd.read_csv(metrics_file)
    
    # Get unique case IDs from V-Net and SegResNet
    vnet_cases = set(metrics_df[metrics_df['model'] == 'V-Net']['case_id'].unique())
    segresnet_cases = set(metrics_df[metrics_df['model'] == 'SegResNet']['case_id'].unique())
    
    # Get available 3D U-Net predictions
    pred_dir = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/all_four_validations/mandatory_artifacts_3dunet/predictions/test")
    pred_files = list(pred_dir.glob("*_pred_mask.nii.gz"))
    dunet_cases = set()
    
    for file in pred_files:
        case_id = int(file.name.split('_')[0])
        dunet_cases.add(case_id)
    
    print(f"V-Net cases: {len(vnet_cases)} ({sorted(vnet_cases)[:5]}...)")
    print(f"SegResNet cases: {len(segresnet_cases)} ({sorted(segresnet_cases)[:5]}...)")
    print(f"3D U-Net cases: {len(dunet_cases)} ({sorted(dunet_cases)[:5]}...)")
    
    # Find overlaps
    vnet_segresnet_overlap = vnet_cases.intersection(segresnet_cases)
    all_three_overlap = vnet_segresnet_overlap.intersection(dunet_cases)
    
    print(f"\nV-Net ↔ SegResNet overlap: {len(vnet_segresnet_overlap)} cases")
    print(f"All three models overlap: {len(all_three_overlap)} cases")
    
    if all_three_overlap:
        print(f"\nOverlapping case IDs: {sorted(all_three_overlap)}")
        
        # Save overlapping cases
        overlap_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/overlap_cases.txt")
        with open(overlap_file, 'w') as f:
            for case_id in sorted(all_three_overlap):
                f.write(f"{case_id}\n")
        
        print(f"Saved overlapping cases to: {overlap_file}")
        
        # Create case mapping for these overlapping cases
        create_overlap_mapping(all_three_overlap)
        
        return all_three_overlap
    else:
        print("\n❌ No overlapping cases found between all three models!")
        
        # Find best alternative overlaps
        print("\nAlternative Options:")
        print(f"1. V-Net ↔ SegResNet: {len(vnet_segresnet_overlap)} cases (already have metrics)")
        print(f"2. Use 3D U-Net cases and run V-Net/SegResNet inference: {len(dunet_cases)} cases")
        
        return set()

def create_overlap_mapping(overlap_cases):
    """Create case mapping for overlapping cases"""
    
    # Load existing case mapping
    mapping_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/_archive/20260204_2301/evaluation_results_3dunet/case_mapping.csv")
    mapping_df = pd.read_csv(mapping_file)
    
    # Filter for overlapping cases
    overlap_mapping = mapping_df[mapping_df['Original_ID'].isin(overlap_cases)]
    
    # Save overlap mapping
    overlap_mapping_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/overlap_case_mapping.csv")
    overlap_mapping.to_csv(overlap_mapping_file, index=False)
    
    print(f"Created overlap mapping: {overlap_mapping_file}")
    print(f"Contains {len(overlap_mapping)} overlapping cases")

def main():
    parser = argparse.ArgumentParser(description='Find overlapping cases between all three models')
    args = parser.parse_args()
    
    overlap_cases = find_overlapping_cases()
    
    if overlap_cases:
        print(f"\n✅ Found {len(overlap_cases)} overlapping cases!")
        print("Next steps:")
        print("1. Copy 3D U-Net predictions for these cases")
        print("2. Compute metrics for 3D U-Net")
        print("3. Update comparison tables")
    else:
        print("\n❌ No overlapping cases found. Consider alternative strategies.")

if __name__ == '__main__':
    main()
