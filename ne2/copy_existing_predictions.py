#!/usr/bin/env python3
"""
Copy and Rename Existing 3D U-Net Predictions for Target Cases
Uses existing predictions from mandatory_artifacts_3dunet
"""

import os
import shutil
from pathlib import Path
import argparse

def copy_predictions_for_target_cases(source_dir, target_dir, target_cases_file):
    """Copy existing predictions for target cases"""
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    # Load target cases
    with open(target_cases_file, 'r') as f:
        target_cases = [int(line.strip()) for line in f if line.strip()]
    
    print(f"Looking for predictions for {len(target_cases)} target cases")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_path}")
    
    copied_count = 0
    missing_cases = []
    
    for case_id in target_cases:
        # Look for prediction file
        pred_filename = f"{case_id}_pred_mask.nii.gz"
        source_file = source_path / pred_filename
        
        if source_file.exists():
            target_file = target_path / pred_filename
            shutil.copy2(source_file, target_file)
            print(f"Copied: {pred_filename}")
            copied_count += 1
        else:
            missing_cases.append(case_id)
            print(f"Missing: {pred_filename}")
    
    print(f"\nSummary:")
    print(f"✅ Copied: {copied_count}/{len(target_cases)} predictions")
    print(f"❌ Missing: {len(missing_cases)} cases")
    
    if missing_cases:
        print(f"Missing case IDs: {missing_cases}")
    
    return copied_count, missing_cases

def main():
    parser = argparse.ArgumentParser(description='Copy existing 3D U-Net predictions for target cases')
    parser.add_argument('--source_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/all_four_validations/mandatory_artifacts_3dunet/predictions/test',
                       help='Directory containing existing predictions')
    parser.add_argument('--target_dir', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/3dunet_predictions_target',
                       help='Directory to save copied predictions')
    parser.add_argument('--target_cases', type=str,
                       default='c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/target_cases.txt',
                       help='Text file with target case IDs')
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not Path(args.source_dir).exists():
        print(f"ERROR: Source directory does not exist: {args.source_dir}")
        return
    
    # Copy predictions
    copied_count, missing_cases = copy_predictions_for_target_cases(
        args.source_dir, args.target_dir, args.target_cases
    )
    
    if copied_count > 0:
        print(f"\n✅ Ready to compute metrics!")
        print(f"Run: python compute_3dunet_metrics_target.py")
    else:
        print(f"\n❌ No predictions copied. Check source directory and target cases.")

if __name__ == '__main__':
    main()
