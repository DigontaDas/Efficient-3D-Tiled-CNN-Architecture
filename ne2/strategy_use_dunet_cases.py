#!/usr/bin/env python3
"""
Strategy C: Use 3D U-Net Cases (851-1000) for Cross-Model Comparison
This will give us 150 cases for comparison instead of just 5
"""

import pandas as pd
from pathlib import Path
import argparse

def create_dunet_based_strategy():
    """Create strategy using 3D U-Net cases as the base"""
    
    # Get available 3D U-Net predictions
    pred_dir = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/all_four_validations/mandatory_artifacts_3dunet/predictions/test")
    pred_files = list(pred_dir.glob("*_pred_mask.nii.gz"))
    dunet_cases = sorted([int(f.name.split('_')[0]) for f in pred_files])
    
    print(f"3D U-Net has predictions for {len(dunet_cases)} cases: {dunet_cases[:5]}...{dunet_cases[-5:]}")
    
    # Save these as our target cases
    target_cases_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/target_cases_dunet.txt")
    with open(target_cases_file, 'w') as f:
        for case_id in dunet_cases:
            f.write(f"{case_id}\n")
    
    print(f"Saved target cases to: {target_cases_file}")
    
    # Check which of these cases have ground truth
    gt_dir = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Dataset_Main/images_label")
    available_gt = []
    
    for case_id in dunet_cases:
        gt_file = gt_dir / f"{case_id}.label.nii.gz"
        if gt_file.exists():
            available_gt.append(case_id)
    
    print(f"Ground truth available for {len(available_gt)}/{len(dunet_cases)} cases")
    
    # Create case mapping for these cases
    create_dunet_case_mapping(dunet_cases)
    
    return dunet_cases, available_gt

def create_dunet_case_mapping(dunet_cases):
    """Create case mapping for 3D U-Net cases"""
    
    mapping_data = []
    for i, case_id in enumerate(dunet_cases):
        mapping_data.append({
            'Original_ID': case_id,
            'SegResNet_Format': f'Case_{i}'
        })
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/dunet_case_mapping.csv")
    mapping_df.to_csv(mapping_file, index=False)
    
    print(f"Created case mapping: {mapping_file}")
    print(f"Maps {len(dunet_cases)} cases to Case_0 to Case_{len(dunet_cases)-1}")

def create_evaluation_plan():
    """Create evaluation plan for all three models"""
    
    plan = """
# EVALUATION PLAN: Strategy C - Use 3D U-Net Cases (851-1000)

## Current Status:
DONE 3D U-Net: 150 predictions (851-1000) - READY FOR METRICS
TODO V-Net: No predictions for 851-1000 - NEED INFERENCE
TODO SegResNet: No predictions for 851-1000 - NEED INFERENCE

## Required Actions:

### Step 1: Compute 3D U-Net Metrics (Immediate)
- Input: 150 predictions + 150 ground truth labels
- Output: metrics for 150 cases
- Command: python compute_3dunet_metrics_dunet_cases.py

### Step 2: Run V-Net Inference on 851-1000
- Need: V-Net model + training script
- Input: 150 images (851-1000)
- Output: 150 predictions
- Status: NEED MODEL ACCESS

### Step 3: Run SegResNet Inference on 851-1000  
- Need: SegResNet model + training script
- Input: 150 images (851-1000)
- Output: 150 predictions
- Status: NEED MODEL ACCESS

### Step 4: Compute V-Net and SegResNet Metrics
- Input: Their predictions + ground truth
- Output: metrics for 150 cases each

### Step 5: Create Unified Comparison
- All 3 models with 150 cases each
- Statistical analysis and plotting

## Expected Outcome:
- 150 cases for cross-model comparison (vs current 5)
- Robust statistical analysis
- Comprehensive thesis results

## Alternative: Partial Strategy
If V-Net/SegResNet models unavailable:
- Use 3D U-Net metrics alone (150 cases)
- Compare with existing V-Net/SegResNet metrics (different cases)
- Note limitation in thesis
"""
    
    plan_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/evaluation_plan_strategy_c.md")
    with open(plan_file, 'w') as f:
        f.write(plan)
    
    print(f"Created evaluation plan: {plan_file}")

def main():
    parser = argparse.ArgumentParser(description='Create strategy using 3D U-Net cases')
    args = parser.parse_args()
    
    dunet_cases, available_gt = create_dunet_based_strategy()
    create_evaluation_plan()
    
    print(f"\n✅ Strategy C Ready!")
    print(f"Target: {len(dunet_cases)} cases for cross-model comparison")
    print(f"Next: Compute 3D U-Net metrics immediately")

if __name__ == '__main__':
    main()
