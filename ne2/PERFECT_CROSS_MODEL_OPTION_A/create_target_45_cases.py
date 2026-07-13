#!/usr/bin/env python3
"""
Create Target 45 Cases List for Perfect Cross-Model Validation
Combines V-Net's 45 cases for unified evaluation
"""

import pandas as pd
from pathlib import Path

def create_target_45_cases():
    """Create the target 45 cases list"""
    
    print("🎯 Creating Target 45 Cases List...")
    
    # Load existing V-Net metrics to get case IDs
    metrics_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/_archive/20260204_2301/evaluation_results/metrics_summary.csv")
    if not metrics_file.exists():
        print("❌ V-Net metrics file not found")
        return
    
    metrics_df = pd.read_csv(metrics_file)
    vnet_cases = metrics_df[metrics_df['model'] == 'V-Net']['case_id'].unique()
    
    # Sort cases
    vnet_cases_sorted = sorted(vnet_cases)
    
    print(f"📊 Found {len(vnet_cases_sorted)} V-Net cases:")
    print(f"   Cases: {vnet_cases_sorted}")
    
    # Save target cases
    target_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/target_45_cases.txt")
    with open(target_file, 'w') as f:
        for case_id in vnet_cases_sorted:
            f.write(f"{case_id}\n")
    
    print(f"✅ Target 45 cases saved to: {target_file}")
    
    # Analyze case distribution
    print(f"\n📈 Case Distribution Analysis:")
    print(f"   Single-digit cases: {[c for c in vnet_cases_sorted if c < 10]}")
    print(f"   Teen cases: {[c for c in vnet_cases_sorted if 10 <= c < 20]}")
    print(f"   100+ cases: {[c for c in vnet_cases_sorted if c >= 100]}")
    
    # Check which cases already have SegResNet metrics
    segresnet_cases = metrics_df[metrics_df['model'] == 'SegResNet']['case_id'].unique()
    overlap_cases = set(vnet_cases_sorted).intersection(set(segresnet_cases))
    
    print(f"\n🔄 SegResNet Coverage:")
    print(f"   SegResNet has: {len(segresnet_cases)} cases")
    print(f"   Overlap with V-Net: {len(overlap_cases)} cases")
    print(f"   Need to evaluate: {len(vnet_cases_sorted) - len(overlap_cases)} additional cases")
    
    return vnet_cases_sorted, overlap_cases

def create_case_analysis():
    """Create detailed case analysis file"""
    
    # Load metrics
    metrics_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/_archive/20260204_2301/evaluation_results/metrics_summary.csv")
    metrics_df = pd.read_csv(metrics_file)
    
    # Create analysis
    analysis_data = []
    
    for case_id in sorted(metrics_df['case_id'].unique()):
        case_data = {'case_id': case_id}
        
        # Check V-Net metrics
        vnet_row = metrics_df[(metrics_df['model'] == 'V-Net') & (metrics_df['case_id'] == case_id)]
        if not vnet_row.empty:
            case_data['vnet_dice'] = vnet_row['dice'].iloc[0]
            case_data['vnet_available'] = True
        else:
            case_data['vnet_dice'] = None
            case_data['vnet_available'] = False
        
        # Check SegResNet metrics
        segresnet_row = metrics_df[(metrics_df['model'] == 'SegResNet') & (metrics_df['case_id'] == case_id)]
        if not segresnet_row.empty:
            case_data['segresnet_dice'] = segresnet_row['dice'].iloc[0]
            case_data['segresnet_available'] = True
        else:
            case_data['segresnet_dice'] = None
            case_data['segresnet_available'] = False
        
        # Check 3D U-Net availability (will be processed later)
        case_data['dunet_available'] = False
        case_data['dunet_dice'] = None
        
        analysis_data.append(case_data)
    
    # Save analysis
    analysis_df = pd.DataFrame(analysis_data)
    analysis_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/case_45_analysis.csv")
    analysis_df.to_csv(analysis_file, index=False)
    
    print(f"✅ Case analysis saved to: {analysis_file}")
    
    # Print summary
    vnet_available = analysis_df['vnet_available'].sum()
    segresnet_available = analysis_df['segresnet_available'].sum()
    
    print(f"\n📊 Current Coverage Summary:")
    print(f"   V-Net: {vnet_available}/45 cases available")
    print(f"   SegResNet: {segresnet_available}/45 cases available")
    print(f"   3D U-Net: 0/45 cases (to be processed)")
    
    return analysis_df

def main():
    print("🚀 Creating Perfect Cross-Model Validation - Target Cases")
    print("=" * 60)
    
    # Create target cases list
    target_cases, overlap_cases = create_target_45_cases()
    
    # Create detailed analysis
    analysis_df = create_case_analysis()
    
    print(f"\n🎯 TARGET 45 CASES READY!")
    print(f"   Total cases: {len(target_cases)}")
    print(f"   Range: {min(target_cases)} - {max(target_cases)}")
    print(f"   Ready for 3D U-Net and SegResNet processing")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Run 3D U-Net inference on these 45 cases")
    print(f"   2. Expand SegResNet evaluation to missing cases")
    print(f"   3. Create unified comparison")

if __name__ == '__main__':
    main()
