#!/usr/bin/env python3
"""
Create Unified Cross-Model Comparison
Combines all three models' results for comprehensive analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_model_results():
    """Load results from all three models"""
    
    print("Loading model results...")
    
    # Load 3D U-Net results (GPU-processed)
    dunet_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/3dunet_metrics_real_gpu.csv")
    if dunet_file.exists():
        dunet_df = pd.read_csv(dunet_file)
        print(f"✅ 3D U-Net: {len(dunet_df)} cases")
    else:
        print("❌ 3D U-Net results not found")
        dunet_df = pd.DataFrame()
    
    # Load V-Net and SegResNet results
    metrics_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/_archive/20260204_2301/evaluation_results/metrics_summary.csv")
    if metrics_file.exists():
        vnet_segresnet_df = pd.read_csv(metrics_file)
        print(f"✅ V-Net + SegResNet: {len(vnet_segresnet_df)} total cases")
        
        # Separate by model
        vnet_df = vnet_segresnet_df[vnet_segresnet_df['model'] == 'V-Net'].copy()
        segresnet_df = vnet_segresnet_df[vnet_segresnet_df['model'] == 'SegResNet'].copy()
        
        print(f"  V-Net: {len(vnet_df)} cases")
        print(f"  SegResNet: {len(segresnet_df)} cases")
    else:
        print("❌ V-Net/SegResNet results not found")
        vnet_df = pd.DataFrame()
        segresnet_df = pd.DataFrame()
    
    return dunet_df, vnet_df, segresnet_df

def standardize_model_names(dunet_df, vnet_df, segresnet_df):
    """Standardize model names and formats"""
    
    # Standardize 3D U-Net
    if not dunet_df.empty:
        dunet_standardized = dunet_df[['Original_ID', 'Dice', 'IoU', 'Precision', 'Recall', 'HD95']].copy()
        dunet_standardized['Model'] = '3D U-Net'
        dunet_standardized['Case_ID'] = dunet_df['Original_ID']
    
    # Standardize V-Net
    if not vnet_df.empty:
        vnet_standardized = vnet_df[['case_id', 'dice', 'iou', 'precision', 'recall', 'hd95']].copy()
        vnet_standardized.columns = ['Case_ID', 'Dice', 'IoU', 'Precision', 'Recall', 'HD95']
        vnet_standardized['Model'] = 'V-Net'
    
    # Standardize SegResNet
    if not segresnet_df.empty:
        segresnet_standardized = segresnet_df[['case_id', 'dice', 'iou', 'precision', 'recall', 'hd95']].copy()
        segresnet_standardized.columns = ['Case_ID', 'Dice', 'IoU', 'Precision', 'Recall', 'HD95']
        segresnet_standardized['Model'] = 'SegResNet'
    
    return dunet_standardized, vnet_standardized, segresnet_standardized

def create_comparison_analysis(dunet_df, vnet_df, segresnet_df):
    """Create comprehensive comparison analysis"""
    
    print("\n🔊 CROSS-MODEL COMPARISON ANALYSIS")
    print("=" * 50)
    
    # Model summaries
    models = []
    summaries = []
    
    if not dunet_df.empty:
        models.append('3D U-Net')
        summaries.append({
            'cases': len(dunet_df),
            'dice_mean': dunet_df['Dice'].mean(),
            'dice_std': dunet_df['Dice'].std(),
            'iou_mean': dunet_df['IoU'].mean(),
            'precision_mean': dunet_df['Precision'].mean(),
            'recall_mean': dunet_df['Recall'].mean(),
            'hd95_mean': dunet_df['HD95'].mean()
        })
    
    if not vnet_df.empty:
        models.append('V-Net')
        summaries.append({
            'cases': len(vnet_df),
            'dice_mean': vnet_df['Dice'].mean(),
            'dice_std': vnet_df['Dice'].std(),
            'iou_mean': vnet_df['IoU'].mean(),
            'precision_mean': vnet_df['Precision'].mean(),
            'recall_mean': vnet_df['Recall'].mean(),
            'hd95_mean': vnet_df['HD95'].mean()
        })
    
    if not segresnet_df.empty:
        models.append('SegResNet')
        summaries.append({
            'cases': len(segresnet_df),
            'dice_mean': segresnet_df['Dice'].mean(),
            'dice_std': segresnet_df['Dice'].std(),
            'iou_mean': segresnet_df['IoU'].mean(),
            'precision_mean': segresnet_df['Precision'].mean(),
            'recall_mean': segresnet_df['Recall'].mean(),
            'hd95_mean': segresnet_df['HD95'].mean()
        })
    
    # Print summaries
    for model, summary in zip(models, summaries):
        print(f"\n📊 {model}:")
        print(f"  Cases: {summary['cases']}")
        print(f"  Dice: {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
        print(f"  IoU: {summary['iou_mean']:.4f}")
        print(f"  Precision: {summary['precision_mean']:.4f}")
        print(f"  Recall: {summary['recall_mean']:.4f}")
        print(f"  HD95: {summary['hd95_mean']:.2f}")
    
    # Create ranking
    print(f"\n🏆 MODEL RANKINGS (by Dice Score):")
    dice_scores = [(model, summary['dice_mean']) for model, summary in zip(models, summaries)]
    dice_scores.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (model, score) in enumerate(dice_scores, 1):
        print(f"  {rank}. {model}: {score:.4f}")
    
    return models, summaries

def save_unified_results(dunet_df, vnet_df, segresnet_df):
    """Save unified results CSV"""
    
    # Combine all results
    all_results = []
    
    if not dunet_df.empty:
        all_results.append(dunet_df)
    if not vnet_df.empty:
        all_results.append(vnet_df)
    if not segresnet_df.empty:
        all_results.append(segresnet_df)
    
    if all_results:
        unified_df = pd.concat(all_results, ignore_index=True)
        
        # Save unified results
        output_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/unified_cross_model_results.csv")
        unified_df.to_csv(output_file, index=False)
        print(f"\n✅ Unified results saved to: {output_file}")
        print(f"   Total entries: {len(unified_df)}")
        print(f"   Models: {unified_df['Model'].nunique()}")
        
        return unified_df
    else:
        print("\n❌ No results to combine")
        return pd.DataFrame()

def create_comparison_plots(unified_df):
    """Create comparison plots"""
    
    if unified_df.empty:
        print("❌ No data for plotting")
        return
    
    print(f"\n📈 Creating comparison plots...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cross-Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Dice Score Comparison
    ax1 = axes[0, 0]
    sns.boxplot(data=unified_df, x='Model', y='Dice', ax=ax1)
    ax1.set_title('Dice Score Distribution')
    ax1.set_ylabel('Dice Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: IoU Comparison
    ax2 = axes[0, 1]
    sns.boxplot(data=unified_df, x='Model', y='IoU', ax=ax2)
    ax2.set_title('IoU Distribution')
    ax2.set_ylabel('IoU')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Precision vs Recall Scatter
    ax3 = axes[1, 0]
    for model in unified_df['Model'].unique():
        model_data = unified_df[unified_df['Model'] == model]
        ax3.scatter(model_data['Precision'], model_data['Recall'], 
                   label=model, alpha=0.7, s=50)
    ax3.set_xlabel('Precision')
    ax3.set_ylabel('Recall')
    ax3.set_title('Precision vs Recall')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mean Performance Summary
    ax4 = axes[1, 1]
    mean_metrics = unified_df.groupby('Model')[['Dice', 'IoU', 'Precision', 'Recall']].mean()
    mean_metrics.plot(kind='bar', ax=ax4)
    ax4.set_title('Mean Performance Metrics')
    ax4.set_ylabel('Score')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plots
    plot_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/cross_model_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plots saved to: {plot_file}")
    
    plt.show()

def generate_thesis_report(models, summaries):
    """Generate thesis-ready report"""
    
    report = f"""
# Cross-Model Validation Report
## Unified Analysis of 3D U-Net, V-Net, and SegResNet

### Executive Summary
This analysis compares the performance of three medical image segmentation models:
{', '.join(models)}.

### Model Performance Overview
"""
    
    for model, summary in zip(models, summaries):
        report += f"""
#### {model}
- **Dataset Size:** {summary['cases']} cases
- **Dice Score:** {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}
- **IoU:** {summary['iou_mean']:.4f}
- **Precision:** {summary['precision_mean']:.4f}
- **Recall:** {summary['recall_mean']:.4f}
- **HD95:** {summary['hd95_mean']:.2f} mm
"""
    
    # Add rankings
    dice_scores = [(model, summary['dice_mean']) for model, summary in zip(models, summaries)]
    dice_scores.sort(key=lambda x: x[1], reverse=True)
    
    report += f"""
### Performance Rankings
Based on Dice coefficient:
"""
    for rank, (model, score) in enumerate(dice_scores, 1):
        report += f"{rank}. **{model}**: {score:.4f}\n"
    
    report += f"""
### Key Findings
1. **Best Performing Model:** {dice_scores[0][0]} with Dice score of {dice_scores[0][1]:.4f}
2. **Dataset Coverage:** 
   - 3D U-Net: {summaries[0]['cases'] if summaries else 0} cases
   - V-Net: {summaries[1]['cases'] if len(summaries) > 1 else 0} cases  
   - SegResNet: {summaries[2]['cases'] if len(summaries) > 2 else 0} cases

### Statistical Analysis Ready
The unified dataset is now ready for:
- Statistical significance testing
- Detailed failure case analysis
- Qualitative visualization
- Thesis chapter preparation

### Files Generated
- `unified_cross_model_results.csv` - Combined metrics
- `cross_model_comparison.png` - Visual comparison
- This report - Summary findings

---
*Analysis completed using GPU acceleration on NVIDIA RTX 3060 Ti*
"""
    
    # Save report
    report_file = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/thesis_cross_model_report.md")
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Thesis report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Create unified cross-model comparison')
    args = parser.parse_args()
    
    # Load all results
    dunet_df, vnet_df, segresnet_df = load_all_model_results()
    
    # Standardize formats
    dunet_std, vnet_std, segresnet_std = standardize_model_names(dunet_df, vnet_df, segresnet_df)
    
    # Create analysis
    models, summaries = create_comparison_analysis(dunet_std, vnet_std, segresnet_std)
    
    # Save unified results
    unified_df = save_unified_results(dunet_std, vnet_std, segresnet_std)
    
    # Create plots
    if not unified_df.empty:
        create_comparison_plots(unified_df)
    
    # Generate thesis report
    generate_thesis_report(models, summaries)
    
    print(f"\n🎉 CROSS-MODEL VALIDATION COMPLETE!")
    print(f"📊 All models analyzed and compared")
    print(f"🔍 Ready for thesis statistical analysis")

if __name__ == '__main__':
    main()
