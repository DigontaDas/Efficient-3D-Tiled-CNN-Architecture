#!/usr/bin/env python3
"""
4-Model Comparison Framework - Ready for Integration
Includes 3D U-Net with validation-based performance estimate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_model_results():
    """Load results from all 4 models"""
    
    # 3D U-Net (validation estimate)
    unet_3d = pd.read_csv('e:/Thesis/evaluation_results/final_3dunet_evaluation.csv')
    
    # SegResNet (assuming this path)
    try:
        segresnet = pd.read_csv('e:/Thesis/SegResNet_model_IMGcas/new_robust_results.csv')
        print(f"✅ SegResNet loaded: {len(segresnet)} cases")
    except:
        print("⚠️  SegResNet file not found at expected path")
        segresnet = None
    
    # Placeholder for other 2 models
    print("📝 Other models (Model_3, Model_4) - add paths when ready")
    
    return unet_3d, segresnet

def create_comparison_summary():
    """Create comparison summary with available data"""
    
    unet_3d, segresnet = load_all_model_results()
    
    # Create comparison table
    comparison_data = []
    
    # Add 3D U-Net
    comparison_data.append({
        'Model': '3D U-Net',
        'Dice': 0.0313,
        'IoU': 0.0159,
        'HD95': 266.1391,
        'Precision': 0.0294,
        'Recall': 0.0335,
        'DataSource': 'Validation Case 701',
        'Cases_Evaluated': 1,
        'Status': 'Estimate'
    })
    
    # Add SegResNet if available
    if segresnet is not None and len(segresnet) > 0:
        comparison_data.append({
            'Model': 'SegResNet',
            'Dice': segresnet['Dice'].mean(),
            'IoU': segresnet['IoU'].mean(),
            'HD95': segresnet['HD95'].mean(),
            'Precision': segresnet['Precision'].mean(),
            'Recall': segresnet['Recall'].mean(),
            'DataSource': 'Test Set',
            'Cases_Evaluated': len(segresnet),
            'Status': 'Complete'
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison
    comparison_df.to_csv('e:/Thesis/evaluation_results/4_model_comparison_summary.csv', index=False)
    
    print("\n=== 4-MODEL COMPARISON SUMMARY ===")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

def create_performance_plot(comparison_df):
    """Create performance comparison plot"""
    
    if len(comparison_df) < 2:
        print("Need at least 2 models for comparison plot")
        return
    
    # Metrics to plot (excluding HD95 which is distance)
    metrics = ['Dice', 'IoU', 'Precision', 'Recall']
    
    # Create subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot bars
        bars = ax.bar(comparison_df['Model'], comparison_df[metric])
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_df[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('e:/Thesis/evaluation_results/4_model_performance_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Performance comparison plot saved")

def rank_models(comparison_df):
    """Rank models by each metric"""
    
    if len(comparison_df) < 2:
        print("Need at least 2 models for ranking")
        return
    
    # Metrics where higher is better
    higher_better = ['Dice', 'IoU', 'Precision', 'Recall']
    # Metrics where lower is better
    lower_better = ['HD95']
    
    rankings = {}
    
    for metric in higher_better:
        if metric in comparison_df.columns:
            rankings[metric] = comparison_df.sort_values(metric, ascending=False)['Model'].tolist()
    
    for metric in lower_better:
        if metric in comparison_df.columns:
            rankings[metric] = comparison_df.sort_values(metric, ascending=True)['Model'].tolist()
    
    # Create ranking table
    ranking_df = pd.DataFrame(rankings)
    ranking_df.index = [f'Rank {i+1}' for i in range(len(ranking_df))]
    
    ranking_df.to_csv('e:/Thesis/evaluation_results/model_rankings.csv')
    
    print("\n=== MODEL RANKINGS ===")
    print(ranking_df.to_string())
    
    return ranking_df

def main():
    """Main function"""
    print("4-Model Comparison Framework")
    print("=" * 50)
    print("3D U-Net evaluation complete using Option 1 (Validation Metrics)")
    print()
    
    # Create comparison summary
    comparison_df = create_comparison_summary()
    
    # Create performance plot
    create_performance_plot(comparison_df)
    
    # Rank models
    ranking_df = rank_models(comparison_df)
    
    print("\n✅ 4-MODEL COMPARISON READY")
    print("Files generated:")
    print("- 4_model_comparison_summary.csv")
    print("- 4_model_performance_comparison.png")
    print("- model_rankings.csv")
    print()
    print("📝 Note: Add other 2 models when available for complete comparison")

if __name__ == "__main__":
    main()
