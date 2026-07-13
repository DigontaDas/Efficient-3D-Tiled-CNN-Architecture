import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load metrics
df = pd.read_csv('evaluation_results/metrics_summary.csv')
summary = pd.read_csv('evaluation_results/model_summary.csv')

# Set style
sns.set_style('whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Metrics to plot
metrics = ['dice', 'iou', 'precision', 'recall', 'hd95']
colors = {'V-Net': '#e74c3c', 'SegResNet': '#3498db'}

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model][metric]
        if len(model_data) > 1:  # Only plot if we have data
            ax.hist(model_data, bins=15, alpha=0.6, label=model, color=colors.get(model, 'gray'))
    
    ax.set_xlabel(metric.upper())
    ax.set_ylabel('Count')
    ax.set_title(f'{metric.upper()} Distribution')
    ax.legend()

# Summary bar chart for mean Dice
ax = axes[1, 2]
models = summary['model'].values
mean_dice = summary['dice_mean'].values
colors_bar = [colors.get(m, 'gray') for m in models]
bars = ax.bar(models, mean_dice, color=colors_bar, edgecolor='black')
ax.set_ylabel('Mean Dice Score')
ax.set_title('Mean Dice Comparison')
ax.set_ylim(0, 1)

# Add value labels on bars
for bar, val in zip(bars, mean_dice):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('evaluation_results/comparative_analysis.png', dpi=150, bbox_inches='tight')
print('Comparative analysis plot saved to: evaluation_results/comparative_analysis.png')

# Print summary table
print('\n' + '='*60)
print('EVALUATION SUMMARY')
print('='*60)
for _, row in summary.iterrows():
    print(f"\n{row['model']}:")
    print(f"  Mean Dice: {row['dice_mean']:.4f} ± {row['dice_std']:.4f}")
    print(f"  Mean IoU: {row['iou_mean']:.4f} ± {row['iou_std']:.4f}")
    print(f"  Mean Precision: {row['precision_mean']:.4f} ± {row['precision_std']:.4f}")
    print(f"  Mean Recall: {row['recall_mean']:.4f} ± {row['recall_std']:.4f}")
    print(f"  Cases evaluated: {row['num_cases']}")
print('='*60)
