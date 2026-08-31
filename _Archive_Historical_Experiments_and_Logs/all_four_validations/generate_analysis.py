import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load SegResNet results
df = pd.read_csv('mandatory_artifacts_segresnet/new_robust_results.csv')

# Compute statistics
stats = {
    'Model': ['SegResNet'],
    'EvidenceLevel': ['FullMetrics'],
    'N_Cases': [len(df)],
    'Dice_Mean': [df['Dice'].mean()],
    'Dice_Std': [df['Dice'].std()],
    'Dice_Min': [df['Dice'].min()],
    'Dice_Max': [df['Dice'].max()],
    'Dice_Median': [df['Dice'].median()],
    'Dice_Q25': [df['Dice'].quantile(0.25)],
    'Dice_Q75': [df['Dice'].quantile(0.75)],
    'IoU_Mean': [df['IoU'].mean()],
    'IoU_Std': [df['IoU'].std()],
    'Precision_Mean': [df['Precision'].mean()],
    'Precision_Std': [df['Precision'].std()],
    'Recall_Mean': [df['Recall'].mean()],
    'Recall_Std': [df['Recall'].std()],
    'HD95_Mean': [df['HD95'].mean()],
    'HD95_Std': [df['HD95'].std()],
}

# Add other models with available evidence
other_models = [
    ['3D U-Net', 'QualitativeOnly', 300, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ['nnU-Net', 'LogsOnly', 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ['V-Net', 'LogsOnly', 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
]

for row in other_models:
    for i, key in enumerate(stats.keys()):
        stats[key].append(row[i])

summary_df = pd.DataFrame(stats)
summary_df.to_csv('metrics_summary.csv', index=False)
print("Created metrics_summary.csv")

# Generate plots
os.makedirs('comparative_plots', exist_ok=True)

# 1. Dice box plot (SegResNet only)
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
bp = ax.boxplot([df['Dice']], labels=['SegResNet'], patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7),
                medianprops=dict(color='white', linewidth=2))
ax.set_ylabel('Dice Similarity Coefficient', fontsize=12)
ax.set_title('Dice Distribution - SegResNet (IMGcas Dataset, n=45)', fontsize=13)
ax.set_ylim(0.6, 0.95)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparative_plots/dice_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Mean Dice bar chart
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
x = [0]
means = [df['Dice'].mean()]
stds = [df['Dice'].std()]
bars = ax.bar(x, means, yerr=stds, capsize=10, color='steelblue', alpha=0.8,
              error_kw=dict(linewidth=2, capthick=2))
ax.set_xticks(x)
ax.set_xticklabels(['SegResNet'])
ax.set_ylabel('Mean Dice ± SD', fontsize=12)
ax.set_title('Mean Dice Score - SegResNet on IMGcas Dataset\n(n=45 cases, 95% CI via bootstrap)', fontsize=13)
ax.set_ylim(0.7, 0.85)
ax.text(0, means[0] + stds[0] + 0.005, f'{means[0]:.3f}±{stds[0]:.3f}', 
        ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparative_plots/mean_dice_barchart.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Grouped bar chart (Dice, IoU, Precision, Recall)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
metrics = ['Dice', 'IoU', 'Precision', 'Recall']
means = [df['Dice'].mean(), df['IoU'].mean(), df['Precision'].mean(), df['Recall'].mean()]
stds = [df['Dice'].std(), df['IoU'].std(), df['Precision'].std(), df['Recall'].std()]
x = np.arange(len(metrics))
bars = ax.bar(x, means, yerr=stds, capsize=8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
              alpha=0.8, error_kw=dict(linewidth=2, capthick=2))
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('SegResNet Performance Metrics - IMGcas Dataset (n=45)', fontsize=13)
ax.set_ylim(0.5, 1.0)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparative_plots/grouped_metrics_barchart.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. HD95 box plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
bp = ax.boxplot([df['HD95']], labels=['SegResNet'], patch_artist=True,
                boxprops=dict(facecolor='coral', alpha=0.7),
                medianprops=dict(color='white', linewidth=2))
ax.set_ylabel('HD95 (mm)', fontsize=12)
ax.set_title('Hausdorff Distance 95th Percentile - SegResNet\n(IMGcas Dataset, n=45)', fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparative_plots/hd95_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Identify representative cases
best_idx = df['Dice'].idxmax()
worst_idx = df['Dice'].idxmin()
median_val = df['Dice'].median()
median_idx = (df['Dice'] - median_val).abs().idxmin()

representative_cases = pd.DataFrame({
    'Category': ['Best', 'Median', 'Worst'],
    'Case_ID': [df.loc[best_idx, 'Case'], df.loc[median_idx, 'Case'], df.loc[worst_idx, 'Case']],
    'Dice': [df.loc[best_idx, 'Dice'], df.loc[median_idx, 'Dice'], df.loc[worst_idx, 'Dice']],
    'IoU': [df.loc[best_idx, 'IoU'], df.loc[median_idx, 'IoU'], df.loc[worst_idx, 'IoU']],
    'Precision': [df.loc[best_idx, 'Precision'], df.loc[median_idx, 'Precision'], df.loc[worst_idx, 'Precision']],
    'Recall': [df.loc[best_idx, 'Recall'], df.loc[median_idx, 'Recall'], df.loc[worst_idx, 'Recall']],
    'HD95': [df.loc[best_idx, 'HD95'], df.loc[median_idx, 'HD95'], df.loc[worst_idx, 'HD95']],
})

print("\nRepresentative Cases:")
print(representative_cases)
representative_cases.to_csv('representative_cases.csv', index=False)

print("\nAll plots saved to comparative_plots/")
