import os
import pandas as pd
import numpy as np
from scipy import stats

primary_dir = r'H:\Thesis_Trainings\Q1_Publication_Package\matched_200ep_benchmark\evaluation_results'
unseen_dir = r'H:\Thesis_Trainings\3d Cas Validations and doings\results\unseen_66_cohort\metrics'

df_ras_p = pd.read_csv(os.path.join(primary_dir, 'metrics_rasnet_200ep.csv'))
df_seg_p = pd.read_csv(os.path.join(primary_dir, 'metrics_segresnet_200ep.csv'))
df_nnu_p = pd.read_csv(os.path.join(primary_dir, 'metrics_nnu_net_v2_200ep.csv'))
df_vnt_p = pd.read_csv(os.path.join(primary_dir, 'metrics_v_net_200ep.csv'))
df_3du_p = pd.read_csv(os.path.join(primary_dir, 'metrics_3d_u_net_200ep.csv'))

df_ras_u = pd.read_csv(os.path.join(unseen_dir, 'unseen_66_rasnet_thresh05_case_metrics.csv'))
df_seg_u = pd.read_csv(os.path.join(unseen_dir, 'unseen_66_segresnet_case_metrics.csv'))
df_nnu_u = pd.read_csv(os.path.join(unseen_dir, 'unseen_66_nnunet_case_metrics.csv'))
df_vnt_u = pd.read_csv(os.path.join(unseen_dir, 'unseen_66_vnet_case_metrics.csv'))
df_3du_u = pd.read_csv(os.path.join(unseen_dir, 'unseen_66_3dunet_case_metrics.csv'))

def compute_wilcoxon_effect(x, y):
    diff = x - y
    res = stats.wilcoxon(x, y, alternative='two-sided')
    w_stat = res.statistic
    n = len(diff[diff != 0])
    # Compute normal approximation Z
    # Mean of W = n*(n+1)/4
    # Var of W = n*(n+1)*(2*n+1)/24
    mean_w = n * (n + 1) / 4.0
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_stat - mean_w) / std_w
    r = abs(z) / np.sqrt(n)
    return {
        'mean_diff': np.mean(diff),
        'med_diff': np.median(diff),
        'w_stat': w_stat,
        'p_raw': res.pvalue,
        'z_stat': z,
        'effect_r': r
    }

print("=== PRIMARY PAIRED EFFECTS (N=150) ===")
for bname, df_b in [('nnU-Net V2', df_nnu_p), ('SegResNet', df_seg_p), ('V-Net', df_vnt_p), ('3D U-Net', df_3du_p)]:
    for met in ['dice', 'hd95', 'precision', 'recall', 'cldice']:
        c_ras = [c for c in df_ras_p.columns if met in c.lower()][0]
        c_b = [c for c in df_b.columns if met in c.lower()][0]
        eff = compute_wilcoxon_effect(df_ras_p[c_ras], df_b[c_b])
        print(f"Primary | RASNet vs {bname:10s} | {met.upper():9s} | Diff: {eff['mean_diff']:+.4f} (Med: {eff['med_diff']:+.4f}) | W={eff['w_stat']:7.1f} | Z={eff['z_stat']:+6.2f} | r={eff['effect_r']:.3f} | p={eff['p_raw']:.3e}")

print("\n=== UNSEEN 66 PAIRED EFFECTS (N=66) ===")
for bname, df_b in [('SegResNet', df_seg_u), ('nnU-Net', df_nnu_u), ('V-Net', df_vnt_u), ('3D U-Net', df_3du_u)]:
    for met in ['dice', 'hd95', 'precision', 'recall', 'cldice']:
        c_ras = [c for c in df_ras_u.columns if met in c.lower()][0]
        c_b = [c for c in df_b.columns if met in c.lower()][0]
        eff = compute_wilcoxon_effect(df_ras_u[c_ras], df_b[c_b])
        print(f"Unseen  | RASNet vs {bname:10s} | {met.upper():9s} | Diff: {eff['mean_diff']:+.4f} (Med: {eff['med_diff']:+.4f}) | W={eff['w_stat']:7.1f} | Z={eff['z_stat']:+6.2f} | r={eff['effect_r']:.3f} | p={eff['p_raw']:.3e}")
