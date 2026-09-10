import os
import glob
import pandas as pd
import numpy as np
from scipy import stats

def get_stats(arr):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'p25': np.nan, 'p75': np.nan, 'iqr': np.nan,
            'min': np.nan, 'max': np.nan
        }
    p25 = np.percentile(arr, 25)
    p75 = np.percentile(arr, 75)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'median': float(np.median(arr)),
        'p25': float(p25),
        'p75': float(p75),
        'iqr': float(p75 - p25),
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }

def main():
    primary_dir = r'H:\Thesis_Trainings\Q1_Publication_Package\matched_200ep_benchmark\evaluation_results'
    unseen_dir = r'H:\Thesis_Trainings\3d Cas Validations and doings\results\unseen_66_cohort\metrics'
    cas3d_dir = r'H:\Thesis_Trainings\3d Cas Validations and doings\results'

    primary_files = {
        'RASNet (Champion)': os.path.join(primary_dir, 'metrics_rasnet_200ep.csv'),
        'SegResNet': os.path.join(primary_dir, 'metrics_segresnet_200ep.csv'),
        'nnU-Net V2': os.path.join(primary_dir, 'metrics_nnu_net_v2_200ep.csv'),
        'V-Net': os.path.join(primary_dir, 'metrics_v_net_200ep.csv'),
        '3D U-Net': os.path.join(primary_dir, 'metrics_3d_u_net_200ep.csv'),
    }

    unseen_files = {
        'RASNet (tau=0.60)': os.path.join(unseen_dir, 'unseen_66_rasnet_case_metrics.csv'),
        'RASNet (tau=0.50)': os.path.join(unseen_dir, 'unseen_66_rasnet_thresh05_case_metrics.csv'),
        'SegResNet': os.path.join(unseen_dir, 'unseen_66_segresnet_case_metrics.csv'),
        'V-Net': os.path.join(unseen_dir, 'unseen_66_vnet_case_metrics.csv'),
        'nnU-Net': os.path.join(unseen_dir, 'unseen_66_nnunet_case_metrics.csv'),
        '3D U-Net': os.path.join(unseen_dir, 'unseen_66_3dunet_case_metrics.csv'),
    }

    metrics_list = ['dice', 'hd95', 'iou', 'precision', 'recall', 'cldice', 'asd']

    rows = []

    # Process Primary
    for mname, mpath in primary_files.items():
        if not os.path.exists(mpath):
            continue
        df = pd.read_csv(mpath)
        col_map = {c.lower(): c for c in df.columns}
        for met in metrics_list:
            matching_cols = [c for c in df.columns if met in c.lower()]
            if not matching_cols:
                continue
            col = matching_cols[0]
            st = get_stats(df[col].dropna())
            rows.append({
                'Dataset': 'Primary ImageCAS (N=150)',
                'Model': mname,
                'Metric': met.upper(),
                'N': len(df[col].dropna()),
                'Mean': st['mean'],
                'Std': st['std'],
                'Median': st['median'],
                'Q1_25pct': st['p25'],
                'Q3_75pct': st['p75'],
                'IQR': st['iqr'],
                'Min': st['min'],
                'Max': st['max']
            })

    # Process Unseen 66
    for mname, mpath in unseen_files.items():
        if not os.path.exists(mpath):
            continue
        df = pd.read_csv(mpath)
        for met in metrics_list:
            matching_cols = [c for c in df.columns if met in c.lower()]
            if not matching_cols:
                continue
            col = matching_cols[0]
            st = get_stats(df[col].dropna())
            rows.append({
                'Dataset': 'External 3D CAS Unseen (N=66)',
                'Model': mname,
                'Metric': met.upper(),
                'N': len(df[col].dropna()),
                'Mean': st['mean'],
                'Std': st['std'],
                'Median': st['median'],
                'Q1_25pct': st['p25'],
                'Q3_75pct': st['p75'],
                'IQR': st['iqr'],
                'Min': st['min'],
                'Max': st['max']
            })

    res_df = pd.DataFrame(rows)
    out_csv = r'H:\Thesis_Trainings\results\case_level_model_comparison.csv'
    res_df.to_csv(out_csv, index=False)
    print(f"Saved case level comparison to {out_csv} with {len(res_df)} rows.")

    # Also compute paired comparisons and effect sizes
    print("\n=== PAIRED WILCOXON & EFFECT SIZES (PRIMARY N=150) ===")
    df_ras = pd.read_csv(primary_files['RASNet (Champion)'])
    for bname in ['SegResNet', 'nnU-Net V2', 'V-Net', '3D U-Net']:
        df_b = pd.read_csv(primary_files[bname])
        for met in ['dice', 'hd95', 'precision', 'recall', 'cldice']:
            c_ras = [c for c in df_ras.columns if met in c.lower()][0]
            c_b = [c for c in df_b.columns if met in c.lower()][0]
            diff = df_ras[c_ras] - df_b[c_b]
            res = stats.wilcoxon(df_ras[c_ras], df_b[c_b], alternative='two-sided')
            # rank biserial correlation or Wilcoxon r = Z / sqrt(N)
            n = len(diff)
            # Normal approximation for Wilcoxon Z
            # stats.wilcoxon returns statistic W
            # Scipy ranksums or manual Z:
            mean_d = np.mean(diff)
            med_d = np.median(diff)
            print(f"Primary | RASNet vs {bname:10s} | {met.upper():9s} | Mean Diff: {mean_d:+.4f} | Med Diff: {med_d:+.4f} | W: {res.statistic:8.1f} | p-val: {res.pvalue:.4e}")

    print("\n=== PAIRED WILCOXON & EFFECT SIZES (UNSEEN 66) ===")
    df_ras_66 = pd.read_csv(unseen_files['RASNet (tau=0.50)'])
    df_ras_66_06 = pd.read_csv(unseen_files['RASNet (tau=0.60)'])
    for bname in ['SegResNet', 'V-Net', 'nnU-Net', '3D U-Net']:
        df_b = pd.read_csv(unseen_files[bname])
        for met in ['dice', 'hd95', 'precision', 'recall', 'cldice']:
            c_ras = [c for c in df_ras_66.columns if met in c.lower()][0]
            c_b = [c for c in df_b.columns if met in c.lower()][0]
            diff = df_ras_66[c_ras] - df_b[c_b]
            res = stats.wilcoxon(df_ras_66[c_ras], df_b[c_b], alternative='two-sided')
            mean_d = np.mean(diff)
            med_d = np.median(diff)
            print(f"Unseen66 (tau=0.50) | RASNet vs {bname:10s} | {met.upper():9s} | Mean Diff: {mean_d:+.4f} | Med Diff: {med_d:+.4f} | W: {res.statistic:8.1f} | p-val: {res.pvalue:.4e}")

if __name__ == '__main__':
    main()
