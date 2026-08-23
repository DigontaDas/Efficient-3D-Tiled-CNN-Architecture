#!/usr/bin/env python3
"""
01_significance_testing.py
=============================================================================
Statistical Significance & Uncertainty Estimation for RASNet vs. Baselines.

Author: Digonta Das / Nafis Mehedi
Project: Efficient 3D Tiled CNN Architecture (RASNet, ImageCAS Dataset)
Target: Q1 Medical Imaging Journal Submission

Methods:
  - Paired two-sided Wilcoxon signed-rank test (scipy.stats.wilcoxon)
  - Holm-Bonferroni family-wise error rate (FWER) correction
  - Non-parametric Percentile Bootstrap 95% Confidence Intervals (N=2000 resamples)
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

# Fix random seed for exact reproducibility
RANDOM_SEED = 42
N_BOOTSTRAP = 2000

# Base directory paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAININGS_DIR = os.path.join(REPO_ROOT, "Thesis_Trainings", "Thesis_Trainings")
OUTPUT_DIR = os.path.join(REPO_ROOT, "Q1_Publication_Package", "stats")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metric file paths
RASNET_CSV = os.path.join(TRAININGS_DIR, "results-after-hallucin-fix", "metrics_rasnet.csv")
SEGRESNET_CSV = os.path.join(TRAININGS_DIR, "all_four_validations", "metrics_segresnet.csv")
NNUNET_CSV = os.path.join(TRAININGS_DIR, "all_four_validations", "mandatory_artifacts_nnunet", "metrics_nnunet.csv")
UNET3D_CSV = os.path.join(TRAININGS_DIR, "all_four_validations", "metrics_3d_unet.csv")

METRICS = ["dice", "iou", "precision", "recall", "hd95"]
METRIC_NAMES = {
    "dice": "Dice Similarity Coefficient (DSC)",
    "iou": "Intersection-over-Union (IoU)",
    "precision": "Precision (PPV)",
    "recall": "Recall / Sensitivity",
    "hd95": "95% Hausdorff Distance (HD95, mm)"
}


def load_and_align_datasets() -> Dict[str, pd.DataFrame]:
    """Load per-case metric CSV files and verify case alignment."""
    files = {
        "RASNet (Ours)": RASNET_CSV,
        "SegResNet": SEGRESNET_CSV,
        "nnU-Net V2": NNUNET_CSV,
        "3D U-Net": UNET3D_CSV
    }
    
    dfs = {}
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required metric file for {name}: {path}")
        df = pd.read_csv(path)
        df["case_id"] = df["case_id"].astype(int)
        df = df.sort_values("case_id").reset_index(drop=True)
        dfs[name] = df
        print(f"Loaded {name}: {len(df)} cases (Cases {df['case_id'].min()} to {df['case_id'].max()})")
        
    # Verify alignment
    ref_cases = dfs["RASNet (Ours)"]["case_id"].tolist()
    for name, df in dfs.items():
        if df["case_id"].tolist() != ref_cases:
            raise ValueError(f"Case ID mismatch detected between RASNet and {name}!")
            
    return dfs


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply step-down Holm-Bonferroni adjustment to a list of p-values.
    Controls family-wise error rate (FWER) without parametric assumptions.
    """
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        k = m - rank
        adj_p = min(1.0, p * k)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max
        
    return adjusted


def get_significance_stars(p_val: float) -> str:
    """Return standard academic significance notation."""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "n.s."


def compute_wilcoxon_tests(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Perform paired two-sided Wilcoxon signed-rank tests for RASNet vs. each baseline.
    """
    rasnet_df = dfs["RASNet (Ours)"]
    baselines = ["SegResNet", "nnU-Net V2", "3D U-Net"]
    
    records = []
    p_values_to_correct = []
    
    for baseline_name in baselines:
        base_df = dfs[baseline_name]
        
        for metric in METRICS:
            r_vals = rasnet_df[metric].to_numpy()
            b_vals = base_df[metric].to_numpy()
            diff = r_vals - b_vals
            
            # Paired two-sided Wilcoxon signed-rank test
            # zero_method='wilcox' discards zero differences
            try:
                res = stats.wilcoxon(r_vals, b_vals, alternative='two-sided')
                stat = res.statistic
                p_val = res.pvalue
            except Exception as e:
                stat = np.nan
                p_val = 1.0
                
            r_mean = np.mean(r_vals)
            r_std = np.std(r_vals, ddof=1)
            b_mean = np.mean(b_vals)
            b_std = np.std(b_vals, ddof=1)
            delta_mean = r_mean - b_mean
            
            p_values_to_correct.append(p_val)
            records.append({
                "Comparison": f"RASNet vs. {baseline_name}",
                "Metric": metric,
                "Metric_Label": METRIC_NAMES[metric],
                "RASNet (Mean ± SD)": f"{r_mean:.4f} ± {r_std:.4f}",
                "Baseline (Mean ± SD)": f"{b_mean:.4f} ± {b_std:.4f}",
                "Delta (Mean)": f"{delta_mean:+.4f}",
                "Wilcoxon_W": stat,
                "Raw_p_value": p_val,
            })
            
    # Apply Holm-Bonferroni correction across all tests
    corrected_p = holm_bonferroni_correction(p_values_to_correct)
    
    for i, rec in enumerate(records):
        adj_p = corrected_p[i]
        rec["Holm_Corrected_p"] = adj_p
        rec["Significance"] = get_significance_stars(adj_p)
        
    res_df = pd.DataFrame(records)
    return res_df


def compute_bootstrap_ci(data: np.ndarray, n_boot: int = N_BOOTSTRAP, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute non-parametric bootstrap 95% confidence interval of the mean.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    boot_means = np.empty(n_boot)
    n = len(data)
    
    for b in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[b] = np.mean(sample)
        
    alpha = (1.0 - ci) / 2.0
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1.0 - alpha) * 100)
    orig_mean = np.mean(data)
    
    return orig_mean, lower, upper


def generate_bootstrap_table(rasnet_df: pd.DataFrame) -> pd.DataFrame:
    """Generate 95% Bootstrap CI table for all metrics of RASNet."""
    records = []
    for metric in METRICS:
        vals = rasnet_df[metric].to_numpy()
        mean_val, ci_low, ci_high = compute_bootstrap_ci(vals)
        std_val = np.std(vals, ddof=1)
        median_val = np.median(vals)
        iqr_val = np.percentile(vals, 75) - np.percentile(vals, 25)
        
        records.append({
            "Metric": metric,
            "Metric_Label": METRIC_NAMES[metric],
            "N": len(vals),
            "Mean": mean_val,
            "SD": std_val,
            "Bootstrap_95%_CI_Lower": ci_low,
            "Bootstrap_95%_CI_Upper": ci_high,
            "Mean (95% CI)": f"{mean_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]",
            "Median [IQR]": f"{median_val:.4f} [{iqr_val:.4f}]"
        })
    return pd.DataFrame(records)


def main():
    print("=" * 80)
    print("STEP 1: STATISTICAL SIGNIFICANCE TESTING & BOOTSTRAP UNCERTAINTY ESTIMATION")
    print("=" * 80)
    
    dfs = load_and_align_datasets()
    
    # 1. Wilcoxon Signed-Rank Test with Holm-Bonferroni Correction
    print("\nExecuting paired two-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction...")
    sig_df = compute_wilcoxon_tests(dfs)
    
    csv_sig_path = os.path.join(OUTPUT_DIR, "stats_significance_table.csv")
    md_sig_path = os.path.join(OUTPUT_DIR, "stats_significance_table.md")
    sig_df.to_csv(csv_sig_path, index=False)
    
    # Create clean Markdown table
    md_cols = [
        "Comparison", "Metric_Label", "RASNet (Mean ± SD)",
        "Baseline (Mean ± SD)", "Delta (Mean)", "Raw_p_value", "Holm_Corrected_p", "Significance"
    ]
    md_display_df = sig_df[md_cols].copy()
    md_display_df["Raw_p_value"] = md_display_df["Raw_p_value"].apply(lambda p: f"{p:.4e}" if p < 0.0001 else f"{p:.4f}")
    md_display_df["Holm_Corrected_p"] = md_display_df["Holm_Corrected_p"].apply(lambda p: f"{p:.4e}" if p < 0.0001 else f"{p:.4f}")
    md_display_df.columns = [
        "Comparison", "Metric", "RASNet (Mean ± SD)",
        "Baseline (Mean ± SD)", "Δ Mean", "Raw p-value", "Holm-Adj p-value", "Sig."
    ]
    
    with open(md_sig_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Statistical Significance Testing: RASNet vs. SOTA Baselines\n\n")
        f.write(f"- **Test Cohort**: ImageCAS Reserved Test Set ($N=150$, Cases 851–1000)\n")
        f.write("- **Statistical Test**: Paired Two-Sided Wilcoxon Signed-Rank Test (`scipy.stats.wilcoxon`)\n")
        f.write("- **Multiple Hypothesis Adjustment**: Step-Down Holm-Bonferroni Family-Wise Correction\n")
        f.write("- **Significance Thresholds**: `***` $p < 0.001$, `**` $p < 0.01$, `*` $p < 0.05$, `n.s.` $p \\ge 0.05$\n\n")
        f.write(md_display_df.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write("*Note: Statistical testing performed on paired per-case outputs across identical physical 3D coordinate volumes.*\n")
        
    print(f"Saved: {csv_sig_path}")
    print(f"Saved: {md_sig_path}")
    
    # 2. Non-Parametric Bootstrap 95% Confidence Intervals
    print("\nComputing Non-Parametric Bootstrap 95% Confidence Intervals (N=2000)...")
    boot_df = generate_bootstrap_table(dfs["RASNet (Ours)"])
    
    csv_boot_path = os.path.join(OUTPUT_DIR, "bootstrap_CI_table.csv")
    md_boot_path = os.path.join(OUTPUT_DIR, "bootstrap_CI_table.md")
    boot_df.to_csv(csv_boot_path, index=False)
    
    with open(md_boot_path, "w", encoding="utf-8") as f:
        f.write("# 📈 RASNet Performance with Non-Parametric 95% Bootstrap Confidence Intervals\n\n")
        f.write(f"- **Evaluation Dataset**: ImageCAS Test Split ($N=150$ Cases)\n")
        f.write(f"- **Bootstrap Resamples**: $B = {N_BOOTSTRAP}$ iterations (Percentile Method, Seed = {RANDOM_SEED})\n\n")
        display_boot = boot_df[["Metric_Label", "N", "Mean (95% CI)", "Median [IQR]"]].copy()
        display_boot.columns = ["Evaluation Metric", "N", "Mean (95% Bootstrap CI)", "Median [IQR]"]
        f.write(display_boot.to_markdown(index=False))
        f.write("\n")
        
    print(f"Saved: {csv_boot_path}")
    print(f"Saved: {md_boot_path}")
    print("\nSummary of Key Significance Findings:")
    for _, row in md_display_df.iterrows():
        print(f"  [{row['Sig.']:>4}] {row['Comparison']:<30} | {row['Metric']:<35} | p = {row['Holm-Adj p-value']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
