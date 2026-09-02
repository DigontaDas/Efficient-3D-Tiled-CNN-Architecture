import pandas as pd
import numpy as np

CSV_PATH = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"
df = pd.read_csv(CSV_PATH)

df["abs_diff"] = df["difference"].abs()
df_sorted = df.sort_values(by="abs_diff", ascending=False).reset_index(drop=True)

print("=" * 85)
print("TOP OUTLIERS IN HOSPITAL COHORT (SORTED BY |DIFFERENCE| DESCENDING):")
print("=" * 85)
print(f"{'Rank':<5} | {'Case ID':<8} | {'Rad %DS':<9} | {'AI %DS':<9} | {'Diff %':<8} | {'Target Location':<18} | {'Rad CAD-RADS':<20}")
print("-" * 85)

for i in range(min(8, len(df_sorted))):
    row = df_sorted.iloc[i]
    print(f"#{i+1:<4} | {row['case_id']:<8} | {row['radiologist_stenosis_pct']:<9.1f} | {row['rasnet_stenosis_pct']:<9.1f} | {row['difference']:+7.1f}% | {str(row['lesion_location']):<18} | {str(row['radiologist_cad_rads']):<20}")

print("=" * 85)
