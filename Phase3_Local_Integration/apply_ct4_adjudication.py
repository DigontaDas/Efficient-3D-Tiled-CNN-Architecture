"""
apply_ct4_adjudication.py

Adjudicates CT4 ground-truth under SCCT CAD-RADS 2.0 worst-lesion rule:
Source: H:\Thesis_CT_scans_Labeled\CT4\4.2.png
Badges present:
1. Proximal LCx: 70-99% (Severe, midpoint 85.0%, CAD-RADS 4)
2. Distal LCx: 50-69% (Moderate, midpoint 60.0%, CAD-RADS 3)
Under CAD-RADS 2.0, the patient-level classification is governed by the highest-grade lesion (85.0%, CAD-RADS 4).
"""

import os
import pandas as pd

AGREEMENT_CSV = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"

def main():
    df = pd.read_csv(AGREEMENT_CSV)
    
    idx = df[df['case_id'] == 'CT4'].index[0]
    old_ds = df.loc[idx, 'radiologist_stenosis_pct']
    old_cad = df.loc[idx, 'radiologist_cad_rads']
    old_loc = df.loc[idx, 'lesion_location']
    
    print("=" * 70)
    print("STEP 4: CT4 GROUND TRUTH ADJUDICATION")
    print("=" * 70)
    print(f"Case: CT4")
    print(f"Previous GT: {old_ds}% | {old_cad} | Loc: {old_loc}")
    
    df.loc[idx, 'radiologist_stenosis_pct'] = 85.0
    df.loc[idx, 'radiologist_cad_rads'] = "CAD-RADS 4 (Severe)"
    df.loc[idx, 'lesion_location'] = "LCx (Proximal 70-99%)"
    
    print(f"Updated GT:  85.0% | CAD-RADS 4 (Severe) | Loc: LCx (Proximal 70-99%)")
    print("Clinical Rationale: CAD-RADS 2.0 standard mandates patient-level classification")
    print("by the maximal stenosis (4.2.png shows Proximal LCx 70-99% and Distal LCx 50-69%).")
    print("=" * 70)
    
    df.to_csv(AGREEMENT_CSV, index=False)
    print(f"Saved updated agreement CSV to: {AGREEMENT_CSV}")

if __name__ == "__main__":
    main()
