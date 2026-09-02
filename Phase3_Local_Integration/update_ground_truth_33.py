import os
import pandas as pd

AGREEMENT_CSV = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"

# Complete 33-case audited dataset mapping:
# case_id, radiologist_stenosis_pct, radiologist_cad_rads, lesion_location
audited_cases = [
    {"case_id": "CT1",  "radiologist_stenosis_pct": 10.0,  "radiologist_cad_rads": "CAD-RADS 1 (Minimal)",  "lesion_location": "LAD Patent"},
    {"case_id": "CT4",  "radiologist_stenosis_pct": 85.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "LCx"},
    {"case_id": "CT5",  "radiologist_stenosis_pct": 100.0, "radiologist_cad_rads": "CAD-RADS 5 (Occlusion)","lesion_location": "LAD Occlusion"},
    {"case_id": "CT61", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "RCA/PDA"},
    {"case_id": "CT62", "radiologist_stenosis_pct": 10.0,  "radiologist_cad_rads": "CAD-RADS 1 (Minimal)",  "lesion_location": "LAD Patent"},
    {"case_id": "CT63", "radiologist_stenosis_pct": 85.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "LCx Plaque"},
    {"case_id": "CT64", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "Mid-LAD"},
    {"case_id": "CT65", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "Prox-LAD"},
    {"case_id": "CT66", "radiologist_stenosis_pct": 37.5,  "radiologist_cad_rads": "CAD-RADS 2 (Mild)",     "lesion_location": "PDA/RCA"},
    {"case_id": "CT67", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "LAD/LCx"},
    {"case_id": "CT68", "radiologist_stenosis_pct": 85.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "LAD/LCx"},
    {"case_id": "CT70", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "RCA"},
    {"case_id": "CT71", "radiologist_stenosis_pct": 37.5,  "radiologist_cad_rads": "CAD-RADS 2 (Mild)",     "lesion_location": "LCx"},
    {"case_id": "CT72", "radiologist_stenosis_pct": 80.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "LAD"},
    {"case_id": "CT73", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "LCx/LAD"},
    {"case_id": "CT74", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "RCA"},
    {"case_id": "CT75", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "LAD"},
    {"case_id": "CT76", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "LAD"},
    {"case_id": "CT77", "radiologist_stenosis_pct": 85.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "LAD/RCA"},
    {"case_id": "CT78", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "LCx"},
    {"case_id": "CT79", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "LAD"},
    {"case_id": "CT80", "radiologist_stenosis_pct": 80.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "RCA"},
    {"case_id": "CT81", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "LAD"},
    {"case_id": "CT82", "radiologist_stenosis_pct": 10.0,  "radiologist_cad_rads": "CAD-RADS 1 (Minimal)",  "lesion_location": "LAD Patent"},
    {"case_id": "CT83", "radiologist_stenosis_pct": 80.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "LAD/LCx"},
    {"case_id": "CT84", "radiologist_stenosis_pct": 85.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "LAD"},
    {"case_id": "CT85", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "RCA"},
    {"case_id": "CT86", "radiologist_stenosis_pct": 60.0,  "radiologist_cad_rads": "CAD-RADS 3 (Moderate)", "lesion_location": "LAD"},
    {"case_id": "CT87", "radiologist_stenosis_pct": 80.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "LAD/RCA"},
    {"case_id": "CT88", "radiologist_stenosis_pct": 85.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "LAD"},
    {"case_id": "CT89", "radiologist_stenosis_pct": 85.0,  "radiologist_cad_rads": "CAD-RADS 4 (Severe)",   "lesion_location": "LAD"},
    {"case_id": "CT90", "radiologist_stenosis_pct": 37.5,  "radiologist_cad_rads": "CAD-RADS 2 (Mild)",     "lesion_location": "LCx"}
]

df = pd.DataFrame(audited_cases)
df["rasnet_stenosis_pct"] = 0.0
df["rasnet_cad_rads"] = "Pending"
df["difference"] = 0.0

df.to_csv(AGREEMENT_CSV, index=False)
print(f"Saved audited ground-truth dataset with {len(df)} cases to {AGREEMENT_CSV}")
