"""
audit_case_provenance.py

Performs the formal Step 0 Provenance Audit across all 32 cases in the Ibrahim Cardiac Hospital cohort.
Checks:
1. Patient Identifier visible on screenshot and matching DICOM patient
2. PACS caliper measurement or CAD-RADS badge visible and legible
Enforces strict quarantine: Any case failing either check is quarantined to the 150-case expansion.
"""

import os
import pandas as pd

AUDIT_DATA = [
    {"case_id": "CT1",  "id_visible": "no",  "badge_visible": "no",  "verified": "no",  "reason": "Excluded: No screenshot folder exists in labeled root"},
    {"case_id": "CT4",  "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: EMRUL KAYES (16020889), Badges 70-99% & 50-69%"},
    {"case_id": "CT5",  "id_visible": "no",  "badge_visible": "yes", "verified": "no",  "reason": "Excluded: Patient mismatch (DICOM: MD.ASHEQUR RAHMAN vs Screenshot: MD ENAMUL HAQUE)"},
    {"case_id": "CT61", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: MASUMA AKTAR (14100448), Badge 50-69%"},
    {"case_id": "CT62", "id_visible": "yes", "badge_visible": "no",  "verified": "no",  "reason": "Excluded: Patient verified (RAZIA SULTANA) but no stenosis badge exists (unbadged patent)"},
    {"case_id": "CT63", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: MD AFZALUR RAHMAN (MR251004225), Badge 70-99%"},
    {"case_id": "CT64", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: DR.MAHBUBUR RAHMAN (MR251004314), Badge 70-99%"},
    {"case_id": "CT65", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: TASLIMA AKHTER (25100453), Badge 25-49%"},
    {"case_id": "CT66", "id_visible": "no",  "badge_visible": "yes", "verified": "no",  "reason": "Excluded: Badge visible (25-49%) but patient ID header cropped off"},
    {"case_id": "CT67", "id_visible": "yes", "badge_visible": "no",  "verified": "no",  "reason": "Excluded: Patient verified (RATNA PARVEEN) but no badge exists (crosshair only)"},
    {"case_id": "CT68", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: MD ABDUS SALAM (18090938), Badge 70-99%"},
    {"case_id": "CT70", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: ROWSHAN ARA BEGUM (16090010), Badge 90-99%"},
    {"case_id": "CT71", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: M ZIAUL HOQE (MR251013147), Badge 25-49%"},
    {"case_id": "CT72", "id_visible": "no",  "badge_visible": "yes", "verified": "no",  "reason": "Excluded: Badge visible (50-69%) but patient ID header cropped off"},
    {"case_id": "CT73", "id_visible": "no",  "badge_visible": "yes", "verified": "no",  "reason": "Excluded: Badges visible (50-69%, 25-49%) but patient ID header cropped off"},
    {"case_id": "CT74", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: DIPTI ROY (25102441), Badge 25-49%"},
    {"case_id": "CT75", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: MIR ASADUZZAMAN CHOWDHURY (MR251012323), Badge 50-69%"},
    {"case_id": "CT76", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: MD HASIBUL ISLAM (MR250814465), Badge 50-69%"},
    {"case_id": "CT77", "id_visible": "no",  "badge_visible": "yes", "verified": "no",  "reason": "Excluded: Badges visible (70-99%) but patient ID header cropped off on all screenshots"},
    {"case_id": "CT78", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: MD AMINUR (MR251011415), Badge 90-99%"},
    {"case_id": "CT79", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: MRS.JHORNA (MR251010886), Badge 25-49%"},
    {"case_id": "CT80", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: MD.ANWAR HOSSAIN (MR251010840), Badge 90-99%"},
    {"case_id": "CT81", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: REBAKA SULTANA (17120535), Badge 70-99%"},
    {"case_id": "CT82", "id_visible": "yes", "badge_visible": "no",  "verified": "no",  "reason": "Excluded: Patient verified (MRS. SHIRIN AKHTER) but no badge exists (unbadged patent)"},
    {"case_id": "CT83", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: SHAH MUHAMMAD AMIRUZZAMAN (MR251010297), Badge 90-99%"},
    {"case_id": "CT84", "id_visible": "no",  "badge_visible": "yes", "verified": "no",  "reason": "Excluded: Badge visible (50-69%) but patient ID header cropped off"},
    {"case_id": "CT85", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: MD MOHIUDDIN (25100858), Badge 70-99%"},
    {"case_id": "CT86", "id_visible": "yes", "badge_visible": "no",  "verified": "no",  "reason": "Excluded: Patient verified (MD SAYDUZZAMAN) but no badge exists (unbadged reformat)"},
    {"case_id": "CT87", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: SALMA RAHMAN (25100872), Badge 25-49%"},
    {"case_id": "CT88", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: KOHINUR BEGUM (25110953), Badge 50-69%"},
    {"case_id": "CT89", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: ABDUL KAYUM (MR251105307), Badge 70-99%"},
    {"case_id": "CT90", "id_visible": "yes", "badge_visible": "yes", "verified": "yes", "reason": "Verified: MUSTAFIZUR RAHMAN (25110644), Badge 25-49% on 90.3.png"}
]

def main():
    df = pd.DataFrame(AUDIT_DATA)
    out_dir = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation"
    os.makedirs(out_dir, exist_ok=True)
    
    audit_csv = os.path.join(out_dir, "provenance_audit_results.csv")
    df.to_csv(audit_csv, index=False)
    print(f"Saved audit to {audit_csv}")
    
    verified_df = df[df['verified'] == 'yes'].copy()
    verified_csv = os.path.join(out_dir, "verified_cohort_cases.csv")
    verified_df.to_csv(verified_csv, index=False)
    print(f"Saved {len(verified_df)} verified cases to {verified_csv}")
    
    unverified_df = df[df['verified'] == 'no'].copy()
    unverified_csv = os.path.join(out_dir, "quarantined_cohort_cases.csv")
    unverified_df.to_csv(unverified_csv, index=False)
    print(f"Saved {len(unverified_df)} quarantined cases to {unverified_csv}")
    
    print("\n" + "="*80)
    print("STEP 0 PROVENANCE AUDIT SUMMARY")
    print("="*80)
    print(f"Total Cohort Scans Inspected: {len(df)}")
    print(f"VERIFIED Cases: {len(verified_df)}")
    print(f"QUARANTINED Cases (Pending Expansion): {len(unverified_df)}")
    print("="*80)
    
    print("\nQuarantined Cases Detail:")
    for _, row in unverified_df.iterrows():
        print(f"  - {row['case_id']}: {row['reason']}")

if __name__ == "__main__":
    main()
