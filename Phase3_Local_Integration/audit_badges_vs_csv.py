import os
import pydicom
import pandas as pd
from match_all_badges import results as badge_results

DICOM_ROOT = r"H:\CT_Scans_Thesis"
LABELED_ROOT = r"H:\Thesis_CT_scans_Labeled"
gt_csv = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"
old_df = pd.read_csv(gt_csv)

print(f"{'Case':<6} | {'DICOM Patient':<20} | {'Old CSV %':<10} | {'True Badges':<22} | {'True Rad % Range'} | {'AI %DS':<8}")
print("-" * 95)

audited_rows = []

# Mapping from CAD-RADS badge to standard clinical stenosis percentage
badge_map = {
    "25-49%": 37.5,
    "50-69%": 60.0,
    "70-99%": 85.0
}

for cid in range(61, 91):
    cname = f"CT{cid}"
    dicom_folder = os.path.join(DICOM_ROOT, f"CT {cid}", "A")
    pname = "UNKNOWN"
    if os.path.exists(dicom_folder):
        files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if os.path.isfile(os.path.join(dicom_folder, f))]
        if files:
            try:
                ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
                pname = str(getattr(ds, "PatientName", "UNKNOWN"))
            except Exception:
                pass
                
    badges = list(set(badge_results.get(cname, [])))
    
    old_row = old_df[old_df["case_id"] == cname]
    old_pct = float(old_row["radiologist_stenosis_pct"].values[0]) if len(old_row) > 0 else 0.0
    ai_pct = float(old_row["rasnet_stenosis_pct"].values[0]) if len(old_row) > 0 else 0.0
    target_loc = str(old_row["lesion_location"].values[0]) if len(old_row) > 0 else "LAD"
    
    # Determine true radiologist ground truth percentage
    if len(badges) == 0:
        if "PATENT" in target_loc.upper() or cname in ["CT62", "CT82"]:
            true_pct = 10.0
            badge_str = "None (Patent)"
        else:
            true_pct = old_pct
            badge_str = "Unlabeled/Text"
    elif "70-99%" in badges:
        true_pct = 85.0
        badge_str = "70-99% (Severe)"
    elif "50-69%" in badges:
        true_pct = 60.0
        badge_str = "50-69% (Moderate)"
    elif "25-49%" in badges:
        true_pct = 37.5
        badge_str = "25-49% (Mild)"
    else:
        true_pct = old_pct
        badge_str = str(badges)
        
    audited_rows.append({
        "case_id": cname,
        "patient_name": pname,
        "old_csv_pct": old_pct,
        "true_pct": true_pct,
        "badge_str": badge_str,
        "ai_pct": ai_pct,
        "target_loc": target_loc
    })
    print(f"{cname:<6} | {pname:<20} | {old_pct:<10.1f} | {badge_str:<22} | {true_pct:<18.1f} | {ai_pct:<8.1f}")
