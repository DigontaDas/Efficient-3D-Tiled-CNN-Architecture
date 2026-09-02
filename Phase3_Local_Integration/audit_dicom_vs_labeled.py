import pydicom
import os
import pandas as pd

DICOM_ROOT = r"H:\CT_Scans_Thesis"
LABELED_ROOT = r"H:\Thesis_CT_scans_Labeled"
gt_csv = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"
gt_df = pd.read_csv(gt_csv)

print(f"{'Case':<6} | {'DICOM Patient Name':<24} | {'Patient ID':<14} | {'StudyDate':<10} | {'CSV Target':<14} | {'CSV Rad %':<10} | {'Screenshots'}")
print("-" * 110)

for cid in range(61, 91):
    cname = f"CT{cid}"
    dicom_folder = os.path.join(DICOM_ROOT, f"CT {cid}", "A")
    labeled_folder = os.path.join(LABELED_ROOT, cname)
    
    pname, pid, sdate = "MISSING", "MISSING", "MISSING"
    if os.path.exists(dicom_folder):
        files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if os.path.isfile(os.path.join(dicom_folder, f))]
        if files:
            try:
                ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
                pname = str(getattr(ds, "PatientName", "UNKNOWN"))
                pid = str(getattr(ds, "PatientID", "UNKNOWN"))
                sdate = str(getattr(ds, "StudyDate", "UNKNOWN"))
            except Exception:
                pass
                
    scr_files = os.listdir(labeled_folder) if os.path.exists(labeled_folder) else []
    
    row = gt_df[gt_df["case_id"] == cname]
    csv_target = str(row["lesion_location"].values[0]) if len(row) > 0 else "N/A"
    csv_rad = str(row["radiologist_stenosis_pct"].values[0]) if len(row) > 0 else "N/A"
    
    print(f"{cname:<6} | {pname:<24} | {pid:<14} | {sdate:<10} | {csv_target:<14} | {csv_rad:<10} | {str(scr_files)}")
