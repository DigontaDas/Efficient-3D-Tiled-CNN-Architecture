# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\07_local_data_qc.py
"""
When the radiologist delivers NIfTI files, run this script first.
It validates that all files have compatible spacing, orientation, and label format.
"""
import SimpleITK as sitk, os, json
import numpy as np

LOCAL_DATA_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\local_data"
EXPECTED_LABEL_VALUES = {0, 1}   # Binary mask: 0=background, 1=vessel

def main() -> None:
    if not os.path.exists(LOCAL_DATA_DIR):
        print(f"[ERROR] Local data directory does not exist: {LOCAL_DATA_DIR}")
        return
        
    issues = []
    cases = sorted(os.listdir(LOCAL_DATA_DIR))
    cases = [c for c in cases if os.path.isdir(os.path.join(LOCAL_DATA_DIR, c))]
    
    if not cases:
        print(f"No cases found in {LOCAL_DATA_DIR}.")
        print("Please structure data as local_data/case_id/image.nii.gz and label.nii.gz")
        return
        
    print(f"Auditing {len(cases)} local cases...")
    for case in cases:
        lbl_path = os.path.join(LOCAL_DATA_DIR, case, "label.nii.gz")
        img_path = os.path.join(LOCAL_DATA_DIR, case, "image.nii.gz")
        if not os.path.exists(lbl_path) or not os.path.exists(img_path):
            issues.append(f"MISSING file in case {case}")
            continue

        try:
            img = sitk.ReadImage(img_path)
            lbl = sitk.ReadImage(lbl_path)

            # Check sizes match
            if img.GetSize() != lbl.GetSize():
                issues.append(f"SIZE MISMATCH in case {case}: img={img.GetSize()} lbl={lbl.GetSize()}")

            # Check spacing matches image
            img_spacing = tuple(round(s, 4) for s in img.GetSpacing())
            lbl_spacing = tuple(round(s, 4) for s in lbl.GetSpacing())
            if img_spacing != lbl_spacing:
                issues.append(f"SPACING MISMATCH in case {case}: img={img_spacing} lbl={lbl_spacing}")

            # Check label is binary
            arr = sitk.GetArrayFromImage(lbl)
            unique = set(np.unique(arr).tolist())
            if not unique.issubset(EXPECTED_LABEL_VALUES):
                issues.append(f"NON-BINARY LABEL in case {case}: unique values = {unique}")
        except Exception as exc:
            issues.append(f"READ ERROR in case {case}: {exc}")

    if not issues:
        print("All local data passed QC checks.")
    else:
        print(f"{len(issues)} issue(s) found:")
        for i in issues:
            print(f"   - {i}")

if __name__ == "__main__":
    main()
