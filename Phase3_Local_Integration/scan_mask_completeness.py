import os
import sys
import SimpleITK as sitk
import numpy as np
import pandas as pd
import cc3d

LOCAL_DATA_DIR = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data"
gt_csv = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"
gt_df = pd.read_csv(gt_csv)

print(f"{'Case':<6} | {'Target Loc':<14} | {'Rad %DS':<8} | {'Mask Voxels':<12} | {'Num Comp':<10} | {'Top 2 Comp Sizes':<22} | {'Diagnosis'}")
print("-" * 105)

for idx, row in gt_df.iterrows():
    cname = row['case_id']
    target_loc = str(row['lesion_location'])
    rad_ds = row['radiologist_stenosis_pct']
    pred_path = os.path.join(LOCAL_DATA_DIR, cname, "pred_mask.nii.gz")
    if not os.path.exists(pred_path):
        print(f"{cname:<6} | {target_loc:<14} | {rad_ds:<8.1f} | MISSING MASK")
        continue
    
    pred_sitk = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(np.uint8)
    spacing = pred_sitk.GetSpacing()
    total_vox = int(np.sum(pred_arr))
    
    labeled, N = cc3d.connected_components(pred_arr, return_N=True)
    sizes = np.bincount(labeled.flat)[1:]
    top_sizes = sorted(sizes, reverse=True)[:2] if len(sizes) > 0 else []
    
    status = "Normal / Full"
    if total_vox < 600:
        status = "PARTIAL/TINY MASK (<600 vox)"
    elif len(top_sizes) == 1:
        if "RCA" in target_loc or "PDA" in target_loc:
            status = "MISSING RCA (Target is RCA, only 1 tree)"
        else:
            status = "Single Tree (LCA only)"
    elif len(top_sizes) >= 2:
        if top_sizes[1] < 150 and ("RCA" in target_loc or "PDA" in target_loc):
            status = "TINY SECOND TREE (<150 vox)"
            
    print(f"{cname:<6} | {target_loc:<14} | {rad_ds:<8.1f} | {total_vox:<12} | {N:<10} | {str(top_sizes):<22} | {status}")
