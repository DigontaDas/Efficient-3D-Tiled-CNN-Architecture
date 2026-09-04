"""
audit_3d_mask_completeness.py

Performs Step 3 Mask Completeness Audit across the verified cohort cases.
Measures:
- Total voxel count and physical volume (mm³)
- Physical bounding box extents (X, Y, Z in mm)
- Connected component distribution
- Coronary arterial branch identification (LAD, LCx, RCA) based on spatial geometry
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label

LOCAL_DATA_DIR = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data"
VERIFIED_CSV = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\verified_cohort_cases.csv"
OUT_CSV = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\mask_completeness_audit.csv"

def audit_case_mask(case_id):
    mask_path = os.path.join(LOCAL_DATA_DIR, case_id, "pred_mask.nii.gz")
    if not os.path.exists(mask_path):
        return None
        
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img).astype(bool)
    sp = img.GetSpacing() # (sx, sy, sz) in mm
    vox_vol = sp[0] * sp[1] * sp[2]
    
    vox_count = int(np.sum(arr))
    vol_mm3 = round(vox_count * vox_vol, 1)
    
    if vox_count == 0:
        return {
            "case_id": case_id,
            "voxel_count": 0,
            "volume_mm3": 0.0,
            "extent_x_mm": 0.0,
            "extent_y_mm": 0.0,
            "extent_z_mm": 0.0,
            "num_cc": 0,
            "lad_present": False,
            "lcx_present": False,
            "rca_present": False,
            "status": "EMPTY_MASK"
        }
        
    coords = np.argwhere(arr) # (z, y, x)
    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)
    
    extent_x = round((max_x - min_x + 1) * sp[0], 1)
    extent_y = round((max_y - min_y + 1) * sp[1], 1)
    extent_z = round((max_z - min_z + 1) * sp[2], 1)
    
    # Connected components
    labeled, num_cc = label(arr)
    
    # Anatomical system identification:
    # Coronary anatomy in standard axial/cardiac orientation:
    # RCA is right-sided (lower x in voxel space or depending on orientation),
    # LAD is anterior (anterior descending), LCx is posterior/lateral.
    # We partition based on midpoints:
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    
    # Substantial components (>50 voxels)
    rca_vox = np.sum(coords[:, 2] < mid_x)
    lad_vox = np.sum((coords[:, 2] >= mid_x) & (coords[:, 1] < mid_y))
    lcx_vox = np.sum((coords[:, 2] >= mid_x) & (coords[:, 1] >= mid_y))
    
    lad_present = lad_vox > 200
    lcx_present = lcx_vox > 150
    rca_present = rca_vox > 200
    
    branches = []
    if lad_present: branches.append("LAD")
    if lcx_present: branches.append("LCx")
    if rca_present: branches.append("RCA")
    
    status = "COMPLETE_TREE" if len(branches) >= 3 else ("PARTIAL_TREE" if len(branches) >= 2 else "TRUNCATED")
    
    return {
        "case_id": case_id,
        "voxel_count": vox_count,
        "volume_mm3": vol_mm3,
        "extent_x_mm": extent_x,
        "extent_y_mm": extent_y,
        "extent_z_mm": extent_z,
        "num_cc": int(num_cc),
        "lad_present": lad_present,
        "lcx_present": lcx_present,
        "rca_present": rca_present,
        "branches_detected": "+".join(branches),
        "status": status
    }

def main():
    ver_df = pd.read_csv(VERIFIED_CSV)
    cases = ver_df['case_id'].tolist()
    
    print("=" * 90)
    print("STEP 3: 3D MASK COMPLETENESS AUDIT (VERIFIED COHORT)")
    print("=" * 90)
    
    rows = []
    for cid in cases:
        r = audit_case_mask(cid)
        if r:
            rows.append(r)
            
    df = pd.DataFrame(rows)
    df = df.sort_values(by="voxel_count").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved audit results to: {OUT_CSV}\n")
    
    print(f"{'Case':<6} | {'Voxels':<8} | {'Vol(mm³)':<9} | {'Extent (X,Y,Z mm)':<20} | {'Branches':<12} | {'Status'}")
    print("-" * 90)
    for _, r in df.iterrows():
        ext = f"{r['extent_x_mm']}x{r['extent_y_mm']}x{r['extent_z_mm']}"
        print(f"{r['case_id']:<6} | {r['voxel_count']:<8d} | {r['volume_mm3']:<9.1f} | {ext:<20s} | {r['branches_detected']:<12s} | {r['status']}")

if __name__ == "__main__":
    main()
