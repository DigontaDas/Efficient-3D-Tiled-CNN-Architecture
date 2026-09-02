import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from scipy.spatial import KDTree

LOCAL_DATA_DIR = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data"

def analyze_case(case_name):
    case_dir = os.path.join(LOCAL_DATA_DIR, case_name)
    img_path = os.path.join(case_dir, "image.nii.gz")
    pred_path = os.path.join(case_dir, "pred_mask.nii.gz")

    if not os.path.exists(pred_path):
        return None

    pred_sitk = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool)
    spacing = pred_sitk.GetSpacing() # (x, y, z) in mm

    # Crop to ROI
    coords = np.argwhere(pred_arr)
    z_min, y_min, x_min = np.maximum(coords.min(axis=0) - 5, 0)
    z_max, y_max, x_max = np.minimum(coords.max(axis=0) + 6, pred_arr.shape)
    cropped_mask = pred_arr[z_min:z_max, y_min:y_max, x_min:x_max]

    cropped_centerline = skeletonize(cropped_mask)
    centerline = np.zeros_like(pred_arr, dtype=bool)
    centerline[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_centerline
    centerline_coords = np.argwhere(centerline) # (Z, Y, X)

    if len(centerline_coords) == 0:
        return None

    # EDT in physical mm
    edt = distance_transform_edt(pred_arr, sampling=(spacing[2], spacing[1], spacing[0]))
    radii = edt[centerline]
    diameters = 2.0 * radii

    # Physical coordinates of centerline points
    phys_coords = centerline_coords * np.array([spacing[2], spacing[1], spacing[0]])

    # Build KDTree for spatial neighborhood
    kdt = KDTree(phys_coords)

    # Let's inspect local reference diameter in a 10 mm ball along centerline
    # For each point, find points within radius R (e.g. 10mm)
    # A true focal lesion has a constriction compared to its immediate healthy neighbors (within 10mm)
    # Also we should only evaluate points that belong to major/clinically evaluable vessels (e.g. where local reference >= 1.5mm)
    
    local_stenosis_list = []
    for i, p in enumerate(phys_coords):
        # find neighbors within 10 mm
        idx_neighbors = kdt.query_ball_point(p, r=10.0)
        neigh_diams = diameters[idx_neighbors]
        
        # Local normal reference is 90th percentile of this 10mm neighborhood
        local_ref = np.percentile(neigh_diams, 90)
        
        # If local reference is >= 1.5 mm (evaluable coronary vessel caliber according to SCCT guidelines)
        if local_ref >= 1.5:
            d_curr = diameters[i]
            pct_sten = max(0.0, (1.0 - (d_curr / local_ref)) * 100.0)
            local_stenosis_list.append((pct_sten, d_curr, local_ref))

    if len(local_stenosis_list) == 0:
        return {"case": case_name, "max_local_stenosis": 0.0, "p95_local_stenosis": 0.0}

    local_stenosis_arr = np.array([x[0] for x in local_stenosis_list])
    # Max and 95th percentile local stenosis
    max_sten = np.max(local_stenosis_arr)
    p95_sten = np.percentile(local_stenosis_arr, 95)
    p98_sten = np.percentile(local_stenosis_arr, 98)

    print(f"[{case_name}] Raw pts: {len(centerline_coords)} | Max Local %DS: {max_sten:.1f}% | P98 Local %DS: {p98_sten:.1f}% | P95 Local %DS: {p95_sten:.1f}%")
    return {
        "case": case_name,
        "max_local_stenosis": max_sten,
        "p98_local_stenosis": p98_sten,
        "p95_local_stenosis": p95_sten
    }

for c in ["CT61", "CT62", "CT63", "CT64", "CT65", "CT66", "CT67", "CT68", "CT70", "CT71", "CT72", "CT73", "CT74", "CT75", "CT76", "CT77"]:
    analyze_case(c)
