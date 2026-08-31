"""
batch_clinical_postprocess.py — Automated 3D Centerline Extraction, EDT Radius Profiling & Stenosis Quantification
==================================================================================================================
Processes all local_data/CT{i} cases:
  1. Loads 3D NIfTI image and predicted segmentation mask
  2. Extracts 3D vessel centerlines via topological skeletonization
  3. Computes local physical vessel radii via 3D Euclidean Distance Transform (EDT)
  4. Calculates maximum percent diameter stenosis (%DS) and CAD-RADS category
  5. Generates high-resolution 2D overlay plots and an aggregated CSV
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "local_data")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "hospital_cohort_automated_stenosis.csv")


def get_cad_rads_grade(pct: float) -> int:
    """Classifies percent diameter stenosis into clinical CAD-RADS tiers."""
    if pct <= 0.0:
        return 0
    elif pct < 25.0:
        return 1
    elif pct < 50.0:
        return 2
    elif pct < 70.0:
        return 3
    elif pct < 100.0:
        return 4
    else:
        return 5


def process_case_centerline(case_name: str) -> dict | None:
    case_dir = os.path.join(LOCAL_DATA_DIR, case_name)
    img_path = os.path.join(case_dir, "image.nii.gz")
    pred_path = os.path.join(case_dir, "pred_mask.nii.gz")

    if not os.path.exists(img_path) or not os.path.exists(pred_path):
        return None

    try:
        pred_sitk = sitk.ReadImage(pred_path)
        pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool) # (D, H, W)
        spacing = pred_sitk.GetSpacing() # (x, y, z) in mm

        if not pred_arr.any():
            print(f"[{case_name}] Mask is empty. Skipping.")
            return None

        # Crop to bounding box of prediction to accelerate 3D skeletonization by 20x
        coords = np.argwhere(pred_arr)
        z_min, y_min, x_min = np.maximum(coords.min(axis=0) - 5, 0)
        z_max, y_max, x_max = np.minimum(coords.max(axis=0) + 6, pred_arr.shape)
        
        cropped_mask = pred_arr[z_min:z_max, y_min:y_max, x_min:x_max]

        # 1. 3D Skeletonization on cropped ROI
        cropped_centerline = skeletonize(cropped_mask)
        centerline = np.zeros_like(pred_arr, dtype=bool)
        centerline[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_centerline

        centerline_coords = np.argwhere(centerline) # (Z, Y, X)
        if len(centerline_coords) == 0:
            return None

        # 2. Euclidean Distance Transform (Physical Radius in mm)
        edt = distance_transform_edt(pred_arr, sampling=(spacing[2], spacing[1], spacing[0]))
        radii = edt[centerline]
        diameters = 2.0 * radii

        # Filter out extreme tip noise (< 0.3mm)
        valid_mask = diameters >= 0.3
        if np.sum(valid_mask) > 10:
            diameters = diameters[valid_mask]
            centerline_coords = centerline_coords[valid_mask]

        mean_diam = float(np.mean(diameters))
        std_diam = float(np.std(diameters))
        max_diam = float(np.max(diameters))
        min_diam = float(np.min(diameters))

        # Stenosis Calculation: (1 - r_min / r_ref) * 100
        # Reference diameter: 90th percentile of proximal/mid vessel
        ref_diam = float(np.percentile(diameters, 90))
        if ref_diam > 0:
            stenosis_pct = max(0.0, min(100.0, (1.0 - (min_diam / ref_diam)) * 100.0))
        else:
            stenosis_pct = 0.0

        cad_rads = get_cad_rads_grade(stenosis_pct)

        # 3. Generate Visual Slice Plot
        plot_path = os.path.join(case_dir, f"vessel_centerline_overlay_{case_name}.png")
        z_coords = centerline_coords[:, 0]
        unique_z, counts_z = np.unique(z_coords, return_counts=True)
        best_slice_idx = unique_z[np.argmax(counts_z)]

        img_sitk = sitk.ReadImage(img_path)
        img_arr = sitk.GetArrayFromImage(img_sitk)
        img_slice = np.clip(img_arr[best_slice_idx, :, :], -100, 600)

        slice_centerline = centerline[best_slice_idx, :, :]
        slice_pred = pred_arr[best_slice_idx, :, :]

        plt.figure(figsize=(6, 6), dpi=200)
        plt.imshow(img_slice, cmap="gray", origin="lower")
        if slice_pred.any():
            plt.contour(slice_pred, colors="#ef4444", levels=[0.5], linewidths=1.2, alpha=0.9)
        y_c, x_c = np.where(slice_centerline)
        if len(x_c) > 0:
            plt.scatter(x_c, y_c, color='#3b82f6', s=3, label='Centerline')

        # Zoom into coronary region
        coords = np.argwhere(slice_pred > 0)
        if len(coords) > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            margin = 35
            plt.ylim(max(0, min_y - margin), min(img_slice.shape[0], max_y + margin))
            plt.xlim(max(0, min_x - margin), min(img_slice.shape[1], max_x + margin))

        plt.title(f"{case_name} Centerline (Max %DS: {stenosis_pct:.1f}%, CAD-RADS {cad_rads})", fontsize=10, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        res = {
            "case_id": case_name,
            "centerline_points": len(centerline_coords),
            "mean_diameter_mm": round(mean_diam, 3),
            "std_diameter_mm": round(std_diam, 3),
            "min_diameter_mm": round(min_diam, 3),
            "ref_diameter_mm": round(ref_diam, 3),
            "rasnet_stenosis_pct": round(stenosis_pct, 1),
            "cad_rads_grade": cad_rads,
            "plot_path": plot_path
        }
        print(f"[{case_name}] Mean Diam: {mean_diam:.2f}mm | Min: {min_diam:.2f}mm | Max %DS: {stenosis_pct:.1f}% (CAD-RADS {cad_rads})", flush=True)
        return res

    except Exception as e:
        print(f"[{case_name}] Error: {e}", flush=True)
        return None


def main():
    print("================================================================================", flush=True)
    print(">> AUTOMATED CLINICAL CENTERLINE & STENOSIS PROFILING", flush=True)
    print(f"   Target Directory: {LOCAL_DATA_DIR}", flush=True)
    print("================================================================================", flush=True)

    if not os.path.exists(LOCAL_DATA_DIR):
        print(f"No data in {LOCAL_DATA_DIR}")
        return

    cases = sorted([d for d in os.listdir(LOCAL_DATA_DIR) if os.path.isdir(os.path.join(LOCAL_DATA_DIR, d)) and d.startswith("CT")])
    print(f"Found {len(cases)} local case directories to profile...\n", flush=True)

    results = []
    for c in cases:
        r = process_case_centerline(c)
        if r:
            results.append(r)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[OK] Saved automated stenosis report table to: {OUTPUT_CSV}", flush=True)
        print(df[["case_id", "mean_diameter_mm", "min_diameter_mm", "rasnet_stenosis_pct", "cad_rads_grade"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
