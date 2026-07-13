# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\11_clinical_postprocess.py
"""
Task 3.4 — Clinical Post-Processing Pipeline
==============================================
Simulates clinical vessel analysis workflow:
  1. Loads a 3D NIfTI segmentation prediction volume.
  2. Extracts the 3D centerline (skeletonization) using scikit-image morphology.
  3. Computes local vessel radius using the Euclidean Distance Transform (EDT).
  4. Identifies potential stenosis points (constrictions where local radius is < 50% of the mean).
  5. Generates a clinical analysis report and centerline visual plot.

Saves output assets to:
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\clinical_postprocess\
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# Output paths
OUTPUT_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\clinical_postprocess"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default prediction directory points to the custom RASNet predictions folder
DEFAULT_PRED_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\rasnet_development\predictions"

import argparse

def main():
    print("==================================================")
    print("      CLINICAL SKELETONIZATION & POST-PROCESS     ")
    print("==================================================")
    
    parser = argparse.ArgumentParser(description="Clinical skeletonization and post-processing")
    parser.add_argument("--case-id", type=int, default=934, help="Case ID to analyze")
    parser.add_argument("--pred-dir", type=str, default=DEFAULT_PRED_DIR, help="Directory containing predictions")
    args = parser.parse_args()
    
    test_case = args.case_id
    pred_path = os.path.join(args.pred_dir, f"{test_case}.nii.gz")
    
    # Handle possible fallback to segresnet if the target file isn't in rasnet yet
    if not os.path.exists(pred_path):
        fallback_dir = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_segresnet\predictions\test"
        pred_path_fallback = os.path.join(fallback_dir, f"{test_case}.nii.gz")
        if os.path.exists(pred_path_fallback):
            print(f"[INFO] Custom prediction for Case {test_case} not found in {args.pred_dir}. Falling back to SegResNet baseline predictions.")
            pred_path = pred_path_fallback
        else:
            print(f"[ERROR] Prediction mask for Case {test_case} not found in either {args.pred_dir} or {fallback_dir}.")
            return

    print(f"Loading prediction mask for analysis: Case {test_case} ({pred_path})")
    
    # Setup output file paths dynamically per case ID
    report_path = os.path.join(OUTPUT_DIR, f"clinical_vessel_report_{test_case}.md")
    plot_path = os.path.join(OUTPUT_DIR, f"vessel_centerline_extraction_{test_case}.png")
    pred_sitk = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool)
    spacing = pred_sitk.GetSpacing() # (x, y, z) in mm
    
    if not pred_arr.any():
        print("[WARNING] The prediction volume is empty. Skeletonization aborted.")
        return

    # 2. Extract 3D Centerline (Skeletonization)
    print("Extracting 3D vessel centerline (skeletonization)...")
    centerline = skeletonize(pred_arr)
    centerline_coords = np.argwhere(centerline) # coordinates in (Z, Y, X)
    print(f"Centerline extracted: {len(centerline_coords)} voxels.")

    # 3. Compute Euclidean Distance Transform (EDT) for local radii
    # EDT computes the distance from each foreground voxel to the nearest background voxel.
    # The EDT value at a centerline voxel represents the local radius of the vessel at that point.
    print("Computing Euclidean Distance Transform for vessel radius measurements...")
    # Scale by spacing to get physical distance in mm
    edt = distance_transform_edt(pred_arr, sampling=(spacing[2], spacing[1], spacing[0]))
    
    # Radii along the centerline
    radii = edt[centerline]
    diameters = 2 * radii # mm
    
    mean_diameter = np.mean(diameters)
    std_diameter = np.std(diameters)
    min_diameter = np.min(diameters)
    max_diameter = np.max(diameters)

    print(f"Vessel Diameter Stats:")
    print(f"  - Mean: {mean_diameter:.2f} mm")
    print(f"  - Min:  {min_diameter:.2f} mm")
    print(f"  - Max:  {max_diameter:.2f} mm")

    # 4. Clinical Stenosis Detection
    # A stenosis is marked where the local diameter is less than 50% of the mean diameter
    stenosis_threshold = 0.50 * mean_diameter
    stenosis_indices = np.where(diameters < stenosis_threshold)[0]
    stenosis_coords = centerline_coords[stenosis_indices]
    stenosis_pct = (len(stenosis_indices) / len(centerline_coords)) * 100

    print(f"Stenosis Detection (Threshold: < {stenosis_threshold:.2f} mm):")
    print(f"  - Detected stenosis points: {len(stenosis_indices)} voxels ({stenosis_pct:.2f}% of vessel length)")

    # 5. Save Report
    with open(report_path, "w") as f:
        f.write("# Clinical Coronary Artery Centerline Analysis Report\n\n")
        f.write(f"- **Patient Case ID**: {test_case}\n")
        f.write(f"- **Voxel Spacing (X, Y, Z)**: {spacing[0]:.4f} x {spacing[1]:.4f} x {spacing[2]:.4f} mm\n")
        f.write(f"- **Vessel Centerline Length**: {len(centerline_coords)} points ({len(centerline_coords)*spacing[0]:.2f} mm total length)\n\n")
        f.write("## Diameter Profile\n\n")
        f.write(f"- **Mean Diameter**: {mean_diameter:.3f} mm\n")
        f.write(f"- **Std Dev**: {std_diameter:.3f} mm\n")
        f.write(f"- **Max Diameter**: {max_diameter:.3f} mm\n")
        f.write(f"- **Min Diameter**: {min_diameter:.3f} mm\n\n")
        f.write("## Stenosis Analysis\n\n")
        f.write(f"- **Reference Mean Diameter**: {mean_diameter:.2f} mm\n")
        f.write(f"- **Stenosis Threshold (50% of Mean)**: < {stenosis_threshold:.2f} mm\n")
        f.write(f"- **Stenosis Detections**: {len(stenosis_indices)} points ({stenosis_pct:.2f}% of centerline)\n")
        if len(stenosis_indices) > 0:
            f.write("- **Diagnosis**: Significant constriction detected along vessel tree. Action recommended.\n")
        else:
            f.write("- **Diagnosis**: Normal vessel diameter profile. No significant stenotic regions detected.\n")
    print(f"[OK] Saved Clinical Report: {report_path}")

    # 6. Plot Visualization
    # Extract 2D projection or single slice to plot overlays
    # Find the axial slice with the maximum centerline points for clear representation
    z_coords = centerline_coords[:, 0]
    unique_z, counts_z = np.unique(z_coords, return_counts=True)
    best_slice_idx = unique_z[np.argmax(counts_z)]
    
    # Load original image slice for background
    sys.path.insert(0, os.path.dirname(__file__))
    import dataset_paths
    img_path = dataset_paths.find_image(test_case)
    img_sitk = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    img_clipped = np.clip(img_arr[best_slice_idx, :, :], -100, 600)

    # Get slices for centerline and prediction mask
    slice_centerline = centerline[best_slice_idx, :, :]
    slice_pred = pred_arr[best_slice_idx, :, :]
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    
    ax.imshow(img_clipped, cmap="gray", origin="lower")
    
    # Draw prediction contour (red)
    if slice_pred.any():
        ax.contour(slice_pred, colors="#ef4444", levels=[0.5], linewidths=1.2, alpha=0.9)
    
    # Draw centerline points (blue dots)
    y_c, x_c = np.where(slice_centerline)
    if len(x_c) > 0:
        ax.scatter(x_c, y_c, color='#3b82f6', s=3, label='Centerline', alpha=0.9)
    
    # Identify stenosis points on this specific slice and plot them (yellow stars)
    stenosis_in_slice = [c for c in stenosis_coords if c[0] == best_slice_idx]
    if len(stenosis_in_slice) > 0:
        ys_s = [c[1] for c in stenosis_in_slice]
        xs_s = [c[2] for c in stenosis_in_slice]
        ax.scatter(xs_s, ys_s, color='#eab308', marker='*', s=15, label='Detected Stenosis (< 50% Mean)', zorder=5)

    # Crop around the coronary artery bounding box for zoom look
    coords = np.argwhere(slice_pred > 0)
    if len(coords) > 0:
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        margin = 35
        ax.set_ylim(max(0, min_y - margin), min(img_clipped.shape[0], max_y + margin))
        ax.set_xlim(max(0, min_x - margin), min(img_clipped.shape[1], max_x + margin))

    ax.set_title(f"Vessel Centerline & Stenosis Overlay (Case {test_case}, Slice {best_slice_idx})", fontsize=10, fontweight="bold")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved Plot: {plot_path}")

    print("==================================================")
    print("[OK] Centerline extraction completed successfully.")
    print("==================================================")

if __name__ == "__main__":
    main()
