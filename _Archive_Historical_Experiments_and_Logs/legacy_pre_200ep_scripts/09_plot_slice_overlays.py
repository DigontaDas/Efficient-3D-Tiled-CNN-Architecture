"""
Task 3.2 — Qualitative Slice Overlays
=======================================
Generates a 3-panel qualitative overlay figure showing the axial slices with
contours of the ground-truth (green) and SegResNet prediction (red).
Uses the representative cases:
  - Best-case: Case 934 (Dice = 0.8516)
  - Median-case: Case 974 (Dice = 0.7887)
  - Worst-case: Case 931 (Dice = 0.5538)
Saves the figure to c:\\Thesis_RASNET\\Thesis_Trainings\\Thesis_Trainings\\Final_Generated_assets\\slice_overlays.png.

Run with:
    c:\\Thesis_RASNET\\Thesis_Trainings\\Thesis_Trainings\\.venv_cuda\\Scripts\\python.exe Phase3_Local_Integration\\09_plot_slice_overlays.py
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR = os.path.join(ROOT_DIR, "Final_Generated_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Add current directory to path for dataset_paths
sys.path.insert(0, os.path.dirname(__file__))
import dataset_paths

PRED_DIR = os.path.join(ROOT_DIR, "all_four_validations", "mandatory_artifacts_segresnet", "predictions", "test")

# Representative cases
cases = {
    "Best Case (Case 934, DSC = 0.852)": {"id": 934, "dice": 0.8516},
    "Median Case (Case 974, DSC = 0.789)": {"id": 974, "dice": 0.7887},
    "Worst Case (Case 931, DSC = 0.554)": {"id": 931, "dice": 0.5538},
}


def main() -> None:
    print("Generating qualitative slice overlays...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=300)
    
    for ax_idx, (title, info) in enumerate(cases.items()):
        case_id = info["id"]
        ax = axes[ax_idx]
        
        # Load image, label, and prediction
        img_path = dataset_paths.find_image(case_id)
        gt_path = dataset_paths.find_label(case_id)
        pred_path = os.path.join(PRED_DIR, f"{case_id}.nii.gz")
        
        if not all(os.path.exists(p) for p in [img_path, gt_path, pred_path]):
            print(f"[WARNING] Missing files for case {case_id}, skipping panel.")
            ax.text(0.5, 0.5, "Missing Data", ha="center", va="center")
            continue
            
        img_sitk = sitk.ReadImage(img_path)
        gt_sitk = sitk.ReadImage(gt_path)
        pred_sitk = sitk.ReadImage(pred_path)
        
        # Convert to numpy arrays (Z, Y, X)
        img_arr = sitk.GetArrayFromImage(img_sitk)
        gt_arr = sitk.GetArrayFromImage(gt_sitk)
        pred_arr = sitk.GetArrayFromImage(pred_sitk)
        
        # Clip intensities for better contrast windowing (soft tissue / vessels window)
        # e.g., window around [-100, 600] HU
        img_clipped = np.clip(img_arr, -100, 600)
        
        # Find the slice with the maximum amount of vessel in the ground truth
        slice_sums = gt_arr.sum(axis=(1, 2))
        best_slice_idx = int(np.argmax(slice_sums))
        
        # Extract slices
        slice_img = img_clipped[best_slice_idx, :, :]
        slice_gt = gt_arr[best_slice_idx, :, :]
        slice_pred = pred_arr[best_slice_idx, :, :]
        
        # Display image slice
        ax.imshow(slice_img, cmap="gray", origin="lower")
        
        # Draw contours for GT (green) and Prediction (red)
        # We check if there's any foreground to draw contours
        if slice_gt.any():
            ax.contour(slice_gt, colors="#10b981", levels=[0.5], linewidths=1.5, alpha=0.95)
        if slice_pred.any():
            ax.contour(slice_pred, colors="#ef4444", levels=[0.5], linewidths=1.2, alpha=0.95)
            
        # Add legend indicators manually using proxy artists
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(edgecolor="#10b981", facecolor="none", label="Ground Truth"),
            Patch(edgecolor="#ef4444", facecolor="none", label="SegResNet Pred")
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.5)
        
        # Crop to a bounding box around the vessel to make it a zoom-in look, 
        # so the details of the coronary artery are visible rather than a tiny dot in 512x512
        coords = np.argwhere(slice_gt > 0)
        if len(coords) > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            # Add margin
            margin = 40
            y_start = max(0, min_y - margin)
            y_end = min(slice_img.shape[0], max_y + margin)
            x_start = max(0, min_x - margin)
            x_end = min(slice_img.shape[1], max_x + margin)
            
            ax.set_ylim(y_start, y_end)
            ax.set_xlim(x_start, x_end)
            
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        
    plt.suptitle("Qualitative Segmentation Comparison on ImageCAS Test Set", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(ASSETS_DIR, "slice_overlays.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Qualitative overlays plot successfully saved -> {output_path}")


if __name__ == "__main__":
    main()
