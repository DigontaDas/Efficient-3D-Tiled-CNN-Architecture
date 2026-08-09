# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\run_postprocess_after_fix.py
r"""
RASNet Post-Processing & Visualization Runner (Post-Hallucination Fix).
Generates qualitative slice overlays and clinical centerline stenosis reports for target cases
(e.g., 851, 860, 900, 920, 934) saved directly in:
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\results-after-hallucin-fix\
"""
import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from skimage.morphology import skeletonize
import dataset_paths

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "results-after-hallucin-fix")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "qualitative_overlays")
CLINICAL_DIR = os.path.join(OUTPUT_DIR, "clinical_postprocess")

def generate_overlays(target_cases=[851, 860, 900, 920, 934]):
    print("==================================================")
    print("  QUALITATIVE SLICE OVERLAY GENERATION           ")
    print("==================================================")
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    for cid in target_cases:
        img_p = dataset_paths.find_image(cid)
        gt_p = dataset_paths.find_label(cid)
        pred_p = os.path.join(PRED_DIR, f"{cid}.nii.gz")

        if not (os.path.exists(img_p) and os.path.exists(gt_p) and os.path.exists(pred_p)):
            print(f"[SKIP] Case {cid}: Missing NIfTI volumes.")
            continue

        img_sitk = sitk.ReadImage(img_p)
        gt_sitk = sitk.ReadImage(gt_p)
        pred_sitk = sitk.ReadImage(pred_p)

        img_arr = sitk.GetArrayFromImage(img_sitk)
        gt_arr = sitk.GetArrayFromImage(gt_sitk)
        pred_arr = sitk.GetArrayFromImage(pred_sitk)

        # Clip HU intensity for display
        img_clipped = np.clip(img_arr, -100, 600)

        # Find slice with maximum ground truth vessel area
        slice_sums = gt_arr.sum(axis=(1, 2))
        if slice_sums.max() == 0:
            slice_idx = img_arr.shape[0] // 2
        else:
            slice_idx = int(np.argmax(slice_sums))

        slice_img = img_clipped[slice_idx, :, :]
        slice_gt = gt_arr[slice_idx, :, :]
        slice_pred = pred_arr[slice_idx, :, :]

        plt.figure(figsize=(6.5, 6.5), dpi=200)
        plt.imshow(slice_img, cmap="gray", origin="lower")

        # Plot contours
        if slice_gt.any():
            plt.contour(slice_gt, colors="#10b981", levels=[0.5], linewidths=1.5, alpha=0.95)
        if slice_pred.any():
            plt.contour(slice_pred, colors="#ef4444", levels=[0.5], linewidths=1.2, alpha=0.95)

        # Bounding box crop around vessel
        coords = np.argwhere(slice_gt > 0)
        if len(coords) > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            margin = 45
            plt.ylim(max(0, min_y - margin), min(slice_img.shape[0], max_y + margin))
            plt.xlim(max(0, min_x - margin), min(slice_img.shape[1], max_x + margin))

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(edgecolor="#10b981", facecolor="none", label="Ground Truth (Green)"),
            Patch(edgecolor="#ef4444", facecolor="none", label="RASNet Prediction (Red)")
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.7)
        plt.title(f"Case {cid} Qualitative Overlay (Slice {slice_idx})", fontsize=10, fontweight="bold")
        plt.axis("off")

        out_img_path = os.path.join(OVERLAY_DIR, f"case_{cid}_overlay.png")
        plt.savefig(out_img_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved Slice Overlay: {out_img_path}")

        # ALSO Generate 3-Panel MIP Overlay (Axial, Coronal, Sagittal)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#0d0d0d")

        views = [
            ("Axial (Z)", np.max(img_clipped, axis=2), np.max(gt_arr, axis=2), np.max(pred_arr, axis=2)),
            ("Coronal (Y)", np.max(img_clipped, axis=1), np.max(gt_arr, axis=1), np.max(pred_arr, axis=1)),
            ("Sagittal (X)", np.max(img_clipped, axis=0), np.max(gt_arr, axis=0), np.max(pred_arr, axis=0)),
        ]

        for ax, (title, img_mip, gt_mip, pred_mip) in zip(axes, views):
            ax.set_facecolor("#0d0d0d")
            ax.imshow(img_mip.T, cmap="gray", origin="lower", aspect="equal")

            gt_rgba = np.zeros((*gt_mip.T.shape, 4), dtype=np.float32)
            gt_rgba[..., 1] = gt_mip.T.astype(np.float32)  # green
            gt_rgba[..., 3] = (gt_mip.T > 0).astype(np.float32) * 0.55
            ax.imshow(gt_rgba, origin="lower", aspect="equal")

            pred_rgba = np.zeros((*pred_mip.T.shape, 4), dtype=np.float32)
            pred_rgba[..., 0] = pred_mip.T.astype(np.float32)  # red
            pred_rgba[..., 1] = pred_mip.T.astype(np.float32) * 0.4
            pred_rgba[..., 3] = (pred_mip.T > 0).astype(np.float32) * 0.55
            ax.imshow(pred_rgba, origin="lower", aspect="equal")

            ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=8)
            ax.axis("off")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(color=(0, 1, 0, 0.8), label="Ground Truth (Green)"),
            Patch(color=(1, 0.4, 0, 0.8), label="RASNet Prediction (Red/Orange)")
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=12, facecolor="#1a1a1a", edgecolor="#555", labelcolor="white")
        fig.suptitle(f"Case {cid} 3D Maximum Intensity Projection (MIP) Overlay", color="white", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        mip_out_path = os.path.join(OVERLAY_DIR, f"case_{cid}_mip_overlay.png")
        plt.savefig(mip_out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"[OK] Saved MIP Overlay: {mip_out_path}")

def generate_clinical_reports(target_cases=[851, 860, 900, 920, 934]):
    print("\n==================================================")
    print("  CLINICAL CENTERLINE & STENOSIS ANALYSIS        ")
    print("==================================================")
    os.makedirs(CLINICAL_DIR, exist_ok=True)

    for cid in target_cases:
        pred_p = os.path.join(PRED_DIR, f"{cid}.nii.gz")
        if not os.path.exists(pred_p):
            print(f"[SKIP] Case {cid}: Missing prediction mask.")
            continue

        pred_sitk = sitk.ReadImage(pred_p)
        pred_arr = sitk.GetArrayFromImage(pred_sitk)  # (Z, Y, X)
        spacing = pred_sitk.GetSpacing()  # (dx, dy, dz)

        # Skeletonization
        skeleton = skeletonize(pred_arr > 0)
        centerline_points = np.argwhere(skeleton)
        num_points = len(centerline_points)

        # Distance transform to measure radius
        edt = distance_transform_edt(pred_arr > 0, sampling=(spacing[2], spacing[1], spacing[0]))
        radii_mm = edt[skeleton] if num_points > 0 else np.array([0])
        diameters_mm = radii_mm * 2.0
        mean_diameter = float(np.mean(diameters_mm))

        # Stenosis points (<50% mean diameter)
        stenosis_mask = diameters_mm < (0.5 * mean_diameter)
        stenosis_count = int(np.sum(stenosis_mask))
        stenosis_pct = (stenosis_count / num_points * 100.0) if num_points > 0 else 0.0

        # Markdown report
        report_path = os.path.join(CLINICAL_DIR, f"clinical_vessel_report_{cid}.md")
        with open(report_path, "w") as f:
            f.write(f"# Clinical Vessel Centerline & Stenosis Report: Case {cid}\n\n")
            f.write(f"- **Centerline Points**: {num_points}\n")
            f.write(f"- **Mean Vessel Diameter**: {mean_diameter:.2f} mm\n")
            f.write(f"- **Stenosis Points (<50% mean diameter)**: {stenosis_count} ({stenosis_pct:.2f}% of vessel length)\n")
            f.write(f"- **Voxel Spacing (dx, dy, dz)**: {spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.3f} mm\n")

        print(f"[OK] Saved Clinical Report for Case {cid}: {report_path}")

if __name__ == "__main__":
    generate_overlays()
    generate_clinical_reports()
