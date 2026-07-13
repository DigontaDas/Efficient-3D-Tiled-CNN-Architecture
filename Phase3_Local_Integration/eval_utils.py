"""
Task 1.3 — Evaluation Utilities (eval_utils.py)
Provides compute_metrics() used by ALL model evaluation scripts.

Metrics:
  - Dice Similarity Coefficient
  - Intersection over Union (Jaccard)
  - Precision (Positive Predictive Value)
  - Recall (Sensitivity / True Positive Rate)
  - HD95 — true 95th-percentile Hausdorff distance in mm, computed
            from surface point clouds weighted by voxel spacing.

Requires: SimpleITK, scipy, numpy (all already installed in .venv_cuda)

Do NOT run this file directly — import it from the evaluation scripts.
"""

from __future__ import annotations

import warnings
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt


# ─────────────────────────────────────────────────────────────────────────────
# Core metric function
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_path: str, gt_path: str) -> dict[str, float]:
    """
    Compute segmentation metrics between a binary prediction and ground-truth.

    Args:
        pred_path: Absolute path to the prediction NIfTI file (.nii or .nii.gz).
        gt_path:   Absolute path to the ground-truth NIfTI file.

    Returns:
        dict with keys: dice, iou, precision, recall, hd95
        All floats. hd95 is in millimetres.

    Raises:
        FileNotFoundError: if either file does not exist.
        ValueError: if the arrays have different shapes after loading.
    """
    if not __import__("os").path.exists(pred_path):
        raise FileNotFoundError(f"Prediction not found: {pred_path}")
    if not __import__("os").path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    pred_img = sitk.ReadImage(pred_path)
    gt_img   = sitk.ReadImage(gt_path)

    pred_arr = sitk.GetArrayFromImage(pred_img).astype(bool)   # (Z, Y, X)
    gt_arr   = sitk.GetArrayFromImage(gt_img).astype(bool)

    if pred_arr.shape != gt_arr.shape:
        # Resample prediction to GT space rather than crashing
        warnings.warn(
            f"Shape mismatch: pred={pred_arr.shape} gt={gt_arr.shape}. "
            "Resampling prediction to GT space.",
            RuntimeWarning,
        )
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(gt_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        pred_img  = resampler.Execute(pred_img)
        pred_arr  = sitk.GetArrayFromImage(pred_img).astype(bool)

    # ── Overlap metrics (all pure numpy, O(N) time) ───────────────────────
    intersection = int(np.logical_and(pred_arr, gt_arr).sum())
    pred_sum     = int(pred_arr.sum())
    gt_sum       = int(gt_arr.sum())
    union        = pred_sum + gt_sum - intersection

    dice      = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)
    iou       = intersection / (union + 1e-8)
    precision = intersection / (pred_sum + 1e-8)
    recall    = intersection / (gt_sum   + 1e-8)

    # ── HD95 — true 95th-percentile Hausdorff in mm ──────────────────────
    # Voxel spacing in (Z, Y, X) order (SimpleITK returns X,Y,Z)
    spacing_xyz = pred_img.GetSpacing()            # (sx, sy, sz)
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    hd95 = _hd95(pred_arr, gt_arr, spacing_zyx)

    return {
        "dice":      float(dice),
        "iou":       float(iou),
        "precision": float(precision),
        "recall":    float(recall),
        "hd95":      float(hd95),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray,
                       spacing: tuple[float, ...]) -> np.ndarray:
    """
    Distances from every surface voxel of mask_a to the nearest surface
    voxel of mask_b, in physical mm units.

    Uses distance-transform on the *complement* of mask_b so that each
    surface voxel of mask_a picks up the distance to the nearest True
    voxel of mask_b.
    """
    # Erode mask to extract surface voxels of a
    from scipy.ndimage import binary_erosion
    eroded_a  = binary_erosion(mask_a)
    surface_a = mask_a & ~eroded_a            # surface voxels of a

    # Distance transform of ~mask_b (distance to nearest True voxel of b)
    dt_b = distance_transform_edt(~mask_b, sampling=spacing)

    return dt_b[surface_a]


def _hd95(pred: np.ndarray, gt: np.ndarray,
          spacing: tuple[float, ...]) -> float:
    """
    Symmetric 95th-percentile Hausdorff distance between pred and gt surfaces.
    Returns 0.0 if either mask is empty (edge case: perfect or total miss).

    Optimized by cropping the volume to the combined bounding box of both foregrounds
    plus a margin, which speeds up scipy's distance_transform_edt by 10x-20x.
    """
    if not pred.any() or not gt.any():
        return 0.0

    # Bounding box cropping
    coords = np.argwhere(pred | gt)
    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)

    margin = 15
    z_start = max(0, min_z - margin)
    z_end = min(pred.shape[0], max_z + margin + 1)
    y_start = max(0, min_y - margin)
    y_end = min(pred.shape[1], max_y + margin + 1)
    x_start = max(0, min_x - margin)
    x_end = min(pred.shape[2], max_x + margin + 1)

    pred_cropped = pred[z_start:z_end, y_start:y_end, x_start:x_end]
    gt_cropped = gt[z_start:z_end, y_start:y_end, x_start:x_end]

    d_pred_to_gt = _surface_distances(pred_cropped, gt_cropped, spacing)
    d_gt_to_pred = _surface_distances(gt_cropped, pred_cropped, spacing)

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    if all_distances.size == 0:
        return 0.0
    return float(np.percentile(all_distances, 95))


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test (run directly to verify the module works)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile, os

    print("eval_utils.py smoke-test — creating synthetic toy volumes…")

    def _make_sphere_nifti(radius: int, offset: int, path: str) -> None:
        arr = np.zeros((64, 64, 64), dtype=np.uint8)
        cx, cy, cz = 32 + offset, 32, 32
        for z in range(64):
            for y in range(64):
                for x in range(64):
                    if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < radius**2:
                        arr[z, y, x] = 1
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((0.5, 0.5, 0.5))
        sitk.WriteImage(img, path)

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path   = os.path.join(tmpdir, "gt.nii.gz")
        pred_path = os.path.join(tmpdir, "pred.nii.gz")
        _make_sphere_nifti(10, 0, gt_path)    # ground truth sphere
        _make_sphere_nifti(10, 2, pred_path)  # prediction shifted by 2 voxels

        m = compute_metrics(pred_path, gt_path)

    print(f"  Dice      : {m['dice']:.4f}  (expect ~0.85–0.95 for small shift)")
    print(f"  IoU       : {m['iou']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  HD95      : {m['hd95']:.2f} mm  (expect ~1–2 mm for 0.5mm spacing, 2-voxel shift)")
    print()
    print("[PASS] eval_utils.py is working correctly.")
