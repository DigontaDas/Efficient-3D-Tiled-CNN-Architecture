"""
eval_3d_unet.py — Coordinate-aware evaluation of the 3D U-Net baseline.

Pipeline (per case):
  1. Load raw .img.nii.gz via dataset_paths.find_image(case_id)
  2. Store original SimpleITK metadata (size, spacing, origin, direction)
  3. HU clip [-100, 800] → scale [0,1] → resize to 128³ (matches training)
  4. Forward pass → sigmoid → threshold 0.5 → 128³ binary mask
  5. Upsample 128³ mask back to ORIGINAL voxel space with SimpleITK nearest-
     neighbour resampling (preserving spacing/origin/direction)
  6. Save predicted NIfTI
  7. Compute Dice, IoU, Precision, Recall, HD95 via eval_utils.compute_metrics
  8. Aggregate over 150 test cases, save CSV

Usage:
  python eval_3d_unet.py              # full 150-case evaluation
  python eval_3d_unet.py --case 934   # single-case dry run
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import csv
import time
from typing import Optional

import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import zoom

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PHASE3_DIR = os.path.join(_THIS_DIR, "..", "Phase3_Local_Integration")
sys.path.insert(0, _PHASE3_DIR)
sys.path.insert(0, _THIS_DIR)

import dataset_paths          # find_image / find_label
from eval_utils import compute_metrics  # Dice, IoU, Precision, Recall, HD95
from model_3d_unet import get_unet3d_model

# ── Paths ─────────────────────────────────────────────────────────────────────
_SPLITS_FILE  = os.path.join(_PHASE3_DIR, "splits_final.json")
_RESULTS_DIR  = os.path.join(_THIS_DIR,   "results")
_PRED_DIR     = os.path.join(_RESULTS_DIR, "predictions")
_METRICS_LOCAL = os.path.join(_RESULTS_DIR, "metrics_3d_unet.csv")
_METRICS_ALL  = os.path.join(
    _THIS_DIR, "..", "all_four_validations", "metrics_3d_unet.csv"
)
_BEST_MODEL   = os.path.join(_RESULTS_DIR, "checkpoints", "best_model.pth")

# ── Constants matching training ───────────────────────────────────────────────
TARGET_SIZE: tuple[int, int, int] = (128, 128, 128)
HU_MIN: float = -100.0
HU_MAX: float =  800.0
THRESHOLD: float = 0.5


# ── Model loader ─────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> torch.nn.Module:
    """Load best_model.pth into a 3D U-Net and set to eval mode."""
    model = get_unet3d_model(n_channels=1, n_classes=1, base_filters=16, bilinear=True)
    if not os.path.exists(_BEST_MODEL):
        raise FileNotFoundError(f"best_model.pth not found at {_BEST_MODEL}")
    ckpt = torch.load(_BEST_MODEL, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    logging.info(
        f"Loaded best_model.pth (epoch {ckpt.get('epoch', '?')}, "
        f"val_dice={ckpt.get('best_val_dice', '?'):.4f})"
    )
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_for_inference(img_path: str) -> tuple[torch.Tensor, sitk.Image]:
    """
    Load a NIfTI image, apply normalization, resize to TARGET_SIZE.

    Returns:
        tensor: (1, 1, D, H, W) float32 ready for the model.
        original_sitk: the raw SimpleITK image (used as reference for resampling).
    
    Note: Current checkpoint uses image / image.max() normalization. 
    Retraining with HU [-100, 800] clipping is recommended for better robustness.
    """
    original_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(original_sitk).astype(np.float32)  # (Z, Y, X)

    # HU clip + scale [0, 1] matching the optimized 70-epoch training setup
    arr = np.clip(arr, HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)

    # Transpose from (Z, Y, X) -> (X, Y, Z) to match training (W, H, D)
    arr = np.transpose(arr, (2, 1, 0))

    # Resize to 128³
    if arr.shape != TARGET_SIZE:
        zoom_factors = [t / s for t, s in zip(TARGET_SIZE, arr.shape)]
        arr = zoom(arr, zoom_factors, order=1)

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,W,H,D)
    return tensor, original_sitk


# ── Postprocessing ────────────────────────────────────────────────────────────

def upsample_to_original_space(
    pred_128: np.ndarray,
    reference_img: sitk.Image,
) -> sitk.Image:
    """
    Resample a 128³ binary prediction back to the original image coordinate space.

    Uses SimpleITK nearest-neighbour interpolation so that binary labels are
    preserved exactly — identical approach to RASNet's eval_rasnet.py Invertd.
    """
    # Build a SimpleITK image from the 128³ mask.
    # We assign unit spacing here; the Resample step will map it to the
    # correct physical space via the reference image geometry.
    pred_sitk = sitk.GetImageFromArray(pred_128.astype(np.uint8))

    orig_size    = reference_img.GetSize()          # (X, Y, Z) in sitk convention
    orig_spacing = reference_img.GetSpacing()

    # Compute and set a pseudo-spacing so physical extents match
    # (the resampler only cares about the reference anyway, but this avoids
    # degenerate cases where spacing=0).
    pseudo_spacing = tuple(
        orig_spacing[i] * orig_size[i] / TARGET_SIZE[2 - i]   # note: sitk X=axis2
        for i in range(3)
    )
    pred_sitk.SetSpacing(pseudo_spacing)
    pred_sitk.SetOrigin(reference_img.GetOrigin())
    pred_sitk.SetDirection(reference_img.GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampled = resampler.Execute(pred_sitk)
    return resampled


# ── Single-case inference ─────────────────────────────────────────────────────

def run_case(
    case_id: int,
    model: torch.nn.Module,
    device: torch.device,
    save_pred: bool = True,
    is_dry_run: bool = False,
) -> Optional[dict]:
    """
    Run inference on one case and return metric dict, or None on failure.
    """
    img_path = dataset_paths.find_image(case_id)
    gt_path  = dataset_paths.find_label(case_id)

    if not img_path or not os.path.exists(img_path):
        logging.warning(f"Case {case_id}: image not found — skipping.")
        return None
    if not gt_path or not os.path.exists(gt_path):
        logging.warning(f"Case {case_id}: label not found — skipping.")
        return None

    t0 = time.time()

    # 1. Preprocess
    tensor, original_sitk = preprocess_for_inference(img_path)
    tensor = tensor.to(device)

    # 2. Inference
    with torch.no_grad():
        logits = model(tensor)                           # (1,1,W,H,D)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()   # (W,H,D)

    # 3. Threshold → binary 128³ mask in training orientation (W,H,D)
    pred_model = (prob >= THRESHOLD).astype(np.uint8)

    # Transpose back from (W, H, D) -> (D, H, W) to match SimpleITK orientation
    pred_128 = np.transpose(pred_model, (2, 1, 0))

    # 4. Upsample to original coordinate space
    pred_native = upsample_to_original_space(pred_128, original_sitk)

    # 5. Save predicted NIfTI
    pred_path = os.path.join(_PRED_DIR, f"{case_id}.nii.gz")
    if save_pred:
        os.makedirs(_PRED_DIR, exist_ok=True)
        sitk.WriteImage(pred_native, pred_path)

    # 6. Compute metrics (same function as RASNet eval)
    metrics = compute_metrics(pred_path, gt_path)
    elapsed = time.time() - t0

    # Print shape diagnostics if dry-run check is active
    if is_dry_run:
        gt_sitk = sitk.ReadImage(gt_path)
        print("Shape Diagnostics:")
        print(f"  - Input shape (sitk size): {original_sitk.GetSize()}")
        print(f"  - Resized shape (to model): {TARGET_SIZE}")
        print(f"  - Prediction shape BEFORE upsampling: {pred_128.shape}")
        print(f"  - Prediction shape AFTER upsampling (sitk size): {pred_native.GetSize()}")
        print(f"  - Ground truth shape (sitk size): {gt_sitk.GetSize()}")

    logging.info(
        f"Case {case_id:04d} | Dice={metrics.get('dice', 0):.4f} "
        f"HD95={metrics.get('hd95', 999):.2f}mm | {elapsed:.1f}s"
    )
    return {"case_id": case_id, **metrics}


# ── Full evaluation ───────────────────────────────────────────────────────────

def run_full_eval(device: torch.device) -> None:
    """Evaluate best_model.pth on all 150 test cases from splits_final.json."""
    import json

    with open(_SPLITS_FILE, "r") as fh:
        splits = json.load(fh)
    test_ids = [int(x) for x in splits["test"]]
    logging.info(f"Evaluating on {len(test_ids)} test cases...")

    model = load_model(device)

    rows: list[dict] = []
    for i, case_id in enumerate(test_ids, 1):
        logging.info(f"[{i}/{len(test_ids)}] Case {case_id}")
        result = run_case(case_id, model, device)
        if result:
            rows.append(result)

    if not rows:
        logging.error("No results — check dataset paths and model checkpoint.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    keys = [k for k in rows[0] if k != "case_id"]
    means = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}

    print("\n" + "=" * 65)
    print("3D U-Net Baseline — Final Results on 150 Test Cases")
    print("=" * 65)
    for k, v in means.items():
        print(f"  Mean {k:15s}: {v:.4f}")
    print("=" * 65)

    if means.get("dice", 0) < 0.50:
        print("\nWARNING: Mean Dice < 0.50 - consider continuing training.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    for out_path in [_METRICS_LOCAL, _METRICS_ALL]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logging.info(f"Metrics saved to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(_RESULTS_DIR, "logs", "eval_3d_unet.log")),
        ],
    )

    parser = argparse.ArgumentParser(description="3D U-Net Evaluation")
    parser.add_argument(
        "--case", type=int, default=None,
        help="Single case ID for dry-run check (e.g. 934). Omit for full eval."
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving predicted NIfTI files (faster dry run)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    os.makedirs(os.path.join(_RESULTS_DIR, "logs"), exist_ok=True)

    if args.case is not None:
        # ── Dry-run mode: single case ─────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"DRY-RUN: Case {args.case}")
        print(f"{'='*50}")
        model  = load_model(device)
        result = run_case(args.case, model, device, save_pred=not args.no_save, is_dry_run=True)
        if result:
            dice = result.get("dice", 0)
            hd95 = result.get("hd95", 999)
            passed = dice > 0.01 and hd95 < 200
            status = "[PASS]" if passed else "[FAIL]"
            print(f"\nCase {args.case} | Dice={dice:.4f} | HD95={hd95:.2f}mm | {status}")
            if not passed:
                sys.exit(1)
        else:
            print("Case failed — check logs above.")
            sys.exit(1)
    else:
        # ── Full evaluation ───────────────────────────────────────────────────
        run_full_eval(device)


if __name__ == "__main__":
    main()
