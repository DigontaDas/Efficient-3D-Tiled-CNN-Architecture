# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\eval_curriculum.py
"""
Evaluation and Inference Pipeline for Curriculum learning checkpoints.
Can be imported as a module or executed directly.
"""
from __future__ import annotations

import os
import sys
import json
import csv
import torch
import numpy as np
import SimpleITK as sitk
import monai.transforms as mt
from monai.inferers import sliding_window_inference
import cc3d
import concurrent.futures
import matplotlib.pyplot as plt

# Add current dir to path to import rasnet & dataset_paths
sys.path.insert(0, os.path.dirname(__file__))
from rasnet_model import RASNet
from eval_utils import compute_metrics
import dataset_paths

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)

# Output base directory
OUTPUT_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\rasnet_development"

# Transforms
pre_trans = mt.Compose([
    mt.LoadImaged(keys=["image"]),
    mt.EnsureChannelFirstd(keys=["image"]),
    mt.Orientationd(keys=["image"], axcodes="RAS"),
    mt.Spacingd(keys=["image"], pixdim=(0.5, 0.5, 0.5), mode="bilinear"),
    mt.ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
    mt.EnsureTyped(keys=["image"])
])

post_trans = mt.Compose([
    mt.Invertd(
        keys=["pred"],
        transform=pre_trans,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True,
    )
])

def topological_postprocess(pred_mask: np.ndarray, min_size=50) -> np.ndarray:
    """Removes small isolated blobs, keeping at most the top-2 components."""
    labeled, N = cc3d.connected_components(pred_mask, return_N=True)
    if N == 0:
        return np.zeros_like(pred_mask, dtype=np.uint8)
        
    sizes = np.bincount(labeled.flat)
    valid_components = []
    for comp_id in range(1, len(sizes)):
        if sizes[comp_id] >= min_size:
            valid_components.append((comp_id, sizes[comp_id]))
            
    valid_components.sort(key=lambda x: x[1], reverse=True)
    top_components = valid_components[:2]
    
    cleaned = np.zeros_like(pred_mask, dtype=np.uint8)
    for comp_id, _ in top_components:
        cleaned[labeled == comp_id] = 1
        
    return cleaned

def evaluate_single_case(case_id: int, out_path: str, gt_path: str) -> dict:
    try:
        m = compute_metrics(out_path, gt_path)
        m["case_id"] = case_id
        m["status"] = "ok"
        return m
    except Exception as exc:
        return {"case_id": case_id, "status": "error", "error": str(exc)}

def evaluate_checkpoint(
    ckpt_path: str,
    output_csv: str,
    pred_dir: str,
    use_tta: bool = True,
    threshold: float = 0.6
) -> dict[str, float]:
    """
    Runs inference on all 150 test cases and saves NIfTI files and calculates mean metrics.
    """
    print("=" * 60)
    print(f"EVALUATING CHECKPOINT: {ckpt_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Pred Dir: {pred_dir}")
    print(f"TTA: {use_tta} | Threshold: {threshold}")
    print("=" * 60)

    os.makedirs(pred_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Load splits
    splits_file = os.path.join(os.path.dirname(__file__), "splits_final.json")
    with open(splits_file) as f:
        splits = json.load(f)
    test_ids = splits.get("test", [])
    print(f"Test cases found: {len(test_ids)}")

    # Instantiate model
    model = RASNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    )
    print("Loading weights...")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=True)
    model = model.to(DEVICE, memory_format=torch.channels_last_3d)
    model.eval()

    eval_tasks = []
    for idx, case_id in enumerate(sorted(test_ids)):
        img_path = dataset_paths.find_image(case_id)
        gt_path = dataset_paths.find_label(case_id)
        out_path = os.path.join(pred_dir, f"{case_id}.nii.gz")

        if not img_path or not gt_path:
            print(f"[{idx+1}/{len(test_ids)}] Case {case_id}: Missing files, skipping.")
            continue

        # If already exists, skip inference
        if os.path.exists(out_path):
            eval_tasks.append((case_id, out_path, gt_path))
            continue

        try:
            batch = pre_trans({"image": img_path})
            input_tensor = batch["image"].unsqueeze(0).to(DEVICE, memory_format=torch.channels_last_3d)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    if use_tta:
                        # Original
                        logits = sliding_window_inference(
                            input_tensor, roi_size=PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                        )
                        prob_avg = torch.softmax(logits, dim=1)

                        # Flip axis 0 (dim 2 of input tensor)
                        input_flip0 = torch.flip(input_tensor, dims=[2])
                        logits_flip0 = sliding_window_inference(
                            input_flip0, roi_size=PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                        )
                        prob_avg += torch.softmax(torch.flip(logits_flip0, dims=[2]), dim=1)

                        # Flip axis 1 (dim 3)
                        input_flip1 = torch.flip(input_tensor, dims=[3])
                        logits_flip1 = sliding_window_inference(
                            input_flip1, roi_size=PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                        )
                        prob_avg += torch.softmax(torch.flip(logits_flip1, dims=[3]), dim=1)

                        # Flip axis 2 (dim 4)
                        input_flip2 = torch.flip(input_tensor, dims=[4])
                        logits_flip2 = sliding_window_inference(
                            input_flip2, roi_size=PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                        )
                        prob_avg += torch.softmax(torch.flip(logits_flip2, dims=[4]), dim=1)

                        prob_avg /= 4.0
                    else:
                        logits = sliding_window_inference(
                            input_tensor, roi_size=PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                        )
                        prob_avg = torch.softmax(logits, dim=1)

            batch["pred"] = prob_avg.squeeze(0)
            batch = post_trans(batch)

            pred_probs = batch["pred"]
            fg_prob = pred_probs[1]
            pred_mask = (fg_prob > threshold).cpu().numpy().astype(np.uint8)

            pred_mask_cleaned = topological_postprocess(pred_mask, min_size=50)

            # Save NIfTI
            original_img = sitk.ReadImage(img_path)
            pred_mask_sitk = np.transpose(pred_mask_cleaned, (2, 1, 0)) # (Z, Y, X)
            
            pred_sitk = sitk.GetImageFromArray(pred_mask_sitk)
            pred_sitk.CopyInformation(original_img)
            sitk.WriteImage(pred_sitk, out_path)

            eval_tasks.append((case_id, out_path, gt_path))
            if (idx + 1) % 20 == 0 or (idx + 1) == len(test_ids):
                print(f"  [{idx+1}/{len(test_ids)}] Completed GPU inference.")

        except Exception as exc:
            print(f"[ERROR] Case {case_id} failed GPU inference: {exc}")

    # Compute metrics in parallel
    print("\nComputing metrics on CPU cores...")
    rows = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(evaluate_single_case, cid, out_p, gt_p): cid for cid, out_p, gt_p in eval_tasks}
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res["status"] == "ok":
                rows.append(res)
            else:
                print(f"[WARNING] Case {res['case_id']} failed metrics: {res.get('error')}")

    rows.sort(key=lambda x: x["case_id"])

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "dice", "iou", "precision", "recall", "hd95"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "case_id": r["case_id"],
                "dice": r["dice"],
                "iou": r["iou"],
                "precision": r["precision"],
                "recall": r["recall"],
                "hd95": r["hd95"]
            })

    # Summary
    dices = [r["dice"] for r in rows]
    ious = [r["iou"] for r in rows]
    precisions = [r["precision"] for r in rows]
    recalls = [r["recall"] for r in rows]
    hds = [r["hd95"] for r in rows]

    summary = {
        "dice": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "iou": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "precision": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "recall": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "hd95": float(np.mean(hds)),
        "hd95_std": float(np.std(hds)),
    }

    print(f"\nEvaluation Results:")
    print(f"  Dice:      {summary['dice']:.4f} +/- {summary['dice_std']:.4f}")
    print(f"  Precision: {summary['precision']:.4f} +/- {summary['precision_std']:.4f}")
    print(f"  Recall:    {summary['recall']:.4f} +/- {summary['recall_std']:.4f}")
    print(f"  HD95:      {summary['hd95']:.2f} +/- {summary['hd95_std']:.2f} mm")
    print("=" * 60)

    return summary

def generate_overlay_pngs(pred_dir: str, output_dir: str, cases: list[int] = [1, 5, 6, 10, 13]) -> None:
    """Generates slice overlays for target cases."""
    print(f"Generating slice overlays for cases {cases}...")
    os.makedirs(output_dir, exist_ok=True)

    for case_id in cases:
        img_path = dataset_paths.find_image(case_id)
        gt_path = dataset_paths.find_label(case_id)
        pred_path = os.path.join(pred_dir, f"{case_id}.nii.gz")

        if not all(os.path.exists(p) for p in [img_path, gt_path, pred_path]):
            print(f"[WARNING] Missing files for overlay case {case_id}, skipping.")
            continue

        img_sitk = sitk.ReadImage(img_path)
        gt_sitk = sitk.ReadImage(gt_path)
        pred_sitk = sitk.ReadImage(pred_path)

        img_arr = sitk.GetArrayFromImage(img_sitk)
        gt_arr = sitk.GetArrayFromImage(gt_sitk)
        pred_arr = sitk.GetArrayFromImage(pred_sitk)

        # Clip intensity
        img_clipped = np.clip(img_arr, -100, 600)

        # Max vessel slice
        slice_sums = gt_arr.sum(axis=(1, 2))
        best_slice_idx = int(np.argmax(slice_sums))

        slice_img = img_clipped[best_slice_idx, :, :]
        slice_gt = gt_arr[best_slice_idx, :, :]
        slice_pred = pred_arr[best_slice_idx, :, :]

        plt.figure(figsize=(6, 6), dpi=150)
        plt.imshow(slice_img, cmap="gray", origin="lower")

        # Plot contours
        if slice_gt.any():
            plt.contour(slice_gt, colors="#10b981", levels=[0.5], linewidths=1.5, alpha=0.95)
        if slice_pred.any():
            plt.contour(slice_pred, colors="#ef4444", levels=[0.5], linewidths=1.2, alpha=0.95)

        # Bounding box crop
        coords = np.argwhere(slice_gt > 0)
        if len(coords) > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            margin = 40
            y_start = max(0, min_y - margin)
            y_end = min(slice_img.shape[0], max_y + margin)
            x_start = max(0, min_x - margin)
            x_end = min(slice_img.shape[1], max_x + margin)
            plt.ylim(y_start, y_end)
            plt.xlim(x_start, x_end)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(edgecolor="#10b981", facecolor="none", label="Ground Truth (Green)"),
            Patch(edgecolor="#ef4444", facecolor="none", label="Curriculum Model (Red)")
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.6)
        plt.title(f"Case {case_id} qualitative overlay (Slice {best_slice_idx})", fontsize=10, fontweight="bold")
        plt.axis("off")

        out_path = os.path.join(output_dir, f"case_{case_id}_overlay.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Saved overlay: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python eval_curriculum.py <ckpt_path> <output_csv> <pred_dir>")
        sys.exit(1)
        
    evaluate_checkpoint(
        ckpt_path=sys.argv[1],
        output_csv=sys.argv[2],
        pred_dir=sys.argv[3],
        use_tta=True,
        threshold=0.6
    )
