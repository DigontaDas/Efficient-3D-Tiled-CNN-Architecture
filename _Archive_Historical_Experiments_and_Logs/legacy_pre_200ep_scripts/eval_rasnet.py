# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\eval_rasnet.py
"""
Evaluation and Inference Pipeline for the Custom RASNet Model.
1. Loads the trained RASNet weights (strict=True) from development folder.
2. Runs sliding-window inference on the GPU.
3. Inverts transforms via mt.Invertd to return predictions to original coordinate grids.
4. Applies Topology-Aware Post-Processing via cc3d to remove disconnected background blobs.
5. Computes DSC, IoU, Precision, Recall, and HD95 metrics.

Supports:
  --single-case: Runs evaluation on Case 934 only for pipeline verification.
  (Otherwise evaluates all 150 test cases).

Saves outputs to:
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\rasnet_development\
"""
import os
import sys
import json
import torch
import numpy as np
import SimpleITK as sitk
import monai.transforms as mt
from monai.inferers import sliding_window_inference
from skimage.morphology import remove_small_objects
import cc3d
import concurrent.futures

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

# Paths
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Final_Generated_assets", "imagecas_pipeline_validation", "rasnet_development"))
CKPT_PATH = os.path.join(OUTPUT_DIR, "rasnet_best.pth")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
SPLITS_FILE = os.path.join(os.path.dirname(__file__), "splits_final.json")
REPORT_PATH = os.path.join(OUTPUT_DIR, "rasnet_evaluation_report.md")

# Preprocessing transforms (nnU-Net style spacing and intensity clipping)
pre_trans = mt.Compose([
    mt.LoadImaged(keys=["image"]),
    mt.EnsureChannelFirstd(keys=["image"]),
    mt.Orientationd(keys=["image"], axcodes="RAS"),
    mt.Spacingd(keys=["image"], pixdim=(0.5, 0.5, 0.5), mode="bilinear"),
    mt.ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
    mt.EnsureTyped(keys=["image"])
])

# Post-processing transforms to invert spatial operations
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
    """
    Removes small isolated connected component blobs and preserves only the 
    top-2 largest connected components (Left & Right Coronary Arteries) 
    that meet the min_size threshold.
    """
    # Find connected components in 3D
    labeled, N = cc3d.connected_components(pred_mask, return_N=True)
    if N == 0:
        return np.zeros_like(pred_mask, dtype=np.uint8)
        
    # Calculate sizes of each component (index 0 is background)
    sizes = np.bincount(labeled.flat)
    
    # Filter by min_size and find the top-2 largest components
    valid_components = []
    for comp_id in range(1, len(sizes)):
        if sizes[comp_id] >= min_size:
            valid_components.append((comp_id, sizes[comp_id]))
            
    # Sort by size descending
    valid_components.sort(key=lambda x: x[1], reverse=True)
    
    # Keep only the top-2 largest components
    top_components = valid_components[:2]
    
    # Create the output mask
    cleaned = np.zeros_like(pred_mask, dtype=np.uint8)
    for comp_id, _ in top_components:
        cleaned[labeled == comp_id] = 1
        
    return cleaned

def evaluate_single_case(case_id: int, out_path: str, gt_path: str) -> dict:
    """Helper process execution target to evaluate metrics on CPU in parallel."""
    try:
        m = compute_metrics(out_path, gt_path)
        m["case_id"] = case_id
        m["status"] = "ok"
        return m
    except Exception as exc:
        return {"case_id": case_id, "status": "error", "error": str(exc)}

def run_evaluation():
    single_case = "--single-case" in sys.argv
    
    print("==================================================")
    print("      RASNET CUSTOM MODEL EVALUATION RUN          ")
    print("==================================================")
    print(f"Device: {DEVICE}")
    print(f"Mode: {'Single Case 934 Verification' if single_case else 'Full Test Set (N=150)'}")
    print("--------------------------------------------------")

    os.makedirs(PRED_DIR, exist_ok=True)

    if not os.path.exists(CKPT_PATH):
        print(f"[ERROR] Trained RASNet checkpoint not found at: {CKPT_PATH}")
        print("Please run train_rasnet.py first.")
        return

    # 1. Load splits to determine case IDs
    if not os.path.exists(SPLITS_FILE):
        print(f"[ERROR] Splits file not found: {SPLITS_FILE}")
        return

    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    
    if single_case:
        test_ids = [934]
    else:
        test_ids = splits.get("test", [])

    print(f"Instantiating RASNet and loading weights from {CKPT_PATH}...")
    model = RASNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    )
    # Load model weights (strict=True since we are loading weights trained on this exact architecture)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE), strict=True)
    model = model.to(DEVICE, memory_format=torch.channels_last_3d)
    model.eval()

    print(f"\nRunning test inference on {len(test_ids)} cases...")
    
    eval_tasks = []

    for idx, case_id in enumerate(sorted(test_ids)):
        img_path = dataset_paths.find_image(case_id)
        gt_path = dataset_paths.find_label(case_id)
        out_path = os.path.join(PRED_DIR, f"{case_id}.nii.gz")

        if not img_path or not gt_path:
            print(f"[SKIP] Case {case_id}: Missing raw files.")
            continue

        # Resume inference capability: skip GPU forward pass if NIfTI prediction is already on disk
        if os.path.exists(out_path):
            print(f"[{idx+1}/{len(test_ids)}] Prediction already exists for Case {case_id}. Skipping GPU inference.")
            eval_tasks.append((case_id, out_path, gt_path))
            continue

        print(f"[{idx+1}/{len(test_ids)}] GPU Inference for Case {case_id}...")
        
        try:
            # A. Preprocessing
            batch = pre_trans({"image": img_path})
            input_tensor = batch["image"].unsqueeze(0).to(DEVICE, memory_format=torch.channels_last_3d)

            # B. Sliding Window Inference with Test Time Augmentation (TTA)
            # We average the softmax probabilities of original + 3 flipped versions
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    # Pass 1: Original volume
                    logits = sliding_window_inference(
                        input_tensor,
                        roi_size=PATCH_SIZE,
                        sw_batch_size=8,
                        predictor=model,
                        overlap=0.5
                    )
                    prob_avg = torch.softmax(logits, dim=1)
                    
                    # Pass 2: Flip axis 0 (dim 2 of input tensor [B, C, D, H, W])
                    input_flip0 = torch.flip(input_tensor, dims=[2])
                    logits_flip0 = sliding_window_inference(
                        input_flip0,
                        roi_size=PATCH_SIZE,
                        sw_batch_size=8,
                        predictor=model,
                        overlap=0.5
                    )
                    logits_flip0_rect = torch.flip(logits_flip0, dims=[2])
                    prob_avg += torch.softmax(logits_flip0_rect, dim=1)
                    
                    # Pass 3: Flip axis 1 (dim 3)
                    input_flip1 = torch.flip(input_tensor, dims=[3])
                    logits_flip1 = sliding_window_inference(
                        input_flip1,
                        roi_size=PATCH_SIZE,
                        sw_batch_size=8,
                        predictor=model,
                        overlap=0.5
                    )
                    logits_flip1_rect = torch.flip(logits_flip1, dims=[3])
                    prob_avg += torch.softmax(logits_flip1_rect, dim=1)
                    
                    # Pass 4: Flip axis 2 (dim 4)
                    input_flip2 = torch.flip(input_tensor, dims=[4])
                    logits_flip2 = sliding_window_inference(
                        input_flip2,
                        roi_size=PATCH_SIZE,
                        sw_batch_size=8,
                        predictor=model,
                        overlap=0.5
                    )
                    logits_flip2_rect = torch.flip(logits_flip2, dims=[4])
                    prob_avg += torch.softmax(logits_flip2_rect, dim=1)
                    
                    # Average probabilities
                    prob_avg /= 4.0
            
            # C. Invert transforms
            batch["pred"] = prob_avg.squeeze(0)
            batch = post_trans(batch)

            # D. Threshold at 0.6 confidence for tighter boundary matching
            pred_probs = batch["pred"]
            # pred_probs is already softmaxed from TTA; channel 1 = foreground
            fg_prob = pred_probs[1]
            pred_mask = (fg_prob > 0.6).cpu().numpy().astype(np.uint8)

            # E. Topology-Aware Post-Processing
            pred_mask_cleaned = topological_postprocess(pred_mask, min_size=50)

            # F. Save SimpleITK image with metadata
            original_img = sitk.ReadImage(img_path)
            pred_mask_sitk = np.transpose(pred_mask_cleaned, (2, 1, 0)) # (Z, Y, X)
            
            pred_sitk = sitk.GetImageFromArray(pred_mask_sitk)
            pred_sitk.CopyInformation(original_img)
            sitk.WriteImage(pred_sitk, out_path)

            eval_tasks.append((case_id, out_path, gt_path))

        except Exception as exc:
            print(f"[ERROR] Case {case_id} failed GPU Inference: {exc}")

    # Now compute metrics in parallel over CPU processes to speed up HD95
    print("\nCalculating metrics in parallel on CPU cores...")
    rows = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(evaluate_single_case, cid, out_p, gt_p): cid for cid, out_p, gt_p in eval_tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            cid = res["case_id"]
            if res["status"] == "ok":
                rows.append(res)
                print(f"[{i+1}/{len(eval_tasks)}] [OK] Case {cid:4d} Dice={res['dice']:.4f} | HD95={res['hd95']:.2f} mm")
            else:
                print(f"[{i+1}/{len(eval_tasks)}] [WARNING] Case {cid} failed metric calculation: {res.get('error')}")

    # Sort rows by case_id for consistency
    rows.sort(key=lambda x: x["case_id"])

    # 2. Write Report
    if rows:
        dices = [r["dice"] for r in rows]
        ious = [r["iou"] for r in rows]
        hds = [r["hd95"] for r in rows]
        
        mean_dice = np.mean(dices)
        mean_iou = np.mean(ious)
        mean_hd = np.mean(hds)

        print("--------------------------------------------------")
        print(f"Evaluation Completed. Mean Dice: {mean_dice:.4f} | Mean HD95: {mean_hd:.2f} mm")
        
        # Save CSV Metrics
        import csv
        csv_path = os.path.join(os.path.dirname(__file__), "metrics_rasnet.csv")
        csv_path_dev = os.path.join(OUTPUT_DIR, "metrics_rasnet.csv")
        for p in [csv_path, csv_path_dev]:
            with open(p, "w", newline="") as f:
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
        print(f"[OK] Saved CSV Metrics to:\n  {csv_path}\n  {csv_path_dev}")

        
        # Save Report
        with open(REPORT_PATH, "w") as f:
            f.write("# RASNet Custom Model Evaluation Report\n\n")
            f.write(f"- **Scope**: {'Single Case Verification' if single_case else 'Test Set (N=150)'}\n")
            f.write(f"- **Mean Dice Coefficient**: {mean_dice:.4f} +/- {np.std(dices):.4f}\n")
            f.write(f"- **Mean IoU**: {mean_iou:.4f} +/- {np.std(ious):.4f}\n")
            f.write(f"- **Mean HD95 (mm)**: {mean_hd:.2f} +/- {np.std(hds):.2f} mm\n\n")
            f.write("## Individual Case Metrics\n\n")
            f.write("| Case ID | Dice | IoU | Precision | Recall | HD95 (mm) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for r in rows:
                f.write(f"| {r['case_id']} | {r['dice']:.4f} | {r['iou']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['hd95']:.2f} mm |\n")
        
        print(f"[OK] Saved Evaluation Report: {REPORT_PATH}")
    print("==================================================")

if __name__ == "__main__":
    run_evaluation()
