# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\eval_segresnet.py
"""
Evaluation and Inference Pipeline for SegResNet.
Mirrors eval_rasnet.py exactly for a fair comparison:
1. Loads the pre-trained SegResNet checkpoint.
2. Runs sliding-window inference with Test Time Augmentation (TTA) on GPU.
3. Inverts transforms via mt.Invertd to return predictions to original coordinate space.
4. Applies Topology-Aware Post-Processing via cc3d.
5. Computes DSC, IoU, Precision, Recall, and HD95 metrics.
"""
import os
import sys
import json
import time
import torch
import numpy as np
import SimpleITK as sitk
import monai.transforms as mt
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
import cc3d
import concurrent.futures

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import compute_metrics
import dataset_paths

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)

# Paths
ROOT_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings"
CKPT_PATH = os.path.join(ROOT_DIR, "all_four_validations", "mandatory_artifacts_segresnet", "best_resumed.pt")
OUTPUT_DIR = os.path.join(ROOT_DIR, "all_four_validations", "mandatory_artifacts_segresnet")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
SPLITS_FILE = os.path.join(os.path.dirname(__file__), "splits_final.json")
REPORT_PATH = os.path.join(OUTPUT_DIR, "segresnet_evaluation_report.md")

# Preprocessing transforms (mirroring eval_rasnet.py spacing and HU range)
pre_trans = mt.Compose([
    mt.LoadImaged(keys=["image"]),
    mt.EnsureChannelFirstd(keys=["image"]),
    mt.Orientationd(keys=["image"], axcodes="RAS"),
    mt.Spacingd(keys=["image"], pixdim=(0.8, 0.8, 0.8), mode="bilinear"),
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
    """Removes small isolated blobs and preserves the top-2 largest connected components."""
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

def run_evaluation():
    single_case = "--single-case" in sys.argv
    
    print("==================================================")
    print("      SEGRESNET FAIR EVALUATION RUN               ")
    print("==================================================")
    print(f"Device: {DEVICE}")
    print(f"Mode: {'Single Case 934 Verification' if single_case else 'Full Test Set (N=150)'}")
    print("--------------------------------------------------")

    os.makedirs(PRED_DIR, exist_ok=True)

    if not os.path.exists(CKPT_PATH):
        sys.exit(f"[ERROR] SegResNet checkpoint not found at: {CKPT_PATH}")

    # Load splits
    if not os.path.exists(SPLITS_FILE):
        sys.exit(f"[ERROR] Splits file not found: {SPLITS_FILE}")

    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    
    if single_case:
        test_ids = [934]
    else:
        test_ids = splits.get("test", [])

    print("Instantiating SegResNet...")
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    )
    
    # Load model weights
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
        
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

        if not single_case and os.path.exists(out_path):
            print(f"[{idx+1}/{len(test_ids)}] Prediction already exists for Case {case_id}. Skipping inference.")
            eval_tasks.append((case_id, out_path, gt_path))
            continue

        print(f"[{idx+1}/{len(test_ids)}] GPU Inference for Case {case_id}...")
        
        try:
            # A. Preprocessing
            batch = pre_trans({"image": img_path})
            input_tensor = batch["image"].unsqueeze(0).to(DEVICE, memory_format=torch.channels_last_3d)

            # B. Sliding Window Inference with Test Time Augmentation (TTA) - sw_batch_size=8 for GPU saturation
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
                    
                    # Pass 2: Flip axis 0
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
                    
                    # Pass 3: Flip axis 1
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
                    
                    # Pass 4: Flip axis 2
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

            # D. Threshold and get binary mask
            pred_probs = batch["pred"]
            pred_mask = torch.argmax(pred_probs, dim=0).cpu().numpy().astype(np.uint8)

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

    # Compute metrics in parallel
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

    rows.sort(key=lambda x: x["case_id"])

    if rows:
        dices = [r["dice"] for r in rows]
        ious = [r["iou"] for r in rows]
        hds = [r["hd95"] for r in rows]
        precisions = [r["precision"] for r in rows]
        recalls = [r["recall"] for r in rows]
        
        mean_dice = np.mean(dices)
        mean_iou = np.mean(ious)
        mean_hd = np.mean(hds)

        print("--------------------------------------------------")
        print(f"Evaluation Completed. Mean Dice: {mean_dice:.4f} | Mean HD95: {mean_hd:.2f} mm")
        
        # Save CSV Metrics
        import csv
        csv_path = os.path.join(ROOT_DIR, "all_four_validations", "metrics_segresnet.csv")
        csv_path_dev = os.path.join(OUTPUT_DIR, "metrics_segresnet.csv")
        for p in [csv_path, csv_path_dev]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["case_id", "dice", "iou", "precision", "recall", "hd95"])
                writer.writeheader()
                for r in rows:
                    writer.writerow({
                        "case_id": r["case_id"],
                        "dice": f"{r['dice']:.6f}",
                        "iou": f"{r['iou']:.6f}",
                        "precision": f"{r['precision']:.6f}",
                        "recall": f"{r['recall']:.6f}",
                        "hd95": f"{r['hd95']:.6f}"
                    })
        print(f"Saved SegResNet metrics -> {csv_path}")

if __name__ == "__main__":
    run_evaluation()
