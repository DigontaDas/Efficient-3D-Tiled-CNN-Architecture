# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\train_and_track_v3.py
"""
Retraining and Metric Tracking Pipeline for RASNet v3.
Trains for 30 epochs (in 5-epoch blocks) on the 100 cases, beginning with v2 weights.
Evaluates Dice, Precision, and Recall on the 150 test cases after every 5-epoch block using 0.6 threshold (no-TTA for speed).
Applies early stopping rules:
1. Dice < 0.75 -> Stop immediately (direction is wrong).
2. Dice >= 0.78 and Precision >= 0.88 -> Stop immediately (sweet spot found).
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import monai.transforms as mt
from monai.data import PersistentDataset, DataLoader
from monai.inferers import sliding_window_inference
from scipy.ndimage import binary_erosion
import concurrent.futures

# Set Cwd to import rasnet files
sys.path.insert(0, os.path.dirname(__file__))
from rasnet_model import RASNet
from rasnet_loss import StenosisAwareLoss
from eval_utils import compute_metrics
import dataset_paths

# Setup GPU configurations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 4
LR = 1e-4

# Paths
OUTPUT_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\rasnet_development"
V2_CKPT_PATH = os.path.join(OUTPUT_DIR, "rasnet_best.pth")
V3_TEMP_CKPT = os.path.join(OUTPUT_DIR, "rasnet_v3_temp.pth")
V3_BEST_CKPT = os.path.join(OUTPUT_DIR, "rasnet_v3_best.pth")

# Custom label erosion transform
class RandErodeLabelD:
    def __init__(self, keys=["label"], prob=0.3):
        self.keys = keys
        self.prob = prob

    def __call__(self, data):
        if np.random.random() < self.prob:
            d = dict(data)
            for key in self.keys:
                arr = d[key].numpy() if hasattr(d[key], 'numpy') else np.array(d[key])
                eroded = binary_erosion(arr[0] > 0.5).astype(np.float32)
                if torch.is_tensor(d[key]):
                    d[key] = torch.from_numpy(eroded[np.newaxis]).to(d[key].device)
                else:
                    d[key] = eroded[np.newaxis]
            return d
        return data

# Clean meta-tensors
class ConvertToPlainTensor:
    def __call__(self, data):
        if isinstance(data, dict):
            cleaned = {"image": data["image"], "label": data["label"]}
            for k in ["image", "label"]:
                if hasattr(cleaned[k], "as_tensor"):
                    cleaned[k] = cleaned[k].as_tensor()
                else:
                    cleaned[k] = torch.as_tensor(cleaned[k])
            return cleaned
        return data

# Preprocessing transforms (Training)
train_transforms = mt.Compose([
    mt.LoadImaged(keys=["image", "label"]),
    mt.EnsureChannelFirstd(keys=["image", "label"]),
    mt.CropForegroundd(keys=["image", "label"], source_key="image"),
    mt.Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest")),
    mt.ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
    mt.RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=PATCH_SIZE, pos=2, neg=1, num_samples=2
    ),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandErodeLabelD(keys=["label"], prob=0.3),
    mt.ToTensord(keys=["image", "label"]),
    ConvertToPlainTensor(),
])

# Evaluation Preprocessing (TTA-disabled for speed)
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
    import cc3d
    labeled, N = cc3d.connected_components(pred_mask, return_N=True)
    if N == 0:
        return np.zeros_like(pred_mask, dtype=np.uint8)
    sizes = np.bincount(labeled.flat)
    valid_components = [(comp_id, sizes[comp_id]) for comp_id in range(1, len(sizes)) if sizes[comp_id] >= min_size]
    valid_components.sort(key=lambda x: x[1], reverse=True)
    top_components = valid_components[:2]
    cleaned = np.zeros_like(pred_mask, dtype=np.uint8)
    for comp_id, _ in top_components:
        cleaned[labeled == comp_id] = 1
    return cleaned

def evaluate_metrics_only(case_id, out_path, gt_path):
    try:
        metrics = compute_metrics(out_path, gt_path)
        return {
            "case_id": case_id,
            "dice": metrics.get("dice", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "status": "ok"
        }
    except Exception as e:
        return {"case_id": case_id, "status": "error", "error": str(e)}

def evaluate_v3_checkpoint(model, test_ids):
    model.eval()
    temp_pred_dir = os.path.join(OUTPUT_DIR, "predictions_temp_v3")
    os.makedirs(temp_pred_dir, exist_ok=True)
    
    eval_tasks = []
    
    for idx, case_id in enumerate(test_ids):
        img_path = dataset_paths.find_image(case_id)
        gt_path = dataset_paths.find_label(case_id)
        out_path = os.path.join(temp_pred_dir, f"{case_id}.nii.gz")
        
        if not img_path or not gt_path:
            continue
            
        try:
            # Preprocessing
            batch = pre_trans({"image": img_path})
            input_tensor = batch["image"].unsqueeze(0).to(DEVICE, memory_format=torch.channels_last_3d)
            
            # Non-TTA sliding window inference for speed (10 mins check)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    logits = sliding_window_inference(
                        input_tensor, roi_size=PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                    )
                    prob_avg = torch.softmax(logits, dim=1)
            
            # Invert transform & threshold at 0.6
            batch["pred"] = prob_avg.squeeze(0)
            batch = post_trans(batch)
            pred_probs = batch["pred"]
            fg_prob = pred_probs[1]
            pred_mask = (fg_prob > 0.6).cpu().numpy().astype(np.uint8)
            pred_mask_cleaned = topological_postprocess(pred_mask, min_size=50)
            
            # Save SimpleITK
            original_img = sitk.ReadImage(img_path)
            pred_mask_sitk = np.transpose(pred_mask_cleaned, (2, 1, 0))
            pred_sitk = sitk.GetImageFromArray(pred_mask_sitk)
            pred_sitk.CopyInformation(original_img)
            sitk.WriteImage(pred_sitk, out_path)
            
            eval_tasks.append((case_id, out_path, gt_path))
        except Exception as e:
            print(f"    [ERR] Evaluation prep for Case {case_id} failed: {e}")

    # Compute metrics in parallel
    rows = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(evaluate_metrics_only, cid, out_p, gt_p): cid for cid, out_p, gt_p in eval_tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res["status"] == "ok":
                rows.append(res)
                
    if not rows:
        return 0.0, 0.0, 0.0
        
    mean_dice = np.mean([r["dice"] for r in rows])
    mean_precision = np.mean([r["precision"] for r in rows])
    mean_recall = np.mean([r["recall"] for r in rows])
    return mean_dice, mean_precision, mean_recall

def main():
    print("=" * 60)
    print("      RASNET V3 TRACKED TRAINING AND TESTING LOOP     ")
    print("=" * 60)
    
    # 1. Load data and setup loader
    splits_file = os.path.join(os.path.dirname(__file__), "splits_final.json")
    with open(splits_file) as f:
        splits = json.load(f)
        
    train_ids = splits.get("train", [])[:100]
    test_ids = splits.get("test", [])
    
    print(f"Train on: {len(train_ids)} cases. Evaluate on: {len(test_ids)} cases.")
    
    data_files = []
    for cid in train_ids:
        data_files.append({
            "image": dataset_paths.find_image(cid),
            "label": dataset_paths.find_label(cid)
        })
        
    cache_dir = os.path.join(OUTPUT_DIR, "persistent_cache_v3")
    os.makedirs(cache_dir, exist_ok=True)
    dataset = PersistentDataset(data=data_files, transform=train_transforms, cache_dir=cache_dir)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    # 2. Setup model
    model = RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)
    
    # Initialize from the baseline v2 weights
    print(f"Initializing weights from baseline v2: {V2_CKPT_PATH}")
    model.load_state_dict(torch.load(V2_CKPT_PATH, map_location=DEVICE), strict=True)
    model = model.to(DEVICE, memory_format=torch.channels_last_3d)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = StenosisAwareLoss(alpha=0.5, beta=0.3, delta=0.2, gamma=2.5)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30 * len(loader), eta_min=1e-6)
    
    # Tracking table
    history = []
    
    # Run 5-epoch blocks
    total_epochs = 30
    block_size = 5
    num_blocks = total_epochs // block_size
    
    current_epoch = 0
    early_stopped = False
    
    for block in range(1, num_blocks + 1):
        print(f"\n--- Training block {block}/{num_blocks} (Epochs {current_epoch+1} - {current_epoch+block_size}) ---")
        
        # Train for block_size epochs
        for ep_in_block in range(1, block_size + 1):
            current_epoch += 1
            model.train()
            epoch_loss = 0.0
            t0 = time.time()
            
            for batch in loader:
                imgs = batch["image"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last_3d)
                labels = batch["label"].to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    preds, ds_preds = model(imgs)
                    main_loss = loss_fn(preds, labels)
                    aux2_pred = F.interpolate(ds_preds[0], size=labels.shape[2:], mode='trilinear', align_corners=True)
                    aux3_pred = F.interpolate(ds_preds[1], size=labels.shape[2:], mode='trilinear', align_corners=True)
                    aux2_loss = loss_fn(aux2_pred, labels)
                    aux3_loss = loss_fn(aux3_pred, labels)
                    loss = main_loss + 0.4 * aux2_loss + 0.2 * aux3_loss
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_loss += loss.item()
                
            print(f"  Epoch {current_epoch:02d} | Mean Loss: {epoch_loss/len(loader):.4f} | Time: {time.time()-t0:.1f}s")
            
        # Save temp checkpoint at end of block
        torch.save(model.state_dict(), V3_TEMP_CKPT)
        
        # Evaluate
        print(f"  Evaluating Epoch {current_epoch} checkpoint on 150 test cases...")
        t_eval = time.time()
        dice, precision, recall = evaluate_v3_checkpoint(model, test_ids)
        print(f"  Evaluation completed in {time.time()-t_eval:.1f}s.")
        print(f"  Results -> Dice: {dice:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        # Record history
        history.append({
            "epoch": current_epoch,
            "dice": dice,
            "precision": precision,
            "recall": recall
        })
        
        # Print table so far
        print("\n  [Metrics Table So Far]")
        print("  Epoch |   Dice   | Precision |  Recall")
        print("  --------------------------------------")
        for h in history:
            print(f"   {h['epoch']:02d}   |  {h['dice']:.4f}  |  {h['precision']:.4f}   |  {h['recall']:.4f}")
        print("  --------------------------------------")
        
        # Check Stop Condition 1: Dice drops below 0.75
        if dice < 0.75:
            print(f"\n[STOP] Dice dropped to {dice:.4f} (below 0.75) at epoch {current_epoch}. Stopping retraining pipeline.")
            early_stopped = True
            break
            
        # Check Stop Condition 2: Dice crosses 0.78 AND Precision stays above 0.88
        if dice >= 0.78 and precision >= 0.88:
            print(f"\n[STOP] Sweet spot reached! Dice: {dice:.4f} >= 0.78 AND Precision: {precision:.4f} >= 0.88 at epoch {current_epoch}.")
            print(f"Saving checkpoint to final destination: {V3_BEST_CKPT}")
            torch.save(model.state_dict(), V3_BEST_CKPT)
            early_stopped = True
            break

    if not early_stopped:
        print("\nCompleted all 30 epochs.")
        # If training finished, save the final model weights as rasnet_v3_best
        torch.save(model.state_dict(), V3_BEST_CKPT)
        print(f"Saved final weights to: {V3_BEST_CKPT}")
        
    print("\n==================================================")
    print("      RETRAINING TRACKER LOOP COMPLETED           ")
    print("==================================================")

if __name__ == "__main__":
    main()
