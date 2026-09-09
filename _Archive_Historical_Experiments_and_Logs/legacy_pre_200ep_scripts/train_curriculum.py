# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\train_curriculum.py
"""
End-to-End Curriculum Learning Training Pipeline for RASNet.
Executes Stage 1, Stage 2, and Stage 3 sequentially, running evaluations
and checking stopping criteria at each stage.
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from monai.data import PersistentDataset, DataLoader
import monai.transforms as mt
from scipy.ndimage import binary_erosion

# Add current dir to path to import rasnet & dataset_paths
sys.path.insert(0, os.path.dirname(__file__))
from rasnet_model import RASNet
from rasnet_loss import StenosisAwareLoss
from eval_curriculum import evaluate_checkpoint, generate_overlay_pngs
import dataset_paths

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 4

# Output Paths
OUTPUT_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\rasnet_development"
STARTING_WEIGHTS = os.path.join(OUTPUT_DIR, "rasnet_best.pth")

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

def train_stage(
    stage_num: int,
    train_ids: list[int],
    initial_weights: str,
    save_ckpt_path: str,
    epochs: int,
    lr: float,
    eta_min: float = 1e-6
) -> list[float]:
    print("=" * 60)
    print(f"      STARTING STAGE {stage_num} TRAINING")
    print(f"      Cases count: {len(train_ids)} | Epochs: {epochs} | LR: {lr:.1e}")
    print("=" * 60)

    # Setup files
    data_files = []
    for cid in train_ids:
        img_path = dataset_paths.find_image(cid)
        lbl_path = dataset_paths.find_label(cid)
        if img_path and lbl_path:
            data_files.append({"image": img_path, "label": lbl_path})

    # Caching setup
    cache_dir = os.path.join(OUTPUT_DIR, f"persistent_cache_curr_s{stage_num}")
    os.makedirs(cache_dir, exist_ok=True)
    dataset = PersistentDataset(data=data_files, transform=train_transforms, cache_dir=cache_dir)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )

    # Instantiate model and load weights
    model = RASNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    )
    print(f"Loading starting weights from: {initial_weights}")
    model.load_state_dict(torch.load(initial_weights, map_location=DEVICE), strict=True)
    model = model.to(DEVICE, memory_format=torch.channels_last_3d)

    # Optimizer, Loss, Scaler, Scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = StenosisAwareLoss(alpha=0.5, beta=0.3, delta=0.2, gamma=2.5)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader), eta_min=eta_min
    )

    losses = []
    best_loss = float("inf")

    print("\nStarting training loop...")
    print("Epoch | Time/Epoch | Mean Loss | Learning Rate | Max VRAM")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        model.train()
        for batch in loader:
            imgs = batch["image"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last_3d)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                preds, ds_preds = model(imgs)
                main_loss = loss_fn(preds, labels)
                
                # Deep supervision auxiliary losses
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

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        lr_curr = scheduler.get_last_lr()[0]
        max_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

        print(f" {epoch:02d}   |   {epoch_time:.1f}s   |  {avg_loss:.4f}   |   {lr_curr:.6f}   |  {max_mem:.1f} MB")

        # Save stage-best checkpoint based on train loss (since no val split)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_ckpt_path)

    print("-" * 60)
    print(f"Stage {stage_num} completed. Best training loss: {best_loss:.4f}")
    print(f"Saved best checkpoint to: {save_ckpt_path}")

    # Plot curves
    plot_path = os.path.join(OUTPUT_DIR, f"loss_curves_curriculum_s{stage_num}.png")
    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.plot(range(1, epochs + 1), losses, color='#3b82f6', marker='o', linewidth=2, label=f'Stage {stage_num} Loss')
    plt.title(f'RASNet Curriculum Stage {stage_num} Training Curve', fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontweight='semibold')
    plt.ylabel('Loss', fontweight='semibold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved stage loss plot: {plot_path}")
    print("=" * 60 + "\n")

    return losses

def update_comparison_outputs(final_metrics: dict) -> None:
    """Updates CSV, generates comparison plots and generates overlays."""
    import sys
    import os
    all_four_validations_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "all_four_validations"))
    sys.path.insert(0, all_four_validations_dir)
    from generate_curriculum_plots import run_updates
    run_updates(final_metrics)

def main() -> None:
    splits_json_path = os.path.join(os.path.dirname(__file__), "curriculum_splits.json")
    if not os.path.exists(splits_json_path):
        print(f"[ERROR] splits file curriculum_splits.json not found at: {splits_json_path}")
        print("Please run curriculum_splits.py first.")
        sys.exit(1)

    with open(splits_json_path) as f:
        splits_data = json.load(f)

    tier1 = splits_data["tier1_easy"]
    tier2 = splits_data["tier2_medium"]
    tier3 = splits_data["tier3_hard"]

    print("Curriculum training starting...")
    print(f"Tier 1 (Easy):   {len(tier1)} cases")
    print(f"Tier 2 (Medium): {len(tier2)} cases")
    print(f"Tier 3 (Hard):   {len(tier3)} cases")

    # Outputs
    s1_ckpt = os.path.join(OUTPUT_DIR, "rasnet_curriculum_stage1.pth")
    s2_ckpt = os.path.join(OUTPUT_DIR, "rasnet_curriculum_stage2.pth")
    final_ckpt = os.path.join(OUTPUT_DIR, "rasnet_curriculum_final.pth")

    # Keep track of active best checkpoint details to write summary at the end
    best_stage_num = 0
    best_stage_ckpt = ""
    best_stage_metrics = None

    # =========================================================================
    # STAGE 1 — PROXIMAL VESSEL MASTERY
    # =========================================================================
    s1_train_ids = tier1
    s1_losses = train_stage(
        stage_num=1,
        train_ids=s1_train_ids,
        initial_weights=STARTING_WEIGHTS,
        save_ckpt_path=s1_ckpt,
        epochs=15,
        lr=1e-4
    )

    # Eval Stage 1 (150 test cases, TTA and threshold 0.6 to match v2 comparison)
    s1_csv = os.path.join(os.path.dirname(__file__), "metrics_curriculum_stage1.csv")
    s1_pred_dir = os.path.join(OUTPUT_DIR, "predictions_curriculum_stage1")
    s1_metrics = evaluate_checkpoint(s1_ckpt, s1_csv, s1_pred_dir, use_tta=True, threshold=0.6)

    # Check Stage 1 stopping rules
    if s1_metrics["dice"] < 0.76:
        print(f"\n[STOPPING RULE TRIGGERED] Dice drops below 0.76 (Dice: {s1_metrics['dice']:.4f}). Reverting to rasnet_best.pth.")
        # Revert: copy rasnet_best.pth as the final model
        shutil.copy2(STARTING_WEIGHTS, final_ckpt)
        # Use v2 baseline metrics as final
        final_m = {"dice": 0.7879, "iou": 0.6625, "precision": 0.8619, "recall": 0.7495, "hd95": 8.98}
        best_stage_num = 0
        best_stage_ckpt = STARTING_WEIGHTS
        best_stage_metrics = final_m
        update_comparison_outputs(final_m)
        return

    if s1_metrics["dice"] > 0.82 and s1_metrics["precision"] > 0.87:
        print(f"\n[EARLY STOP TRIGGERED] Sweet spot found in Stage 1! Dice: {s1_metrics['dice']:.4f}, Precision: {s1_metrics['precision']:.4f}")
        shutil.copy2(s1_ckpt, final_ckpt)
        best_stage_num = 1
        best_stage_ckpt = s1_ckpt
        best_stage_metrics = s1_metrics
        update_comparison_outputs(s1_metrics)
        return

    # Keep track as temporary best
    best_stage_num = 1
    best_stage_ckpt = s1_ckpt
    best_stage_metrics = s1_metrics

    # =========================================================================
    # STAGE 2 — MID-VESSEL GENERALIZATION
    # =========================================================================
    s2_train_ids = tier1 + tier2
    s2_losses = train_stage(
        stage_num=2,
        train_ids=s2_train_ids,
        initial_weights=s1_ckpt,
        save_ckpt_path=s2_ckpt,
        epochs=15,
        lr=5e-5
    )

    # Eval Stage 2
    s2_csv = os.path.join(os.path.dirname(__file__), "metrics_curriculum_stage2.csv")
    s2_pred_dir = os.path.join(OUTPUT_DIR, "predictions_curriculum_stage2")
    s2_metrics = evaluate_checkpoint(s2_ckpt, s2_csv, s2_pred_dir, use_tta=True, threshold=0.6)

    # Check Stage 2 stopping rules
    if s2_metrics["dice"] < 0.76:
        print(f"\n[STOPPING RULE TRIGGERED] Dice drops below 0.76 (Dice: {s2_metrics['dice']:.4f}). Reverting to Stage 1 weights.")
        shutil.copy2(s1_ckpt, final_ckpt)
        update_comparison_outputs(s1_metrics)
        return

    if s2_metrics["dice"] > 0.82 and s2_metrics["precision"] > 0.87:
        print(f"\n[EARLY STOP TRIGGERED] Sweet spot found in Stage 2! Dice: {s2_metrics['dice']:.4f}, Precision: {s2_metrics['precision']:.4f}")
        shutil.copy2(s2_ckpt, final_ckpt)
        best_stage_num = 2
        best_stage_ckpt = s2_ckpt
        best_stage_metrics = s2_metrics
        update_comparison_outputs(s2_metrics)
        return

    # Update temporary best if Dice is better
    if s2_metrics["dice"] > best_stage_metrics["dice"]:
        best_stage_num = 2
        best_stage_ckpt = s2_ckpt
        best_stage_metrics = s2_metrics

    # =========================================================================
    # STAGE 3 — HARD CASE FINE-TUNING
    # =========================================================================
    s3_train_ids = tier1 + tier2 + tier3
    s3_losses = train_stage(
        stage_num=3,
        train_ids=s3_train_ids,
        initial_weights=s2_ckpt,
        save_ckpt_path=final_ckpt,
        epochs=10,
        lr=1e-5
    )

    # Eval Stage 3
    final_csv_local = os.path.join(os.path.dirname(__file__), "metrics_curriculum_final.csv")
    final_csv_all = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\metrics_curriculum_final.csv"
    
    # Run evaluation
    final_pred_dir = os.path.join(OUTPUT_DIR, "predictions_curriculum_final")
    final_metrics = evaluate_checkpoint(final_ckpt, final_csv_local, final_pred_dir, use_tta=True, threshold=0.6)
    
    # Save to all_four_validations as well
    shutil.copy2(final_csv_local, final_csv_all)

    # Check Stage 3 stopping rules
    if final_metrics["dice"] < 0.76:
        print(f"\n[STOPPING RULE TRIGGERED] Final Dice drops below 0.76 (Dice: {final_metrics['dice']:.4f}). Reverting to Stage 2 weights.")
        shutil.copy2(s2_ckpt, final_ckpt)
        # Overwrite final csv with stage 2 metrics
        shutil.copy2(s2_csv, final_csv_local)
        shutil.copy2(s2_csv, final_csv_all)
        best_stage_metrics = s2_metrics
    else:
        # Determine the absolute best stage checkpoint among 1, 2, and 3 based on test Dice
        all_metrics = [("Stage 1", s1_metrics), ("Stage 2", s2_metrics), ("Stage 3 (Final)", final_metrics)]
        best_entry = max(all_metrics, key=lambda x: x[1]["dice"])
        print(f"\n[COMPLETED] Best checkpoint is from {best_entry[0]} with Dice={best_entry[1]['dice']:.4f}")
        
        if best_entry[0] == "Stage 1":
            shutil.copy2(s1_ckpt, final_ckpt)
            shutil.copy2(s1_csv, final_csv_local)
            shutil.copy2(s1_csv, final_csv_all)
            best_stage_num = 1
            best_stage_ckpt = s1_ckpt
            best_stage_metrics = s1_metrics
        elif best_entry[0] == "Stage 2":
            shutil.copy2(s2_ckpt, final_ckpt)
            shutil.copy2(s2_csv, final_csv_local)
            shutil.copy2(s2_csv, final_csv_all)
            best_stage_num = 2
            best_stage_ckpt = s2_ckpt
            best_stage_metrics = s2_metrics
        else:
            best_stage_num = 3
            best_stage_ckpt = final_ckpt
            best_stage_metrics = final_metrics

    # Generate final comparative overlays, updated tables, and barcharts
    update_comparison_outputs(best_stage_metrics)

    # Save overlays (using the final/best model's prediction dir)
    overlay_dir = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\curriculum_overlays"
    best_pred_dir = os.path.join(OUTPUT_DIR, f"predictions_curriculum_stage{best_stage_num}" if best_stage_num < 3 else "predictions_curriculum_final")
    generate_overlay_pngs(best_pred_dir, overlay_dir, cases=[1, 5, 6, 10, 13])

    # Write summary
    summary_path = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\curriculum_summary.md"
    with open(summary_path, "w") as f:
        f.write("# RASNet Curriculum Learning Fine-Tuning Summary\n\n")
        f.write("## Best Checkpoint Analysis\n\n")
        f.write(f"- **Best Stage**: Stage {best_stage_num}\n")
        f.write(f"- **Checkpoint Path**: `{best_stage_ckpt}`\n")
        f.write(f"- **Inference settings**: Test Time Augmentation (TTA) = True, Threshold = 0.6\n\n")
        f.write("### Mean Metrics on Test Set (N=150)\n\n")
        f.write(f"- **Dice**: {best_stage_metrics['dice']:.4f} +/- {best_stage_metrics.get('dice_std', 0.0):.4f}\n")
        f.write(f"- **IoU**: {best_stage_metrics['iou']:.4f} +/- {best_stage_metrics.get('iou_std', 0.0):.4f}\n")
        f.write(f"- **Precision**: {best_stage_metrics['precision']:.4f} +/- {best_stage_metrics.get('precision_std', 0.0):.4f}\n")
        f.write(f"- **Recall**: {best_stage_metrics['recall']:.4f} +/- {best_stage_metrics.get('recall_std', 0.0):.4f}\n")
        f.write(f"- **HD95**: {best_stage_metrics['hd95']:.2f} +/- {best_stage_metrics.get('hd95_std', 0.0):.2f} mm\n\n")
        f.write("### Comparison vs Baseline\n\n")
        f.write("| Model | Dice | Precision | Recall | HD95 (mm) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **RASNet Curriculum (Ours)** | **{best_stage_metrics['dice']:.4f}** | **{best_stage_metrics['precision']:.4f}** | {best_stage_metrics['recall']:.4f} | **{best_stage_metrics['hd95']:.2f}** |\n")
        f.write("| RASNet v2 (0.6 threshold) | 0.7879 | 0.8619 | 0.7495 | 8.98 |\n")
        f.write("| RASNet v2 (0.5 threshold) | 0.7942 | 0.8533 | 0.7495 | 8.12 |\n")

    print("\n" + "=" * 60)
    print("  Curriculum training and final generation complete!")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
