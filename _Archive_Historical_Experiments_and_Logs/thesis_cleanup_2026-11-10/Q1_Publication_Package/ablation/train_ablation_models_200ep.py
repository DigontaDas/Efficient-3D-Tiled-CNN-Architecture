#!/usr/bin/env python3
"""
train_ablation_models_200ep.py — Full Authentic 200-Epoch Training for Ablation Models
========================================================================================
Trains the isolated intermediate ablation architectures from scratch on ImageCAS (690 train):
  1. Variant 'attngate_only' (Step 2 in Table 3):
     - SegResNet backbone + 3-Level AttentionGate3D
     - Deep Supervision: Disabled
     - Loss: Standard DiceCELoss (plain loss, no focal weighting, no stenosis modulation)
     - Checkpoint: checkpoints/ablation_attngate_best.pth

  2. Variant 'deepsup_plain' (Step 3 in Table 3):
     - SegResNet backbone + 3-Level AttentionGate3D + Deep Supervision (aux2, aux3)
     - Deep Supervision: Enabled (main + 0.4*aux2 + 0.2*aux3)
     - Loss: Standard DiceCELoss on all heads
     - Checkpoint: checkpoints/ablation_deepsup_best.pth

Hardware Acceleration (RTX 4080 SUPER 16GB):
  - TF32 math on Ampere/Ada Lovelace tensor cores
  - PyTorch Automatic Mixed Precision (AMP FP16)
  - PersistentDataset with shared preprocessing cache (cache/segresnet)
  - Pin memory + multi-worker prefetching
  - Sliding-window batch size 16 for validation
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import monai
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import Dataset, PersistentDataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged,
    Spacingd, RandCropByPosNegLabeld, ToTensord, CropForegroundd,
    RandFlipd, Activations, AsDiscrete
)

# Shared paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "Phase3_Local_Integration"))
from rasnet_model import RASNet
import dataset_paths

# Hardware performance flags for RTX 4080 SUPER
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_variant(variant: str, args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "matched_200ep_benchmark"))
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    cache_dir = os.path.join(base_dir, "cache", "segresnet")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    if variant == "attngate_only":
        best_ckpt_path = os.path.join(ckpt_dir, "ablation_attngate_best.pth")
        last_ckpt_path = os.path.join(ckpt_dir, "ablation_attngate_last.pth")
        log_file = os.path.join(log_dir, "ablation_attngate_200ep_log.csv")
        model_desc = "SegResNet + AttentionGate3D Only (Plain DiceCELoss, No Deep Sup)"
        use_deepsup = False
    elif variant == "deepsup_plain":
        best_ckpt_path = os.path.join(ckpt_dir, "ablation_deepsup_best.pth")
        last_ckpt_path = os.path.join(ckpt_dir, "ablation_deepsup_last.pth")
        log_file = os.path.join(log_dir, "ablation_deepsup_200ep_log.csv")
        model_desc = "SegResNet + AttentionGate3D + Deep Supervision (Plain DiceCELoss)"
        use_deepsup = True
    else:
        raise ValueError(f"Unknown variant: {variant}")

    print("=" * 80)
    print(f"[*] TRAINING ABLATION MODEL: {variant.upper()}")
    print(f"[*] Description: {model_desc}")
    print(f"[*] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    splits_file = os.path.join(REPO_ROOT, "Phase3_Local_Integration", "splits_final.json")
    with open(splits_file) as f:
        splits = json.load(f)
    train_ids = splits["train"]
    val_ids = splits["val"]

    if args.dry_run:
        train_ids = train_ids[:4]
        val_ids = val_ids[:2]
        total_epochs = 1
        print(f"[*] DRY RUN: {len(train_ids)} train, {len(val_ids)} val cases, 1 epoch.")
    else:
        total_epochs = args.epochs

    train_files = []
    for cid in train_ids:
        img = dataset_paths.find_image(cid)
        lbl = dataset_paths.find_label(cid)
        if img and lbl:
            train_files.append({"image": img, "label": lbl})

    val_files = []
    for cid in val_ids:
        img = dataset_paths.find_image(cid)
        lbl = dataset_paths.find_label(cid)
        if img and lbl:
            val_files.append({"image": img, "label": lbl})

    print(f"[*] Dataset: {len(train_files)} Train Volumes, {len(val_files)} Val Volumes")

    # Identical pre-processing pipeline matching matched 200ep benchmark
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(96, 96, 96), pos=2, neg=1, num_samples=1
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        ToTensord(keys=["image", "label"]),
        ConvertToPlainTensor()
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image", "label"]),
        ConvertToPlainTensor()
    ])

    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=cache_dir)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers if not args.dry_run else 0,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 and not args.dry_run else None,
        persistent_workers=True if args.num_workers > 0 and not args.dry_run else False
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Initialize model
    model = RASNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    ).to(device)

    # Plain Dice+CE Loss for isolated architectural benchmark
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = AsDiscrete(to_onehot=2)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_val_dice = -1.0
    start_total_time = time.time()
    history = []
    start_epoch = 1

    # Auto-resume check
    if os.path.exists(last_ckpt_path) and os.path.exists(log_file) and not args.dry_run:
        try:
            existing_df = pd.read_csv(log_file)
            if len(existing_df) > 0 and len(existing_df) < total_epochs:
                history = existing_df.to_dict("records")
                last_logged_epoch = int(history[-1]["epoch"])
                start_epoch = last_logged_epoch + 1
                model.load_state_dict(torch.load(last_ckpt_path, map_location=device))
                for ep in range(1, start_epoch):
                    scheduler.step()
                val_dices = [float(r["val_dice"]) for r in history if not np.isnan(r.get("val_dice", np.nan))]
                best_val_dice = max(val_dices) if val_dices else -1.0
                cum_hrs = float(history[-1].get("cumulative_time_hours", 0.0))
                start_total_time = time.time() - (cum_hrs * 3600.0)
                print(f"[*] AUTO-RESUME: Loaded {last_ckpt_path}, resuming from Epoch {start_epoch} -> {total_epochs}")
            elif len(existing_df) >= total_epochs:
                print(f"[*] Variant '{variant}' already completed {len(existing_df)} epochs. Skipping.")
                return
        except Exception as exc:
            print(f"[!] Warning: Auto-resume failed ({exc}). Starting from scratch.")
            history = []
            start_epoch = 1

    print(f"[*] Model Parameters: {sum(p.numel() for p in model.parameters()):,} (Scratch / Random Init)")
    print(f"[*] Schedule: Epochs {start_epoch} to {total_epochs} | lr={args.lr}")

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch in train_loader:
            step += 1
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out = model(images)
                if use_deepsup:
                    main_out, ds_outputs = out
                    loss_main = criterion(main_out, labels)
                    aux2 = torch.nn.functional.interpolate(ds_outputs[0], size=labels.shape[2:], mode="trilinear", align_corners=True)
                    aux3 = torch.nn.functional.interpolate(ds_outputs[1], size=labels.shape[2:], mode="trilinear", align_corners=True)
                    total_loss = loss_main + 0.4 * criterion(aux2, labels) + 0.2 * criterion(aux3, labels)
                else:
                    main_out = out[0] if isinstance(out, (tuple, list)) else out
                    total_loss = criterion(main_out, labels)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            if args.dry_run and step >= 2:
                break

        train_loss = epoch_loss / max(step, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Validation
        val_dice = np.nan
        val_loss = np.nan
        if epoch % args.val_interval == 0 or epoch % 50 == 0 or epoch == total_epochs or args.dry_run:
            model.eval()
            val_epoch_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_img = v_batch["image"].to(device)
                    v_lbl = v_batch["label"].to(device)

                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        val_logits = sliding_window_inference(
                            v_img, (96, 96, 96), sw_batch_size=args.sw_batch_size,
                            predictor=model, overlap=0.25
                        )
                        if isinstance(val_logits, (tuple, list)):
                            val_logits = val_logits[0]
                        v_loss = criterion(val_logits, v_lbl)
                        val_epoch_loss += v_loss.item()
                        val_steps += 1

                        val_p = [post_pred(i) for i in decollate_batch(val_logits)]
                        val_l = [post_label(i) for i in decollate_batch(v_lbl)]
                        dice_metric(y_pred=val_p, y=val_l)

                    if args.dry_run and val_steps >= 2:
                        break

            val_dice = float(dice_metric.aggregate().item())
            dice_metric.reset()
            val_loss = val_epoch_loss / max(val_steps, 1)

            if val_dice > best_val_dice and not args.dry_run:
                best_val_dice = val_dice
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  [>>> BEST VAL DICE] Epoch {epoch:03d}: {val_dice:.4f} -> Saved to {best_ckpt_path}")

        ep_duration = time.time() - t0
        cum_hours = (time.time() - start_total_time) / 3600.0

        if not args.dry_run:
            torch.save(model.state_dict(), last_ckpt_path)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5) if not np.isnan(val_loss) else "",
            "val_dice": round(val_dice, 5) if not np.isnan(val_dice) else "",
            "learning_rate": round(current_lr, 7),
            "epoch_time_seconds": round(ep_duration, 2),
            "cumulative_time_hours": round(cum_hours, 4)
        })
        pd.DataFrame(history).to_csv(log_file, index=False)

        val_str = f" | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}" if not np.isnan(val_dice) else ""
        if epoch % 5 == 0 or epoch == 1 or epoch == total_epochs or not np.isnan(val_dice):
            print(f"[{variant}] Epoch {epoch:03d}/{total_epochs:03d} | Train Loss: {train_loss:.4f}{val_str} | Time: {ep_duration:.1f}s | Cum: {cum_hours:.2f}h")

    print(f"\n[OK] Training completed for {variant}. Best Val Dice: {best_val_dice:.4f}")
    print(f"[OK] Checkpoints: {best_ckpt_path} | Logs: {log_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Ablation Models 200-Epoch Matched Training")
    parser.add_argument("--variant", type=str, default="all", choices=["attngate_only", "deepsup_plain", "all"],
                        help="Which ablation variant to train (default: 'all' runs both sequentially)")
    parser.add_argument("--dry-run", action="store_true", help="Quick verification step")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--num-workers", type=int, default=10, help="DataLoader workers (default: 10)")
    parser.add_argument("--epochs", type=int, default=200, help="Target epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--val-interval", type=int, default=20, help="Val interval in epochs")
    parser.add_argument("--sw-batch-size", type=int, default=16, help="Sliding window batch size")
    args = parser.parse_args()

    if args.variant == "all":
        variants = ["attngate_only", "deepsup_plain"]
    else:
        variants = [args.variant]

    for v in variants:
        train_single_variant(v, args)


if __name__ == "__main__":
    main()
