#!/usr/bin/env python3
"""
train_segresnet_200ep.py — SegResNet 200-Epoch Matched Benchmark Training
========================================================================
- Model: MONAI SegResNet (spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)
- Schedule: Genuine 200 epochs with CosineAnnealingLR (T_max=200, eta_min=1e-6)
- Loss: DiceFocalLoss (lambda_dice=1.0, lambda_focal=2.0)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Hardware: Optimized for RTX 4080 SUPER 16GB (AMP FP16, TF32, batch_size=4)
- Telemetry: Output per-epoch metrics to logs/segresnet_200ep_log.csv
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
from monai.networks.nets import SegResNet
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import Dataset, PersistentDataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Spacingd, RandCropByPosNegLabeld, ToTensord, CropForegroundd,
    RandFlipd, Activations, AsDiscrete
)

# Shared dataset resolver
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Phase3_Local_Integration")))
import dataset_paths

# Hardware Acceleration Flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="SegResNet 200-Epoch Matched Benchmark Training")
    parser.add_argument("--dry-run", action="store_true", help="Quick 1-step verification run")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--epochs", type=int, default=200, help="Target epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-interval", type=int, default=10, help="Validation interval in epochs (default: 10)")
    parser.add_argument("--sw-batch-size", type=int, default=8, help="Sliding window batch size for validation (default: 8)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    cache_dir = os.path.join(base_dir, "cache", "segresnet")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    best_ckpt_path = os.path.join(ckpt_dir, "segresnet_best.pt")
    last_ckpt_path = os.path.join(ckpt_dir, "segresnet_last.pt")
    log_file = os.path.join(log_dir, "segresnet_200ep_log.csv")

    # Load Splits
    splits_file = os.path.join(base_dir, "..", "..", "Phase3_Local_Integration", "splits_final.json")
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
        print(f"[*] FULL RUN: {len(train_ids)} train, {len(val_ids)} val cases, {total_epochs} epochs.")

    train_files = [{"image": dataset_paths.find_image(cid), "label": dataset_paths.find_label(cid)} for cid in train_ids]
    val_files = [{"image": dataset_paths.find_image(cid), "label": dataset_paths.find_label(cid)} for cid in val_ids]

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(96, 96, 96), pos=2, neg=1, num_samples=2
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ToTensord(keys=["image", "label"]),
        ConvertToPlainTensor()
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image", "label"]),
        ConvertToPlainTensor()
    ])

    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=cache_dir)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4 if not args.dry_run else 0, pin_memory=True
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Initialize SegResNet
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    ).to(device)

    loss_fn = DiceFocalLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=1.0,
        lambda_focal=2.0
    )

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

    # Check for existing checkpoint to resume
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
                print(f"[*] AUTO-RESUME: Successfully loaded {last_ckpt_path}")
                print(f"[*] Resuming from Epoch {start_epoch} -> {total_epochs} (Best Val Dice so far: {best_val_dice:.4f})")
                print(f"[*] LR fast-forwarded to: {optimizer.param_groups[0]['lr']:.2e}")
            elif len(existing_df) >= total_epochs:
                print(f"[*] SegResNet already completed {len(existing_df)} epochs. Skipping to complete.")
                return
        except Exception as e:
            print(f"[!] Warning: Auto-resume check failed ({e}). Starting fresh.")
            history = []
            start_epoch = 1

    print(f"[*] Model SegResNet Initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"[*] Training schedule: {start_epoch} to {total_epochs} epochs | CosineAnnealingLR(T_max={total_epochs})")
    print(f"[*] Speed Optimizations: val_interval={args.val_interval} | sw_batch_size={args.sw_batch_size}")

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
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if args.dry_run and step >= 2:
                break

        train_loss = epoch_loss / max(step, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Validation
        val_dice = np.nan
        val_loss = np.nan
        if epoch % args.val_interval == 0 or epoch == total_epochs or args.dry_run:
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
                        v_loss = loss_fn(val_logits, v_lbl)

                    val_epoch_loss += v_loss.item()
                    val_steps += 1

                    v_outputs = [post_pred(i) for i in decollate_batch(val_logits)]
                    v_labels = [post_label(i) for i in decollate_batch(v_lbl)]
                    dice_metric(y_pred=v_outputs, y=v_labels)

                    if args.dry_run and val_steps >= 1:
                        break

            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            val_loss = val_epoch_loss / max(val_steps, 1)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), best_ckpt_path)

        # Save Last Checkpoint Every Epoch
        torch.save(model.state_dict(), last_ckpt_path)

        # Save Checkpoint Every 50 Epochs
        if epoch % 50 == 0:
            periodic_ckpt_path = os.path.join(ckpt_dir, f"segresnet_epoch_{epoch}.pt")
            torch.save(model.state_dict(), periodic_ckpt_path)
            print(f"[*] Checkpoint saved at Epoch {epoch}: {periodic_ckpt_path}")

        epoch_time = time.time() - t0
        cum_hours = (time.time() - start_total_time) / 3600.0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
            "cumulative_time_hours": cum_hours
        }
        history.append(row)
        pd.DataFrame(history).to_csv(log_file, index=False)

        val_str = f"Val Dice: {val_dice:.4f}" if not np.isnan(val_dice) else "Val: (skipped)"
        print(f"Epoch {epoch:03d}/{total_epochs} | Train Loss: {train_loss:.4f} | {val_str} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

    print("==================================================")
    print(f"SegResNet Complete. Best Val Dice: {best_val_dice:.4f}")
    print(f"Best Checkpoint: {best_ckpt_path}")
    print(f"Log: {log_file}")
    print("==================================================")

if __name__ == "__main__":
    main()
